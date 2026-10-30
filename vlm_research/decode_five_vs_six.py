"""One incremental Decode-only condition; historical Full/five-layer are read-only."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.run import prepared
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import atomic_json, tensor_hashes
from paper_rq1_early_joint_attention.engine import generate, JointAttention
from paper_rq1_early_joint_attention.protocol import OUT as OLD, ROOT, GPUS, MODEL_PATH, CONDITIONS
from paper_rq1_early_joint_attention.run import verify as verify_old, checkpoint_valid, preflight
from paper_rq1_early_joint_attention.statistics import bootstrap, mcnemar_exact
from vlm_common import load_model, gpu_memory

OUT = ROOT / 'results/decode_five_vs_six'
NEW = dict(name='L0-L5 Decode-only', layers=list(range(6)), mode='Decode-only')
BASE = '7c2475717ffaacc1f351978bf59eeaf0197c3109'
ENV_KEYS = ['python', 'torch', 'transformers', 'cuda_runtime', 'cudnn', 'gpu',
            'model_revision', 'model_path', 'parameter_dtypes', 'cuda_visible_devices']


def read(path):
    return json.loads(path.read_text())


def manifest():
    value = read(OLD / 'sample_manifest.json')
    assert len(value['samples']) == 1859
    assert len({(s['task'], s['question_id']) for s in value['samples']}) == 1859
    return value


def old_record(sample, key):
    path = OLD / f"generation/shard{sample['rank']}/checkpoints/{sample['task']}__{sample['question_id']}/{key}.json"
    binding = digest(read(OLD / 'execution_manifest.json'))
    if not checkpoint_valid(path, binding, sample, key):
        raise ValueError('Missing historical paired result')
    return read(path)


def code_hashes():
    return {name: sha256(ROOT / name) for name in ['decode_five_vs_six.py', 'test_decode_five_vs_six.py']}


def cpu():
    OUT.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in
        ['test_early_joint_attention', 'test_early_joint_analysis', 'test_decode_five_vs_six'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    value = dict(passed=result.wasSuccessful(), tests=result.testsRun, output=stream.getvalue(), source=code_hashes())
    freeze_json(OUT / 'cpu_validation.json', value)
    print(stream.getvalue(), flush=True)
    if not result.wasSuccessful():
        raise AssertionError('CPU tests failed')


def freeze():
    baseline = verify_old()
    check = read(OUT / 'cpu_validation.json')
    assert check['passed'] and check['source'] == code_hashes()
    assert subprocess.check_output(['git', 'merge-base', 'HEAD', BASE], cwd=ROOT, text=True).strip() == BASE
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *code_hashes()], cwd=ROOT, text=True).strip()
    samples = manifest()['samples']
    for sample in samples:
        for key in ['S0', 'S3']:
            old_record(sample, key)
    freeze_json(OUT / 'execution_manifest.json', dict(base_commit=BASE,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source=code_hashes(), historical_binding=digest(baseline),
        sample_manifest_sha256=sha256(OLD / 'sample_manifest.json'),
        cpu_sha256=sha256(OUT / 'cpu_validation.json'), condition=NEW, samples=1859,
        generation=dict(max_new_tokens=32, eos_token_id=2, greedy=True, own_cache=True),
        comparisons=['five-Full', 'six-Full', 'six-five'], bootstrap_repetitions=200000, seed=20261031,
        bootstrap='Paired whole-image resampling, question-weighted; ordinary95 and Bonferroni family3.',
        noninferiority_pp=-1, criterion='family3 lower bound strictly greater than -1pp',
        recommendation='Prefer six only if NI vs Full and vs five, direct six-five CI includes0 or is positive, '
          'and no excess EOS/truncation/repetition flags; otherwise retain five. No speed claim.'))
    print('Frozen1859 samples; only new six-layer Decode-only', flush=True)


def verify():
    config = read(OUT / 'execution_manifest.json')
    assert config['source'] == code_hashes() and config['condition'] == NEW
    assert config['historical_binding'] == digest(verify_old())
    assert config['sample_manifest_sha256'] == sha256(OLD / 'sample_manifest.json')
    assert config['cpu_sha256'] == sha256(OUT / 'cpu_validation.json')
    return config


def validate_smoke(model, processor, rank):
    checks = []
    prior = read(OLD / f'smoke/shard{rank}/validation.json')
    assert sha256(prior['raw_path']) == prior['raw_sha256']
    with np.load(prior['raw_path']) as arrays:
        historical_logits = {k: arrays[k].copy() for k in arrays.files}
    for sample in manifest()['smoke']:
        prompt = prepared(model, processor, sample)
        before = tensor_hashes(prompt)
        full, logits = generate(model, prompt, CONDITIONS['S0'], collect=True, audit=True)
        previous = next(c for c in prior['checks'] if c['question_id'] == sample['question_id'])
        assert full['token_ids'] == previous['full']['token_ids']
        difference = float(np.max(np.abs(torch.stack(logits).numpy() - historical_logits[sample['question_id']])))
        assert difference == 0, ('Historical Full logits differ', difference)
        five, _ = generate(model, prompt, CONDITIONS['S3'])
        assert five['token_ids'] == next(c for c in previous['conditions'] if c['condition'] == 'S3')['generation']['token_ids']
        with torch.inference_mode():
            native = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
                                    use_cache=True, eos_token_id=2)[0, prompt['input_ids'].shape[1]:].tolist()
        assert native == full['token_ids']
        noop, no_logits = generate(model, prompt, NEW, noop=True, collect=True)
        assert noop['token_ids'] == full['token_ids'] and len(no_logits) == len(logits)
        assert all(torch.equal(a, b) for a, b in zip(no_logits, logits))
        counts = [0] * 32
        handles = []
        try:
            for k, block in enumerate(model_layers(model)):
                def observe(module, args, out, k=k):
                    counts[k] += 1
                    assert torch.isfinite(out).all()
                handles.append(block.mlp.register_forward_hook(observe))
            six, six_logits = generate(model, prompt, NEW, collect=True, audit=True)
        finally:
            for handle in handles:
                handle.remove()
        assert all(c == six['length'] for c in counts)
        assert torch.equal(six_logits[0], logits[0])
        assert six['initial_cache_sha256'] == full['initial_cache_sha256']
        assert six['token_ids'][0] == full['token_ids'][0]
        assert len(six['events']) == 6 * six['length']
        assert {e['layer'] for e in six['events']} == set(range(6))
        assert all(e['active'] == e['decode'] for e in six['events'])
        with JointAttention(model, NEW['layers'], NEW['mode'], prompt['input_ids'].shape[1]):
            with torch.inference_mode():
                native_six = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
                    use_cache=True, eos_token_id=2)[0, prompt['input_ids'].shape[1]:].tolist()
        assert native_six == six['token_ids'] and tensor_hashes(prompt) == before
        checks.append(dict(question_id=sample['question_id'], historical_max_logit_diff=difference,
            historical_full_five_tokens_exact=True, noop_all_logits_exact=True, native_hf_tokens_exact=True,
            native_hf_six_tokens_exact=True, prefill_logits_and_cache_exact=True,
            all32_mlp_native=True, six=six))
    # Also replay two formal pairs on their original physical GPU, not just smoke images.
    replay = []
    for sample in [s for s in manifest()['samples'] if s['rank'] == rank][:2]:
        prompt = prepared(model, processor, sample)
        for key in ['S0', 'S3']:
            result, _ = generate(model, prompt, CONDITIONS[key])
            assert result['token_ids'] == old_record(sample, key)['generation']['token_ids']
            replay.append(dict(question_id=sample['question_id'], condition=key, token_ids_exact=True))
    return dict(passed=True, checks=checks, formal_baseline_replay=replay)


def gate():
    config = verify()
    records = [read(OUT / f'shard{k}/smoke.json') for k in range(3)]
    for record in records:
        assert record['binding'] == digest(config) and record['validation']['passed']
    shared = [k for k in ENV_KEYS if k != 'cuda_visible_devices']
    assert len({digest({k: r['environment'][k] for k in shared}) for r in records}) == 1
    freeze_json(OUT / 'gpu_gate.json', dict(passed=True, binding=digest(config),
        smoke_hashes=[sha256(OUT / f'shard{k}/smoke.json') for k in range(3)]))
    print('Three GPU smoke/baseline reuse gates passed', flush=True)


def worker(rank):
    config = verify()
    binding = digest(config)
    folder = OUT / f'shard{rank}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pre = preflight(rank)
    model, processor, env = load_model(MODEL_PATH)
    previous = read(next((OLD / f'generation/shard{rank}').glob('environment_*.json')))
    assert {k: env[k] for k in ENV_KEYS} == {k: previous[k] for k in ENV_KEYS}
    env.update(physical_gpu=rank + 1, preflight=pre, code_commit=config['code_commit'])
    footprint = hook_footprint(model)
    versions = {k: p._version for k, p in model.named_parameters()}
    if not (folder / 'smoke.json').exists():
        validation = validate_smoke(model, processor, rank)
        assert footprint == hook_footprint(model)
        atomic_json(folder / 'smoke.json', dict(binding=binding, environment=env, validation=validation))
    else:
        assert read(folder / 'smoke.json')['binding'] == binding
    print('Smoke ready', rank, flush=True)
    deadline = time.monotonic() + 900
    while not (OUT / 'gpu_gate.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Main GPU smoke acceptance gate absent')
        time.sleep(1)
    assert read(OUT / 'gpu_gate.json')['binding'] == binding
    selected = [s for s in manifest()['samples'] if s['rank'] == rank]
    start = time.perf_counter()
    resumed = 0
    for i, sample in enumerate(selected):
        path = folder / 'checkpoints' / f"{sample['task']}__{sample['question_id']}.json"
        if checkpoint_valid(path, binding, sample, 'six'):
            resumed += 1
            continue
        prompt = prepared(model, processor, sample)
        inputs = tensor_hashes(prompt)
        tick = time.perf_counter()
        generation, _ = generate(model, prompt, NEW)
        assert generation['token_ids'][0] == old_record(sample, 'S0')['generation']['token_ids'][0]
        text = processor.tokenizer.decode(generation['token_ids'], skip_special_tokens=True).strip()
        generation.update(text=text, prediction=gqa_prediction(text), score=int(gqa_prediction(text) == sample['answer']))
        assert tensor_hashes(prompt) == inputs and hook_footprint(model) == footprint
        assert versions == {k: p._version for k, p in model.named_parameters()}
        atomic_json(path, dict(binding=binding, sample_sha256=digest(sample), condition='six',
            generation=generation, result_sha256=digest(generation), seconds=time.perf_counter() - tick))
        if (i + 1) % 50 == 0 or i + 1 == len(selected):
            print(json.dumps(dict(rank=rank, done=i + 1, total=len(selected), seconds=time.perf_counter() - start)), flush=True)
    verify()
    completion = dict(status='complete', binding=binding, samples=len(selected), resumed=resumed,
        generation_seconds=time.perf_counter() - start, hooks_clean=hook_footprint(model) == footprint,
        parameters_unchanged=versions == {k: p._version for k, p in model.named_parameters()}, memory=gpu_memory())
    if not (folder / 'completion.json').exists():
        atomic_json(folder / 'completion.json', completion)
    print('Complete', rank, completion, flush=True)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from paper_rq1_early_joint_attention.analyze import anomaly
    config = verify()
    binding = digest(config)
    for rank in range(3):
        assert read(OUT / f'shard{rank}/completion.json')['status'] == 'complete'
        assert read(OUT / f'shard{rank}/completion.json')['binding'] == binding
    rows, scores, identities = [], [], []
    eos = [0, 0, 0]; truncated = [0, 0, 0]; abnormal = [0, 0, 0]
    for sample in manifest()['samples']:
        path = OUT / f"shard{sample['rank']}/checkpoints/{sample['task']}__{sample['question_id']}.json"
        assert checkpoint_valid(path, binding, sample, 'six')
        gens = [old_record(sample, 'S0')['generation'], old_record(sample, 'S3')['generation'], read(path)['generation']]
        record = {k: sample[k] for k in ['task', 'question_id', 'image_id', 'answer']}
        for k, (name, gen) in enumerate(zip(['Full', 'five', 'six'], gens)):
            assert gen['score'] == int(gqa_prediction(gen['text']) == sample['answer'])
            assert gen['prediction'] == gqa_prediction(gen['text'])
            ids = gen['token_ids']
            assert 0 < len(ids) <= 32 and (2 not in ids[:-1])
            assert gen['eos'] == (ids[-1] == 2) and gen['truncated'] == (len(ids) == 32 and ids[-1] != 2)
            assert gen['edits'] == (0 if k == 0 else (5 if k == 1 else 6) * (len(ids) - 1))
            flags = anomaly(ids, gen['text'], gen['eos'])
            eos[k] += int(gen['eos']); truncated[k] += int(gen['truncated'])
            abnormal[k] += int(flags['empty_answer'] or flags['repetition_flag'])
            record.update({name + '_text': gen['text'], name + '_score': gen['score'],
                name + '_tokens': json.dumps(ids), name + '_eos': gen['eos'], name + '_abnormal': bool(flags['empty_answer'] or flags['repetition_flag'])})
        assert gens[0]['token_ids'][0] == gens[1]['token_ids'][0] == gens[2]['token_ids'][0]
        scores.append([g['score'] for g in gens]); identities.append(sample['image_identity']); rows.append(record)
    assert sum(len(list((OUT / f'shard{rank}/checkpoints').glob('*.json'))) for rank in range(3)) == 1859
    values = np.asarray(scores, dtype=float)
    deltas = np.column_stack([values[:, 1] - values[:, 0], values[:, 2] - values[:, 0], values[:, 2] - values[:, 1]])
    stats = bootstrap(deltas, identities, [3, 3, 3])
    contrasts = []
    for j, (name, a, b) in enumerate([('five-Full', 1, 0), ('six-Full', 2, 0), ('six-five', 2, 1)]):
        harm = int(((values[:, b] == 1) & (values[:, a] == 0)).sum())
        gain = int(((values[:, b] == 0) & (values[:, a] == 1)).sum())
        s = stats[j]
        contrasts.append(dict(comparison=name, n=1859, images=len(set(identities)), delta_pp=100 * s['mean'],
            correct_to_wrong=harm, wrong_to_correct=gain, ci95_low_pp=100 * s['ci95_low'], ci95_high_pp=100 * s['ci95_high'],
            family3_low_pp=100 * s['family95_low'], family3_high_pp=100 * s['family95_high'],
            noninferiority=bool(s['family95_low'] > -.01), mcnemar_exact_p=mcnemar_exact(harm, gain)))
    summary = [dict(condition=name, n=1859, correct=int(values[:, k].sum()), accuracy_pct=100 * float(values[:, k].mean()),
        full_correct_to_wrong=int(((values[:, 0] == 1) & (values[:, k] == 0)).sum()),
        full_wrong_to_correct=int(((values[:, 0] == 0) & (values[:, k] == 1)).sum()),
        eos=eos[k], truncated=truncated[k], abnormal=abnormal[k]) for k, name in enumerate(['Full', 'L0-L4 Decode-only', 'L0-L5 Decode-only'])]
    recommend_six = (contrasts[1]['noninferiority'] and contrasts[2]['noninferiority']
        and contrasts[2]['ci95_high_pp'] >= 0 and eos[2] >= eos[1] and truncated[2] <= truncated[1] and abnormal[2] <= abnormal[1])
    write_csv('accuracy.csv', summary); write_csv('paired_differences.csv', contrasts); write_csv('per_question.csv', rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1, 1.5]})
    colors = ['#59636e', '#16887d', '#c24a49']
    accuracy = [r['accuracy_pct'] for r in summary]
    axes[0].bar(range(3), accuracy, color=colors, width=.58)
    axes[0].set_xticks(range(3), ['Full', 'L0-L4', 'L0-L5'])
    axes[0].set_ylim(min(accuracy) - 1.2, max(accuracy) + .65)
    axes[0].set_ylabel('Exact-answer accuracy (%)')
    axes[0].set_title('Same frozen 1,859 questions')
    for k, row in enumerate(summary):
        axes[0].text(k, accuracy[k] + .09, f"{accuracy[k]:.2f}%\n{row['correct']}/1859", ha='center', fontsize=9)
    for k, row in enumerate(contrasts):
        axes[1].errorbar(row['delta_pp'], k, xerr=[[row['delta_pp'] - row['ci95_low_pp']], [row['ci95_high_pp'] - row['delta_pp']]],
            fmt='o', color=colors[k], capsize=4, label=None)
        axes[1].text(row['delta_pp'], k + .23, f"{row['delta_pp']:+.3f} pp; harm/fix {row['correct_to_wrong']}/{row['wrong_to_correct']}", ha='center', fontsize=8)
    axes[1].set_yticks(range(3), ['L0-L4 - Full', 'L0-L5 - Full', 'L0-L5 - L0-L4'])
    axes[1].axvline(0, color='grey', linewidth=.8)
    axes[1].axvline(-1, color='#ba4949', linestyle='--', linewidth=1, label='NI boundary: -1 pp')
    axes[1].set_ylim(2.65, -.6); axes[1].set_xlabel('Paired accuracy difference (pp), image-cluster 95% CI')
    axes[1].set_title('Only six-layer condition newly generated'); axes[1].legend(loc='lower left', fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x' if ax is axes[1] else 'y', alpha=.15); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.png', dpi=220); fig.savefig(OUT / 'comparison.pdf'); plt.close(fig)
    result = dict(summary=summary, comparisons=contrasts, recommend='six' if recommend_six else 'five',
        binding=binding, checkpoints=1859, image_clusters=len(set(identities)),
        worker_seconds=[read(OUT / f'shard{k}/completion.json')['generation_seconds'] for k in range(3)])
    freeze_json(OUT / 'result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cpu', 'freeze', 'worker', 'gate', 'analyze'])
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    try:
        if args.stage == 'worker':
            worker(args.rank)
        else:
            globals()[args.stage]()
    except BaseException:
        if args.stage == 'worker':
            freeze_json(OUT / f'failures/shard{args.rank}_{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
        raise

import argparse
from collections import Counter
import fcntl
from functools import lru_cache
import io
import json
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import torch
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.run import prepared, atomic_json, tensor_hashes
from paper_rq1_single_layer_gqa6.protocol import OUTPUT as ORIGINAL, TASKS
from paper_rq1_early_joint_attention.engine import JointAttention, generate
from paper_rq1_early_joint_attention.protocol import ROOT, MODEL_PATH, GPUS
from paper_rq1_early_joint_attention.run import verify as verify_engine, preflight, checkpoint_valid
from decode_five_vs_six import ENV_KEYS
from vlm_common import load_model, gpu_memory

OUT = ROOT / 'results/figure3_decode_window_comparison'
ORIGINAL_COMMIT = '221cdde5'
CONDITIONS = {'Full': dict(layers=[], mode='Full')}
CONDITIONS.update({f'W{k}': dict(layers=list(range(start, start + 6)), mode='Decode-only')
    for k, start in enumerate([0, 6, 12, 18, 26])})
SEED = 20261031


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def cohort():
    value = read(ORIGINAL / 'sample_manifest.json')
    samples = value['samples']
    assert len(samples) == 3000 and Counter(s['task'] for s in samples) == Counter({t: 500 for t in TASKS})
    assert len({s['question_id'] for s in samples}) == 3000
    return samples


def rank_for(sample):
    # Keep identical images, including any duplicate image IDs, on one physical GPU.
    return int(digest(['figure3-worker', SEED, sample['image_sha256']]), 16) % 3


def sources():
    names = ['figure3_decode_windows/' + n for n in ['__init__.py', 'run.py', 'analyze.py']]
    names += ['test_figure3_decode_windows.py', 'decode_five_vs_six.py',
              'paper_rq1_early_joint_attention/statistics.py', 'paper_rq1_single_layer_gqa6/run.py']
    return {n: sha256(ROOT / n) for n in names}


def cpu():
    OUT.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ['test_early_joint_attention', 'test_early_joint_analysis', 'test_decode_five_vs_six', 'test_figure3_decode_windows'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    freeze_json(OUT / 'cpu_validation.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        source=sources(), output=stream.getvalue()))
    print(stream.getvalue(), flush=True)
    if not result.wasSuccessful():
        raise AssertionError('CPU acceptance failed')


def freeze():
    original_path = 'vlm_research/results/paper_rq1_single_layer_gqa6/sample_manifest.json'
    blob = subprocess.check_output(['git', 'rev-parse', f'{ORIGINAL_COMMIT}:{original_path}'], cwd=ROOT, text=True).strip()
    assert blob == subprocess.check_output(['git', 'hash-object', ORIGINAL / 'sample_manifest.json'], cwd=ROOT, text=True).strip()
    tested = read(OUT / 'cpu_validation.json')
    assert tested['passed'] and tested['source'] == sources()
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *sources()], cwd=ROOT, text=True).strip()
    old = verify_engine()
    scorer_path = ORIGINAL / 'scorer_validation.json'
    for p, h in read(scorer_path)['source_sha256'].items():
        assert sha256(p) == h
    assignments = [dict(task=s['task'], question_id=s['question_id'], image_id=s['image_id'],
        image_sha256=s['image_sha256'], sample_sha256=digest(s), worker=rank_for(s)) for s in cohort()]
    freeze_json(OUT / 'assignments.json', assignments)
    freeze_json(OUT / 'execution_manifest.json', dict(
        original_commit=ORIGINAL_COMMIT, original_manifest_git_blob=blob,
        original_manifest_sha256=sha256(ORIGINAL / 'sample_manifest.json'),
        original_scorer_sha256=sha256(scorer_path), assignments_sha256=sha256(OUT / 'assignments.json'),
        engine_binding=digest(old), source=sources(), cpu_sha256=sha256(OUT / 'cpu_validation.json'),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        model_revision=old['protocol']['model_revision'], conditions=CONDITIONS,
        samples=3000, per_task={t: 500 for t in TASKS}, images=len({s['image_sha256'] for s in cohort()}),
        shard_questions=dict(Counter(str(a['worker']) for a in assignments)),
        generation=dict(do_sample=False, num_beams=1, max_new_tokens=32, eos_token_id=2,
            use_cache=True, independent_prefill_and_prefix=True, native_first_answer=True),
        full_reuse=False, reason='Original full generations cover only480/3000. Regenerate all3000 Full on paired hardware.',
        statistics=dict(seed=SEED, bootstrap_repetitions=200000, unit='Whole image SHA256 clusters, paired question-weighted ratio',
            primary_family=5, noninferiority_pp=-1, criterion='Corrected lower bound strictly greater than -1pp',
            supplementary_direct_family=4, direct_contrasts='W1/W2/W3/W4 minus W0; separate auxiliary family, not joint9-way claim',
            taskwise='Five-window correction separately within each task; descriptive, not global30-test inference',
            full_length_groups=['EOS-only tail: Full has exactly1 non-EOS token',
                'Content tail: Full has>=2 non-EOS tokens', 'Immediate EOS', 'Full missing EOS'],
            lengthwise='Five-window correction within each prespecified Full length group; descriptive conditioning on Full',
            caveat='CI including0 is not equivalence. Degenerate resampling is flagged, not zero population risk.'),
        boundaries='Only Full/W0-W4, original3000 questions, no training, no new data, no layer search or acceleration claim.'))
    print('Frozen3000 original questions:', Counter(a['worker'] for a in assignments), flush=True)


def verify():
    c = read(OUT / 'execution_manifest.json')
    assert c['source'] == sources() and c['conditions'] == CONDITIONS
    assert c['engine_binding'] == digest(verify_engine())
    assert c['original_manifest_sha256'] == sha256(ORIGINAL / 'sample_manifest.json')
    assert c['original_scorer_sha256'] == sha256(ORIGINAL / 'scorer_validation.json')
    assert c['assignments_sha256'] == sha256(OUT / 'assignments.json')
    assert c['cpu_sha256'] == sha256(OUT / 'cpu_validation.json')
    for p, h in read(ORIGINAL / 'scorer_validation.json')['source_sha256'].items():
        assert sha256(p) == h
    return c


def smoke(model, processor):
    # Use the first frozen input of each task, never choose by predicted outcome.
    samples = [next(s for s in cohort() if s['task'] == t) for t in TASKS]
    checks = []
    for index, sample in enumerate(samples):
        prompt = prepared(model, processor, sample)
        before = tensor_hashes(prompt)
        full, logits = generate(model, prompt, CONDITIONS['Full'], collect=True, audit=True)
        n = prompt['input_ids'].shape[1]
        with torch.inference_mode():
            native = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
                use_cache=True, eos_token_id=2)[0, n:].tolist()
        assert native == full['token_ids']
        # One real image per window exercises all fixed layer sets on each GPU.
        key = f'W{index % 5}'
        condition = CONDITIONS[key]
        noop, no_logits = generate(model, prompt, condition, noop=True, collect=True)
        assert noop['token_ids'] == full['token_ids'] and len(no_logits) == len(logits)
        assert all(torch.equal(a, b) for a, b in zip(logits, no_logits))
        calls = [0] * 32
        handles = []
        try:
            for k, block in enumerate(model_layers(model)):
                def observe(module, args, output, k=k):
                    calls[k] += 1
                    assert torch.isfinite(output).all()
                handles.append(block.mlp.register_forward_hook(observe))
            changed, other = generate(model, prompt, condition, collect=True, audit=True)
        finally:
            for handle in handles:
                handle.remove()
        assert all(count == changed['length'] for count in calls)
        assert torch.equal(logits[0], other[0]) and changed['token_ids'][0] == full['token_ids'][0]
        assert changed['initial_cache_sha256'] == full['initial_cache_sha256']
        assert {e['layer'] for e in changed['events']} == set(condition['layers'])
        assert len(changed['events']) == 6 * changed['length']
        assert all(e['active'] == e['decode'] for e in changed['events'])
        with JointAttention(model, condition['layers'], 'Decode-only', n):
            with torch.inference_mode():
                native_changed = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
                    use_cache=True, eos_token_id=2)[0, n:].tolist()
        assert native_changed == changed['token_ids'] and tensor_hashes(prompt) == before
        checks.append(dict(task=sample['task'], question_id=sample['question_id'], condition=key,
            full=full, changed=changed, full_initial_logits_sha256=digest(logits[0].tolist()),
            noop_logits_exact=True, prefill_logits_and_cache_exact=True, native_full_and_changed_tokens_exact=True,
            all32_mlp_normal=True))
    return dict(passed=True, checks=checks)


def gate():
    c = verify()
    records = [read(OUT / f'shard{k}/smoke.json') for k in range(3)]
    shared = [k for k in ENV_KEYS if k != 'cuda_visible_devices']
    assert len({digest({k: r['environment'][k] for k in shared}) for r in records}) == 1
    for record in records:
        assert record['binding'] == digest(c) and record['validation']['passed']
        assert record['validation'] == records[0]['validation']
    freeze_json(OUT / 'gpu_gate.json', dict(passed=True, binding=digest(c), workers=3,
        cross_gpu_smoke_tokens_and_initial_logits_exact=True,
        smoke_sha256=[sha256(OUT / f'shard{k}/smoke.json') for k in range(3)]))
    print('All three GPU smoke gates passed', flush=True)


def worker(rank):
    c = verify()
    binding = digest(c)
    folder = OUT / f'shard{rank}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pre = preflight(rank)
    model, processor, environment = load_model(MODEL_PATH)
    assert environment['transformers'] == '4.49.0'
    assert environment['parameter_dtypes'] == ['torch.float16']
    assert environment['model_revision'] == c['model_revision']
    environment.update(physical_gpu=rank + 1, code_commit=c['code_commit'], preflight=pre)
    versions = {name: p._version for name, p in model.named_parameters()}
    footprint = hook_footprint(model)
    if not (folder / 'smoke.json').exists():
        validation = smoke(model, processor)
        assert footprint == hook_footprint(model)
        atomic_json(folder / 'smoke.json', dict(binding=binding, environment=environment, validation=validation))
    else:
        assert read(folder / 'smoke.json')['binding'] == binding
    print('Smoke ready', rank, flush=True)
    deadline = time.monotonic() + 900
    while not (OUT / 'gpu_gate.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('GPU acceptance gate missing')
        time.sleep(1)
    assert read(OUT / 'gpu_gate.json')['binding'] == binding
    selected = [s for s in cohort() if rank_for(s) == rank]
    started = time.perf_counter()
    resumed = 0
    for index, sample in enumerate(selected):
        folder_q = folder / 'checkpoints' / f"{sample['task']}__{sample['question_id']}"
        prompt = None
        full_first_token = None
        for key, condition in CONDITIONS.items():
            path = folder_q / f'{key}.json'
            if checkpoint_valid(path, binding, sample, key):
                if key == 'Full':
                    full_first_token = read(path)['generation']['token_ids'][0]
                resumed += 1
                continue
            if prompt is None:
                prompt = prepared(model, processor, sample)
                inputs = tensor_hashes(prompt)
            tick = time.perf_counter()
            generation, _ = generate(model, prompt, condition)
            if key == 'Full':
                full_first_token = generation['token_ids'][0]
            assert generation['token_ids'][0] == full_first_token
            text = processor.tokenizer.decode(generation['token_ids'], skip_special_tokens=True).strip()
            generation.update(text=text, prediction=gqa_prediction(text), score=int(gqa_prediction(text) == sample['answer']))
            assert tensor_hashes(prompt) == inputs and hook_footprint(model) == footprint
            assert versions == {name: p._version for name, p in model.named_parameters()}
            atomic_json(path, dict(binding=binding, sample_sha256=digest(sample), condition=key,
                task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                generation=generation, result_sha256=digest(generation), seconds=time.perf_counter() - tick))
        if (index + 1) % 25 == 0 or index + 1 == len(selected):
            print(json.dumps(dict(rank=rank, questions=index + 1, total=len(selected),
                generations=(index + 1) * 6, seconds=time.perf_counter() - started)), flush=True)
    verify()
    value = dict(status='complete', binding=binding, questions=len(selected), generations=6 * len(selected),
        resumed=resumed, generation_seconds=time.perf_counter() - started, memory=gpu_memory(),
        hooks_clean=hook_footprint(model) == footprint,
        parameters_unchanged=versions == {name: p._version for name, p in model.named_parameters()})
    if not (folder / 'completion.json').exists():
        atomic_json(folder / 'completion.json', value)
    print('Complete', rank, value, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cpu', 'freeze', 'gate', 'worker'])
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

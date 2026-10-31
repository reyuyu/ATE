import argparse
from collections import Counter
import fcntl
import io
import json
import os
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
from paper_rq1_early_joint_attention.engine import generate
from paper_rq1_early_joint_attention.protocol import ROOT, MODEL_PATH, GPUS
from paper_rq1_early_joint_attention.run import preflight, checkpoint_valid
from first_answer_attention import run as previous
from first_answer_attention.engine import supplement_generate
from decode_five_vs_six import ENV_KEYS
from vlm_common import load_model, gpu_memory

OUT = ROOT / 'results/four_layer_p1_decode_scan'
BASE = 'e495adf4fd93337a3638981bbb66c2bbac81242d'
TASKS, SEED, cohort, rank_for, read = previous.TASKS, previous.SEED, previous.cohort, previous.rank_for, previous.read
CONDITIONS = {f'W{s:02d}': dict(layers=list(range(s, s + 4)), decode=True) for s in range(29)}
NEW = {key: c for key, c in CONDITIONS.items() if key != 'W02'}


def sources():
    names = ['four_layer_attention/' + n for n in ['__init__.py', 'run.py', 'analyze.py']]
    names += ['test_four_layer_attention.py']
    return {n: sha256(ROOT / n) for n in names}


def reused_path(sample, key):
    if key == 'Full': return previous.baseline_path(sample)
    assert key == 'W02'
    return previous.OUT / f"shard{rank_for(sample)}/checkpoints/{sample['task']}__{sample['question_id']}/E.json"


def reused(sample, key, config):
    old_binding = config['figure3_binding'] if key == 'Full' else config['previous_binding']
    assert checkpoint_valid(reused_path(sample, key), old_binding, sample, 'Full' if key == 'Full' else 'E')
    return read(reused_path(sample, key))['generation']


def cpu():
    stream = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ['test_early_joint_attention', 'test_early_joint_analysis', 'test_decode_five_vs_six',
         'test_figure3_decode_windows', 'test_first_answer_attention', 'test_four_layer_attention'])
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    freeze_json(OUT / 'cpu_validation.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        source=sources(), output=stream.getvalue()))
    print(stream.getvalue(), flush=True)
    assert result.wasSuccessful()


def freeze():
    old = previous.verify()
    tested = read(OUT / 'cpu_validation.json')
    assert tested['passed'] and tested['source'] == sources()
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *sources()], cwd=ROOT, text=True).strip()
    assert CONDITIONS['W02'] == previous.CONDITIONS['E']
    old_result = read(previous.OUT / 'result.json')
    assert old_result['status'] == 'complete' and old_result['binding'] == digest(old)
    bindings = dict(previous_binding=digest(old), figure3_binding=old['previous_binding'])
    records, counts = [], Counter()
    for s in cohort():
        record = dict(task=s['task'], question_id=s['question_id'], image_sha256=s['image_sha256'],
            sample_sha256=digest(s), worker=rank_for(s))
        for key in ['Full', 'W02']:
            g = reused(s, key, bindings)
            assert g['prediction'] == gqa_prediction(g['text']) and g['score'] == int(g['prediction'] == s['answer'])
            counts[key] += g['score']
            record[key + '_sha256'] = sha256(reused_path(s, key))
        records.append(record)
    assert counts == dict(Full=2627, W02=2630)
    freeze_json(OUT / 'reuse_bindings.json', records)
    freeze_json(OUT / 'execution_manifest.json', dict(base_commit=BASE, **bindings,
        source=sources(), cpu_sha256=sha256(OUT / 'cpu_validation.json'),
        reuse_bindings_sha256=sha256(OUT / 'reuse_bindings.json'),
        original_manifest_sha256=old['original_manifest_sha256'], conditions=CONDITIONS,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        model_revision=old['model_revision'], questions=3000, images=2409, per_task={t: 500 for t in TASKS},
        shard_questions=dict(Counter(str(r['worker']) for r in records)),
        new_generations=84000, reused_generations=6000,
        generation=old['generation'],
        scope='Every window: only original Prefill P-1 plus all subsequent Decode; other Prefill queries and all MLPs normal.',
        hardware='Original image-to-GPU1/2/3 assignment; each worker waits for its own GPU to become completely idle.',
        smoke='Independent per-GPU gate with exact replay of historical Full/E and all-step no-op checks; no allocation on busy GPUs.',
        statistics=dict(seed=SEED, repetitions=1000000, primary_family=29, noninferiority_pp=-1,
            unit='Whole-image SHA256 paired clusters; question-weighted accuracy ratio',
            criterion='Bonferroni29 corrected lower bound strictly greater than -1pp',
            taskwise='Scores and flips only, descriptive; no additional significance claims.'),
        boundary='Exploratory on existing3000, not independent confirmation of L2-L5; no training, BOS/ROI interventions or acceleration claims.'))
    print('Frozen29 windows; Full/W02 reused;84000 new generations', flush=True)


def verify():
    c = read(OUT / 'execution_manifest.json')
    assert c['source'] == sources() and c['conditions'] == CONDITIONS
    assert c['cpu_sha256'] == sha256(OUT / 'cpu_validation.json')
    assert c['reuse_bindings_sha256'] == sha256(OUT / 'reuse_bindings.json')
    assert c['previous_binding'] == digest(previous.verify())
    return c


def smoke(model, processor, config, rank):
    samples = [next(s for s in cohort() if s['task'] == t) for t in list(TASKS)[:2]]
    old_smoke = read(previous.OUT / f'shard{rank}/smoke.json')['validation']['records']
    checks = []
    footprint = hook_footprint(model)
    for index, sample in enumerate(samples):
        prompt = prepared(model, processor, sample)
        inputs = tensor_hashes(prompt)
        n = prompt['input_ids'].shape[1]
        full, logits = generate(model, prompt, dict(layers=[], mode='Full'), collect=True)
        assert full['token_ids'] == reused(sample, 'Full', config)['token_ids']
        assert digest(logits[0].tolist()) == old_smoke[index]['full_first_logits_sha256']
        native_e, first, _ = supplement_generate(model, prompt, CONDITIONS['W02'])
        old_e = reused(sample, 'W02', config)
        assert native_e['token_ids'] == old_e['token_ids'] and digest(first.tolist()) == old_e['first_logits_sha256']
        tested = []
        if index == 0:
            untouched, handles = {}, []
            try:
                for k, block in enumerate(model_layers(model)):
                    def save(module, args, output, k=k): untouched[k] = output[0][:, :-1].detach().clone()
                    handles.append(block.register_forward_hook(save))
                previous.reference(model, prompt)
            finally:
                for h in handles: h.remove()
            for key in ['W00', 'W01', 'W14', 'W28']:
                condition = CONDITIONS[key]
                noop, _, no_logits = supplement_generate(model, prompt, condition, noop=True, collect=True)
                assert noop['token_ids'] == full['token_ids'] and len(logits) == len(no_logits)
                assert all(torch.equal(a, b) for a, b in zip(logits, no_logits))
                counts, handles = [0] * 32, []
                try:
                    for k, block in enumerate(model_layers(model)):
                        def observe(module, args, output, k=k):
                            if output[0].shape[1] == n: assert torch.equal(output[0][:, :-1], untouched[k])
                        def mlp(module, args, output, k=k):
                            counts[k] += 1
                            assert torch.isfinite(output).all()
                        handles += [block.register_forward_hook(observe), block.mlp.register_forward_hook(mlp)]
                    changed, first, _ = supplement_generate(model, prompt, condition, audit=True)
                finally:
                    for h in handles: h.remove()
                assert all(v == changed['length'] for v in counts)
                assert {e['layer'] for e in changed['events']} == set(condition['layers'])
                assert all(e['target'] == 'original_P-1' and not e['decode'] for e in changed['events'] if e['active'])
                assert all(e['active'] == e['decode'] for e in changed['decode_events'])
                assert changed['edits'] == 4 * changed['length']
                assert hook_footprint(model) == footprint
                tested.append(dict(condition=key, token_ids=changed['token_ids'], edits=changed['edits'],
                    other_prefill_queries_all32_exact=True, all32_mlp_normal=True, noop_all_logits_exact=True))
        assert torch.equal(previous.reference(model, prompt), logits[0])
        assert tensor_hashes(prompt) == inputs and hook_footprint(model) == footprint
        checks.append(dict(question_id=sample['question_id'], native_full_replay_exact=True,
            reused_E_tokens_and_first_logits_exact=True, historical_full_first_logits_exact=True,
            native_recovery_exact=True, tested=tested))
    return dict(passed=True, checks=checks)


def wait_for_idle(rank):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPUS[rank]: raise ValueError('Wrong GPU UUID; not a resource wait')
    last = 0
    while True:
        try: return preflight(rank)
        except ValueError as error:
            if time.monotonic() - last >= 300 or not last:
                print(json.dumps(dict(status='waiting_for_idle_gpu', physical_gpu=rank + 1,
                    reason=str(error), cuda_allocated=False, unix_time=time.time())), flush=True)
                last = time.monotonic()
            time.sleep(30)


def worker(rank):
    c = verify()
    binding = digest(c)
    folder = OUT / f'shard{rank}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'completion.json').exists():
        assert read(folder / 'completion.json')['binding'] == binding
        print('Already complete', rank, flush=True)
        return
    pre = wait_for_idle(rank)
    model, processor, environment = load_model(MODEL_PATH)
    old_env = read(previous.OUT / f'shard{rank}/smoke.json')['environment']
    assert all(environment[k] == old_env[k] for k in ENV_KEYS)
    assert environment['transformers'] == '4.49.0' and environment['parameter_dtypes'] == ['torch.float16']
    environment.update(physical_gpu=rank + 1, code_commit=c['code_commit'], preflight=pre)
    versions = {n: p._version for n, p in model.named_parameters()}
    footprint = hook_footprint(model)
    validation = smoke(model, processor, c, rank)
    assert versions == {n: p._version for n, p in model.named_parameters()}
    assert hook_footprint(model) == footprint
    if not (folder / 'smoke.json').exists():
        atomic_json(folder / 'smoke.json', dict(binding=binding, environment=environment, validation=validation))
    else: assert read(folder / 'smoke.json')['binding'] == binding
    print('Per-GPU smoke passed; formal generation starting', rank, flush=True)
    records = {(r['task'], r['question_id']): r for r in read(OUT / 'reuse_bindings.json')}
    samples = [s for s in cohort() if rank_for(s) == rank]
    started, resumed = time.perf_counter(), 0
    for index, sample in enumerate(samples):
        record = records[sample['task'], sample['question_id']]
        for key in ['Full', 'W02']:
            assert sha256(reused_path(sample, key)) == record[key + '_sha256']
        full = reused(sample, 'Full', c)
        directory = folder / 'checkpoints' / f"{sample['task']}__{sample['question_id']}"
        pending = [key for key in NEW if not checkpoint_valid(directory / f'{key}.json', binding, sample, key)]
        resumed += len(NEW) - len(pending)
        if pending:
            prompt = prepared(model, processor, sample)
            inputs = tensor_hashes(prompt)
            for key in pending:
                tick = time.perf_counter()
                generation, first, _ = supplement_generate(model, prompt, NEW[key])
                assert int(first.argmax()) == generation['token_ids'][0]
                text = processor.tokenizer.decode(generation['token_ids'], skip_special_tokens=True).strip()
                generation.update(text=text, prediction=gqa_prediction(text), score=int(gqa_prediction(text) == sample['answer']),
                    first_token=generation['token_ids'][0], full_first_token=full['token_ids'][0],
                    first_top1_flip=generation['token_ids'][0] != full['token_ids'][0], first_logits_sha256=digest(first.tolist()))
                atomic_json(directory / f'{key}.json', dict(binding=binding, sample_sha256=digest(sample), condition=key,
                    task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                    generation=generation, result_sha256=digest(generation), seconds=time.perf_counter() - tick))
            assert tensor_hashes(prompt) == inputs and hook_footprint(model) == footprint
            assert versions == {n: p._version for n, p in model.named_parameters()}
        if (index + 1) % 10 == 0 or index + 1 == len(samples):
            print(json.dumps(dict(rank=rank, questions=index + 1, total=len(samples),
                generations=(index + 1) * 28, seconds=time.perf_counter() - started)), flush=True)
    verify()
    atomic_json(folder / 'completion.json', dict(status='complete', binding=binding, questions=len(samples),
        generations=len(samples) * 28, resumed=resumed, seconds=time.perf_counter() - started,
        memory=gpu_memory(), hooks_clean=hook_footprint(model) == footprint,
        parameters_unchanged=versions == {n: p._version for n, p in model.named_parameters()}))
    print('Complete', rank, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cpu', 'freeze', 'worker'])
    parser.add_argument('--rank', type=int, choices=range(3), default=0)
    args = parser.parse_args()
    try:
        if args.stage == 'worker': worker(args.rank)
        else: globals()[args.stage]()
    except BaseException:
        if args.stage == 'worker':
            freeze_json(OUT / f'failures/shard{args.rank}_{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
        raise

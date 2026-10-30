import argparse
from contextlib import ExitStack
import fcntl
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
from paper_rq1_early_joint_attention.engine import JointAttention, generate
from paper_rq1_early_joint_attention.protocol import ROOT, MODEL_PATH
from paper_rq1_early_joint_attention.run import preflight, checkpoint_valid
from figure3_decode_windows import run as previous
from decode_five_vs_six import ENV_KEYS
from vlm_common import load_model, gpu_memory
from .engine import CONDITIONS, supplement_generate, first_metrics

OUT = ROOT / 'results/first_answer_early_attention'
BASE = 'c25001e135408ad72d9f797da03265464b01f7b1'
TASKS, SEED, cohort, rank_for, read = previous.TASKS, previous.SEED, previous.cohort, previous.rank_for, previous.read
DIRECT = [('B', 'A'), ('D', 'C'), ('C', 'A'), ('D', 'B'), ('E', 'C'), ('E', 'D')]


def sources():
    names = ['first_answer_attention/' + n for n in ['__init__.py', 'engine.py', 'run.py', 'analyze.py']]
    names += ['test_first_answer_attention.py', 'paper_rq1_early_joint_attention/engine.py', 'paper_rq1/engine.py']
    return {n: sha256(ROOT / n) for n in names}


def baseline_path(sample):
    return previous.OUT / f"shard{rank_for(sample)}/checkpoints/{sample['task']}__{sample['question_id']}/Full.json"


def baseline(sample, binding):
    path = baseline_path(sample)
    assert checkpoint_valid(path, binding, sample, 'Full')
    row = read(path)
    return row['generation']


def cpu():
    stream = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ['test_early_joint_attention', 'test_early_joint_analysis', 'test_decode_five_vs_six',
         'test_figure3_decode_windows', 'test_first_answer_attention'])
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
    full = [baseline(s, digest(old)) for s in cohort()]
    assert len(full) == 3000 and sum(g['score'] for g in full) == 2627
    records = [dict(task=s['task'], question_id=s['question_id'], image_sha256=s['image_sha256'],
        sample_sha256=digest(s), worker=rank_for(s), full_checkpoint_sha256=sha256(baseline_path(s))) for s in cohort()]
    freeze_json(OUT / 'baseline_bindings.json', records)
    freeze_json(OUT / 'execution_manifest.json', dict(base_commit=BASE, previous_binding=digest(old),
        model_revision=old['model_revision'], source=sources(), cpu_sha256=sha256(OUT / 'cpu_validation.json'),
        baseline_bindings_sha256=sha256(OUT / 'baseline_bindings.json'),
        original_manifest_sha256=old['original_manifest_sha256'], conditions=CONDITIONS,
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        questions=3000, images=2409, per_task={t: 500 for t in TASKS},
        full_accuracy_reused=True, full_correct=2627, full_generation_replayed_only_in_smoke=True,
        reference='One native cached Prefill per question obtains on-machine first-token logits, not free Full generation.',
        temporal_scope='Only original P-1 in Prefill; C/D/E additionally all subsequent Decode queries. Never all Prefill queries.',
        generation=dict(greedy=True, use_cache=True, max_new_tokens=32, eos_token_id=2, own_condition_prefix=True),
        statistics=dict(seed=SEED, bootstrap_repetitions=200000, primary_family=5, noninferiority_pp=-1,
            direct_contrasts=DIRECT, direct_family=6, unit='Whole-image SHA256 paired clusters; question-weighted ratio',
            taskwise='Descriptive; five-comparison correction within task, not global30 simultaneous inference.',
            first_logp_loss='logP_Full(native first Top1) - logP_condition(native first Top1); signed FP32 log_softmax.',
            criterion='Corrected lower bound strictly greater than -1pp; CI containing0 is not equivalence.')))
    print('Frozen3000; reused Full2627; five new conditions', flush=True)


def verify():
    c = read(OUT / 'execution_manifest.json')
    assert c['source'] == sources() and c['conditions'] == CONDITIONS
    assert c['cpu_sha256'] == sha256(OUT / 'cpu_validation.json')
    assert c['baseline_bindings_sha256'] == sha256(OUT / 'baseline_bindings.json')
    assert c['previous_binding'] == digest(previous.verify())
    return c


@torch.inference_mode()
def reference(model, prompt):
    output = model(**prompt, past_key_values=None, use_cache=True, output_attentions=False, return_dict=True)
    logits = output.logits[0, -1].detach().float().cpu().clone()
    assert torch.isfinite(logits).all()
    return logits


def smoke(model, processor, previous_binding):
    selected = [next(s for s in cohort() if s['task'] == t) for t in TASKS[:2]]
    records = []
    footprint = hook_footprint(model)
    for index, sample in enumerate(selected):
        prompt = prepared(model, processor, sample)
        inputs = tensor_hashes(prompt)
        n = prompt['input_ids'].shape[1]
        full, logits = generate(model, prompt, dict(layers=[], mode='Full'), collect=True, audit=True)
        assert full['token_ids'] == baseline(sample, previous_binding)['token_ids']
        ref = reference(model, prompt)
        assert torch.equal(ref, logits[0])
        native = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
            use_cache=True, eos_token_id=2)[0, n:].tolist()
        assert native == full['token_ids']
        checks = []
        for key, condition in CONDITIONS.items():
            noop, first, no_logits = supplement_generate(model, prompt, condition, noop=True, collect=True)
            assert noop['token_ids'] == full['token_ids'] and torch.equal(first, ref)
            assert len(no_logits) == len(logits) and all(torch.equal(a, b) for a, b in zip(logits, no_logits))
            # Causal earlier Prefill queries must remain identical throughout all32 layers.
            untouched, counts, handles = {}, [0] * 32, []
            try:
                for k, block in enumerate(model_layers(model)):
                    def full_query(module, args, output, k=k):
                        h = output[0]
                        if h.shape[1] == n:
                            untouched[k] = h[:, :-1].detach().clone()
                    handles.append(block.register_forward_hook(full_query))
                reference(model, prompt)
            finally:
                for h in handles: h.remove()
            handles = []
            try:
                for k, block in enumerate(model_layers(model)):
                    def check_query(module, args, output, k=k):
                        h = output[0]
                        if h.shape[1] == n:
                            assert torch.equal(h[:, :-1], untouched[k])
                    def mlp(module, args, output, k=k):
                        counts[k] += 1
                        assert torch.isfinite(output).all()
                    handles += [block.register_forward_hook(check_query), block.mlp.register_forward_hook(mlp)]
                changed, first, trace = supplement_generate(model, prompt, condition, audit=True, collect=True)
            finally:
                for h in handles: h.remove()
            assert all(c == changed['length'] for c in counts)
            assert {e['layer'] for e in changed['events']} == set(condition['layers'])
            assert all(e['target'] == 'original_P-1' and not e['decode'] for e in changed['events'] if e['active'])
            if condition['decode']:
                assert all(e['active'] == e['decode'] for e in changed['decode_events'])
            with ExitStack() as stack:
                stack.enter_context(JointAttention(model, condition['layers'], 'P-1-only', n))
                if condition['decode']:
                    stack.enter_context(JointAttention(model, condition['layers'], 'Decode-only', n))
                expected = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
                    use_cache=True, eos_token_id=2)[0, n:].tolist()
            assert expected == changed['token_ids'] and torch.equal(first, trace[0])
            assert hook_footprint(model) == footprint and tensor_hashes(prompt) == inputs
            # Running native again after all condition-owned caches must recover exactly.
            assert torch.equal(reference(model, prompt), ref)
            checks.append(dict(condition=key, generation=changed, first_logits_sha256=digest(first.tolist()),
                first_metrics=first_metrics(ref, first), noop_all_logits_exact=True,
                other_prefill_queries_all32layers_exact=True, all32_mlp_normal=True,
                independent_native_hf_tokens_exact=True, native_recovery_exact=True))
        records.append(dict(task=sample['task'], question_id=sample['question_id'], full_tokens=full['token_ids'],
            full_first_logits_sha256=digest(ref.tolist()), checks=checks))
    return dict(passed=True, records=records)


def gate():
    c = verify()
    records = [read(OUT / f'shard{k}/smoke.json') for k in range(3)]
    shared = [k for k in ENV_KEYS if k != 'cuda_visible_devices']
    assert len({digest({k: r['environment'][k] for k in shared}) for r in records}) == 1
    assert all(r['binding'] == digest(c) and r['validation']['passed'] for r in records)
    assert all(r['validation'] == records[0]['validation'] for r in records)
    freeze_json(OUT / 'gpu_gate.json', dict(passed=True, binding=digest(c),
        cross_gpu_smoke_exact=True, old_full_replay_exact=True,
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
    assert environment['transformers'] == '4.49.0' and environment['parameter_dtypes'] == ['torch.float16']
    assert environment['model_revision'] == c['model_revision']
    old_env = read(previous.OUT / f'shard{rank}/smoke.json')['environment']
    assert all(environment[k] == old_env[k] for k in ENV_KEYS)
    environment.update(physical_gpu=rank + 1, code_commit=c['code_commit'], preflight=pre)
    versions = {n: p._version for n, p in model.named_parameters()}
    footprint = hook_footprint(model)
    if not (folder / 'smoke.json').exists():
        value = smoke(model, processor, c['previous_binding'])
        assert hook_footprint(model) == footprint
        atomic_json(folder / 'smoke.json', dict(binding=binding, environment=environment, validation=value))
    else:
        assert read(folder / 'smoke.json')['binding'] == binding
    print('Smoke ready', rank, flush=True)
    deadline = time.monotonic() + 900
    while not (OUT / 'gpu_gate.json').exists():
        if time.monotonic() > deadline: raise TimeoutError('GPU gate missing')
        time.sleep(1)
    assert read(OUT / 'gpu_gate.json')['binding'] == binding
    bindings = {(r['task'], r['question_id']): r for r in read(OUT / 'baseline_bindings.json')}
    selected = [s for s in cohort() if rank_for(s) == rank]
    started, resumed = time.perf_counter(), 0
    for index, sample in enumerate(selected):
        directory = folder / 'checkpoints' / f"{sample['task']}__{sample['question_id']}"
        assert sha256(baseline_path(sample)) == bindings[sample['task'], sample['question_id']]['full_checkpoint_sha256']
        full = baseline(sample, c['previous_binding'])
        pending = [key for key in CONDITIONS if not checkpoint_valid(directory / f'{key}.json', binding, sample, key)]
        resumed += 5 - len(pending)
        if pending:
            prompt = prepared(model, processor, sample)
            inputs = tensor_hashes(prompt)
            ref = reference(model, prompt)
            assert int(ref.argmax()) == full['token_ids'][0]
            reference_hash = digest(ref.tolist())
            for key in pending:
                tick = time.perf_counter()
                generation, first, _ = supplement_generate(model, prompt, CONDITIONS[key])
                generation.update(first_metrics(ref, first))
                generation.update(full_first_logits_sha256=reference_hash, first_logits_sha256=digest(first.tolist()))
                text = processor.tokenizer.decode(generation['token_ids'], skip_special_tokens=True).strip()
                generation.update(text=text, prediction=gqa_prediction(text), score=int(gqa_prediction(text) == sample['answer']))
                assert int(first.argmax()) == generation['token_ids'][0]
                assert hook_footprint(model) == footprint and tensor_hashes(prompt) == inputs
                assert versions == {n: p._version for n, p in model.named_parameters()}
                atomic_json(directory / f'{key}.json', dict(binding=binding, sample_sha256=digest(sample), condition=key,
                    task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                    generation=generation, result_sha256=digest(generation), seconds=time.perf_counter() - tick))
        if (index + 1) % 25 == 0 or index + 1 == len(selected):
            print(json.dumps(dict(rank=rank, questions=index + 1, total=len(selected),
                generations=(index + 1) * 5, seconds=time.perf_counter() - started)), flush=True)
    verify()
    result = dict(status='complete', binding=binding, questions=len(selected), generations=5 * len(selected),
        resumed=resumed, seconds=time.perf_counter() - started, memory=gpu_memory(),
        hooks_clean=hook_footprint(model) == footprint,
        parameters_unchanged=versions == {n: p._version for n, p in model.named_parameters()})
    atomic_json(folder / 'completion.json', result)
    print('Complete', rank, result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cpu', 'freeze', 'gate', 'worker'])
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    try:
        if args.stage == 'worker': worker(args.rank)
        else: globals()[args.stage]()
    except BaseException:
        if args.stage == 'worker':
            freeze_json(OUT / f'failures/shard{args.rank}_{time.time_ns()}.json', dict(traceback=traceback.format_exc()))
        raise

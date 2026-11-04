"""Three smoke inputs, then exactly 300 quality generations and 20 paired timings."""
import os
import time
import traceback
from dataclasses import asdict

import torch

from .common import OUT, atomic, digest, read, verify, score, hardware
from .engine import generate, CONDITIONS
from ..physical_handoff_fastpath import engine as base
from ..fastp2_bypass_writer.engine import generate as legacy_generate, LAYERS
from ..fastp2_bypass_writer.run import CompleteCensus, operators
from ..p1_layerwise_accuracy.engine import Adapter, tensor_hash
from ..physical_visual_handoff.engine import equivalence


def prepare(adapter, sample, spec):
    adapter.prepare(sample, {})
    assert adapter.input_hashes == spec['input_hashes']
    layout = base.Layout.from_adapter(adapter, sample, {})
    assert list(layout.question) == spec['Q'] and list(layout.relay) == spec['R']
    assert list(layout.image) == spec['I'] and layout.cutoff == 20
    return layout


def save(path, payload):
    assert not path.exists(), 'Checkpoint overwrite prohibited'
    atomic(path, dict(payload=payload, sha256=digest(payload)))


def load(path, p, s, spec):
    blob = read(path)
    v = blob['payload']
    assert digest(v) == blob['sha256'] and v['protocol_sha256'] == digest(p)
    assert v['sample_sha256'] == digest(s) and v['spec_sha256'] == digest(spec)
    return v


def smoke(adapter, p, samples, specs):
    numerical, native_checks = [], []
    for q in p['smoke_ids']:
        s, spec = samples[q], specs[q]
        layout = prepare(adapter, s, spec)
        native = generate(adapter, layout, 'C0', spec['writers'], 2, True, True)
        noop = generate(adapter, layout, 'Native-Noop', spec['writers'], 2, True, True)
        assert len(native[2]) == len(noop[2]) >= 2
        assert all(torch.equal(a, b) for a, b in zip(native[2], noop[2]))
        captured = []
        def observe(m, args, out):
            captured.append(out.logits.detach().cpu().clone())
        handle = adapter.model.register_forward_hook(observe)
        try:
            with torch.inference_mode():
                official = adapter.model.generate(**adapter.inputs, do_sample=False, num_beams=1,
                    max_new_tokens=2, use_cache=True, logits_to_keep=0)
        finally:
            handle.remove()
        assert official[0, layout.length:].tolist() == native[0]['token_ids']
        assert len(captured) == len(native[2])
        assert all(torch.equal(a, b) for a, b in zip(captured, native[2]))
        baseline = generate(adapter, layout, 'C1', spec['writers'], 2, True, True)
        old = base.generate(adapter, layout, 'Fast-P2', 2, False, True, True)
        assert all(torch.equal(a, b) for a, b in zip(baseline[2], old[2]))
        a2 = generate(adapter, layout, 'C3', spec['writers'], 2, True, True)
        old_a2 = legacy_generate(adapter, layout, 'A2', {}, 2, False, True, True)
        assert all(torch.equal(a, b) for a, b in zip(a2[2], old_a2[2]))
        conditional = generate(adapter, layout, 'C2', spec['writers'], 2, True, True)
        assert conditional[0]['audit']
        with operators(adapter) as (calls, mlp, op_rows):
            actual = generate(adapter, layout, 'C4', spec['writers'], 2, True, True)
        assert len(actual[0]['token_ids']) >= 2
        assert all(mlp[i] == 2 for i in range(32))
        for i in LAYERS:
            assert all(calls[i, op] == 1 for op in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
        for row in op_rows:
            if row['layer'] in LAYERS:
                expected_rows = layout.length - 577 if row['op'] in ('q_proj', 'o_proj') else layout.length
                assert row['shape'][1] == expected_rows
        for ref in ('C4-ZeroRef', 'C4-DenseRef'):
            expected = generate(adapter, layout, ref, spec['writers'], 2, True, True)
            comparison = equivalence('llava', expected[1][:1], actual[1][:1], expected[3], actual[3])
            comparison.update(question_id=q, reference=ref,
                same_tokens=expected[0]['token_ids'] == actual[0]['token_ids'],
                first_logits_exact=torch.equal(expected[1][0], actual[1][0]))
            numerical.append(comparison)
            assert comparison['passed'] and comparison['same_tokens'], comparison
            del expected
        events = actual[0]['audit']
        assert all(e['input_queries'] == len(layout.text) for e in events if e['prefill'] and e['layer'] >= 20)
        kv = actual[0]['kv']
        assert kv['historical_cache_copies'] == kv['append_concatenations'] == 0
        assert all(r['used_tokens'] == (0 if r['layer'] in LAYERS else len(layout.text) + 1) for r in kv['layers'])
        atomic(OUT / 'smoke_details' / (q + '.json'), dict(numerical=numerical[-2:],
            C4_events=events, C2_events=conditional[0]['audit'], operator_rows=op_rows,
            MLP_calls=dict(mlp), layout=asdict(layout), C4_KV=kv))
        native_checks.append(dict(question_id=q, native_noop_all_logits_equal=True,
            HF_generate_equal=True, frozen_FastP2_equal=True, frozen_A2_equal=True,
            final_mask_probability_zero=True, physical_query_rows_and_L20_exit=True,
            all_MLP_executed=True, own_cache=True, no_history_copy=True))
        adapter.verify()
        del native, noop, captured, baseline, old, a2, old_a2, conditional, actual
        print('SMOKE', q, 'PASS', flush=True)
    atomic(OUT / 'gpu_smoke.json', dict(status='PASS', checks=native_checks, numerical=numerical,
                                      protocol_sha256=digest(p), tolerances=p['numerical_tolerances']))


def execute(adapter, p, samples, specs):
    assert read(OUT / 'gpu_smoke.json')['status'] == 'PASS'
    started = time.time()
    for index, q in enumerate(sorted(samples)):
        s, spec = samples[q], specs[q]
        path = OUT / 'checkpoints' / (q + '.json')
        if path.exists():
            load(path, p, s, spec)
            continue
        layout = prepare(adapter, s, spec)
        results = {}
        for c in CONDITIONS:
            census = CompleteCensus(adapter)
            with census.context():
                v, logits, _, _, _ = generate(adapter, layout, c, spec['writers'], capture=True)
            v['score'] = score(v['text'], s['answers'])
            target = int(logits[0].argmax()) if c == 'C0' else results['C0']['quality']['token_ids'][0]
            v['native_first_token_logp'] = float(logits[0].log_softmax(-1)[0, target])
            v['first_logits_sha256'] = tensor_hash(logits[0])
            v['repeated'] = any(v['token_ids'][i:i+3] == v['token_ids'][j:j+3]
                for i in range(len(v['token_ids']) - 2) for j in range(i + 3, len(v['token_ids']) - 2))
            v.pop('process_memory_snapshot', None)
            results[c] = dict(quality=v, flops=census.rows(), measurements=[])
            del logits
        if q in p['performance_ids']:
            for c in CONDITIONS:
                for _ in range(p['warmups']):
                    generate(adapter, layout, c, spec['writers'])
            for rep in range(p['repetitions']):
                shift = (rep + index) % len(CONDITIONS)
                for c in CONDITIONS[shift:] + CONDITIONS[:shift]:
                    v = generate(adapter, layout, c, spec['writers'])[0]
                    assert v['token_ids'] == results[c]['quality']['token_ids'], 'Nondeterministic greedy'
                    results[c]['measurements'].append(v)
        adapter.verify()
        save(path, dict(protocol_sha256=digest(p), sample_sha256=digest(s), spec_sha256=digest(spec),
                        question_id=q, layout=asdict(layout), conditions=results))
        print('PROGRESS', index + 1, 60, q, 'seconds', round(time.time() - started, 1), flush=True)
    atomic(OUT / 'worker_completion.json', dict(status='COMPLETE', elapsed_seconds=time.time() - started,
        protocol_sha256=digest(p), quality_generations=300, performance_questions=20,
        timed_requests=500, warmup_requests=300))


def main():
    p = verify()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == p['device']['uuid']
    state = hardware()
    assert p['device']['uuid'] not in state['processes'], 'Selected GPU is no longer idle'
    atomic(OUT / 'run_preflight.json', state)
    samples, specs = read(OUT / 'sample_manifest.json'), read(OUT / 'input_specs.json')
    adapter = None
    try:
        adapter = Adapter('llava')
        atomic(OUT / 'environment.json', adapter.environment)
        if not (OUT / 'gpu_smoke.json').exists():
            smoke(adapter, p, samples, specs)
        else:
            assert read(OUT / 'gpu_smoke.json')['protocol_sha256'] == digest(p)
        execute(adapter, p, samples, specs)
        adapter.verify()
    except BaseException:
        atomic(OUT / 'STOP.json', dict(traceback=traceback.format_exc(), time=time.time()))
        raise
    finally:
        del adapter
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

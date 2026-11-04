"""Three smoke inputs, then exactly 300 quality generations and 20 paired timings."""
import os
import time
import traceback
from dataclasses import asdict
from contextlib import contextmanager, ExitStack

import torch

from .common import OUT, atomic, digest, read, verify, score, hardware
from .engine import generate, CONDITIONS, LAYERS
from ..physical_handoff_fastpath import engine as base
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


@contextmanager
def observe_visual(adapter, layout):
    snapshots = {}
    with ExitStack() as stack:
        for i in LAYERS:
            def attention(m, a, out, i=i):
                if out[0].shape[1] == layout.length:
                    snapshots[i, 'attention'] = out[0].detach().float().cpu().clone()
            def hidden(m, a, out, i=i):
                if out[0].shape[1] == layout.length:
                    snapshots[i, 'hidden'] = out[0].detach().float().cpu().clone()
            stack.callback(adapter.layers[i].self_attn.register_forward_hook(attention).remove)
            stack.callback(adapter.layers[i].register_forward_hook(hidden).remove)
        def l6(m, a):
            if a[0].shape[1] == layout.length:
                snapshots[6, 'input'] = a[0].detach().float().cpu().clone()
        stack.callback(adapter.layers[6].register_forward_pre_hook(l6).remove)
        yield snapshots


def compare_visual(question, condition, layout, physical, reference, baseline):
    rows = []
    groups = dict(Image=layout.image, Question=layout.question,
                  Other_R=tuple(sorted(set(layout.relay)-set(layout.question))),
                  Prefix=tuple(i for i in range(layout.length-1)
                               if i not in set(layout.image) | set(layout.relay)),
                  Pminus1=(layout.length-1,))
    dummy = [torch.zeros(1, 1)]
    for (layer, kind), actual in physical.items():
        for group, positions in groups.items():
            if not positions:
                continue
            expected = reference[layer, kind][:, list(positions)]
            observed = actual[:, list(positions)]
            e = equivalence('llava', dummy, dummy, [expected], [observed])
            assert e['passed'], (question, condition, layer, kind, group, e)
            # Identical dense shapes: zeroing the last causal query cannot change earlier queries.
            baseline_exact = torch.equal(expected, baseline[layer, kind][:, list(positions)])
            if group != 'Pminus1':
                assert baseline_exact, (layer, kind, group, 'Reference altered earlier query')
            rows.append(dict(question_id=question, condition=condition, layer=layer,
                tensor=kind, group=group, n_positions=len(positions), reference_vs_baseline_exact=baseline_exact,
                physical_vs_reference_exact=torch.equal(expected, observed),
                max_abs_difference=float((expected-observed).abs().max()), **e))
    assert len(physical) == len(reference) == len(baseline) == 9
    return rows


def smoke(adapter, p, samples, specs):
    numerical, native_checks, preservation = [], [], []
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
        for condition, no_skip in (('C3-V2', 'C1'), ('C4-V2', 'C2')):
            with observe_visual(adapter, layout) as baseline_visual:
                no_skip_result = generate(adapter, layout, no_skip, spec['writers'], 2, True, True)
            with operators(adapter) as (calls, mlp, op_rows), observe_visual(adapter, layout) as visual:
                actual = generate(adapter, layout, condition, spec['writers'], 2, True, True)
            assert len(actual[0]['token_ids']) >= 2
            assert all(mlp[i] == 2 for i in range(32))
            for i in LAYERS:
                assert all(calls[i, op] == 1 for op in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
            for row in op_rows:
                if row['layer'] in LAYERS:
                    expected_rows = layout.length - 1 if row['op'] in ('q_proj', 'o_proj') else layout.length
                    assert row['shape'][1] == expected_rows
            ref = condition + '-ZeroRef'
            with observe_visual(adapter, layout) as ref_visual:
                expected = generate(adapter, layout, ref, spec['writers'], 2, True, True)
            comparison = equivalence('llava', expected[1], actual[1], expected[3], actual[3])
            comparison.update(question_id=q, reference=ref,
                same_tokens=expected[0]['token_ids'] == actual[0]['token_ids'],
                first_logits_exact=torch.equal(expected[1][0], actual[1][0]))
            numerical.append(comparison)
            assert comparison['passed'] and comparison['same_tokens'], comparison
            preservation += compare_visual(q, condition, layout, visual, ref_visual, baseline_visual)
            events = actual[0]['audit']
            assert all(e['input_queries'] == len(layout.text) for e in events if e['prefill'] and e['layer'] >= 20)
            for e in events:
                if e['layer'] in LAYERS:
                    assert e['query_positions'] == (list(range(layout.length-1)) if e['prefill'] else [])
            kv = actual[0]['kv']
            assert kv['historical_cache_copies'] == kv['append_concatenations'] == 0
            assert all(r['used_tokens'] == (0 if r['layer'] in LAYERS else len(layout.text) + 1) for r in kv['layers'])
            atomic(OUT / 'smoke_details' / (q + '-' + condition + '.json'), dict(numerical=comparison,
                events=events, operator_rows=op_rows, MLP_calls=dict(mlp), layout=asdict(layout), KV=kv))
            del expected, actual, no_skip_result, visual, ref_visual, baseline_visual
        native_checks.append(dict(question_id=q, native_noop_all_logits_equal=True,
            HF_generate_equal=True, frozen_FastP2_equal=True,
            final_mask_probability_zero=True, physical_query_rows_and_L20_exit=True,
            all_MLP_executed=True, own_cache=True, no_history_copy=True))
        adapter.verify()
        del native, noop, captured, baseline, old
        print('SMOKE', q, 'PASS', flush=True)
    assert preservation and all(r['passed'] for r in preservation)
    atomic(OUT / 'gpu_smoke.json', dict(status='PASS', checks=native_checks, numerical=numerical,
                                      preservation=preservation, visual_prefill_attention_preserved=True,
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

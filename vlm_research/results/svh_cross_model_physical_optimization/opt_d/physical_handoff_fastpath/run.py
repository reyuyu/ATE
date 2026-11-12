"""Freeze, smoke, and benchmark. One process and one native model per GPU."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch

from ..physical_visual_handoff.common import ROOT, HISTORY, MODELS, atomic, digest, key, read, sha, TOLERANCES
from ..physical_visual_handoff.engine import run as debug_run, equivalence
from ..p1_layerwise_accuracy.engine import Adapter, generate as hf_generate, tensor_hash
from .engine import Layout, generate
from .ledger import Census

OUT = ROOT / 'vlm_research/results/physical_handoff_fastpath_tflops'
CONDITIONS = ['Native Full', 'D1 Dense', 'Fast-P1', 'Fast-P2']
BASE = '7900c44217dab6421bbafff184a0029c01c535dc'


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT / 'vlm_research' / p for p in (
        'physical_visual_handoff/engine.py', 'physical_visual_handoff/common.py',
        'p1_layerwise_accuracy/engine.py', 'p1_layerwise_accuracy/common.py',
        'qwen3vl_early_windows/experiment.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def freeze():
    assert str(ROOT).startswith('/data/') and not str(ROOT).startswith('/data1/')
    OUT.mkdir(parents=True, exist_ok=True)
    hw = subprocess.check_output(['nvidia-smi', '-q'], text=True)
    atomic(OUT / 'hardware_info.json', dict(nvidia_smi_q=hw, disk=subprocess.check_output(['df', '-h', '/data'], text=True),
        ncu='/usr/local/cuda-12.6/bin/ncu', hardware_counters='HARDWARE_TFLOPS_UNAVAILABLE',
        reason='ERR_NVGPUCTRPERM on ncu --query-metrics --devices 2; no permission changes requested',
        theoretical_peak='Not used to estimate measured throughput; dense/sparse theoretical peaks not mixed'))
    for kind in MODELS:
        original = read(ROOT / 'vlm_research/results/physical_visual_handoff_pilot' / kind / 'sample_manifest.json')
        pool = {key(s): s for s in original['samples']}
        cohort = [pool[k] for k in original['pilot60']]
        assert len(cohort) == len({s['image_sha256'] for s in cohort}) == 60
        bindings = read(HISTORY / kind / 'input_bindings.json')
        selected_bindings = {key(s): bindings[key(s)] for s in cohort + original['smoke']}
        for s in cohort + original['smoke']:
            assert sha(s['image_path']) == s['image_sha256']
        manifest = dict(samples=cohort, smoke=original['smoke'][:3], historical_pilot60=original['pilot60'],
                        selection='Exact original physical-handoff pilot60; 10/task; no post-outcome filtering')
        atomic(OUT / kind / 'sample_manifest.json', manifest)
        atomic(OUT / kind / 'input_bindings.json', selected_bindings)
        protocol = dict(base=BASE, model=MODELS[kind], conditions=CONDITIONS, kind=kind,
            manifest_sha256=digest(manifest), bindings_sha256=digest(selected_bindings), sources=sources(),
            old_weights_manifest=read(HISTORY / kind / 'protocol.json')['weights'],
            warmups=5, repetitions=20, max_new_tokens=32, engineering_lengths=[16, 64],
            timing='cold request mask+cache setup included in synchronized E2E; CUDA events for model, vision, Prefill, Decode; processing and disk IO excluded equally',
            order='Five warmups per sample/condition; timed conditions interleaved by repetition and rotated by sample',
            engineering_sample='First frozen smoke sample; beyond EOS explicitly allowed only for fixed16/64',
            logits_to_keep=MODELS[kind]['logits_to_keep'], numerical_tolerances=TOLERANCES[kind],
            numerical_policy='Exact native/noop required; native-precision physical approximation reported, not hidden by stabilizers; semantic failures stop',
            dense='Same D1 writing dependency, full hidden rows and full KV; precomputed masks and in-place KV',
            fast_p1='Fixed physical hidden trim; pre-cutoff KV remains visual+dense; late-layer KV text only; common in-place append',
            fast_p2='Fixed physical hidden trim; direct gather into final text KV during Prefill; zero full persistent visual KV; in-place append',
            production='No diagnostic hooks, FP32 stabilizers, canonical tiles, per-layer sync, torch.isin or dynamic visual-mask construction in Decode',
            profiler='Separate one fixed16 example per condition; hooks/census/events excluded from all latency samples',
            status='FROZEN')
        atomic(OUT / kind / 'protocol.json', protocol)


def verify(kind):
    p = read(OUT / kind / 'protocol.json')
    assert sources() == p['sources'], 'Frozen code changed'
    assert digest(read(OUT / kind / 'sample_manifest.json')) == p['manifest_sha256']
    assert digest(read(OUT / kind / 'input_bindings.json')) == p['bindings_sha256']
    return p


def common_prefix_audit(kind, a, b):
    count = min(len(a), len(b))
    if not count:
        raise RuntimeError('Empty logit audit')
    same = all(torch.equal(x, y) for x, y in zip(a[:count], b[:count]))
    return dict(exact=same, compared_steps=count,
                max_abs_logits=max(float((x - y).abs().max()) for x, y in zip(a[:count], b[:count])),
                allclose=all(torch.allclose(x, y, atol=TOLERANCES[kind]['logit_atol'],
                                            rtol=TOLERANCES[kind]['logit_rtol']) for x, y in zip(a[:count], b[:count])))


def smoke(adapter, kind):
    protocol = verify(kind)
    manifest, bindings = read(OUT / kind / 'sample_manifest.json'), read(OUT / kind / 'input_bindings.json')
    rows = []
    for sample in manifest['smoke']:
        spec = bindings[key(sample)]
        adapter.prepare(sample, spec)
        layout = Layout.from_adapter(adapter, sample, spec)
        native, nlog, nvalid, _, _ = generate(adapter, layout, 'Native Full', max_tokens=4, fixed=True, capture=True, audit=True)
        noop, zlog, zvalid, _, _ = generate(adapter, layout, 'Native-Noop', max_tokens=4, fixed=True, capture=True, audit=True)
        assert all(torch.equal(a, b) for a, b in zip(nvalid, zvalid)), 'Strict native/noop valid logits mismatch'
        assert native['token_ids'] == noop['token_ids']
        dense, dlog, _, dhid, drot = generate(adapter, layout, 'D1 Dense', max_tokens=4, fixed=True, capture=True, audit=True)
        old, olog, ohid = debug_run(adapter, sample, spec, 'D1', max_tokens=4, fixed=True, audit=True, capture=True)
        assert all(torch.equal(a, b) for a, b in zip(olog, dlog)), 'New dense changes diagnostic D1 logits'
        assert all(torch.equal(a, b) for a, b in zip(ohid, dhid)), 'New dense changes diagnostic D1 retained states'
        old_result = common_prefix_audit(kind, olog, dlog)
        generated, _, _, _ = hf_generate(adapter, native=True)
        manual, _, _, _, _ = generate(adapter, layout, 'Native Full')
        assert generated['token_ids'] == manual['token_ids'], 'Manual native greedy differs from HF generate'
        checks = []
        for condition in ('Fast-P1', 'Fast-P2'):
            value, vlog, _, vhid, vrot = generate(adapter, layout, condition, max_tokens=4,
                                                fixed=True, capture=True, audit=True)
            # Different generated prefixes cannot be treated as numerical parity tests.
            shared = 1
            while shared < min(len(dlog), len(vlog)) and dense['token_ids'][shared - 1] == value['token_ids'][shared - 1]:
                shared += 1
            eq = equivalence(kind, dlog[:shared], vlog[:shared], dhid, vhid)
            assert all(torch.equal(drot[0][name], vrot[0][name]) for name in ('position_ids', 'cos', 'sin'))
            for i in range(1, len(vrot)):
                # Both use unchanged logical token positions/mRoPE, independent of token values.
                assert all(torch.equal(drot[i][name], vrot[i][name]) for name in ('position_ids', 'cos', 'sin'))
            layers = value['kv']['layers']
            for layer in layers:
                compact = condition == 'Fast-P2' or layer['layer'] >= layout.cutoff
                expected = list(layout.text) if compact else list(range(layout.length))
                expected += list(range(layout.length, layout.length + 3))
                assert layer['original_positions'] == expected
                assert layer['logical_length'] == layout.length + 3
            assert len(value['audit']) == len(adapter.layers) * 4
            assert all(v['blocked_probability_max'] == 0 for v in value['audit'])
            assert all(v['query_count'] == (len(layout.text) if condition.startswith('Fast') and v['layer'] >= layout.cutoff else layout.length)
                       for v in value['audit'] if v['prefill'])
            checks.append(dict(condition=condition, numerical=eq, same_generated_tokens=value['token_ids'] == dense['token_ids'],
                               shared_prefix_predictions=shared, audit=value['audit'], kv=value['kv']))
        adapter.verify()
        rows.append(dict(sample=key(sample), layout=asdict(layout), native_noop_exact=True,
                         native_generate_tokens_equal=True, old_new_dense=old_result, conditions=checks,
                         hooks_clean=True, weights_and_input_unchanged=True))
        atomic(OUT / kind / 'validation.json', dict(status='SMOKE_IN_PROGRESS', completed=rows,
               protocol_sha256=digest(protocol)))
        print(f'{kind} SMOKE {key(sample)} OK', flush=True)
    atomic(OUT / kind / 'validation.json', dict(status='SEMANTIC_SMOKE_PASSED_NUMERICS_REPORTED', completed=rows,
           protocol_sha256=digest(protocol), historical_native_parity='PARTIAL conclusion retained; no stabilized kernels enabled'))


def scalar_measure(value):
    return {k: value[k] for k in ('latency_ms', 'ttft_ms', 'vision_ms', 'language_prefill_ms', 'decode_ms',
        'gpu_model_ms', 'cpu_setup_and_gaps_ms', 'prediction_positions', 'decode_steps', 'content_tokens',
        'peak_allocated', 'peak_reserved', 'initial_allocated', 'peak_increment', 'eos', 'truncated', 'empty')}


def benchmark(adapter, kind):
    protocol = verify(kind)
    manifest, bindings = read(OUT / kind / 'sample_manifest.json'), read(OUT / kind / 'input_bindings.json')
    start = time.time()
    batches = [('natural', manifest['samples'], 32, False),
               ('fixed16', manifest['smoke'][:1], 16, True), ('fixed64', manifest['smoke'][:1], 64, True)]
    for mode, samples, length, fixed in batches:
        for sample_index, sample in enumerate(samples):
            directory = OUT / kind / 'checkpoints' / mode
            path = directory / (key(sample) + '.json')
            if path.exists():
                previous = read(path)
                assert previous['protocol_sha256'] == digest(protocol) and previous['sample_sha256'] == digest(sample)
                assert all(len(v['measurements']) == 20 for v in previous['conditions'].values())
                continue
            adapter.prepare(sample, bindings[key(sample)])
            layout = Layout.from_adapter(adapter, sample, bindings[key(sample)])
            results = {}
            for condition in CONDITIONS:
                value, scores, _, _, _ = generate(adapter, layout, condition, length, fixed, capture=True)
                # Quality, census and diagnostic collection are outside timing repetitions.
                full_log = None
                if condition == 'Native Full':
                    full_log = scores[0]
                    full_token = int(full_log.argmax())
                    full_lp = float(full_log.log_softmax(-1)[0, full_token])
                    saved_full = full_log
                else:
                    full_log = saved_full
                lp = float(scores[0].log_softmax(-1)[0, full_token])
                value.update(correct=value['text'].rstrip('.').lower() == sample['answer'].lower(),
                             first_token=full_token if condition == 'Native Full' else int(scores[0].argmax()),
                             full_token_logp_loss=full_lp - lp,
                             first_token_flip=int(scores[0].argmax()) != full_token,
                             first_logits_sha256=tensor_hash(scores[0]),
                             first_logits_maxabs_vs_native=float((scores[0] - full_log).abs().max()))
                census = Census(adapter)
                with census.context():
                    counted, _, _, _, _ = generate(adapter, layout, condition, length, fixed)
                assert counted['token_ids'] == value['token_ids'], 'Instrumentation changes tokens'
                results[condition] = dict(quality=value, flop_census=census.rows(), measurements=[])
                for _ in range(protocol['warmups']):
                    generate(adapter, layout, condition, length, fixed)
            # Interleave conditions to reduce thermal/clock drift bias.
            for rep in range(protocol['repetitions']):
                order = CONDITIONS[(rep + sample_index) % 4:] + CONDITIONS[:(rep + sample_index) % 4]
                for condition in order:
                    value, _, _, _, _ = generate(adapter, layout, condition, length, fixed)
                    assert value['token_ids'] == results[condition]['quality']['token_ids'], 'Nondeterministic greedy output'
                    results[condition]['measurements'].append(scalar_measure(value))
            adapter.verify()
            checkpoint = dict(protocol_sha256=digest(protocol), sample_sha256=digest(sample),
                sample=key(sample), image_sha256=sample['image_sha256'], task=sample['task'], question=sample['question'],
                answer=sample['answer'], layout=asdict(layout), mode=mode, conditions=results)
            atomic(path, checkpoint)
            print(f'{kind} {mode} {sample_index + 1}/{len(samples)} {key(sample)} elapsed={time.time() - start:.1f}s', flush=True)
    atomic(OUT / kind / 'completion.json', dict(status='BENCHMARK_COMPLETE', natural=60, fixed_inputs=2,
        conditions=4, repetitions=20, warmups=5, benchmark_elapsed_seconds=time.time() - start,
        protocol_sha256=digest(protocol)))


def profile(adapter, kind):
    verify(kind)
    sample = read(OUT / kind / 'sample_manifest.json')['smoke'][0]
    bindings = read(OUT / kind / 'input_bindings.json')
    adapter.prepare(sample, bindings[key(sample)])
    layout = Layout.from_adapter(adapter, sample, bindings[key(sample)])
    for condition in CONDITIONS:
        generate(adapter, layout, condition, 16, True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as prof:
            result, _, _, _, _ = generate(adapter, layout, condition, 16, True, profile=True)
        events = []
        for event in prof.key_averages():
            events.append(dict(name=event.key, count=event.count, self_cpu_us=event.self_cpu_time_total,
                cpu_us=event.cpu_time_total, self_device_us=event.self_device_time_total,
                device_us=event.device_time_total, device_memory_bytes=event.self_device_memory_usage))
        kernels = {}
        for event in prof.events():
            if str(event.device_type).endswith('CUDA'):
                row = kernels.setdefault(event.name, dict(name=event.name, count=0, device_us=0.0))
                row['count'] += 1
                row['device_us'] += event.device_time_total
        dest = OUT / kind / 'profiler'
        atomic(dest / (condition.replace(' ', '_') + '.json'), dict(events=events, cuda_activities=list(kernels.values()),
            kv_management_ms=result['kv_management_ms'], sample=key(sample), fixed_predictions=16,
            note='Profiler/CUDA-event instrumentation excluded from latency samples'))
        # Full trace can be several MB; keep one representative compact fast path trace.
        if condition == 'Fast-P2':
            prof.export_chrome_trace(str(dest / 'Fast-P2.trace.json'))
        print(f'{kind} PROFILE {condition} complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'smoke', 'benchmark', 'profile', 'all'])
    parser.add_argument('--kind', choices=list(MODELS))
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
        return
    kind = args.kind
    assert kind
    protocol = verify(kind)
    adapter = None
    try:
        atomic(OUT / kind / 'running.json', dict(pid=os.getpid(), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
            action=args.action, started=time.time(), protocol_sha256=digest(protocol)))
        adapter = Adapter(kind)
        if args.action in ('smoke', 'all'):
            smoke(adapter, kind)
        if args.action in ('benchmark', 'all'):
            assert read(OUT / kind / 'validation.json')['status'] == 'SEMANTIC_SMOKE_PASSED_NUMERICS_REPORTED'
            benchmark(adapter, kind)
        if args.action in ('profile', 'all'):
            profile(adapter, kind)
        atomic(OUT / kind / 'process_completion.json', dict(status='COMPLETE', ended=time.time(),
            environment=adapter.environment, cuda_device=torch.cuda.get_device_name(0),
            properties=str(torch.cuda.get_device_properties(0))))
    except BaseException:
        atomic(OUT / kind / 'failure.json', dict(status='STOPPED', traceback=traceback.format_exc(),
            action=args.action, stopped=time.time()))
        raise
    finally:
        del adapter
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

"""Independent delivery audit and missing CLIP bmm FLOPs accounting.

This module is not part of the frozen worker entrypoint and does not modify
running source, checkpoints, latency samples, model outputs, or protocols.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from unittest.mock import patch
import gzip
import shutil

import numpy as np

from ...physical_visual_handoff.common import ROOT, atomic, read, sha, write_csv
from ..run import OUT, CONDITIONS, verify


def cpu_and_artifact_audit():
    for kind in ('llava', 'qwen'):
        log = (OUT / f'cpu_{kind}.log').read_text()
        assert 'Ran 6 tests' in log and log.strip().endswith('OK')
        verify(kind)
        complete = read(OUT / kind / 'process_completion.json')
        assert complete['status'] == 'COMPLETE'
        assert read(OUT / kind / 'completion.json')['status'] == 'BENCHMARK_COMPLETE'
        profiles = [p for p in (OUT / kind / 'profiler').glob('*.json') if not p.name.endswith('trace.json')]
        assert len(profiles) == 4
        trace = OUT / kind / 'profiler/Fast-P2.trace.json'
        assert trace.exists() or trace.with_suffix('.json.gz').exists()


def archive_traces():
    rows = []
    for kind in ('llava', 'qwen'):
        source = OUT / kind / 'profiler/Fast-P2.trace.json'
        dest = source.with_suffix('.json.gz')
        if source.exists():
            with source.open('rb') as src, gzip.GzipFile(filename=str(dest), mode='wb', mtime=0) as dst:
                shutil.copyfileobj(src, dst)
            original_sha = sha(source)
            original_size = source.stat().st_size
        else:
            import hashlib
            h = hashlib.sha256()
            original_size = 0
            with gzip.open(dest, 'rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
                    original_size += len(block)
            original_sha = h.hexdigest()
        rows.append(dict(model=kind, original=str(source.relative_to(OUT)), archive=str(dest.relative_to(OUT)),
                         original_sha256=original_sha, archive_sha256=sha(dest),
                         original_bytes=original_size, archive_bytes=dest.stat().st_size))
    atomic(OUT / 'trace_archives.json', dict(lossless_gzip=True, original_local_files_retained=True,
        files=rows, restore='gzip -dk <model>/profiler/Fast-P2.trace.json.gz'))


def clip_attention_supplement():
    """CLIP eager uses bmm rather than matmul; derive its two products exactly.

    The frozen census records all three actual Q/K/V projection FLOPs and
    the unique executed shapes. Their total determines how many actual
    Attention blocks ran, including blocks after the selected feature layer.
    An explicit config head count is bound to the immutable local model config.
    No latency or hardware-operation-count data are altered.
    """
    config_path = (Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf') /
                   'b234b804b114d9e37bb655e11cbbb5f5e971b7a9/config.json')
    config = read(config_path)['vision_config']
    heads = config['num_attention_heads']
    hidden = config['hidden_size']
    rows = []
    for path in sorted((OUT / 'llava/checkpoints').glob('*/*.json')):
        record = read(path)
        for condition, value in record['conditions'].items():
            projection = next(r for r in value['flop_census'] if r['phase'] == 'vision' and r['component'] == 'qkv')
            # Literal shape strings originate from tuple repr; parse structurally.
            import ast
            shapes = [ast.literal_eval(s) for s in projection['observed_shapes']]
            assert len(shapes) == 1
            input_shape, weight_shape, output_shape = shapes[0]
            batch, sequence, width = input_shape
            assert hidden == width == output_shape[-1]
            assert weight_shape == (hidden, hidden)
            per_layer_qkv = 6 * batch * sequence * hidden * hidden
            layers, remainder = divmod(projection['estimated_flops'], per_layer_qkv)
            assert remainder == 0 and layers == config['num_hidden_layers']
            head_dim = hidden // heads
            # QK^T and AV each cost 2*B*H*S*S*D. Both execute densely.
            attention_flops = 4 * batch * heads * sequence * sequence * head_dim * layers
            rows.append(dict(model='llava', mode=record['mode'], sample=record['sample'], condition=condition,
                phase='vision', component='attention_qk_av_bmm_supplement', estimated_flops=attention_flops,
                batch=batch, sequence=sequence, hidden_size=hidden, heads=heads, head_dim=head_dim,
                executed_attention_blocks=layers, native_primitive='torch.bmm',
                provenance='Actual projection-shape/count census + frozen CLIP architecture; two dense native bmm products/block',
                model_config_sha256=sha(config_path)))
    write_csv(OUT / 'clip_bmm_flops_supplement.csv', rows)
    return {(r['mode'], r['sample'], r['condition']): r['estimated_flops'] for r in rows}


def observed_process_memory():
    records = []
    for kind in ('llava', 'qwen'):
        for path in sorted((OUT / kind / 'checkpoints').glob('*/*.json')):
            value = read(path)
            for condition in CONDITIONS:
                q = value['conditions'][condition]['quality']
                for line in q['process_memory_snapshot']:
                    uuid, pid, mib = [s.strip() for s in line.split(',')]
                    records.append(dict(model=kind, mode=value['mode'], sample=value['sample'], condition=condition,
                        gpu_uuid=uuid, process_pid=int(pid), observed_process_vram_mib=int(mib),
                        live_kv_bytes=q['kv']['live_bytes'], allocated_kv_capacity_bytes=q['kv']['storage_bytes'],
                        note='nvidia-smi snapshot after untimed quality run while KV live; process allocator includes retained workspace; not a sampled peak'))
    write_csv(OUT / 'process_vram_snapshots.csv', records)
    return records


def baseline_and_weight_audit():
    measured = {}
    for line in (OUT / 'actual_weight_sha256.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        measured[name] = digest
    rows, configs = [], {}
    for kind in ('llava', 'qwen'):
        protocol = verify(kind)
        for name, details in protocol['old_weights_manifest'].items():
            assert measured[name] == details['sha256']
            assert Path(name).stat().st_size == details['size']
        history = read(ROOT / 'vlm_research/results/gqa3000_p1_layerwise_accuracy' / kind / 'baselines.json')
        for path in sorted((OUT / kind / 'checkpoints/natural').glob('*.json')):
            record = read(path)
            old = history[record['sample']]
            native = record['conditions']['Native Full']['quality']
            rows.append(dict(model=kind, sample=record['sample'], historical_answer=old['text'],
                current_answer=native['text'], exact_token_ids=old['token_ids'] == native['token_ids'],
                old_correct=bool(old['correct']), current_correct=native['correct'],
                baseline_used='Fresh Native Full in this run; historical values never substituted'))
        model_dir = Path(next(iter(protocol['old_weights_manifest']))).parent
        configs[kind] = {str(p): sha(p) for p in model_dir.glob('*.json')}
    write_csv(OUT / 'historical_native_audit.csv', rows)
    atomic(OUT / 'model_binding_audit.json', dict(actual_weight_sha256=measured,
        original_weight_manifests_equal=True, model_json_sha256=configs,
        historical_native_token_mismatches=sum(not r['exact_token_ids'] for r in rows)))
    return rows


def nvml_review():
    """NVML busy-time samples are NOT hardware SM or DRAM-throughput counters."""
    rows = []
    with (OUT / 'nvml_activity_samples.csv').open() as f:
        for raw in csv.reader(f):
            if len(raw) != 10:
                continue
            timestamp, index, uuid, gpu, memory, mib, sm, memclock, watts, degrees = [s.strip() for s in raw]
            rows.append(dict(timestamp=timestamp, gpu_index=int(index), uuid=uuid,
                gpu_busy_percent=float(gpu), memory_read_write_busy_percent=float(memory),
                used_mib=float(mib), sm_clock_mhz=float(sm), memory_clock_mhz=float(memclock),
                power_w=float(watts), temperature_c=float(degrees)))
    write_csv(OUT / 'nvml_activity_parsed.csv', rows)
    summaries = []
    for index, kind in ((2, 'llava'), (3, 'qwen')):
        selected = [r for r in rows if r['gpu_index'] == index]
        assert selected
        summaries.append(dict(model=kind, gpu_index=index, count=len(selected),
            gpu_busy_percent_mean=np.mean([r['gpu_busy_percent'] for r in selected]),
            memory_busy_percent_mean=np.mean([r['memory_read_write_busy_percent'] for r in selected]),
            sm_clock_mhz_median=np.median([r['sm_clock_mhz'] for r in selected]),
            power_w_mean=np.mean([r['power_w'] for r in selected]),
            provenance='NVML sampled device busy time; whole worker, not per-condition attribution; no SM/TC/GB/s conversion'))
    write_csv(OUT / 'nvml_activity_summary.csv', summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for index, kind, color in ((2, 'LLaVA / GPU2', '#227b96'), (3, 'Qwen / GPU3', '#c65b35')):
        selected = [r for r in rows if r['gpu_index'] == index]
        times = [datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f') for r in selected]
        axes[0].plot([(t - times[0]).total_seconds() / 60 for t in times],
                     [r['gpu_busy_percent'] for r in selected], color=color, lw=.8, alpha=.75, label=kind)
    axes[0].set(xlabel='Minutes since monitor start', ylabel='NVML GPU busy time (%)', ylim=(0, 105))
    axes[0].legend(fontsize=8)
    with (OUT / 'kernel_overhead.csv').open() as f:
        overhead = list(csv.DictReader(f))
    for i, (kind, color) in enumerate((('llava', '#227b96'), ('qwen', '#c65b35'))):
        selected = [next(r for r in overhead if r['model'] == kind and r['condition'] == c) for c in CONDITIONS]
        axes[1].bar(np.arange(4) + (i - .5) * .35,
                    [int(r['cuda_activity_count']) for r in selected], .35, label=kind, color=color)
    axes[1].set(xticks=np.arange(4), xticklabels=['Native', 'Dense', 'Fast-P1', 'Fast-P2'],
                ylabel='Separate-profiler CUDA activity count')
    axes[1].legend(fontsize=8)
    fig.suptitle('DRAM GB/s, hardware SM / Tensor Core counters: unavailable (permission denied)', fontsize=10)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / ('figure_D_bandwidth_unavailable_launch_counts.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)
    return summaries


def live_resource_audit(process_memory):
    xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True)
    root = ET.fromstring(xml)
    current = []
    for i, node in enumerate(root.findall('gpu')):
        processes = [dict(pid=p.findtext('pid'), name=p.findtext('process_name'),
                          memory=p.findtext('used_memory')) for p in node.findall('processes/process_info')]
        current.append(dict(index=i, uuid=node.findtext('uuid'), name=node.findtext('product_name'),
            memory_used=node.findtext('fb_memory_usage/used'), memory_free=node.findtext('fb_memory_usage/free'),
            gpu_util=node.findtext('utilization/gpu_util'), sm_clock=node.findtext('clocks/sm_clock'),
            memory_clock=node.findtext('clocks/mem_clock'), processes=processes))
    pids = {read(OUT / k / 'running.json')['pid'] for k in ('llava', 'qwen')}
    uuid_expected = [read(OUT / k / 'running.json')['cuda_visible'] for k in ('llava', 'qwen')]
    released = all(int(p['pid']) not in pids for g in current for p in g['processes'])
    assert released, 'A worker still owns GPU memory'
    atomic(OUT / 'resource_audit.json', dict(status='OWN_GPU_PROCESSES_RELEASED', used_gpu_uuids=uuid_expected,
        worker_pids=sorted(pids), own_processes_absent=True, final_gpus=current,
        initial_evidence='hardware_info.json nvidia_smi_q before launch; GPU0/GPU1 occupied; GPU2/GPU3 idle',
        others_terminated=False, data_root='/data/lk/ATE-worktrees/physical-handoff-fastpath-tflops',
        data1_used=False, disk=subprocess.check_output(['df', '-h', '/data'], text=True),
        process_memory_observations=len(process_memory),
        process_memory_note='process_vram_snapshots.csv; untimed live-KV snapshots, not profiler counter data'))
    return current


def add_technical_review(supplement):
    # Preserve the worker census verbatim. Add the independently derived missing
    # CLIP bmm operations to a separate complete arithmetic ledger.
    with (OUT / 'estimated_flops.csv').open() as f:
        rows = list(csv.DictReader(f))
    complete = []
    for row in rows:
        row = dict(row)
        if row['model'] == 'llava' and row['phase'] == 'vision' and row['component'] == 'TOTAL':
            extra = supplement[row['mode'], row['sample'], row['condition']]
            row['estimated_flops'] = int(row['estimated_flops']) + extra
            row['estimated_phase_tflops_contribution'] = row['estimated_flops'] / (float(row['phase_gpu_ms']) * 1e9)
            row['provenance'] += '; includes audited CLIP bmm supplement, see clip_bmm_flops_supplement.csv'
        complete.append(row)
    for (mode, sample, condition), extra in supplement.items():
        example = next(r for r in rows if r['model'] == 'llava' and r['mode'] == mode and
                       r['sample'] == sample and r['condition'] == condition and r['phase'] == 'vision' and r['component'] == 'TOTAL')
        ms = float(example['phase_gpu_ms'])
        complete.append(dict(model='llava', mode=mode, sample=sample, condition=condition, phase='vision',
            component='attention_qk_av_bmm_supplement', estimated_flops=extra, phase_gpu_ms=ms,
            estimated_phase_tflops_contribution=extra / (ms * 1e9),
            provenance='Actual CLIP executed QKV shapes/count + frozen architecture, not hardware counters'))
    write_csv(OUT / 'estimated_flops_worker_raw.csv', rows)
    write_csv(OUT / 'estimated_flops.csv', complete)
    write_csv(OUT / 'estimated_flops_complete.csv', complete)
    with (OUT / 'accuracy_60.csv').open() as f:
        quality = list(csv.DictReader(f))
    task_rows = []
    for kind in ('llava', 'qwen'):
        for task in sorted({r['task'] for r in quality}):
            for condition in CONDITIONS:
                selected = [r for r in quality if r['model'] == kind and r['task'] == task and r['condition'] == condition]
                assert len(selected) == 10
                task_rows.append(dict(model=kind, task=task, condition=condition, count=10,
                    correct=sum(r['correct'] == 'True' for r in selected),
                    accuracy=100 * sum(r['correct'] == 'True' for r in selected) / 10))
    write_csv(OUT / 'task_accuracy_60.csv', task_rows)
    write_csv(OUT / 'negative_cases.csv', [r for r in quality if r['harm_vs_native'] == 'True' or
                                          r['truncated'] == 'True' or r['changed_vs_dense'] == 'True'])
    text = (OUT / 'report_zh.md').read_text()
    text += '\n## 独立交付复核\n\n'
    text += ('冻结Census捕获torch.matmul但CLIP原生使用torch.bmm；此处不改运行源码或时延样本，'
             '按每题真实QKV执行Shape及调用数、固定Vision config补计CLIP两次bmm。'
             '详细推导和Config SHA见clip_bmm_flops_supplement.csv，主estimated_flops.csv及complete副本均已补齐。'
             '冻结Worker原始账本保存在estimated_flops_worker_raw.csv供核查；正文语言Prefill/Decode TFLOPS和图A不受该Vision补计影响。\n\n')
    text += ('process_vram_snapshots.csv记录每个未计时质量请求中KV仍存活时的进程显存；'
             '它是nvidia-smi快照而非全过程硬件峰值，不能冒充峰值。'
             '请求实际Peak Allocated/Reserved由未Profile的20次重复记录。'
             '任务分组与全部负面案例分别见task_accuracy_60.csv、negative_cases.csv。\n')
    for kind in ('llava', 'qwen'):
        profiles = []
        for path in sorted((OUT / kind / 'profiler').glob('*.json')):
            if path.name.endswith('trace.json'):
                continue
            record = read(path)
            activities = record['cuda_activities']
            top = sorted(activities, key=lambda r: r['device_us'], reverse=True)[:6]
            profiles.append(dict(condition=path.stem.replace('_', ' '), top_cuda_activities=top))
        atomic(OUT / kind / 'profile_review.json', profiles)
        text += f'\n{kind}独立Profiler最耗时活动保存在profile_review.json，不能从这些活动名直接推出DRAM带宽或Tensor Core利用率。\n'
    (OUT / 'report_zh.md').write_text(text)


def native_kv_profile(kind):
    """An independent, untimed native KV microprofile after the main worker exits."""
    import torch
    from transformers.cache_utils import DynamicCache
    from ...p1_layerwise_accuracy.engine import Adapter
    from ..engine import Layout, generate, InplaceKV
    verify(kind)
    adapter = Adapter(kind)
    sample = read(OUT / kind / 'sample_manifest.json')['smoke'][0]
    spec = read(OUT / kind / 'input_bindings.json')[sample['task'] + '__' + sample['question_id']]
    adapter.prepare(sample, spec)
    layout = Layout.from_adapter(adapter, sample, spec)
    generate(adapter, layout, 'Native Full', 16, True)
    original, counts, observations = DynamicCache.update, defaultdict(int), []
    def update(cache, keys, values, layer_idx, *a, **kw):
        phase = 'prefill' if counts[layer_idx] == 0 else 'decode'
        counts[layer_idx] += 1
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original(cache, keys, values, layer_idx, *a, **kw)
        end.record()
        extra_allocations = sum(int(out.data_ptr() != src.data_ptr()) for out, src in zip(result, (keys, values)))
        observations.append((phase, layer_idx, begin, end, extra_allocations,
                             keys.shape[-2], result[0].shape[-2]))
        return result
    with patch.object(DynamicCache, 'update', update):
        result, _, _, _, _ = generate(adapter, layout, 'Native Full', 16, True)
    torch.cuda.synchronize()
    rows = [dict(phase=p, layer=i, cuda_event_ms=a.elapsed_time(b),
                 cache_managed_new_tensor_count=allocations, query_tokens=q, physical_key_tokens=k)
            for p, i, a, b, allocations, q, k in observations]
    assert len(rows) == len(adapter.layers) * 16
    expected_prefill_allocations = 0 if kind == 'llava' else 2
    assert all(r['cache_managed_new_tensor_count'] == expected_prefill_allocations
               for r in rows if r['phase'] == 'prefill')
    assert all(r['cache_managed_new_tensor_count'] == 2 for r in rows if r['phase'] == 'decode')
    adapter.verify()
    atomic(OUT / kind / 'native_kv_update_audit.json', dict(model=kind, sample=sample['task'] + '__' + sample['question_id'],
        fixed_predictions=16, per_layer_events=rows, token_ids=result['token_ids'],
        management_ms={p: sum(r['cuda_event_ms'] for r in rows if r['phase'] == p) for p in ('prefill', 'decode')},
        decode_history_concat_allocations=sum(r['cache_managed_new_tensor_count'] for r in rows if r['phase'] == 'decode'),
        note='Separate event-instrumented native DynamicCache.update; includes GPU queue/launch gaps, not DRAM bandwidth; no impact on main timing samples'))
    uniform = [dict(condition='Native Full', per_layer_events=rows,
        management_ms={p: sum(r['cuda_event_ms'] for r in rows if r['phase'] == p) for p in ('prefill', 'decode')})]
    original_fast = InplaceKV.update
    for condition in CONDITIONS[1:]:
        generate(adapter, layout, condition, 16, True)
        fast_events = []
        def fast_update(cache, keys, values, layer_idx, *a, **kw):
            phase = 'prefill' if cache.prefill else 'decode'
            before = cache.allocations
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            value = original_fast(cache, keys, values, layer_idx, *a, **kw)
            end.record()
            fast_events.append((phase, layer_idx, begin, end, cache.allocations - before,
                                keys.shape[-2], cache.used[layer_idx]))
            return value
        with patch.object(InplaceKV, 'update', fast_update):
            measured, _, _, _, _ = generate(adapter, layout, condition, 16, True)
        torch.cuda.synchronize()
        events = [dict(phase=p, layer=i, cuda_event_ms=a.elapsed_time(b),
                       cache_managed_new_tensor_count=count, query_tokens=q, physical_key_tokens=k)
                  for p, i, a, b, count, q, k in fast_events]
        assert len(events) == len(adapter.layers) * 16
        assert all(r['cache_managed_new_tensor_count'] == 2 for r in events if r['phase'] == 'prefill')
        assert all(r['cache_managed_new_tensor_count'] == 0 for r in events if r['phase'] == 'decode')
        expected = read(OUT / kind / 'checkpoints/fixed16' / (sample['task'] + '__' + sample['question_id'] + '.json'))
        assert measured['token_ids'] == expected['conditions'][condition]['quality']['token_ids']
        uniform.append(dict(condition=condition, per_layer_events=events,
            management_ms={p: sum(r['cuda_event_ms'] for r in events if r['phase'] == p) for p in ('prefill', 'decode')}))
    atomic(OUT / kind / 'kv_update_uniform_audit.json', dict(model=kind, fixed_predictions=16,
        conditions=uniform, instrumentation='Same light CUDA-event wrapper around Cache.update; no full PyTorch profiler; separate from all latency samples'))
    del adapter
    torch.cuda.empty_cache()
    print(f'{kind} NATIVE_KV_MICROPROFILE_COMPLETE', flush=True)


def incorporate_native_kv_profiles():
    with (OUT / 'kernel_overhead.csv').open() as f:
        rows = list(csv.DictReader(f))
    for kind in ('llava', 'qwen'):
        p = read(OUT / kind / 'native_kv_update_audit.json')
        uniform = read(OUT / kind / 'kv_update_uniform_audit.json')
        for condition in uniform['conditions']:
            row = next(r for r in rows if r['model'] == kind and r['condition'] == condition['condition'])
            row['full_profiler_kv_prefill_ms'] = row['kv_prefill_ms']
            row['full_profiler_kv_decode_ms'] = row['kv_decode_ms']
            row['kv_prefill_ms'] = condition['management_ms']['prefill']
            row['kv_decode_ms'] = condition['management_ms']['decode']
            row['cache_managed_decode_concat_allocations'] = p['decode_history_concat_allocations'] if condition['condition'] == 'Native Full' else 0
            row['kv_event_provenance'] = 'Uniform light Cache.update CUDA-event wrapper; full-profiler timings retained separately, not directly compared'
    write_csv(OUT / 'kernel_overhead.csv', rows)


def seal_final_artifacts():
    """Run after redirected logs close, so the hash manifest seals final bytes."""
    cpu_and_artifact_audit()
    required = ['report_zh.md', 'algorithm.md', 'hardware_info.json', 'estimated_flops.csv',
        'achieved_tflops.csv', 'roofline_metrics.csv', 'dram_bandwidth.csv', 'sm_tensorcore_utilization.csv',
        'kv_cache_memory.csv', 'peak_vram.csv', 'latency_benchmark.csv', 'decode_throughput.csv',
        'kernel_overhead.csv', 'accuracy_60.csv', 'equivalence_audit.csv', 'validation.json', 'resource_audit.json']
    assert all((OUT / p).is_file() for p in required)
    assert read(OUT / 'resource_audit.json')['own_processes_absent']
    with (OUT / 'kernel_overhead.csv').open() as f:
        overhead = list(csv.DictReader(f))
    with (OUT / 'accuracy_60.csv').open() as f:
        quality = list(csv.DictReader(f))
    assert len(quality) == 480
    comparisons = []
    for kind in ('llava', 'qwen'):
        for base, target in [('Native Full', 'D1 Dense'), ('Native Full', 'Fast-P2'),
                             ('D1 Dense', 'Fast-P1'), ('D1 Dense', 'Fast-P2'), ('Fast-P1', 'Fast-P2')]:
            baseline = {r['sample']: r for r in quality if r['model'] == kind and r['condition'] == base}
            candidate = {r['sample']: r for r in quality if r['model'] == kind and r['condition'] == target}
            assert baseline.keys() == candidate.keys() and len(baseline) == 60
            harm = sum(baseline[k]['correct'] == 'True' and candidate[k]['correct'] == 'False' for k in baseline)
            rescue = sum(baseline[k]['correct'] == 'False' and candidate[k]['correct'] == 'True' for k in baseline)
            comparisons.append(dict(model=kind, baseline=base, candidate=target, n=60, harm=harm, rescue=rescue,
                delta_accuracy_pp=100 * (rescue - harm) / 60,
                changed_tokens=sum(baseline[k]['token_ids'] != candidate[k]['token_ids'] for k in baseline),
                interpretation='Exploratory pilot; not noninferiority or equivalence'))
        native = read(OUT / kind / 'native_kv_update_audit.json')
        sample = read(OUT / kind / 'checkpoints/fixed16' / (native['sample'] + '.json'))
        assert native['token_ids'] == sample['conditions']['Native Full']['quality']['token_ids']
    write_csv(OUT / 'paired_quality_comparisons.csv', comparisons)
    with (OUT / 'latency_benchmark.csv').open() as f:
        latency = list(csv.DictReader(f))
    paired_latency = []
    for kind in ('llava', 'qwen'):
        for reference in ('Native Full', 'D1 Dense'):
            for candidate in ('Fast-P1', 'Fast-P2'):
                baseline = {r['sample']: float(r['median']) for r in latency if r['model'] == kind and
                            r['mode'] == 'natural' and r['condition'] == reference and r['metric'] == 'latency_ms'}
                alternative = {r['sample']: float(r['median']) for r in latency if r['model'] == kind and
                               r['mode'] == 'natural' and r['condition'] == candidate and r['metric'] == 'latency_ms'}
                speeds = [baseline[k] / alternative[k] for k in baseline]
                paired_latency.append(dict(model=kind, baseline=reference, candidate=candidate, n=60,
                    baseline_mean_ms=np.mean(list(baseline.values())), candidate_mean_ms=np.mean(list(alternative.values())),
                    mean_latency_speedup=np.mean(list(baseline.values())) / np.mean(list(alternative.values())),
                    paired_speedup_median=np.median(speeds), paired_speedup_p10=np.percentile(speeds, 10),
                    paired_speedup_p90=np.percentile(speeds, 90), paired_speedup_min=min(speeds), paired_speedup_max=max(speeds)))
    write_csv(OUT / 'paired_latency_summary.csv', paired_latency)
    redraw_estimated_tflops()
    text = (OUT / 'report_zh.md').read_text()
    marker = '\n## 统一仪表KV微测与执行耗时\n'
    if marker in text:
        text = text.split(marker)[0]
    text += marker + '\n四路径额外采用同样的轻量CUDA Events包裹Cache.update，避免把带完整Profiler的Fast时间与不带Profiler的Native时间直接比较。'
    text += '原完整Profiler测得的KV时间另列保留，不直接相除；两组结果都不属于正式Latency重复。\n\n'
    text += '|模型|条件|Prefill KV管理 ms|15步Decode KV管理合计 ms|Decode历史拼接新分配数|\n|---|---|---:|---:|---:|\n'
    for row in overhead:
        text += f"|{row['model']}|{row['condition']}|{float(row['kv_prefill_ms']):.3f}|{float(row['kv_decode_ms']):.3f}|{row['cache_managed_decode_concat_allocations']}|\n"
    text += ('\n这些KV时间包括小操作的排队与启动，不是纯内存拷贝时间。即使移除历史KV拼接，'
             '仍有模型权重读取、QKV/MLP/LM Head及原生eager操作，不能据KV缩小直接预测Decode加速。\n\n')
    for kind in ('llava', 'qwen'):
        complete = read(OUT / kind / 'completion.json')
        net_ms = 0
        for path in (OUT / kind / 'checkpoints').glob('*/*.json'):
            record = read(path)
            net_ms += sum(m['latency_ms'] for c in record['conditions'].values() for m in c['measurements'])
        text += (f"- {kind}：基准驱动墙钟{complete['benchmark_elapsed_seconds']/60:.2f}分钟；"
                 f"20次正式重复的请求时延相加为{net_ms/60000:.2f}分钟。前者含预热、质量/形状采集和检查点写入，"
                 '不含实现、CPU测试、模型加载、Smoke、后处理和推送。两模型双GPU并行，不能将两卡时间相加当总墙钟。\n')
    text += ('\n本60题是历史探索工程队列，不是独立外部确认集。TFLOPS下降并不自动意味着优化失败：'
             'LLaVA的估算Prefill TFLOPS由Dense的48.456降至Fast-P2的45.376，但工作量下降且自然请求更快；'
             'Qwen则从24.709降至16.544，同时Latency近乎不变，不能将少算宣称为等幅加速。\n')
    text += '\n另给平均请求时延口径，避免只看跨题中位数掩盖长输入差异（逐题先取20次中位数）：\n'
    for row in paired_latency:
        if row['candidate'] == 'Fast-P2':
            text += (f"- {row['model']} 相对{row['baseline']}：平均请求{row['baseline_mean_ms']:.2f}→"
                     f"{row['candidate_mean_ms']:.2f}ms，均值口径加速{row['mean_latency_speedup']:.3f}×；"
                     f"逐题加速比P10/P90为{row['paired_speedup_p10']:.3f}/{row['paired_speedup_p90']:.3f}。\n")
    text += ('\nQwen不能笼统写成所有输入都无收益：部分较长动态视觉输入的节省更明显，'
             '但主中位数收益很小；不得据此事后挑选有利题目重新定义主结果。\n')
    (OUT / 'report_zh.md').write_text(text)
    validation = read(OUT / 'validation.json')
    validation.update(final_logs_sealed=True, required_artifacts=required,
        paired_quality_comparisons=len(comparisons), native_kv_microprofile_tokens_match=True,
        reporting_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')})
    atomic(OUT / 'validation.json', validation)
    hashes = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}
    atomic(OUT / 'artifact_hashes.json', hashes)
    print(f'FINAL_SEAL_PASSED: {len(hashes)} artifacts, 480 natural quality records, 9920 timed requests', flush=True)


def redraw_estimated_tflops():
    """Separate Prefill/Decode scales so sub-TFLOP Decode bars remain legible."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (OUT / 'estimated_flops.csv').open() as f:
        rows = list(csv.DictReader(f))
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.1))
    for col, kind in enumerate(('llava', 'qwen')):
        for row, phase in enumerate(('prefill', 'decode')):
            ax = axes[row, col]
            values = [float(next(r for r in rows if r['model'] == kind and r['mode'] == 'fixed64' and
                          r['condition'] == condition and r['phase'] == phase and r['component'] == 'TOTAL')
                            ['estimated_phase_tflops_contribution']) for condition in CONDITIONS]
            bars = ax.bar(np.arange(4), values, color=['#4b5563', '#c65b35', '#227b96', '#278354'])
            ax.bar_label(bars, fmt='%.2f' if phase == 'prefill' else '%.3f', padding=3, fontsize=9)
            ax.set(xticks=np.arange(4), xticklabels=['Native', 'Dense', 'Fast-P1', 'Fast-P2'],
                   ylabel='Estimated algorithmic TFLOPS', ylim=(0, max(values) * 1.2),
                   title=f'{kind.upper()} / {phase}')
    fig.suptitle('Estimated throughput (fixed64); hardware counters unavailable\nFLOPs / CUDA-event time only; Prefill and Decode use separate scales', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / ('figure_A_estimated_vs_hardware_tflops.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-kv-profile', choices=['llava', 'qwen'])
    parser.add_argument('--seal-only', action='store_true')
    args = parser.parse_args()
    if args.seal_only:
        seal_final_artifacts()
        return
    if args.native_kv_profile:
        native_kv_profile(args.native_kv_profile)
        return
    cpu_and_artifact_audit()
    supplement = clip_attention_supplement()
    memory = observed_process_memory()
    live_resource_audit(memory)
    add_technical_review(supplement)
    incorporate_native_kv_profiles()
    baselines = baseline_and_weight_audit()
    activity = nvml_review()
    archive_traces()
    text = (OUT / 'report_zh.md').read_text()
    text += '\n## 权重、历史基线及设备活动复核\n\n'
    text += ('两模型7个实际权重文件本轮重新SHA256，与冻结清单逐项一致。'
             f"120个Native历史配对中Token序列变化{sum(not r['exact_token_ids'] for r in baselines)}项；"
             '本轮一律使用新测Native，不拼接历史时延或质量结果。详见model_binding_audit.json、historical_native_audit.csv。\n\n')
    text += ('Figure D另给NVML GPU busy-time采样曲线及独立Profiler CUDA活动数。'
             'NVML的memory utilization表示采样窗口读写忙碌时间比例，不是DRAM带宽，也不是Tensor Core或SM硬件Throughput。'
             '采样覆盖两Worker的部分执行区间，不对单个条件归因。\n\n')
    text += ('更新范数/Softmax/激活等非矩阵操作，以及GEMM内部Tile补齐或其他底层指令，'
             '不包含在形状FLOPs账本中；减少的是张量形状对应的估算矩阵算术量，不能宣称硬件实际浮点指令同比减少。\n\n')
    text += ('CPU核心回归为两环境各6/6通过，独立FLOPs记账测试另2/2通过。'
             '后者主动隐藏所有CUDA设备，CPU Profiler启动时记录的gpuGetDeviceCount code 100为无可见GPU探测，'
             '不是实验GPU错误，两项测试实际均通过。\n\n')
    for row in activity:
        text += f"- {row['model']}：NVML平均设备忙碌{row['gpu_busy_percent_mean']:.1f}%，SM时钟中位数{row['sm_clock_mhz_median']:.0f}MHz。\n"
    text += ('\n厂商理论参考见hardware_reference.json；RTX3090的BF16/FP16且FP32累积Tensor Core稠密参考为71TFLOPS，'
             'FP16累积与结构化稀疏峰值另列，均不是实测，也未用于推导实际带宽或硬件TFLOPS。\n')
    text += ('\n## 分模型决策\n\n'
             'LLaVA：存在可复现的自然短答案时延收益，值得在明确承认近似数值路径、预先冻结质量门槛后进入240题质量验证；'
             '不能以本轮60题的净纠正直接宣称非劣效或质量提升。\n\n'
             'Qwen：KV显著下降，但自然短答案时延收益很小，且旧固定隐藏表示容差未通过。'
             '目前不优先扩大为240题“等价且加速”的验证；先明确性能证据和近似计算接受标准。'
             '这不是自动追加的任务，本轮到此停止。\n')
    (OUT / 'report_zh.md').write_text(text)
    validation = read(OUT / 'validation.json')
    validation.update(delivery_audit='PASSED', gpu_released=True,
        original_checkpoints_untouched=True, complete_vision_bmm_ledger='estimated_flops_complete.csv',
        model_and_runtime_code_commit='daacc49f02ec1ef70ff06fa23e9b4872bc24815d',
        reporting_source_sha256={str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')})
    atomic(OUT / 'validation.json', validation)
    hashes = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}
    atomic(OUT / 'artifact_hashes.json', hashes)
    print('DELIVERY_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    main()

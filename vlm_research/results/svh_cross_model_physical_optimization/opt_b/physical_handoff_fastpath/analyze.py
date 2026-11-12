"""Aggregate only complete, hash-bound checkpoints; no invented hardware counters."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..physical_visual_handoff.common import atomic, digest, read, sha, write_csv
from .run import OUT, CONDITIONS, verify

NA = 'N/A'
COLORS = ['#4b5563', '#c65b35', '#227b96', '#278354']


def save(fig, stem):
    fig.savefig(OUT / (stem + '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(OUT / (stem + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    accuracy, kv, peaks, latency, flops, throughput, equivalence_rows, overhead = [], [], [], [], [], [], [], []
    summaries, all_records = {}, {}
    for kind in ('llava', 'qwen'):
        protocol = verify(kind)
        files = sorted((OUT / kind / 'checkpoints').glob('*/*.json'))
        records = [read(p) for p in files]
        assert sum(r['mode'] == 'natural' for r in records) == 60
        assert sum(r['mode'] != 'natural' for r in records) == 2
        expected = set(read(OUT / kind / 'sample_manifest.json')['historical_pilot60'])
        assert {r['sample'] for r in records if r['mode'] == 'natural'} == expected
        assert len({r['image_sha256'] for r in records if r['mode'] == 'natural'}) == 60
        all_records[kind] = records
        per_condition = defaultdict(list)
        for record in records:
            assert record['protocol_sha256'] == digest(protocol)
            for condition in CONDITIONS:
                value = record['conditions'][condition]
                quality, times = value['quality'], value['measurements']
                assert len(times) == 20
                base = dict(model=kind, mode=record['mode'], sample=record['sample'], condition=condition)
                if record['mode'] == 'natural':
                    full = record['conditions']['Native Full']['quality']
                    dense = record['conditions']['D1 Dense']['quality']
                    accuracy.append(dict(**base, task=record['task'], image_sha256=record['image_sha256'],
                        question=record['question'], gold=record['answer'], answer=quality['text'],
                        token_ids=quality['token_ids'], correct=quality['correct'], eos=quality['eos'],
                        empty=quality['empty'], truncated=quality['truncated'],
                        harm_vs_native=full['correct'] and not quality['correct'],
                        rescue_vs_native=not full['correct'] and quality['correct'],
                        changed_vs_native=quality['token_ids'] != full['token_ids'],
                        changed_vs_dense=quality['token_ids'] != dense['token_ids'],
                        first_token_flip=quality['first_token_flip'],
                        full_token_logp_loss=quality['full_token_logp_loss']))
                    per_condition[condition].append(value)
                for metric in ('latency_ms', 'ttft_ms', 'vision_ms', 'language_prefill_ms', 'decode_ms',
                               'gpu_model_ms', 'cpu_setup_and_gaps_ms'):
                    arr = [r[metric] for r in times]
                    latency.append(dict(**base, metric=metric, median=np.median(arr), p10=np.percentile(arr, 10),
                                        p90=np.percentile(arr, 90), repetitions=20, warmups=5))
                for layer in quality['kv']['layers']:
                    kv.append(dict(**base, **{k: v for k, v in layer.items() if k not in ('original_positions', 'key_stride', 'value_stride')}))
                peaks.append(dict(**base, live_kv_bytes=quality['kv']['live_bytes'],
                    allocated_kv_capacity_bytes=quality['kv']['storage_bytes'],
                    peak_allocated_median=np.median([r['peak_allocated'] for r in times]),
                    peak_reserved_median=np.median([r['peak_reserved'] for r in times]),
                    peak_increment_median=np.median([r['peak_increment'] for r in times]),
                    kv_allocations=quality['kv'].get('allocations'),
                    kv_writes=quality['kv'].get('write_calls'),
                    kv_direct_gathers=quality['kv'].get('direct_gather_calls'),
                    historical_cache_copies=quality['kv'].get('historical_cache_copies'),
                    append_concatenations=quality['kv'].get('append_concatenations')))
                phase_totals = defaultdict(int)
                for row in value['flop_census']:
                    phase_totals[row['phase']] += row['estimated_flops']
                    phase = row['phase']
                    metric = {'vision': 'vision_ms', 'prefill': 'language_prefill_ms', 'decode': 'decode_ms'}[phase]
                    ms = float(np.median([r[metric] for r in times]))
                    flops.append(dict(**base, phase=phase, component=row['component'],
                        estimated_flops=row['estimated_flops'], phase_gpu_ms=ms,
                        estimated_phase_tflops_contribution=row['estimated_flops'] / (ms * 1e9) if ms else NA,
                        provenance=row['provenance'], observed_shapes=row['observed_shapes']))
                for phase, total in phase_totals.items():
                    metric = {'vision': 'vision_ms', 'prefill': 'language_prefill_ms', 'decode': 'decode_ms'}[phase]
                    ms = float(np.median([r[metric] for r in times]))
                    flops.append(dict(**base, phase=phase, component='TOTAL', estimated_flops=total,
                        phase_gpu_ms=ms, estimated_phase_tflops_contribution=total / (ms * 1e9) if ms else NA,
                        provenance='Estimated executed arithmetic / CUDA event phase duration; NOT hardware achieved TFLOPS'))
                if record['mode'] != 'natural':
                    per_token = [r['decode_ms'] / r['decode_steps'] for r in times]
                    throughput.append(dict(**base, prediction_positions=quality['prediction_positions'],
                        decode_steps=quality['decode_steps'], median_ms_per_decode_token=np.median(per_token),
                        p10_ms_per_decode_token=np.percentile(per_token, 10), p90_ms_per_decode_token=np.percentile(per_token, 90),
                        median_decode_tokens_s=1000 / np.median(per_token),
                        live_kv_bytes=quality['kv']['live_bytes'], allocated_kv_bytes=quality['kv']['storage_bytes'],
                        caveat='Engineering only; forced beyond EOS; not normal short-answer latency'))
        for condition in CONDITIONS:
            values = per_condition[condition]
            summary = dict(accuracy=100 * np.mean([v['quality']['correct'] for v in values]))
            for metric in ('latency_ms', 'ttft_ms', 'language_prefill_ms', 'vision_ms', 'decode_ms', 'peak_allocated', 'peak_reserved'):
                summary[metric] = float(np.median([np.median([t[metric] for t in v['measurements']]) for v in values]))
            summary['live_kv_mib'] = float(np.median([v['quality']['kv']['live_bytes'] / 2**20 for v in values]))
            summary['allocated_kv_mib'] = float(np.median([v['quality']['kv']['storage_bytes'] / 2**20 for v in values]))
            summary['flops'] = float(np.median([sum(x['estimated_flops'] for x in v['flop_census']) for v in values]))
            summary['prefill_flops'] = float(np.median([sum(x['estimated_flops'] for x in v['flop_census'] if x['phase'] == 'prefill') for v in values]))
            summaries[kind, condition] = summary
        validation = read(OUT / kind / 'validation.json')
        for example in validation['completed']:
            equivalence_rows.append(dict(model=kind, sample=example['sample'], condition='Native-Noop',
                exact_valid_logits=example['native_noop_exact'], semantics_passed=True))
            equivalence_rows.append(dict(model=kind, sample=example['sample'], condition='D1 Dense vs old D1',
                exact_valid_logits=example['old_new_dense']['exact'], semantics_passed=True))
            for condition in example['conditions']:
                equivalence_rows.append(dict(model=kind, sample=example['sample'], condition=condition['condition'],
                    **condition['numerical'], same_generated_tokens=condition['same_generated_tokens'], semantics_passed=True))
        for path in sorted((OUT / kind / 'profiler').glob('*.json')):
            if path.name.endswith('trace.json'):
                continue
            profile = read(path)
            events = profile['events']
            kernels = profile.get('cuda_activities', [])
            overhead.append(dict(model=kind, condition=path.stem.replace('_', ' '), fixed_predictions=16,
                cuda_activity_count=sum(v['count'] for v in kernels),
                cuda_activity_ms=sum(v['device_us'] for v in kernels) / 1000,
                cpu_cuda_launch_us=sum(v['self_cpu_us'] for v in events if 'LaunchKernel' in v['name']),
                aten_cat_calls=sum(v['count'] for v in events if v['name'] == 'aten::cat'),
                kv_prefill_ms=profile['kv_management_ms'].get('prefill', NA),
                kv_decode_ms=profile['kv_management_ms'].get('decode', NA),
                caveat='Profiler-instrumented; overhead excluded from main latency; aten::cat includes native rotary/vision operations'))

    tables = {'accuracy_60.csv': accuracy, 'kv_cache_memory.csv': kv, 'peak_vram.csv': peaks,
              'latency_benchmark.csv': latency, 'estimated_flops.csv': flops,
              'decode_throughput.csv': throughput, 'equivalence_audit.csv': equivalence_rows,
              'kernel_overhead.csv': overhead}
    unavailable = [dict(model=kind, condition=condition, phase=phase, status='HARDWARE_TFLOPS_UNAVAILABLE',
        value=NA, reason='Nsight Compute ERR_NVGPUCTRPERM; hardware counters denied',
        measured_tflops=NA, dram_read_bytes=NA, dram_write_bytes=NA, dram_gb_s=NA,
        l2_hit_rate=NA, sm_throughput_percent=NA, tensorcore_active_percent=NA,
        arithmetic_intensity=NA, bottleneck='Not established by counters')
        for kind in ('llava', 'qwen') for condition in CONDITIONS for phase in ('prefill', 'decode')]
    for name in ('achieved_tflops.csv', 'roofline_metrics.csv', 'dram_bandwidth.csv', 'sm_tensorcore_utilization.csv'):
        tables[name] = unavailable
    for name, rows in tables.items():
        write_csv(OUT / name, rows)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'axes.grid': True, 'grid.alpha': .18})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, kind in zip(axes, ('llava', 'qwen')):
        for j, phase in enumerate(('prefill', 'decode')):
            ys = [np.median([r['estimated_phase_tflops_contribution'] for r in flops
                if r['model'] == kind and r['mode'] == 'fixed64' and r['condition'] == c and
                r['phase'] == phase and r['component'] == 'TOTAL']) for c in CONDITIONS]
            ax.bar(np.arange(4) + (j - .5) * .34, ys, .34, label=phase)
        ax.set(xticks=np.arange(4), xticklabels=['Native', 'Dense', 'Fast-P1', 'Fast-P2'],
               title=kind.upper(), ylabel='Estimated algorithmic TFLOPS')
        ax.legend()
    fig.suptitle('FLOPs / CUDA-event time only; hardware TFLOPS unavailable', fontsize=11)
    fig.tight_layout()
    save(fig, 'figure_A_estimated_vs_hardware_tflops')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, kind in zip(axes, ('llava', 'qwen')):
        base = summaries[kind, 'D1 Dense']
        for c, color in zip(CONDITIONS, COLORS):
            s = summaries[kind, c]
            ax.scatter(100 * (1 - s['prefill_flops'] / base['prefill_flops']),
                       base['latency_ms'] / s['latency_ms'], color=color, s=55, label=c)
        ax.axhline(1, color='#555', lw=.8)
        ax.set(xlabel='Language Prefill FLOPs reduction vs Dense (%)', ylabel='Natural EOS E2E speedup vs Dense', title=kind.upper())
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'figure_B_flops_vs_speedup')

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for col, kind in enumerate(('llava', 'qwen')):
        vals = [summaries[kind, c] for c in CONDITIONS]
        axes[0, col].bar(np.arange(4) - .17, [s['live_kv_mib'] for s in vals], .34, label='Live KV')
        axes[0, col].bar(np.arange(4) + .17, [s['allocated_kv_mib'] for s in vals], .34, label='KV capacity')
        axes[0, col].set(title=kind.upper(), ylabel='KV MiB')
        axes[0, col].legend()
        axes[1, col].bar(np.arange(4), [s['peak_allocated'] / 2**20 for s in vals], color=COLORS)
        axes[1, col].set(ylabel='Total peak allocated MiB')
        for row in range(2):
            axes[row, col].set(xticks=np.arange(4), xticklabels=['Native', 'Dense', 'Fast-P1', 'Fast-P2'])
    fig.tight_layout()
    save(fig, 'figure_C_kv_vs_total_vram')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, kind in zip(axes, ('llava', 'qwen')):
        vals = [next(r for r in overhead if r['model'] == kind and r['condition'] == c) for c in CONDITIONS]
        ax.bar(np.arange(4), [r['cuda_activity_count'] for r in vals], color=COLORS)
        ax.set(xticks=np.arange(4), xticklabels=['Native', 'Dense', 'Fast-P1', 'Fast-P2'],
               ylabel='CUDA activity count (separate profiler)', title=kind.upper())
        ax.text(.03, .97, 'DRAM / SM / Tensor Core counters: N/A\nERR_NVGPUCTRPERM',
                transform=ax.transAxes, va='top', fontsize=8)
    fig.tight_layout()
    save(fig, 'figure_D_bandwidth_unavailable_launch_counts')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, kind in zip(axes, ('llava', 'qwen')):
        for i, (condition, color) in enumerate(zip(CONDITIONS, COLORS)):
            points = [r for r in latency if r['model'] == kind and r['mode'] == 'natural' and
                      r['condition'] == condition and r['metric'] == 'latency_ms']
            ax.scatter(np.full(len(points), i) + np.linspace(-.1, .1, len(points)),
                       [r['median'] for r in points], color=color, alpha=.28, s=12)
            ax.plot([i - .2, i + .2], [np.median([r['median'] for r in points])] * 2, color=color, lw=3)
        ax.set(title=kind.upper(), xticks=np.arange(4), xticklabels=['Native', 'Dense', 'Fast-P1', 'Fast-P2'],
               ylabel='Natural EOS E2E latency (ms)')
    fig.tight_layout()
    save(fig, 'figure_E_natural_gqa_e2e')

    lines = ['# Visual Handoff Fast Path：60题质量与性能Pilot', '',
        '## 范围与有效性', '',
        '历史原生数值验收仍为PARTIAL，未被覆盖或改写。本轮只使用原冻结60题（每任务10题），不构成非劣效验证。',
        '所有路径原精度/eager，无FP32稳定Linear、无Canonical Tile、无训练。Native Full保留直接视觉读取，D1及Fast的写入依赖与其不同；物理优化的主要功能配对是D1。',
        'Native/No-op全部有效Logits严格一致；新Dense与旧Debug D1逐位一致；两模型各3题Mask/位置/Cache/Hook真实GPU Smoke通过。',
        '每题每条件5次预热、20次计时；条件按轮次交错。E2E含每请求Mask/KV设置和冷Cache，不含磁盘IO/CPU图像预处理；不缓存视觉特征，每次都执行原生视觉编码器。',
        '固定16/64为工程测试，可能越过EOS，不能代替自然短答案时延。Profiler和FLOPs形状采集与时延重复完全分离。', '',
        '## 核心结果', '',
        '|模型|条件|Accuracy|自然E2E ms|语言Prefill ms|KV Live MiB|KV Capacity MiB|总Peak Allocated MiB|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for kind in ('llava', 'qwen'):
        for c in CONDITIONS:
            s = summaries[kind, c]
            lines.append(f"|{kind}|{c}|{s['accuracy']:.2f}%|{s['latency_ms']:.2f}|{s['language_prefill_ms']:.2f}|{s['live_kv_mib']:.2f}|{s['allocated_kv_mib']:.2f}|{s['peak_allocated']/2**20:.1f}|")
    lines += ['', '各列为60题各自20次重复中位数的跨题中位数；不把跨题中位数当成总批次耗时。p10/p90及逐题明细见CSV。', '',
              '## FLOPs、TFLOPS与硬件计数器', '',
              '`HARDWARE_TFLOPS_UNAVAILABLE`：Nsight Compute实际探测返回ERR_NVGPUCTRPERM，保留原日志。DRAM读写、GB/s、L2命中、SM/TC利用率、硬件浮点操作计数均未测得，CSV标N/A。未修改系统权限，未将SM百分比或理论峰值换算为实测TFLOPS。',
              'Estimated Algorithmic TFLOPS来自独立执行形状账本的Linear、Conv、QK和AV FLOPs（每MAC=2）除以对应CUDA Event阶段时间；包含原生Vision、QKV、MLP、投影、LM Head，排除Norm/Softmax/激活/RoPE等非矩阵操作，不是硬件Achieved TFLOPS。', '']
    for kind in ('llava', 'qwen'):
        dense = summaries[kind, 'D1 Dense']
        fast = summaries[kind, 'Fast-P2']
        natives = summaries[kind, 'Native Full']
        lines.append(f"- {kind} Fast-P2相对Dense：语言Prefill估算矩阵FLOPs减少{100*(1-fast['prefill_flops']/dense['prefill_flops']):.1f}%；语言Prefill加速{dense['language_prefill_ms']/fast['language_prefill_ms']:.3f}×；自然EOS E2E加速{dense['latency_ms']/fast['latency_ms']:.3f}×；相对Native E2E为{natives['latency_ms']/fast['latency_ms']:.3f}×。KV Live节省{dense['live_kv_mib']-fast['live_kv_mib']:.2f}MiB；总Peak Allocated下降{100*(1-fast['peak_allocated']/dense['peak_allocated']):.2f}%。")
        for mode in ('fixed16', 'fixed64'):
            vals = [r for r in throughput if r['model'] == kind and r['mode'] == mode]
            lines.append(f"- {kind} {mode} Decode：" + '；'.join(f"{r['condition']} {r['median_ms_per_decode_token']:.3f}ms/token" for r in vals) + '。')
        for phase in ('prefill', 'decode'):
            vals = [r for r in flops if r['model'] == kind and r['mode'] == 'fixed64' and r['phase'] == phase and r['component'] == 'TOTAL']
            lines.append(f"- {kind} fixed64 {phase}估算TFLOPS：" + '；'.join(f"{r['condition']} {r['estimated_phase_tflops_contribution']:.3f}" for r in vals) + '（非硬件计数器）。')
    lines += ['', '## Cache、显存与开销解释', '',
        'Fast-P1在写入截止点真正缩短Hidden，前段仍保留稠密KV。Fast-P2在Prefill直接gather非视觉KV到最终缓冲，不先持久保存整层视觉KV再复制删除；Decode原位写入，历史Cache拼接/整段复制为0。KV Live与预分配容量分别报告。',
        '不存在跨请求KV共享；每个请求独立分配，测量含其开销。原始逻辑位置不压缩；Qwen原生DeepStack/mRoPE保留。KV管理GPU时间来自单独Profiler运行的CUDA Events，包括排队/启动开销，不能冒充纯内存带宽。',
        'GPU进程显存快照见resource_audit.json；总Peak Allocated/Reserved包含权重、临时张量、缓存和工作空间，不能用KV减少比例代替总显存减少。peak_increment为相对请求开始的动态峰值，不是只含临时Tensor的精确分解。',
        'KV缩小不必然让Decode同比加速：每步权重矩阵读取、GEMV/小GEMM、原生Python调度和kernel启动仍存在。独立Profiler提供CUDA活动数和CPU启动时间证据；硬件带宽计数器不可用，不能确定宣称带宽受限或计算受限。',
        '矩阵运算变小可能降低估算TFLOPS，但同时减少工作量和时延；不以TFLOPS单独判定成功。若Decode无改善，保留负结果，不用强制64步掩盖自然短答案结果。', '',
        '## 质量与数值风险', '']
    for kind in ('llava', 'qwen'):
        for c in CONDITIONS[1:]:
            rows = [r for r in accuracy if r['model'] == kind and r['condition'] == c]
            lines.append(f"- {kind} {c}：相对Native损害{sum(r['harm_vs_native'] for r in rows)}题、纠正{sum(r['rescue_vs_native'] for r in rows)}题；完整Token序列变化{sum(r['changed_vs_native'] for r in rows)}题；相对Dense序列变化{sum(r['changed_vs_dense'] for r in rows)}题；EOS未结束/截断{sum(r['truncated'] for r in rows)}题、空答案{sum(r['empty'] for r in rows)}题。")
    failed = [r for r in equivalence_rows if r.get('passed') is False]
    lines += [f'物理路径固定容差的Smoke数值检查失败{len(failed)}项，详见equivalence_audit.csv。数值近似与Mask/位置/Cache语义检查分开；语义通过不代表原生数值等价，亦未为通过而放宽旧容差。', '',
        '## 后续建议', '',
        '只建议在明确接受本轮近似计算、逐题差异和性能条件后，进入固定240题质量验证；该验证仍需预声明质量门槛。当前不应直接宣称原生等价、非劣效、硬件TFLOPS或DRAM优化已获证明。本轮到此停止，不自动扩样。', '',
        '## 可复现性', '',
        '源码、模型权重清单、精度、依赖、原始图片/输入和协议哈希在protocol.json；逐题20次原始测量在checkpoints，聚合CSV与论文PNG/PDF同目录。只在/data运行，未使用/data1。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    atomic(OUT / 'summary.json', {f'{k}/{c}': v for (k, c), v in summaries.items()})
    atomic(OUT / 'validation.json', dict(status='COMPLETE_MEASUREMENTS_WITH_NUMERICAL_LIMITATIONS',
        samples_per_model=60, natural_quality_answers=480, timed_natural_generations=9600,
        timed_fixed_generations=320, warmup_generations=2480, numerical_checks_failed=len(failed),
        hardware_tflops='HARDWARE_TFLOPS_UNAVAILABLE', runtime_hashes_verified=True,
        checkpoint_files={str(p.relative_to(OUT)): sha(p) for p in OUT.glob('*/checkpoints/*/*.json')}))


if __name__ == '__main__':
    main()

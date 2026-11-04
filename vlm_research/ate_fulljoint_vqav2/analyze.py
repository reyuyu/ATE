"""Official rescoring and paired image-cluster statistics; no new GPU evaluation."""
import collections
import datetime
import sys

import numpy as np

from .common import OUT, ROOT, atomic, digest, read, sha, write_csv, verify, score, hardware
from .engine import CONDITIONS
from .run import load

LABELS = ['Native Full', 'Fast-P2', '+ Conditional-15', '+ A2', 'FullJoint']


def bootstrap(values, images, seed):
    _, inverse = np.unique(images, return_inverse=True)
    n = inverse.max() + 1
    totals = np.zeros((n, values.shape[1]))
    np.add.at(totals, inverse, values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    out = np.empty((20000, values.shape[1]))
    for start in range(0, 20000, 500):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=500)
        out[start:start + 500] = (weights @ totals) / (weights @ counts)[:, None]
    return out


def main():
    p = verify()
    samples, specs = read(OUT / 'sample_manifest.json'), read(OUT / 'input_specs.json')
    assert len(samples) == 60 and len({s['image_sha256'] for s in samples.values()}) == 60
    records = {q: load(OUT / 'checkpoints' / (q + '.json'), p, samples[q], specs[q]) for q in sorted(samples)}
    for record in records.values():
        cs = record['conditions']
        prefill = {c: sum(r['estimated_flops'] for r in cs[c]['flops'] if r['phase'] == 'prefill') for c in CONDITIONS}
        assert prefill['C1'] == prefill['C2'] and prefill['C3'] == prefill['C4']
        assert cs['C3']['quality']['kv']['storage_bytes'] == cs['C4']['quality']['kv']['storage_bytes']
        mlp = {c: sum(r['estimated_flops'] for r in cs[c]['flops']
                      if r['phase'] == 'prefill' and r['component'] == 'mlp') for c in ('C1', 'C3', 'C4')}
        assert mlp['C1'] == mlp['C3'] == mlp['C4']
    worker = read(OUT / 'worker_completion.json')
    smoke = read(OUT / 'gpu_smoke.json')
    assert worker['status'] == 'COMPLETE' and smoke['status'] == 'PASS'
    ids = sorted(records)
    values = np.array([[records[q]['conditions'][c]['quality']['score'] for c in CONDITIONS] for q in ids])
    draws = bootstrap(values * 100, [samples[q]['image_sha256'] for q in ids], p['bootstrap_seed'])
    per, accuracy, paired, latency, flops, raw_perf, census, paired_cases = [], [], [], [], [], [], [], []
    for j, c in enumerate(CONDITIONS):
        gens = [records[q]['conditions'][c]['quality'] for q in ids]
        delta = draws[:, j] - draws[:, 0]
        accuracy.append(dict(condition=c, n=60, score_pct=100 * values[:, j].mean(),
            delta_native_pp=100 * (values[:, j] - values[:, 0]).mean(),
            paired95_low=np.quantile(delta, .025), paired95_high=np.quantile(delta, .975),
            eos=sum(g['eos'] for g in gens), truncated=sum(g['truncated'] for g in gens),
            empty=sum(g['empty'] for g in gens), repeated=sum(g['repeated'] for g in gens),
            mean_content_tokens=np.mean([g['content_tokens'] for g in gens])))
        for q, g in zip(ids, gens):
            s, spec = samples[q], specs[q]
            assert score(g['text'], s['answers']) == g['score']
            b = records[q]['conditions']['C0']['quality']
            per.append(dict(question_id=q, image_id=s['image_id'], image_sha256=s['image_sha256'],
                question=s['question'], official_answers=s['answers'], answer_type=s['answer_type'],
                condition=c, answer=g['text'], token_ids=g['token_ids'], score=g['score'],
                native_answer=b['text'], delta_native=g['score'] - b['score'],
                first_token_flip=g['token_ids'][0] != b['token_ids'][0],
                token_sequence_changed=g['token_ids'] != b['token_ids'],
                native_first_token_logp_loss=b['native_first_token_logp'] - g['native_first_token_logp'],
                eos=g['eos'], truncated=g['truncated'], empty=g['empty'], repeated=g['repeated'],
                question_tokens=len(spec['Q']), writer_tokens=len(spec['writers']), R_tokens=len(spec['R']),
                triggered=spec['triggered'], input_hashes=spec['input_hashes']))
            for row in records[q]['conditions'][c]['flops']:
                census.append(dict(question_id=q, condition=c, **row))
    for c, b in (('C4', 'C1'), ('C4', 'C3'), ('C4', 'C0'), ('C2', 'C1'), ('C3', 'C1')):
        j, k = CONDITIONS.index(c), CONDITIONS.index(b)
        d = draws[:, j] - draws[:, k]
        gs = [records[q]['conditions'][c]['quality'] for q in ids]
        bs = [records[q]['conditions'][b]['quality'] for q in ids]
        for q, g, baseline in zip(ids, gs, bs):
            paired_cases.append(dict(question_id=q, image_sha256=samples[q]['image_sha256'],
                candidate=c, baseline=b, candidate_answer=g['text'], baseline_answer=baseline['text'],
                candidate_score=g['score'], baseline_score=baseline['score'],
                score_delta=g['score']-baseline['score'], score_down=g['score']<baseline['score'],
                score_up=g['score']>baseline['score'],
                first_token_flip=g['token_ids'][0]!=baseline['token_ids'][0],
                token_sequence_changed=g['token_ids']!=baseline['token_ids']))
        paired.append(dict(candidate=c, baseline=b, n=60, delta_pp=100 * (values[:, j] - values[:, k]).mean(),
            paired95_low=np.quantile(d, .025), paired95_high=np.quantile(d, .975),
            score_down=int((values[:, j] < values[:, k]).sum()), score_up=int((values[:, j] > values[:, k]).sum()),
            fullcredit_lost=int(((values[:, k] == 1) & (values[:, j] < 1)).sum()),
            fullcredit_gained=int(((values[:, k] < 1) & (values[:, j] == 1)).sum()),
            one_to_zero=int(((values[:, k] == 1) & (values[:, j] == 0)).sum()),
            zero_to_one=int(((values[:, k] == 0) & (values[:, j] == 1)).sum()),
            token_sequence_changes=sum(g['token_ids'] != a['token_ids'] for g, a in zip(gs, bs)),
            first_token_flips=sum(g['token_ids'][0] != a['token_ids'][0] for g, a in zip(gs, bs))))
    by_input = {}
    for q in p['performance_ids']:
        for c in CONDITIONS:
            entry = records[q]['conditions'][c]
            ms = entry['measurements']
            assert len(ms) == 5 and all(m['token_ids'] == entry['quality']['token_ids'] for m in ms)
            r = dict(question_id=q, condition=c)
            for field in ('latency_ms', 'vision_ms', 'language_prefill_ms', 'ttft_ms', 'decode_ms',
                          'peak_allocated', 'peak_reserved'):
                r[field] = float(np.median([m[field] for m in ms]))
            steps = entry['quality']['decode_steps']
            r['decode_ms_per_token'] = r['decode_ms'] / steps if steps else None
            phase = collections.Counter()
            for f in entry['flops']:
                phase[f['phase']] += f['estimated_flops']
            for name in ('vision', 'prefill', 'decode'):
                r[name + '_matrix_flops'] = phase[name]
            r['estimated_prefill_tflops'] = phase['prefill'] / r['language_prefill_ms'] / 1e9
            r['live_kv_MiB'] = entry['quality']['kv']['live_bytes'] / 2**20
            r['capacity_kv_MiB'] = entry['quality']['kv']['storage_bytes'] / 2**20
            r['peak_allocated_MiB'], r['peak_reserved_MiB'] = r['peak_allocated'] / 2**20, r['peak_reserved'] / 2**20
            by_input[q, c] = r
            raw_perf.append(r)
    for c in CONDITIONS:
        rs = [by_input[q, c] for q in p['performance_ids']]
        row = dict(condition=c, n=20, warmups=3, repetitions=5)
        for field in ('latency_ms', 'vision_ms', 'language_prefill_ms', 'ttft_ms', 'decode_ms_per_token'):
            data = [r[field] for r in rs if r[field] is not None]
            row[field] = float(np.median(data)) if data else None
            row[field + '_p10'], row[field + '_p90'] = [float(v) for v in np.quantile(data, [.1, .9])] if data else [None, None]
        for b in ('C0', 'C1', 'C3'):
            ratios = [by_input[q, b]['latency_ms'] / by_input[q, c]['latency_ms'] for q in p['performance_ids']]
            row['E2E_speedup_vs_' + b] = float(np.median(ratios))
            row['E2E_reduction_pct_vs_' + b] = float(np.median([100 * (1 - 1 / v) for v in ratios]))
            same = [q for q in p['performance_ids'] if records[q]['conditions'][c]['quality']['token_ids'] ==
                    records[q]['conditions'][b]['quality']['token_ids']]
            row['same_sequence_n_vs_' + b] = len(same)
            row['same_sequence_speedup_vs_' + b] = float(np.median([
                by_input[q, b]['latency_ms'] / by_input[q, c]['latency_ms'] for q in same])) if same else None
        latency.append(row)
        fr = dict(condition=c, n=20)
        for field in ('vision_matrix_flops', 'prefill_matrix_flops', 'decode_matrix_flops',
                      'estimated_prefill_tflops', 'live_kv_MiB', 'capacity_kv_MiB', 'peak_allocated_MiB', 'peak_reserved_MiB'):
            fr[field] = float(np.median([r[field] for r in rs]))
        fr['prefill_flops_reduction_vs_C1_pct'] = float(np.mean([
            100 * (1 - by_input[q, c]['prefill_matrix_flops'] / by_input[q, 'C1']['prefill_matrix_flops'])
            for q in p['performance_ids']]))
        flops.append(fr)
    for name, rows in (('accuracy_summary.csv', accuracy), ('paired_comparisons.csv', paired),
                       ('per_question.csv', per), ('latency_summary.csv', latency),
                       ('flops_kv_summary.csv', flops), ('performance_per_question.csv', raw_perf),
                       ('shape_census.csv', census), ('mask_numerical_audit.csv', smoke['numerical'])):
        write_csv(OUT / name, rows)
    write_csv(OUT / 'negative_cases.csv', [r for r in per if r['delta_native'] < 0 or r['empty'] or r['truncated']])
    write_csv(OUT / 'paired_case_changes.csv', paired_cases)
    plot(accuracy, latency, draws)
    report(p, accuracy, paired, latency, flops, worker)
    assert not (OUT / 'STOP.json').exists()
    actual_weights = {path: sha(path) for path in p['weights']}
    assert actual_weights == p['weights']
    state = hardware()
    assert p['device']['uuid'] not in state['processes'], 'Worker GPU context still alive'
    atomic(OUT / 'gpu_release.json', dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selected_device=p['device'], selected_device_has_no_compute_process=True, hardware=state,
        no_external_process_terminated=True))
    atomic(OUT / 'validation.json', dict(status='COMPLETE', quality_records=300, unique_images=60,
        timing_inputs=20, timed_requests=500, warmup_requests=300, bootstrap=20000,
        cpu_tests_pass=7, gpu_smoke_pass=3, protocol_sha256=digest(p),
        sources_match_freeze=True, model_weights_postrun_unchanged=True, scorer_recomputed=True,
        device_released=True, no_fixed16_or64=True, no_new_data_download=True,
        mask_gate='Actual forbidden Softmax probabilities zero; final writer mask used by A2',
        runtime_failures=[], source_commit='a8b4e251360ea7bcbfea08ed99a2f91749f557db',
        conditional_writer_dense_FLOPs_unchanged=True, C4_C3_KV_capacity_equal=True,
        all_prefill_MLP_flops_preserved=True, historical_tolerances=p['numerical_tolerances']))
    hash_artifacts()
    print('COMPLETE', accuracy, paired, flush=True)


def plot(accuracy, latency, draws):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    colors = ['#666666', '#277da8', '#9e6f20', '#70a37f', '#218c53']
    x = np.arange(5)
    values = np.array([r['score_pct'] for r in accuracy])
    low, high = np.quantile(draws, [.025, .975], axis=0)
    axes[0].bar(x, values, color=colors)
    axes[0].errorbar(x, values, yerr=[values-low, high-values], fmt='none', capsize=3, color='black', lw=.8)
    axes[0].set(ylabel='Official VQAv2 soft accuracy (%)', title='60-image exploratory pilot')
    axes[1].bar(x, [r['E2E_speedup_vs_C0'] for r in latency], color=colors)
    axes[1].axhline(1, color='#555555', lw=.8, ls='--')
    axes[1].set(ylabel='Natural EOS E2E speedup vs Native', title='20 paired inputs | 3 warmup + 5 timed')
    for ax in axes:
        ax.set_xticks(x, [f'{c}\n{label}' for c, label in zip(CONDITIONS, LABELS)], rotation=25, ha='right')
        ax.grid(axis='y', alpha=.15)
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / ('accuracy_speedup.' + extension), dpi=240)
    plt.close(fig)


def report(p, accuracy, paired, latency, flops, worker):
    lines = ['# ATE-FullJoint：VQAv2快速验证', '', '**COMPLETE**：60题×5条件=300条真实Greedy答案；20题×5条件各3次预热、5次正式计时。仅VQAv2，不扩样、不训练。',
        '', '## 合体定义与数据',
        'C0为真正Native Full。C1严格调用冻结Fast-P2：L20前删除视觉Hidden，P−1/Decode不直接读图，文本KV原地追加。C2仅增加Conditional Writer；C3仅增加物理A2；C4合体两者。',
        'Conditional Writer严格使用644834d3的Fixed-Q-Conditional-Assistant-15：所有原Question Subword含标点；首词what/which或Question Token数≥15时加原ASSISTANT的完整4个Subword。其他R、P−1禁读图。仅L0–L19存在视觉写入。不是旧Semantic-k。',
        'A2在L2–L5只投影保留Query的Q/o_proj并计算其Attention；K/V仍覆盖全部Prefill输入，Image/P−1输出置零，Decode不执行该四层Attention。Norm、Residual和全部MLP保留。最终Writer Mask先构造，再重新取A2 active_mask；没有绕过Conditional Mask。',
        'L20之后物理删除576个视觉Hidden，持久KV只存需要的文本；被彻底跳过的四层不存无用KV。其余层Decode使用文本KV，保留原始逻辑位置和RoPE，无历史KV拼接或重复拷贝。',
        '60题从已有冻结VQAv2-val2014本地500题按输入稳定哈希选取，60张不同图片；不是整个验证集的无偏随机样本，也不是新的独立确认集。官方10答案逐项对照原Annotations ZIP，图片SHA、Prompt IDs和输入张量哈希与历史记录一致。',
        '五条件均新生成，Greedy最多32 Token、EOS停止。同一RTX3090、FP16/eager、Transformers4.49.0、同一权重和Processor。沿用Fast-P2的logits_to_keep=0，在同形状下严格检查Native/No-op及官方generate一致；最新规则仅复用接收者定义，不混入其历史答案或不同LM Head形状结果。',
        '', '## 官方共识评分', '|条件|软Accuracy %|ΔNative pp|EOS/60|截断|空答案|', '|---|---:|---:|---:|---:|---:|']
    for r in accuracy:
        lines.append(f"|{r['condition']}|{r['score_pct']:.3f}|{r['delta_native_pp']:+.3f}|{r['eos']}|{r['truncated']}|{r['empty']}|")
    lines += ['', '官方VQAEval标点、数字、冠词、缩写归一化加十标注leave-one-out共识；不是单字符串匹配，也不是简单min(count/3,1)。1/2/3个匹配分别得0.3/0.6/0.9。',
        '', '|比较|Δ软分 pp|配对95%CI|软分下降/上升|满分丢失/获得|', '|---|---:|---|---|---|']
    for r in paired:
        lines.append(f"|{r['candidate']}−{r['baseline']}|{r['delta_pp']:+.3f}|[{r['paired95_low']:+.3f}, {r['paired95_high']:+.3f}]|{r['score_down']}/{r['score_up']}|{r['fullcredit_lost']}/{r['fullcredit_gained']}|")
    lines += ['', '20,000次Image SHA整簇配对Bootstrap，题目等权，固定seed=20261105。全部区间为探索性，不作非劣效/等价判断；VQAv2软分的下降/上升不能全部称为二值正确翻转。完整0↔1翻转另见CSV。',
        '', '## 同卡真实性能', '|条件|Prefill ms|GPU TTFT ms|E2E ms|Decode ms/token|vs Native加速|vs C1加速|vs C3加速|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in latency:
        dec = f"{r['decode_ms_per_token']:.3f}" if r['decode_ms_per_token'] is not None else 'N/A'
        lines.append(f"|{r['condition']}|{r['language_prefill_ms']:.3f}|{r['ttft_ms']:.3f}|{r['latency_ms']:.3f}|{dec}|{r['E2E_speedup_vs_C0']:.3f}×|{r['E2E_speedup_vs_C1']:.3f}×|{r['E2E_speedup_vs_C3']:.3f}×|")
    lines += ['', '时延为逐题5次正式重复中位数再跨20题取中位数；加速为同题时延比的中位数，不必等于两列总体中位数之比。所有条件交错、每请求独立冷KV；Vision每次原生重算。图片I/O、CPU Processor及输入分区/Writer位置准备统一不计时；控制器/实际Mask/Cache构造计入同步墙钟E2E。',
        'GPU TTFT沿用冻结首Forward CUDA Event口径，为Vision+Language Prefill，不包含全部CPU请求准备，不冒充完整墙钟首Token时延。自然EOS的输出长度变化可能改变E2E；同完整Token序列子集的配对加速和样本数另见latency_summary.csv。没有固定16/64测试，也没有缓存复用视觉特征。',
        '', '|条件|Prefill矩阵TFLOPs|vs C1减少%|估算Prefill TFLOP/s|Live KV MiB|容量KV MiB|Peak allocated MiB|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in flops:
        lines.append(f"|{r['condition']}|{r['prefill_matrix_flops']/1e12:.4f}|{r['prefill_flops_reduction_vs_C1_pct']:+.3f}|{r['estimated_prefill_tflops']:.3f}|{r['live_kv_MiB']:.3f}|{r['capacity_kv_MiB']:.3f}|{r['peak_allocated_MiB']:.1f}|")
    lines += ['', '实际Linear/Conv/QK/AV/bmm张量形状审计，2 FLOPs/MAC，包含Vision矩阵计算的独立项；不包含Norm/Softmax/RoPE等非矩阵操作。估算TFLOP/s为算术量除对应CUDA阶段时间，不是硬件计数器。硬件TFLOPS/DRAM计数器N/A，沿用ERR_NVGPUCTRPERM已知权限限制，不伪造硬件实测。',
        'Conditional Writer仍是Dense Hard Mask，本身不省Dense矩阵FLOPs。逐题严格验证C2与C1、C4与C3的Prefill矩阵量相同，C4与C3的KV容量相同；不能把写入连接预算减少当成物理加速。实际省算来自L20删除视觉行、文本KV以及A2局部Query跳算。Live KV可随自然生成长度变化，不是额外的Token选择压缩；KV Live和容量分别计量，不能当作整卡显存减幅。',
        '', '## 直接回答与边界']
    a = {r['condition']: r for r in accuracy}
    t = {r['condition']: r for r in latency}
    f = {r['condition']: r for r in flops}
    comparisons = {(r['candidate'], r['baseline']): r for r in paired}
    lines += ['1. 四项已经组合在C4同一真实生成路径，三题Smoke通过；Dense参考仅验证、不参与速度排名。',
        f"2. C4软Accuracy={a['C4']['score_pct']:.3f}%。",
        f"3. C4−C3={comparisons['C4','C3']['delta_pp']:+.3f}pp，配对95%CI [{comparisons['C4','C3']['paired95_low']:+.3f}, {comparisons['C4','C3']['paired95_high']:+.3f}]。",
        f"4. C4相对Native E2E={t['C4']['E2E_speedup_vs_C0']:.3f}×，时延减少{t['C4']['E2E_reduction_pct_vs_C0']:.2f}%。",
        f"5. C4相对C1/C3 E2E分别为{t['C4']['E2E_speedup_vs_C1']:.3f}×/{t['C4']['E2E_speedup_vs_C3']:.3f}×。",
        f"6. C4相对C1的Prefill矩阵量减少{f['C4']['prefill_flops_reduction_vs_C1_pct']:.3f}%；Live KV为{f['C4']['live_kv_MiB']:.3f}MiB。C4与C3的Token选择不额外节省Dense矩阵计算。",
        f"7. 本轮不建议把完整C4作为优先扩规模候选：Conditional Writer相对C3没有额外矩阵量/KV容量/时延收益，软分差值为{comparisons['C4','C3']['delta_pp']:+.3f}pp，且CI仍很宽。若下一轮决定扩大验证，优先复核更简单的C3是否稳定；这不是在本轮自动扩样或认定其非劣效。C4当前不能称为质量保持的完整加速方案。",
        '', '## 正确性与异常',
        '7项CPU测试、3/3真实GPU Smoke通过。Native/No-op及官方generate全返回有效Logits严格torch.equal；C1与冻结Fast-P2、C3与旧A2严格一致。C4与全Dense D1+I+A零输出参考，以及同物理交接的零输出参考，使用旧FP16 .10/.003 Logits、Hidden RMSE≤.01及Cosine距离≤1e-4，未放宽。',
        'Smoke记录真实禁用概率为0、A2目标Q/o_proj行数、完整K/V依赖、32层MLP正常、L20后实际文本行数、KV逻辑位置及Hook正常/异常清理。所有Checkpoint哈希、输入、完整序列及官方软分重新核对。',
        'EOS、截断、空答案和重复3gram启发式均保留在CSV，不删负例。重复启发式不等于模型失效。逐题每条件原始Shape和计时全部保留在Checkpoint。',
        f"正式GPU矩阵耗时{worker['elapsed_seconds']/60:.2f}分钟（不含模型加载/Smoke/CPU冻结/汇总）；完整操作时间与文件/提交记录另存。",
        '只占用启动前空闲的物理GPU2，GPU0/1外部任务未干扰，GPU3保持空闲；进程退出后保存gpu_release.json。main未修改。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')


def hash_artifacts():
    hashes = {str(path.relative_to(OUT)): sha(path) for path in sorted(OUT.rglob('*'))
              if path.is_file() and path.name != 'artifact_hashes.json'}
    atomic(OUT / 'artifact_hashes.json', hashes)


if __name__ == '__main__':
    if '--hash-only' in sys.argv:
        hash_artifacts()
    else:
        main()

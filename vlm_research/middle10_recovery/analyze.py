"""Hash-verified, image-cluster paired analysis; never launches inference."""
import collections
import argparse
import json
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import *
from .run import verify_freeze, historical
from ..fastp2_bypass_writer.common import hardware


def quantile(values):
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def quality(freeze):
    samples = read(OUT / 'sample_manifest.json')
    partition = read(OUT / 'partition.json')
    keys = sorted(samples)
    matrix = np.zeros((len(keys), len(LABELS)))
    records, rows, kv_rows, hashes = {}, [], [], {}
    source_audit = {(r['key'], r['condition']): r for r in read(OUT / 'source_manifest.json')['reused_records']}
    for i, key in enumerate(keys):
        s = samples[key]
        image = set(s['input_mapping']['image'])
        n = len(s['input_mapping']['prompt_ids'])
        text_positions = [q for q in range(n) if q not in image]
        for j, label in enumerate(LABELS):
            reused = label in BASELINES
            path = HISTORY / 'checkpoints' / key / (label+'.json') if reused else quality_path(key, label)
            v = load(path)
            if reused:
                assert sha(path) == source_audit[key, label]['sha256']
                assert digest(v['input_binding']) == source_audit[key, label]['input_binding_sha256']
            else:
                assert v['sources'] == freeze['sources']
                assert v['rank'] == partition['image_shard'][s['image_sha256']]
                hashes[str(path.relative_to(OUT))] = sha(path)
            g = v['generation']
            assert v['sample_sha256'] == digest(s) and v['score'] == score(s, g['text'])
            assert g['vision_calls'] == 1 and len(g['token_ids']) <= 32
            assert g['first_top1_token'] == g['token_ids'][0]
            assert g['kv']['retained_image_layers'] == list(KEEP[label])
            for r in g['kv']['layers']:
                absent = label != 'Native' and r['layer'] in (2, 3, 4, 5)
                full = label == 'Native' or r['layer'] in KEEP[label]
                positions = [] if absent else (list(range(n)) if full else text_positions.copy())
                if not absent:
                    positions += list(range(n, n+len(g['token_ids'])-1))
                assert r['used_tokens'] == len(positions)
                capacity = len(positions) if label == 'Native' else 0 if absent else (n if full else n-576)+31
                assert r['capacity_tokens'] == capacity
                if 'original_positions_sha256' in r:
                    assert r['original_positions_sha256'] == digest(positions)
                if not reused or i == 0:
                    kv_rows.append(dict(key=key, condition=label, layer=r['layer'],
                        used_tokens=r['used_tokens'], capacity_tokens=r['capacity_tokens'],
                        image_tokens=r['image_tokens'], text_tokens=r['text_tokens'],
                        live_bytes=r['live_bytes'], storage_bytes=r['storage_bytes'],
                        original_positions_sha256=digest(positions)))
            records[key, label] = v
            matrix[i, j] = v['score']
            rows.append(dict(key=key, question_id=s['question_id'], image_id=s['image_id'],
                image_sha256=s['image_sha256'], question=s['question'], gold_answers=s['answer'],
                condition=label, reused_history=reused, score=v['score'], answer=g['text'],
                token_ids=g['token_ids'], first_top1_token=g['first_top1_token'],
                first_logits_sha256=g['first_logits_sha256'], input_sha256=digest(v['input_binding']),
                eos=g['eos'], truncated=g['truncated'], empty=g['empty'], repeated=g['repeated'],
                content_tokens=g['content_tokens'], decode_steps=g['decode_steps'],
                persistent_image_kv_layers=g['kv']['retained_image_layers']))
    assert len(rows) == 4500 and len(records) == 4500
    for row in rows:
        key, label = row['key'], row['condition']
        g = records[key, label]['generation']
        ng = records[key, 'Native']['generation']
        mg = records[key, 'C3+Middle10']['generation']
        row['first_token_flip_vs_native'] = g['token_ids'][0] != ng['token_ids'][0]
        row['first_token_flip_vs_middle10'] = g['token_ids'][0] != mg['token_ids'][0]
        row['native_fixed_first_token_signed_logp_loss'] = (
            ng['first_top1_logp']-g['native_first_token_logp'] if label in NEW else None)
        row['middle10_fixed_first_token_signed_logp_loss'] = (
            mg['first_top1_logp']-g['middle10_first_token_logp'] if label in NEW else None)
    write_csv(OUT / 'per_question.csv', rows)
    write_csv(OUT / 'physical_kv_audit.csv', kv_rows)
    clusters = sorted({s['image_sha256'] for s in samples.values()})
    index = {h: i for i, h in enumerate(clusters)}
    membership = np.array([index[samples[k]['image_sha256']] for k in keys])
    sizes = np.bincount(membership)
    sums = np.zeros((len(clusters), len(LABELS)))
    np.add.at(sums, membership, matrix)
    weights = np.random.default_rng(SEED).multinomial(len(clusters),
                np.full(len(clusters), 1/len(clusters)), size=20000)
    boot = (weights @ sums)/(weights @ sizes)[:, None]*100
    means = matrix.mean(0)*100
    m10, native = LABELS.index('C3+Middle10'), LABELS.index('Native')
    summary = []
    for j, label in enumerate(LABELS):
        gs = [records[k, label]['generation'] for k in keys]
        lo, hi = quantile(boot[:, j])
        dl, dh = quantile(boot[:, j]-boot[:, m10])
        summary.append(dict(condition=label, questions=500, image_clusters=309,
            accuracy=means[j], ci_low=lo, ci_high=hi,
            delta_middle10_pp=means[j]-means[m10], delta_middle10_ci_low=dl, delta_middle10_ci_high=dh,
            delta_native_pp=means[j]-means[native],
            eos=sum(g['eos'] for g in gs), truncated=sum(g['truncated'] for g in gs),
            empty=sum(g['empty'] for g in gs), repeated=sum(g['repeated'] for g in gs),
            mean_content_tokens=float(np.mean([g['content_tokens'] for g in gs]))))
    write_csv(OUT / 'accuracy_summary.csv', summary)
    comparisons, cases = [], []
    extra = tuple((label, 'Native') for label in NEW)
    for candidate, baseline in COMPARISONS + extra:
        a, b = LABELS.index(candidate), LABELS.index(baseline)
        delta = matrix[:, a]-matrix[:, b]
        lo, hi = quantile(boot[:, a]-boot[:, b])
        comparisons.append(dict(comparison=candidate+' - '+baseline,
            delta_accuracy_pp=float(delta.mean()*100), ci_low=lo, ci_high=hi,
            score_increased=int((delta>1e-12).sum()), score_decreased=int((delta< -1e-12).sum()),
            equal=int((np.abs(delta)<=1e-12).sum()),
            full_score_to_lower=int(((matrix[:, b]==1)&(delta< -1e-12)).sum()),
            lower_to_full_score=int(((matrix[:, a]==1)&(delta>1e-12)).sum()),
            first_token_flips=sum(records[k, candidate]['generation']['token_ids'][0] !=
                                  records[k, baseline]['generation']['token_ids'][0] for k in keys),
            sequence_changes=sum(records[k, candidate]['generation']['token_ids'] !=
                                 records[k, baseline]['generation']['token_ids'] for k in keys),
            exploratory=True, bootstrap_replicates=20000))
        for i, k in enumerate(keys):
            if abs(delta[i]) <= 1e-12:
                continue
            cases.append(dict(comparison=candidate+' - '+baseline, key=k, question=samples[k]['question'],
                image_sha256=samples[k]['image_sha256'], gold_answers=samples[k]['answer'],
                baseline_answer=records[k, baseline]['generation']['text'],
                intervention_answer=records[k, candidate]['generation']['text'],
                baseline_score=matrix[i, b], intervention_score=matrix[i, a], delta_score=delta[i],
                native_answer=records[k, 'Native']['generation']['text'], native_score=matrix[i, native],
                middle10_answer=records[k, 'C3+Middle10']['generation']['text'], middle10_score=matrix[i, m10],
                middle10_harmed_vs_native=bool(matrix[i, m10]<matrix[i, native]-1e-12)))
    write_csv(OUT / 'paired_comparisons.csv', comparisons)
    write_csv(OUT / 'recovery_and_harm_cases.csv', cases)
    factors = np.array([0., 0., 0., 1., -1., -1., 1., 0., 0.])
    assert LABELS == ('Native', 'C3-V2', 'C3+Middle9', 'C3+Middle10', 'E1', 'E2', 'E3', 'E4', 'E5')
    low, high = quantile(boot @ factors)
    interaction = dict(comparison='E3 - E2 - E1 + Middle10', delta_accuracy_pp=float(means @ factors),
                       ci_low=low, ci_high=high, exploratory=True, bootstrap_replicates=20000)
    write_csv(OUT / 'factorial_interaction.csv', [interaction])
    harmed = matrix[:, m10] < matrix[:, native]-1e-12
    strict = (matrix[:, native]==1)&(matrix[:, m10]==0)
    recovery = []
    for label in NEW:
        j = LABELS.index(label)
        d = matrix[:, j]-matrix[:, m10]
        recovery.append(dict(condition=label, middle10_harmed_pool=int(harmed.sum()),
            improved_in_harmed_pool=int((harmed&(d>1e-12)).sum()),
            fully_recovered_to_native=int((harmed&(matrix[:, j]>=matrix[:, native]-1e-12)).sum()),
            signed_net_gain_in_harmed_pool_pp=float(d[harmed].sum()/len(keys)*100),
            strict_native1_middle10zero_pool=int(strict.sum()),
            strict_pool_restored_to1=int((strict&(matrix[:, j]==1)).sum()),
            new_harm_to_middle10_full_score=int(((matrix[:, m10]==1)&(d< -1e-12)).sum()),
            all_score_increases=int((d>1e-12).sum()), all_score_decreases=int((d< -1e-12).sum())))
    write_csv(OUT / 'recovery_summary.csv', recovery)
    examples = []
    for qid in ('34822', '39376', '39441', '39065'):
        k = 'textvqa__'+qid
        examples.append(dict(question_id=qid, question=samples[k]['question'],
            gold_answers=samples[k]['answer'],
            **{label: records[k, label]['generation']['text'] for label in LABELS},
            **{label+'_score': records[k, label]['score'] for label in LABELS}))
    write_csv(OUT / 'four_previous_visual_cases.csv', examples)
    return samples, records, summary, comparisons, interaction, recovery, hashes


def timing(freeze, samples, records, hashes):
    labels = read(OUT / 'performance_selection.json')['conditions']
    partition = read(OUT / 'partition.json')
    rows = []
    for k in partition['performance48'] if labels else []:
        for mode in ('natural_eos', 'fixed16'):
            for repeat in range(-3, 5):
                for label in labels:
                    path = performance_path(k, mode, repeat, label)
                    v = load(path)
                    assert v['sources'] == freeze['sources'] and v['sample_sha256'] == digest(samples[k])
                    assert v['rank'] == partition['image_shard'][samples[k]['image_sha256']]
                    g = v['generation']
                    assert g['vision_calls'] == 1 and g['kv']['retained_image_layers'] == list(KEEP[label])
                    assert g['token_ids'] == records[k, label]['generation']['token_ids'] if mode == 'natural_eos' else len(g['token_ids']) == 16
                    hashes[str(path.relative_to(OUT))] = sha(path)
                    if repeat < 0:
                        continue
                    kv = g['kv']
                    rows.append(dict(key=k, condition=label, mode=mode, repeat=repeat, rank=v['rank'],
                        prefill_ms=g['prefill_ms'], ttft_ms=g['ttft_ms'], vision_ms=g['vision_ms'],
                        language_prefill_ms=g['language_prefill_ms'], e2e_ms=g['e2e_ms'], decode_ms=g['decode_ms'],
                        decode_steps=g['decode_steps'], generated_tokens=len(g['token_ids']),
                        live_kv_mib=kv['live_bytes']/2**20, capacity_kv_mib=kv['storage_bytes']/2**20,
                        image_live_mib=kv['image_live_bytes']/2**20, image_capacity_mib=kv['image_capacity_bytes']/2**20,
                        peak_allocated_mib=g['peak_allocated']/2**20, peak_reserved_mib=g['peak_reserved']/2**20))
    assert len(rows) == (48*2*5*len(labels) if labels else 0)
    if rows:
        write_csv(OUT / 'latency_raw.csv', rows)
    else:
        write_csv(OUT / 'latency_raw.csv', [dict(status='NOT_RUN_NO_POSITIVE_QUALITY_RECOVERY')])
    summaries = []
    metrics = [m for m in rows[0] if m not in ('key', 'condition', 'mode', 'repeat', 'rank')] if rows else []
    for mode in ('natural_eos', 'fixed16'):
        for label in labels:
            own = [r for r in rows if r['mode']==mode and r['condition']==label]
            r = dict(condition=label, mode=mode, questions=48, repetitions=5,
                     **{m: float(np.mean([v[m] for v in own])) for m in metrics})
            r['decode_ms_per_token'] = sum(v['decode_ms'] for v in own)/sum(v['decode_steps'] for v in own)
            same = [k for k in partition['performance48'] if records[k, label]['generation']['token_ids'] == records[k, 'Native']['generation']['token_ids']]
            r['same_sequence_n'] = len(same)
            same_candidate = [v['e2e_ms'] for v in own if v['key'] in same]
            same_native = [v['e2e_ms'] for v in rows if v['mode']==mode and v['condition']=='Native' and v['key'] in same]
            r['same_sequence_speedup'] = float(np.mean(same_native)/np.mean(same_candidate)) if same else None
            summaries.append(r)
    for r in summaries:
        native = next(v for v in summaries if v['mode']==r['mode'] and v['condition']=='Native')
        m10 = next(v for v in summaries if v['mode']==r['mode'] and v['condition']=='C3+Middle10')
        r['e2e_speedup_native'] = native['e2e_ms']/r['e2e_ms']
        r['fixed_or_natural_e2e_overhead_middle10_pct'] = (r['e2e_ms']/m10['e2e_ms']-1)*100
        r['prefill_overhead_middle10_pct'] = (r['prefill_ms']/m10['prefill_ms']-1)*100
        ordered = partition['performance48']
        own_values = np.array([np.mean([v['e2e_ms'] for v in rows
            if v['key']==k and v['mode']==r['mode'] and v['condition']==r['condition']]) for k in ordered])
        baseline_values = np.array([np.mean([v['e2e_ms'] for v in rows
            if v['key']==k and v['mode']==r['mode'] and v['condition']=='C3+Middle10']) for k in ordered])
        resamples = np.random.default_rng(SEED).integers(0, len(ordered), size=(20000, len(ordered)))
        lo, hi = quantile((own_values-baseline_values)[resamples].mean(1))
        r['paired_e2e_minus_middle10_ms_ci_low'] = lo
        r['paired_e2e_minus_middle10_ms_ci_high'] = hi
    write_csv(OUT / 'latency_summary.csv', summaries or [dict(status='NOT_RUN_NO_POSITIVE_QUALITY_RECOVERY')])
    return summaries, labels


def plot(summary, timing_rows):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42, 'ps.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    colors = ['#555555', '#999999', '#927359', '#178079', '#478ec2', '#dd9b40', '#884ea0', '#c35365', '#7b8b4b']
    x = np.arange(len(summary))
    y = np.array([r['accuracy'] for r in summary])
    axes[0].bar(x, y, color=colors)
    axes[0].errorbar(x, y, yerr=np.array([[r['accuracy']-r['ci_low'] for r in summary],
                    [r['ci_high']-r['accuracy'] for r in summary]]), fmt='none', color='black', capsize=2, lw=.7)
    axes[0].set(xticks=x, xticklabels=['Native','C3','M9','M10','E1','E2','E3','E4','E5'],
                ylabel='TextVQA soft accuracy (%)', title='Same frozen 500 questions')
    for mode, marker in (('natural_eos', 'o'), ('fixed16', 's')):
        for r in timing_rows:
            if r['mode'] != mode:
                continue
            i = LABELS.index(r['condition'])
            axes[1].scatter(r['e2e_speedup_native'], y[i], color=colors[i], marker=marker, s=35)
            if mode == 'fixed16':
                axes[1].annotate(['Native','C3','M9','M10','E1','E2','E3','E4','E5'][i],
                    (r['e2e_speedup_native'], y[i]), xytext=(3, 3), textcoords='offset points', fontsize=8)
    axes[1].axvline(1.1, ls=':', color='#777777', lw=.8)
    axes[1].set(xlabel='Actual E2E speedup vs Native', ylabel='500-question soft accuracy (%)',
                title='Fresh paired performance: 48 inputs')
    axes[1].text(.02, .02, 'Square: fixed16 (primary cost) | Circle: natural EOS\nQuality always uses natural-EOS answers',
                 transform=axes[1].transAxes, fontsize=7)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / ('accuracy_speedup.'+extension), dpi=240, bbox_inches='tight')
    plt.close(fig)


def report(summary, comparisons, interaction, recovery, timing_rows, labels):
    by_label = {r['condition']: r for r in summary}
    pairs = {r['comparison']:r for r in comparisons}
    def effect(name):
        r = pairs[name]
        return f"{r['delta_accuracy_pp']:+.2f}pp，95%CI [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
    table = ['|条件|Accuracy|ΔM10 pp|ΔNative pp|', '|---|---:|---:|---:|']
    table += [f"|{r['condition']}|{r['accuracy']:.2f}%|{r['delta_middle10_pp']:+.2f}|{r['delta_native_pp']:+.2f}|" for r in summary]
    paired = ['|配对比较|ΔAccuracy pp|图片簇95%CI|软分升/降|', '|---|---:|---|---:|']
    paired += [f"|{r['comparison']}|{r['delta_accuracy_pp']:+.2f}|[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]|{r['score_increased']}/{r['score_decreased']}|" for r in comparisons]
    perf = ['|条件/模式|Prefill ms|TTFT ms|E2E ms|Decode ms/token|KV Live/Capacity MiB|Native加速|相对M10开销|', '|---|---:|---:|---:|---:|---:|---:|---:|']
    perf += [f"|{r['condition']}/{r['mode']}|{r['prefill_ms']:.2f}|{r['ttft_ms']:.2f}|{r['e2e_ms']:.2f}|{r['decode_ms_per_token']:.2f}|{r['live_kv_mib']:.2f}/{r['capacity_kv_mib']:.2f}|{r['e2e_speedup_native']:.3f}x|{r['fixed_or_natural_e2e_overhead_middle10_pct']:+.2f}%|" for r in timing_rows]
    statuses = []
    m10_time = next((r for r in timing_rows if r['condition']=='C3+Middle10' and r['mode']=='fixed16'), None)
    for label in NEW:
        gain = by_label[label]['delta_middle10_pp']
        t = next((r for r in timing_rows if r['condition']==label and r['mode']=='fixed16'), None)
        status = ('NO_USEFUL_RECOVERY' if gain <= 1e-12 else
                  'QUALITY_AND_SPEED_IMPROVED' if t and t['e2e_ms'] <= m10_time['e2e_ms'] else 'QUALITY_ONLY_IMPROVED')
        statuses.append(dict(condition=label, classification=status,
            limitation='Descriptive point estimates on historical500 and48 paired timing inputs; no external confirmation or latency significance claim.'))
    atomic(OUT / 'outcome.json', statuses)
    best = max((by_label[label] for label in NEW), key=lambda r:r['accuracy'])
    fixed_new = [r for r in timing_rows if r['mode']=='fixed16' and r['condition'] in NEW]
    fastest = max(fixed_new, key=lambda r:r['e2e_speedup_native']) if fixed_new else None
    speed_answer = (f"新增条件中固定16 Token加速最高为{fastest['condition']}：相对Native {fastest['e2e_speedup_native']:.3f}x，相对M10开销{fastest['fixed_or_natural_e2e_overhead_middle10_pct']:+.2f}%。"
                    if fastest else '新增条件均无正向质量改善，按协议未运行性能矩阵。')
    lines = ['# C3-V2 + Middle10剩余TextVQA损失诊断', '', '**状态：COMPLETE。固定E1–E5，没有追加候选或修改Prompt。**', '',
        f"最接近Native的新增条件为{best['condition']}：{best['accuracy']:.2f}%，相对M10 {best['delta_middle10_pp']:+.2f}pp，相对Native {best['delta_native_pp']:+.2f}pp。",
        '', '![Accuracy and fresh actual speed](accuracy_speedup.png)', '', '## 完整质量', '', *table, '',
        'N/C3/M9/M10复用已审计的2,000条历史断点；双卡各3题重新生成，完整Token及首Token Logits哈希精确一致。新增E1–E5共2,500条答案。全部500题、309个图片SHA簇，官方十答案共识软分。', '',
        '## 预声明配对比较', '', *paired, '',
        f"2×2交互 E3−E2−E1+M10：{interaction['delta_accuracy_pp']:+.2f}pp，95%CI [{interaction['ci_low']:+.2f}, {interaction['ci_high']:+.2f}]。联合点估计更高不自动意味着正协同；CI跨0不等于没有交互。", '',
        '## 核心问题的直接回答', '',
        '1. L20 R→Image恢复（E1−M10）：'+effect('E1 - C3+Middle10')+'。点估计有改善，单独效应仍不确定。',
        '2. P−1 Early7回看（E2−M10）：'+effect('E2 - C3+Middle10')+'。是有限样本内的探索性正向证据，未增加持久Image KV。',
        '3. 联合恢复（E3−M10）：'+effect('E3 - C3+Middle10')+'；E3−E1为'+effect('E3 - E1')+'；E3−E2为'+effect('E3 - E2')+'。没有明确正协同证据，不能将两个单独效应直接相加。',
        '4. 再恢复P−1的L2–L5文本Attention（E4−E3）：'+effect('E4 - E3')+'。本轮没有进一步质量收益，点估计反而更低；不把区间跨0解释为严格无作用。',
        '5. 仅后移Hidden退出（E5−M9）：'+effect('E5 - C3+Middle9')+'；500题软分逐题相同，但有1题生成序列不同。同Exit21下M10−E5为'+effect('C3+Middle10 - E5')+'。本轮Middle10相对Middle9的分数增量与L20答案回看一致，而不是仅保留L20视觉Hidden计算；这是固定顺序比较，不是普遍唯一机制归因。',
        f"6. 质量最佳新增条件是{best['condition']}。E3将原10.22pp净缺口缩小1.92pp（18.8%），仍剩8.30pp；18.8%只是分数缺口变化，不是某一路径携带信息的比例。",
        '7–8. '+speed_answer+'逐条件质量/速度判定见下方。自然EOS单列，不拿较短输出制造算子加速。',
        '9. L20写入和早层P−1回看只能解释部分可恢复损失。恢复L2–L5 P−1文本Attention未继续改善，单纯退出边界后移无观察到的分数收益。主要剩余8.30pp来源仍未被这些条件定位，可能涉及未覆盖层的直接图像路径、联合信息路由、表示变化或生成表达；本轮没有继续搜索，不能选定其中任何一种为已证明的主要原因。', '',
        '## 剩余错误与新增损害', '',
        '旧M10相对Native有74题软分下降，其中48题为Native满分且M10零分。',
        *[f"- {r['condition']}：在74题中改善{r['improved_in_harmed_pool']}题、恢复至Native分数{r['fully_recovered_to_native']}题；48道严格失败题中恢复满分{r['strict_pool_restored_to1']}题；新增损害M10满分题{r['new_harm_to_middle10_full_score']}题。" for r in recovery], '',
        '逐题损害与恢复全部保留在recovery_and_harm_cases.csv；此前手机时间、球衣数字、瓶身数字及作者名四例见four_previous_visual_cases.csv。',
        '此前展示的4例，E1–E4都没有恢复到Native满分；E5恢复球衣22，但总体仍低于M10。不能只展示修复成功的案例。',
        '少量原图人工复核（事后选例，不是总体错误标注）：35026瓶身SHEPPY\'S，E3从Alcohol恢复Sheppy\'s；35487硬币ONE POUND，E3从1恢复1 pound；36550标牌SAVE50%，E3从50%变为50且官方分数1→0，属于符号/表达损失，不能直接称数字辨认失败；35882衣服2XU，E3从2xu变为Zoot，软分0.9→0，是新增名称错误。',
        '数字/字符遗漏、名称替换、重复/提前终止可作为输出描述；本轮未对全部图片做人工OCR错误标注，不能把所有变化归因为某种OCR机制。', '',
        '## 新测真实性能', '', *perf, '',
        '沿用原48张不同图片；每条件3次预热、5次交错重复，每请求重算Vision、独立冷KV。固定16 Token是新增计算开销主比较；自然EOS不能排除输出长度影响。旧时延未混入。',
        '另对48张图片的每条件5次重复先取均值，再做配对Bootstrap，E2E−M10的毫秒差区间保存在latency_summary.csv。细小的时延点估计胜负不代表稳健统计优势。',
        'Prefill为Vision+Language首次CUDA前向；TTFT/E2E沿用冻结请求墙钟边界。Peak Allocated/Reserved、同完整序列子集与逐请求值见latency_summary.csv和latency_raw.csv。硬件FLOPS/DRAM计数未采集，不声称硬件实测吞吐。', '',
        '## 物理正确性与解释边界', '',
        '19项CPU测试、双卡各3题GPU Smoke通过；Native/No-op严格torch.equal。新物理路径与Dense参考保持历史FP16门槛并要求Greedy Token一致，未放宽阈值。',
        'E1–E4持久Image KV仅L11–L20；E5仅L11–L19。L2–L5均无持久KV，真正Decode继续物理跳算Attention。E4只恢复原Prefill P−1的文本Attention，不读Image。视觉Hidden均在L21前退出。',
        'E1/E2/E3使用同样执行形状，改变的是来源权限；E4多算四层一个P−1 Query。E5相对M9多算L20的视觉Hidden，但L20不留Image KV；M10−E5是在相同退出边界下恢复L20回看的固定顺序效果，不能当成唯一机制贡献。',
        '实际Projection Shape审计见tensor_shape_audit.csv和Smoke记录；未将Dense Mask视为节省FLOPs。非目标连接、MLP、位置/RoPE、独立KV及正常/异常Hook清理通过。', '',
        'E4的真实P−1 Hidden传播另外在同3道冻结Smoke题上记录32层（e4_p1_hidden_propagation.csv）；在所有测速请求结束后执行，不混入测速或额外质量选择。L0–L1应相同，从L2首次改变；非目标前缀的精确性与数值差异原样记录，不把舍入差异隐藏为严格相等。', '',
        '## 判定', '', *[f"- {r['condition']}：{r['classification']}。" for r in statuses], '',
        '以上分类仅表示当前协议下质量及固定长度时延的点估计，不是通用无损、独立非劣效或统计显著的加速结论。完整历史数据重复用于开发性诊断，不宣称外部独立复现。', '',
        '## 复现与停止', '',
        '运行前重新确认授权与空闲。已有冻结和断点不可覆盖；同源码可精确续跑。',
        '```bash', '/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.middle10_recovery.tests',
        '/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.middle10_recovery.run --driver',
        '/data/lk/深度学习/.envs/ate-vlm-research/bin/python -m vlm_research.middle10_recovery.analyze', '```', '',
        '完整逐断点SHA见result_hashes.json；源码/模型/授权/协议绑定见execution_freeze.json与source_manifest.json。GPU进程已退出，见gpu_release_audit.json。本轮到此停止。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n')


def main():
    freeze = verify_freeze()
    assert read(OUT / 'driver_exit.json')['returncodes'] == [0, 0]
    assert all(read(OUT / f'worker{r}/completion.json')['status']=='COMPLETE' for r in range(2))
    owned = set(read(OUT / 'driver_pids.json')['workers'])
    if (OUT / 'hidden_diagnostic_release.json').exists():
        owned.add(read(OUT / 'hidden_diagnostic_release.json')['pid'])
    resource = hardware()
    active = {int(line.split(',')[1].strip()) for line in resource['processes'].splitlines() if line.strip()}
    assert not owned.intersection(active)
    assert all(not Path('/proc', str(pid)).exists() for pid in owned)
    atomic(OUT / 'gpu_release_audit.json', dict(status='RELEASED', at=now(), workers=list(owned),
        task_pids_exited=True, task_pids_absent_from_nvidia_smi=True, **resource))
    samples, records, summary, comparisons, interaction, recovery, hashes = quality(freeze)
    times, labels = timing(freeze, samples, records, hashes)
    shape_rows, numerical_rows = [], []
    for rank in range(2):
        smokes = read(OUT / f'worker{rank}/smoke.json')
        assert smokes['sources'] == freeze['sources']
        for case in smokes['checks']:
            for condition in case['conditions']:
                numerical_rows.append(dict(rank=rank, key=case['key'], condition=condition['condition'],
                    fixed3_max_abs_logits=condition['numerical']['max_abs_logits'],
                    fixed3_hidden_relative_rmse=condition['numerical']['hidden_relative_rmse'],
                    fixed3_hidden_cosine_distance=condition['numerical']['hidden_cosine_distance'],
                    natural_max_abs_logits=condition['natural_numerical']['max_abs_logits'],
                    fixed3_passed=condition['numerical']['passed'],
                    natural_passed=condition['natural_numerical']['passed'],
                    first_logits_sha256=condition['first_logits_sha256']))
                for row in condition['projection_shapes']:
                    shape_rows.append(dict(rank=rank, key=case['key'], condition=condition['condition'], **row))
    write_csv(OUT / 'tensor_shape_audit.csv', shape_rows)
    write_csv(OUT / 'numerical_audit.csv', numerical_rows)
    plot(summary, times)
    report(summary, comparisons, interaction, recovery, times, labels)
    atomic(OUT / 'result_hashes.json', hashes)
    atomic(OUT / 'validation.json', dict(status='PASS', at=now(), CPU_tests=19,
        GPU_smoke_cases_per_card=3, cards=2, historical_quality_records_reused_verified=2000,
        new_quality_records=2500, all_4500_records_audited=True, images=309,
        performance_conditions=labels, performance_images=48 if labels else 0,
        unchanged_tolerances=TOLERANCES['llava'], source_freeze_verified=True,
        no_missing_no_duplicate_image_shards=True, actual_KV_positions_capacities_verified=True))
    atomic(OUT / 'completion.json', dict(status='COMPLETE', at=now(), new_quality_generations=2500,
        historical_reused_generations=2000, formal_timing_requests=48*2*5*len(labels),
        warmup_timing_requests=48*2*3*len(labels), code_commit=freeze['code_commit']))
    print(json.dumps(dict(accuracy=summary, comparisons=comparisons, interaction=interaction,
                          recovery=recovery, timing=times), indent=2, ensure_ascii=False))


def finish_user_stopped():
    """Publish complete quality without presenting interrupted timing as complete."""
    freeze = verify_freeze()
    pids = read(OUT / 'driver_pids.json')
    owned = [pids['driver'], *pids['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in owned)
    resource = hardware()
    active = {int(line.split(',')[1].strip()) for line in resource['processes'].splitlines() if line.strip()}
    assert not set(owned).intersection(active)
    for rank in range(2):
        assert read(OUT / f'worker{rank}/quality_complete.json')['status'] == 'COMPLETE'
    samples, records, summary, comparisons, interaction, recovery, hashes = quality(freeze)
    counts, timing_index = collections.Counter(), []
    for path in sorted((OUT / 'performance').rglob('*.json')):
        v = load(path)
        assert v['sources'] == freeze['sources']
        assert v['sample_sha256'] == digest(samples[v['key']])
        assert digest(v['input_binding']) == digest(records[v['key'], v['condition']]['input_binding'])
        if v['mode'] == 'natural_eos':
            assert v['generation']['token_ids'] == records[v['key'], v['condition']]['generation']['token_ids']
        else:
            assert len(v['generation']['token_ids']) == 16
        relative = str(path.relative_to(OUT))
        hashes[relative] = sha(path)
        counts['warmup' if v['warmup'] else 'formal'] += 1
        timing_index.append(dict(path=relative, key=v['key'], condition=v['condition'],
            mode=v['mode'], repeat=v['repeat'], warmup=v['warmup'], sha256=hashes[relative],
            status='PARTIAL_NOT_USED_FOR_SPEED_CLAIMS'))
    write_csv(OUT / 'partial_timing_index.csv', timing_index)
    numerical, shapes = [], []
    for rank in range(2):
        smoke = read(OUT / f'worker{rank}/smoke.json')
        assert smoke['sources'] == freeze['sources'] and len(smoke['checks']) == 3
        for case in smoke['checks']:
            for condition in case['conditions']:
                assert condition['numerical']['passed'] and condition['natural_numerical']['passed']
                numerical.append(dict(rank=rank, key=case['key'], condition=condition['condition'],
                    fixed3_max_abs_logits=condition['numerical']['max_abs_logits'],
                    fixed3_hidden_relative_rmse=condition['numerical']['hidden_relative_rmse'],
                    fixed3_hidden_cosine_distance=condition['numerical']['hidden_cosine_distance'],
                    natural_max_abs_logits=condition['natural_numerical']['max_abs_logits'],
                    fixed3_passed=True, natural_passed=True,
                    first_logits_sha256=condition['first_logits_sha256']))
                shapes.extend(dict(rank=rank, key=case['key'], condition=condition['condition'], **r)
                              for r in condition['projection_shapes'])
    write_csv(OUT / 'numerical_audit.csv', numerical)
    write_csv(OUT / 'tensor_shape_audit.csv', shapes)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    names = ['Native', 'C3', 'M9', 'M10', 'E1', 'E2', 'E3', 'E4', 'E5']
    x = np.arange(len(summary))
    y = np.array([r['accuracy'] for r in summary])
    colors = ['#555555', '#999999', '#8e998b', '#168376', '#4684b0', '#cd963d', '#914c9c', '#bc5964', '#8e998b']
    axes[0].bar(x, y, color=colors)
    axes[0].set(xticks=x, xticklabels=names, ylabel='TextVQA soft accuracy (%)', title='Same frozen 500 questions')
    for i, value in enumerate(y):
        axes[0].text(i, value+.4, f'{value:.2f}', ha='center', fontsize=7)
    new = summary[4:]
    gains = np.array([r['delta_middle10_pp'] for r in new])
    axes[1].errorbar(np.arange(5), gains,
        yerr=np.array([[r['delta_middle10_pp']-r['delta_middle10_ci_low'] for r in new],
                       [r['delta_middle10_ci_high']-r['delta_middle10_pp'] for r in new]]),
        fmt='o', color='#305c83', capsize=4)
    axes[1].axhline(0, color='#777777', lw=.8)
    axes[1].set(xticks=np.arange(5), xticklabels=list(NEW), ylabel='Accuracy change vs Middle10 (pp)',
                title='20,000 image-cluster paired bootstrap: 95% CI')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'quality_comparison.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)
    table = ['|条件|Accuracy|ΔMiddle10 pp|ΔNative pp|', '|---|---:|---:|---:|']
    table += [f"|{r['condition']}|{r['accuracy']:.2f}%|{r['delta_middle10_pp']:+.2f}|{r['delta_native_pp']:+.2f}|" for r in summary]
    pairs = ['|比较|ΔAccuracy pp|图片簇95%CI|逐题软分升/降|', '|---|---:|---|---:|']
    pairs += [f"|{r['comparison']}|{r['delta_accuracy_pp']:+.2f}|[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]|{r['score_increased']}/{r['score_decreased']}|" for r in comparisons]
    text = ['# Middle10剩余TextVQA损失与固定恢复实验', '',
        '**质量：全部完成。性能：用户要求提前停止，PARTIAL，不作加速结论。**', '',
        '2026年11月6日，用户指示“如何得分不高，就不要测速了，直接提交”。收到指示后仅向本任务两个Worker发送SIGINT，退出码-2为主动中断，不是数值验证失败。未启动额外GPU诊断。', '',
        '## 结论', '',
        '最佳已测E3为41.38%，比Middle10提高1.92pp，图片簇配对95%CI [0.57, 3.39]；仍比Native低8.30pp。只恢复原10.22pp质量缺口的约18.8%，不能称为解决主要瓶颈，不建议直接将其作为接近Native的方案。', '',
        'E1=L20 R视觉写入；E2=额外开放P−1在L0、L1、L6–L10读图；E3=E1+E2；E4=E3并恢复仅Prefill P−1的L2–L5文本Attention；E5=Middle9仅将视觉Hidden退出推迟至L21前。全部规则、KV和原位置语义在protocol.json中固定。', '',
        '## 完整质量', '', *table, '',
        '500题、309个图片SHA簇。官方十答案共识软分；2,000条历史基线经绑定和双卡Smoke重现审计复用，新增E1–E5共2,500条完整答案。未筛题、未改Prompt和评分。', '',
        '## 配对变化', '', *pairs, '',
        f"2×2交互E3−E2−E1+Middle10为{interaction['delta_accuracy_pp']:+.2f}pp，95%CI [{interaction['ci_low']:+.2f}, {interaction['ci_high']:+.2f}]。没有明确正协同证据，CI跨0不等于没有功能效应。", '',
        'E4比E3低0.86pp，CI跨0，未观察到额外稳定恢复。E5与Middle9的500题软分全部相同，但有1题Token序列改变；单纯推迟退出没有观察到得分收益。Middle10比E5高3.12pp，因此相同L21退出下，L20答案视觉回看比只延长Hidden保留更有用；不外推为唯一信息路径。', '',
        'Middle10低于Native的74题中，E3改善11题，全部恢复到至少Native软分；但对Middle10原满分题新增2题损害。总体相对Middle10升14题、降4题，负面结果完整保留。', '',
        '## 案例与未解释的损失', '',
        '事后查看原图的例子：35026中E3将Alcohol恢复为可见品牌Sheppy\'s；35487将1恢复为1 pound。反例36550从50%变为50而失分，不能当成数值识别错误；35882从2xu变为Zoot，产生新品牌错误。这里只是固定样本中的输出观察，不是OCR机制证明。', '',
        '此前重点展示的34822时钟、39376球衣、39441瓶身数字、39065作者，E1–E4均未恢复到Native满分，见four_previous_visual_cases.csv。未逐题人工归因其余全部损失。', '',
        '## 验证与停止范围', '',
        '19项CPU测试通过；每卡3道真实Smoke通过。Native/No-op全部返回有效Logits严格torch.equal，历史四基线Token与首Logits哈希一致；新物理路径与Dense参考在冻结FP16门槛下通过且生成Token一致。禁止概率为0、非目标Prompt、32层MLP、RoPE、独立KV与Hook清理通过。未放宽容差。', '',
        '真实投影Shape见tensor_shape_audit.csv；数值记录见numerical_audit.csv。E1–E4只持久保留L11–L20的Image KV，E2/E3/E4的Early7 P−1回看使用临时KV而不额外持久缓存。E5只保留L11–L19。完整位置与容量见physical_kv_audit.csv。', '',
        f"已保存性能请求：正式{counts['formal']}条、预热{counts['warmup']}条，均为中断前实际完成记录。固定48图矩阵未完整结束，不聚合或排名其速度，不报告完整性能或质量与速度同时改善。partial_timing_index.csv列出已有记录，原始值保留在断点包。", '',
        '额外E4逐层Hidden传播诊断未运行，标记NOT_RUN_USER_STOP；不得把Smoke中有限的Hidden数值对齐审计冒称完整传播结果。其余未完成项目均不视为已有结果。', '',
        '## 统计边界和复现', '',
        '20,000次按图片SHA整簇配对Bootstrap，问题等权、固定种子20261105，普通探索性95%CI。500题为历史探索队列，未使用未校正扫描结果宣称独立确认、非劣效或无损。', '',
        '![完整质量和配对区间](quality_comparison.png)', '',
        '全部逐题答案见per_question.csv；负面与恢复见recovery_and_harm_cases.csv。源码/模型/输入绑定见execution_freeze.json和source_manifest.json。归档中的原始断点可按相对路径还原；result_hashes.json逐文件校验。', '',
        '仅CPU复算：`python -m vlm_research.middle10_recovery.analyze --quality-only-stop`。原GPU运行命令仅供审阅，当前已停止，不应自动续跑。gpu_release_audit.json记录进程退出和显存释放。', '',
        '本轮按用户要求提交并停止，不追加搜索。']
    (OUT / 'report_zh.md').write_text('\n'.join(text)+'\n')
    atomic(OUT / 'result_hashes.json', hashes)
    atomic(OUT / 'user_stop.json', dict(at=now(), requested_action='Stop timing and submit quality due to insufficient score',
        user_message='如何得分不高，就不要测速了，直接提交', workers_interrupted_by='SIGINT',
        driver_exit=read(OUT / 'driver_exit.json'), no_new_GPU_work=True))
    atomic(OUT / 'gpu_release_audit.json', dict(status='RELEASED', at=now(), task_pids=owned,
        task_pids_exited=True, task_pids_absent_from_nvidia_smi=True, **resource))
    atomic(OUT / 'validation.json', dict(status='QUALITY_PASS_PERFORMANCE_STOPPED_BY_USER', at=now(), CPU_tests=19,
        GPU_smoke_cases_per_card=3, cards=2, historical_records_reused_verified=2000,
        new_quality_records=2500, all_4500_records_audited=True, images=309,
        performance_complete=False, partial_performance=dict(counts), e4_hidden_propagation='NOT_RUN_USER_STOP',
        unchanged_tolerances=TOLERANCES['llava'], source_freeze_verified=True,
        no_missing_no_duplicate_quality_shards=True, actual_KV_positions_capacities_verified=True))
    atomic(OUT / 'completion.json', dict(status='STOPPED_BY_USER', quality_status='COMPLETE',
        performance_status='PARTIAL_CANCELLED', at=now(), new_quality_generations=2500,
        historical_reused_generations=2000, partial_timing_requests=dict(counts),
        code_commit=freeze['code_commit'], remaining_GPU_work_cancelled=True))
    print(json.dumps(dict(status='QUALITY_COMPLETE_PERFORMANCE_CANCELLED', accuracy=summary,
        partial_performance=dict(counts), task_pids_exited=owned), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quality-only-stop', action='store_true')
    if parser.parse_args().quality_only_stop:
        finish_user_stopped()
    else:
        main()

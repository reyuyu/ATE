"""Single post-run aggregation, paired image-cluster inference, and figures."""
import argparse
import csv
import json
import math
import os
import sys
import importlib.metadata
from pathlib import Path
import subprocess

import numpy as np

from .common import (ROOT, OUT, CONDITIONS, COMPARISONS, SEED, read, seal,
                     checkpoint, key, verify, sha256)


def table(path, rows):
    assert rows
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def paired_bootstrap(values, images, repetitions=20000):
    """Resample image clusters once per replicate; retain question weighting."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    unique = sorted(set(images))
    sums = np.array([values[np.array(images) == image].sum(axis=0) for image in unique])
    sizes = np.array([sum(x == image for x in images) for image in unique])
    rng = np.random.default_rng(SEED)
    draws = []
    for start in range(0, repetitions, 500):
        indices = rng.integers(0, len(unique), size=(min(500, repetitions-start), len(unique)))
        draws.append(sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None] * 100)
    return np.concatenate(draws)


def exact_mcnemar(harm, rescue):
    n = harm + rescue
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(harm, rescue)+1)) / 2**n)


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol = verify()
    env = read(OUT/'analysis_environment.json')
    assert env['executable'] == sys.executable
    assert env['versions'] == {p: importlib.metadata.version(p) for p in env['versions']}
    assert not (OUT/'completion.json').exists(), 'Already aggregated; use --check instead'
    complete = read(OUT/'generation_completion.json')
    assert complete['count'] == 1200 and read(OUT/'validation.json')['passed']
    samples = read(OUT/'sample_manifest.json')['samples']
    prepared = {key(s): s for s in read(OUT/'qwen_input_manifest.json')['samples']}
    assert len(list((OUT/'checkpoints').glob('*/*.json'))) == 1200
    per, matrix, records = [], [], {}
    for sample in samples:
        identifier = key(sample)
        results = {c: read(checkpoint(sample, c)) for c in CONDITIONS}
        records[identifier] = results
        full = results['Full']
        first_id = full['first_token_id']
        raw_full = np.load(full['raw_first_logits_path'], allow_pickle=False).astype(np.float64)
        logp_full = raw_full[first_id] - np.logaddexp.reduce(raw_full)
        matrix.append([results[c]['correct'] for c in CONDITIONS])
        for condition in CONDITIONS:
            r = results[condition]
            assert r['condition'] == condition and r['question_id'] == sample['question_id']
            assert r['image_sha256'] == sample['image_sha256'] and r['input_binding'] == prepared[identifier]['binding']
            assert r['protocol_sha256'] == sha256(OUT/'protocol.json')
            assert r['code_commit'] == complete['code_commit']
            assert sha256(r['raw_first_logits_path']) == r['raw_first_logits_sha256']
            assert r['correct'] == int(r['prediction'] == sample['answer'].lower())
            assert r['all_logits_finite'] and r['cache_initial_empty']
            assert r['cache_lengths'] == [prepared[identifier]['layout']['input_length']+len(r['token_ids'])-1]*36
            expected_prefill = {'Full': 0, 'I-Bypass': len(prepared[identifier]['roles']['I']),
                'R-Bypass': len(prepared[identifier]['roles']['R']), 'A-Bypass': 1,
                'I+A-Bypass': len(prepared[identifier]['roles']['I'])+1}[condition]
            assert r['target_prefill_queries'] == expected_prefill
            if condition == 'Full':
                assert not r['hook_events']
            else:
                assert len(r['hook_events']) == 6*len(r['token_ids'])
                for event in r['hook_events']:
                    assert event['layer'] in protocol['target_layers'] and not event['noop']
                    expected = expected_prefill if event['phase'] == 'prefill' else int(condition in ('A-Bypass', 'I+A-Bypass'))
                    assert event['query_count'] == expected and event['non_target_exact']
                    assert event['target_zero'] == bool(expected)
            raw = np.load(r['raw_first_logits_path'], allow_pickle=False).astype(np.float64)
            assert np.isfinite(raw).all()
            logp = raw[first_id] - np.logaddexp.reduce(raw)
            per.append(dict(question_id=sample['question_id'], task=sample['task'], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], question=sample['question'], gold=sample['answer'],
                condition=condition, text=r['text'], prediction=r['prediction'], token_ids=r['token_ids'],
                correct=r['correct'], Full_correct=full['correct'],
                correct_to_wrong=int(full['correct'] and not r['correct']),
                wrong_to_correct=int(not full['correct'] and r['correct']),
                first_token_flip=int(r['first_token_id'] != first_id),
                sequence_changed=int(r['token_ids'] != full['token_ids']),
                signed_first_logp_loss=float(logp_full-logp),
                Full_fixed_first_token=first_id, Full_first_logp=float(logp_full), condition_fixed_first_logp=float(logp),
                eos=r['eos'], empty=r['empty'], truncated=r['truncated'], repeated_trigram=r['repeated_trigram'],
                generated_tokens=len(r['token_ids']), I_queries=r['prefill_role_counts']['I'],
                R_queries=r['prefill_role_counts']['R'], A_queries=1+r['decode_queries'],
                target_prefill_queries=r['target_prefill_queries'], target_decode_queries=r['target_decode_queries'],
                intervened_query_layer_updates=6*(r['target_prefill_queries']+r['target_decode_queries']),
                input_binding=r['input_binding'], code_commit=r['code_commit'],
                raw_first_logits_path=r['raw_first_logits_path'], raw_first_logits_sha256=r['raw_first_logits_sha256']))
    m = np.array(matrix, dtype=int)
    images = [s['image_sha256'] for s in samples]
    deltas = [m[:, CONDITIONS.index(a)]-m[:, CONDITIONS.index(b)] for a, b in COMPARISONS]
    interaction = m[:, 4]-m[:, 1]-m[:, 3]+m[:, 0]
    draws = paired_bootstrap(np.column_stack([m, *deltas, interaction]), images)
    summary, pairs = [], []
    for j, c in enumerate(CONDITIONS):
        rows = [r for r in per if r['condition'] == c]
        ci = np.quantile(draws[:, j], [.025, .975])
        summary.append(dict(condition=c, n=240, correct=int(m[:, j].sum()), accuracy_pct=float(m[:, j].mean()*100),
            accuracy_ci_low=float(ci[0]), accuracy_ci_high=float(ci[1]),
            delta_full_pp=float((m[:, j]-m[:, 0]).mean()*100),
            correct_to_wrong=sum(r['correct_to_wrong'] for r in rows),
            wrong_to_correct=sum(r['wrong_to_correct'] for r in rows),
            first_token_flips=sum(r['first_token_flip'] for r in rows),
            sequence_changes=sum(r['sequence_changed'] for r in rows),
            mean_signed_first_logp_loss=float(np.mean([r['signed_first_logp_loss'] for r in rows])),
            eos=sum(r['eos'] for r in rows), empty=sum(r['empty'] for r in rows),
            truncated=sum(r['truncated'] for r in rows), repeated_trigram=sum(r['repeated_trigram'] for r in rows)))
    for j, (a, b) in enumerate(COMPARISONS):
        x, y = m[:, CONDITIONS.index(a)], m[:, CONDITIONS.index(b)]
        harm, rescue = int(((y == 1) & (x == 0)).sum()), int(((y == 0) & (x == 1)).sum())
        ci = np.quantile(draws[:, 5+j], [.025, .975])
        adjusted = np.quantile(draws[:, 5+j], [.025/7, 1-.025/7])
        pairs.append(dict(comparison=a+' minus '+b, delta_pp=float((x-y).mean()*100),
            ci_low=float(ci[0]), ci_high=float(ci[1]),
            bonferroni7_low=float(adjusted[0]), bonferroni7_high=float(adjusted[1]),
            harm=harm, rescue=rescue, mcnemar_exact_two_sided=exact_mcnemar(harm, rescue),
            bootstrap_repetitions=20000, seed=SEED, clusters=len(set(images))))
    ci = np.quantile(draws[:, -1], [.025, .975])
    adjusted = np.quantile(draws[:, -1], [.025/7, 1-.025/7])
    unique = ((m[:, 0] == 1) & (m[:, 1] == 1) & (m[:, 3] == 1) & (m[:, 4] == 0))
    joint = dict(comparison='IA - I - A + Full', delta_pp=float(interaction.mean()*100),
        ci_low=float(ci[0]), ci_high=float(ci[1]), bonferroni7_low=float(adjusted[0]),
        bonferroni7_high=float(adjusted[1]), unique_harms=int(unique.sum()),
        unique_harm_keys=[key(s) for s, flag in zip(samples, unique) if flag], repetitions=20000, seed=SEED)
    task_rows = []
    for task in sorted({s['task'] for s in samples}):
        for c in CONDITIONS:
            rows = [r for r in per if r['task'] == task and r['condition'] == c]
            task_rows.append(dict(task=task, condition=c, n=len(rows), correct=sum(r['correct'] for r in rows),
                accuracy_pct=float(np.mean([r['correct'] for r in rows])*100),
                harm=sum(r['correct_to_wrong'] for r in rows), rescue=sum(r['wrong_to_correct'] for r in rows)))
    table(OUT/'per_question.csv', per)
    table(OUT/'accuracy_summary.csv', summary)
    table(OUT/'task_accuracy.csv', task_rows)
    table(OUT/'paired_comparisons.csv', pairs)
    table(OUT/'joint_interaction.csv', [joint])
    negative = [r for r in per if r['condition'] != 'Full' and (not r['correct'] or r['empty'] or r['truncated'] or r['repeated_trigram'])]
    table(OUT/'negative_cases.csv', negative)
    harms = [r for r in per if r['correct_to_wrong']]
    # May be empty in other cohorts; preserve schema even with no events.
    table(OUT/'harm_cases.csv', harms if harms else [{k: '' for k in per[0]}])
    joints = []
    for s, row, u in zip(samples, m, unique):
        results = records[key(s)]
        joints.append(dict(question_id=s['question_id'], task=s['task'], image_sha256=s['image_sha256'],
            gold=s['answer'], question=s['question'], joint_unique_harm=bool(u),
            **{c+'_correct': int(v) for c, v in zip(CONDITIONS, row)},
            **{c+'_text': results[c]['text'] for c in CONDITIONS}))
    table(OUT/'joint_per_question.csv', joints)
    with (OUT/'llava_reference.csv').open() as handle:
        llava = {r['task']+'__'+r['question_id']: r for r in csv.DictReader(handle)}
    assert set(llava) == {key(s) for s in samples}
    cross = []
    for s in samples:
        assert llava[key(s)]['image_sha256'] == s['image_sha256']
    for j, c in enumerate(CONDITIONS):
        col = {'Full': 'Full_correct', 'I-Bypass': 'I_correct', 'A-Bypass': 'A_correct', 'I+A-Bypass': 'joint_correct'}.get(c)
        if col:
            lm = np.array([int(llava[key(s)][col]) for s in samples])
        else:
            path = OUT/'llava_formal_reference.csv'
            with path.open() as handle:
                reference = {r['task']+'__'+r['question_id']: r for r in csv.DictReader(handle) if r['condition'] == c}
            assert set(reference) == set(llava)
            lm = np.array([int(reference[key(s)]['correct']) for s in samples])
        cross.append(dict(condition=c, qwen_accuracy_pct=float(m[:, j].mean()*100),
            llava_accuracy_pct=float(lm.mean()*100),
            note='Same cohort, different native prompt/processor/precision/target layers; descriptive cross-model comparison'))
    table(OUT/'cross_model_summary.csv', cross)
    colors = ['#424242', '#23856d', '#ce4c4c', '#4674ad', '#a368aa']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    x = np.arange(5)
    axes[0].bar(x, [r['accuracy_pct'] for r in summary], color=colors)
    axes[0].set(xticks=x, xticklabels=['Full', 'I', 'R', 'A', 'I+A'], ylim=(0,100), ylabel='Complete-answer accuracy (%)')
    for j, row in enumerate(summary):
        axes[0].text(j, row['accuracy_pct']+1.3, f"{row['correct']}/240", ha='center', fontsize=9)
    for j, row in enumerate(pairs[:4]):
        axes[1].errorbar(j, row['delta_pp'], yerr=[[row['delta_pp']-row['ci_low']], [row['ci_high']-row['delta_pp']]],
                         fmt='o', color=colors[j+1], capsize=4)
    axes[1].axhline(0, color='gray', linewidth=.8)
    axes[1].set(xticks=range(4), xticklabels=['I', 'R', 'A', 'I+A'], ylabel='Paired delta vs Full (pp), 95% CI')
    axes[2].bar(x[1:]-.18, [r['correct_to_wrong'] for r in summary[1:]], width=.36, color='#ce4c4c', label='Correct to wrong')
    axes[2].bar(x[1:]+.18, [r['wrong_to_correct'] for r in summary[1:]], width=.36, color='#23856d', label='Wrong to correct')
    axes[2].set(xticks=x[1:], xticklabels=['I', 'R', 'A', 'I+A'], ylabel='Questions')
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle('Qwen3-VL-8B / L1-L6 / fixed 240 GQA images')
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT/('figure_query_roles.'+suffix), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
    ax.plot(range(5), [r['qwen_accuracy_pct'] for r in cross], 'o-', label='Qwen3-VL L1-L6 / BF16')
    ax.plot(range(5), [r['llava_accuracy_pct'] for r in cross], 's--', label='LLaVA L2-L5 / FP16')
    ax.set(xticks=range(5), xticklabels=['Full', 'I', 'R', 'A', 'I+A'], ylabel='Complete-answer accuracy (%)')
    ax.legend(frameon=False)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT/('figure_cross_model.'+suffix), dpi=180)
    plt.close(fig)
    report(summary, pairs, joint, cross, complete, prepared, len(negative))
    seal(OUT/'analysis_validation.json', dict(passed=True, checkpoints=1200, rows=len(per),
        images=240, source_protocol_verified=True, per_condition=240,
        raw_first_logits_sha256_all_verified=True, no_nan_inf=True,
        query_scope_all_checked=True, llava_same_cohort=True, bootstrap_repetitions=20000,
        rng_seed=SEED, comparison_family=7, aggregation_code_commit=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()))
    seal(OUT/'result_hashes.json', {str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.suffix in ('.csv', '.json', '.md', '.png', '.pdf')
        and p.name not in ('result_hashes.json', 'completion.json')})
    seal(OUT/'completion.json', dict(status='COMPLETE', samples=240, conditions=5, generations=1200,
        generation_code_commit=complete['code_commit'], protocol_sha256=sha256(OUT/'protocol.json'),
        report_sha256=sha256(OUT/'report_zh.md'), result_hashes_sha256=sha256(OUT/'result_hashes.json'),
        statistics_complete=True))
    print(json.dumps(dict(accuracy=summary, paired=pairs, joint=joint), indent=2))


def report(summary, pairs, joint, cross, complete, prepared, n_negative):
    lines = ['# Qwen3-VL Query角色敏感性：固定L1–L6、240题正式探索', '',
        '## 结论', '',
        '五条件均为本轮独立真实KV Cache完整Greedy生成，不是首Token命中率。只将指定Query的o_proj输出置零，未跳过实际计算。', '',
        '| 条件 | 正确/240 | Accuracy | ΔFull pp | 损害/纠正 | 首Token翻转 | 平均signed LogP Loss |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['condition']}|{r['correct']}/240|{r['accuracy_pct']:.3f}%|{r['delta_full_pp']:+.3f}|{r['correct_to_wrong']}/{r['wrong_to_correct']}|{r['first_token_flips']}|{r['mean_signed_first_logp_loss']:+.5f}|")
    lines += ['', '## 图像簇配对统计', '',
        '固定seed=20261103，按Image SHA256整簇配对Bootstrap 20,000次，保留题目权重。240张图各一题。六个预设比较加I+A交互共七项，同时报告普通和Bonferroni校正区间。任务分组仅作描述。', '',
        '| 比较 | Δ pp | 普通95%CI | 七项校正95%CI | 损害/纠正 |', '|---|---:|---|---|---:|']
    for r in pairs:
        lines.append(f"|{r['comparison']}|{r['delta_pp']:+.3f}|[{r['ci_low']:.3f}, {r['ci_high']:.3f}]|[{r['bonferroni7_low']:.3f}, {r['bonferroni7_high']:.3f}]|{r['harm']}/{r['rescue']}|")
    lines += ['', f"I+A交互 (IA−I−A+Full)：{joint['delta_pp']:+.3f} pp，95%CI [{joint['ci_low']:.3f}, {joint['ci_high']:.3f}]，校正CI [{joint['bonferroni7_low']:.3f}, {joint['bonferroni7_high']:.3f}]。",
        f"联合独有损害（Full、I、A均正确但I+A错误）：{joint['unique_harms']}题；题号见joint_interaction.csv。", '']
    r_i, r_a = pairs[4:6]
    clear = r_i['ci_high'] < 0 and r_a['ci_high'] < 0
    lines += [f"**R相对I/A是否明显更有害：{'是，两个预设配对比较的普通95%CI均低于0' if clear else '未同时获得两个普通95%CI均低于0的证据，需按区间解释'}。**",
        f"I+A相对Full净变化为{summary[4]['delta_full_pp']:+.3f} pp，损害{summary[4]['correct_to_wrong']}题、纠正{summary[4]['wrong_to_correct']}题，不能仅凭净变化或CI跨0宣称严格无损。",
        '低敏感性是本批数据、固定层段和指定干预下的行为结果；不是Attention不读取信息、不是可物理删除层，也不是已实现推理加速。', '',
        '## 与LLaVA的同题对照', '', '| 条件 | Qwen Accuracy | LLaVA Accuracy |', '|---|---:|---:|']
    for r in cross:
        lines.append(f"|{r['condition']}|{r['qwen_accuracy_pct']:.3f}%|{r['llava_accuracy_pct']:.3f}%|")
    lines += ['',
        'LLaVA历史结果来自8d12f8c05f2b246d029ec4d9ffe45d1f857fa947及8019c6009bb1d8c9f2dc3ffd8c18bf4dfb8c1fd1，逐题ID和图片SHA一致。Qwen保留历史原生Prompt、Processor、BF16和DeepStack，LLaVA为FP16、L2–L5；因此这是角色模式的跨模型对照，不是控制所有架构与层数因素的实验。',
        ('本轮支持跨模型复现“R比I/A更敏感”的方向，但I+A是否仍低损害必须结合上表真实损害及区间单独判断。' if clear else '本轮不能完整确认LLaVA式角色分工跨模型复现，不能将未显著差异解释为角色等价。'), '',
        '## 协议与正确性', '',
        '- 固定Qwen3-VL-8B-Instruct revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b；BF16/eager，Transformers4.57.6，原生分辨率、mRoPE、DeepStack保持不变。',
        '- 原LLaVA样本清单字节不变，SHA256：258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c。未重采样、未按结果筛选。',
        '- I使用实际image-pad位置；R严格位于vision_end之后、P−1之前，包括后图像模板文本，排除vision_end本身；A包括P−1与全部真实Decode Query。没有硬编码576。',
        '- 只关闭0-based L1–L6的Attention o_proj目标行，不改Mask/其余行/MLP/Residual/KV。I/R只作用Prefill；A与I+A在真实Decode继续关闭。',
        '- CPU范围/异常清理测试，以及输入长度最短、中位、最长三题GPU Smoke全部通过；全部返回有效logits在同形状logits_to_keep=1下Full/No-op逐位torch.equal。没有放宽容差。',
        '- Smoke检查独立Cache对象及实际存储、每步36层KV长度、原生BF16残差、三路DeepStack特征与注入、MLP调用、历史A-only一致性、异常Hook清理、输入及权重不变。正式1200检查点再校验范围、有限性、Cache及原始Logits SHA。',
        '- R、I、A干预Query数量不同，不能将总损害直接当作单Query贡献；低净损害不等于严格无损。无训练、无Kernel、无其他层段扫描。', '']
    counts = [len(s['roles']['I']) for s in prepared.values()]
    relay = [len(s['roles']['R']) for s in prepared.values()]
    lines += [f"实际I Query范围：{min(counts)}–{max(counts)}；R Query范围：{min(relay)}–{max(relay)}。",
        f"正式生成耗时{complete['formal_seconds']/60:.2f}分钟（包含结束模型内容校验），显存allocated峰值{complete['peak_allocated_mib']:.0f}MiB、reserved峰值{complete['peak_reserved_mib']:.0f}MiB。",
        f"全部负面结果共{n_negative}个条件-题目行，见negative_cases.csv；真实原对变错见harm_cases.csv；完整答案、Token与LogP见per_question.csv。EOS/空答案/截断/重复诊断见accuracy_summary.csv。",
        f"生成源码Commit：`{complete['code_commit']}`；协议SHA256：`{sha256(OUT/'protocol.json')}`。",
        '原始首步全词表Logits仅保存在服务器外部结果目录，不上传权重或大缓存；对应哈希随检查点交付。', '',
        '![Qwen角色结果](figure_query_roles.png)', '', '![跨模型同题对照](figure_cross_model.png)', '']
    (OUT/'report_zh.md').write_text('\n'.join(lines), encoding='utf-8')


def check():
    verify()
    c = read(OUT/'completion.json')
    assert c['status'] == 'COMPLETE' and c['generations'] == 1200
    assert sha256(OUT/'report_zh.md') == c['report_sha256']
    assert sha256(OUT/'result_hashes.json') == c['result_hashes_sha256']
    for name, expected in read(OUT/'result_hashes.json').items():
        assert sha256(OUT/name) == expected, name
    print('COMPLETE: frozen bindings and all result hashes verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    check() if args.check else analyze()

"""One preregistered paired comparison; historical No-Writer is reference only."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np

from .common import *
from ..qwen3vl_visual_writing_necessity.analyze import paired_bootstrap


def decision(lower_one_sided, low_two_sided, high_two_sided):
    if lower_one_sided > -1.:
        return 'SUPPORTED'
    if high_two_sided < -1.:
        return 'BELOW_MARGIN'
    return 'INSUFFICIENT'


def table(name, rows, fields=None):
    with (OUT/name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def compare(samples, records, baselines, label):
    values = np.array([[baselines[key(s)]['All-Writer']['generation']['correct'],
        records[key(s)]['generation']['correct'], baselines[key(s)]['No-Writer']['generation']['correct']]
        for s in samples], dtype=float)
    draws = paired_bootstrap(values*100, [s['image_sha256'] for s in samples], repetitions=20000, seed=SEED)
    delta = (values[:, 1]-values[:, 0])*100
    paired_draws = draws[:, 1]-draws[:, 0]
    low, high = np.quantile(paired_draws, [.025, .975])
    lower = float(np.quantile(paired_draws, .05))
    gs = [records[key(s)]['generation'] for s in samples]
    fs = [baselines[key(s)]['All-Writer']['generation'] for s in samples]
    result = dict(subset=label, questions=len(samples), image_clusters=len({s['image_sha256'] for s in samples}),
        All_Writer_accuracy_pct=float(values[:, 0].mean()*100), Keep_L0_L21_accuracy_pct=float(values[:, 1].mean()*100),
        No_Writer_reference_accuracy_pct=float(values[:, 2].mean()*100), delta_accuracy_pp=float(delta.mean()),
        ci_95_two_sided_low=float(low), ci_95_two_sided_high=float(high), ci_95_one_sided_lower=lower,
        noninferiority_margin_pp=-1., primary_NI_decision=decision(lower, low, high) if label == 'overall' else 'DESCRIPTIVE_ONLY',
        stricter_minus_0_5_pp_sensitivity=bool(lower > -.5) if label == 'overall' else '',
        correct_to_wrong=int(((values[:, 0] == 1) & (values[:, 1] == 0)).sum()),
        wrong_to_correct=int(((values[:, 0] == 0) & (values[:, 1] == 1)).sum()),
        text_changes=sum(g['text'] != f['text'] for g, f in zip(gs, fs)),
        first_token_flips=sum(g['first_token_flip'] for g in gs),
        eos=sum(g['eos'] for g in gs), truncated=sum(g['truncated'] for g in gs),
        empty=sum(g['empty'] for g in gs), repeated=sum(g['repeated_trigram'] for g in gs))
    np.savez_compressed(OUT/f'bootstrap_{label}.npz', accuracy_pct=draws, delta_accuracy_pp=paired_draws)
    score_intervals = np.quantile(draws, [.025, .975], axis=0)
    return result, values, score_intervals


def audit():
    from tokenizers import Tokenizer
    p = verify()
    assert read(OUT/'historical_reuse_audit.json')['paired_All_Writer'] == 3000
    samples = read(OUT/'sample_manifest.json')['samples']
    specs, baselines = read(OUT/'input_bindings.json'), read(OUT/'historical_baselines.json')
    by = {key(s): s for s in samples}
    shards = read(OUT/'shards.json')['keys']
    image_sets = [{by[k]['image_sha256'] for k in part} for part in shards]
    assert image_sets[0].isdisjoint(image_sets[1])
    scorer = read(OUT/'official_gqa_prediction.json')
    assert sha256(scorer['path']) == scorer['sha256']
    scope = {}
    exec(compile(ast.parse(scorer['function_source']), scorer['path'], 'exec'), scope)
    tokenizer = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    eos = read(MODEL/'generation_config.json')['eos_token_id']
    eos = [eos] if isinstance(eos, int) else eos
    records, hashes, maximum_row_error = {}, [], 0.
    for shard in (0, 1):
        done, smoke = read(OUT/f'shard{shard}/complete.json'), read(OUT/f'shard{shard}/smoke.json')
        assert done['passed'] and smoke['passed'] and done['binding'] == smoke['binding'] == digest(p)
        paths = sorted((OUT/f'shard{shard}/checkpoints').glob('*.json'))
        assert len(paths) == done['generations'] == 1500
        for path in paths:
            r = read(path)
            identifier = key(r)
            s, spec, g = by[identifier], specs[identifier], r['generation']
            assert identifier in shards[shard] and identifier not in records
            assert r['condition'] == CONDITION and r['binding'] == digest(p) and r['code_commit'] == p['code_commit']
            assert r['sample_sha256'] == digest(s) and r['input_binding'] == spec['binding']
            assert r['image_sha256'] == s['image_sha256'] and r['gold'] == s['answer']
            assert r['gpu_uuid'] == p['gpus'][shard]['uuid'] and digest(g) == r['generation_sha256']
            assert g['writer_layers'] == OPEN_LAYERS and g['P1_Decode_image_off'] and g['logits_to_keep'] == 1
            assert g['independent_native_cache'] and g['cache_initial_empty'] and g['all_logits_finite']
            assert 1 <= len(g['token_ids']) <= 32
            text = tokenizer.decode(g['token_ids'], skip_special_tokens=True).strip()
            assert text == g['text'] and scope['gqa_prediction'](text) == g['prediction']
            assert g['correct'] == int(g['prediction'] == s['answer'].lower())
            assert g['eos'] == (g['token_ids'][-1] in eos)
            assert g['truncated'] == (len(g['token_ids']) == 32 and not g['eos'])
            assert g['first_token_flip'] == int(g['token_ids'][0] != baselines[identifier]['All-Writer']['generation']['token_ids'][0])
            n, count = spec['layout']['input_length'], len(spec['roles']['I'])
            a = g['source_audit']
            assert a['layer_events'] == 36*len(g['token_ids']) and a['steps'] == len(g['token_ids'])
            assert a['blocked_probability_max'] == 0 and a['row_sum_max_error'] <= 2**-7
            assert a['prefill_blocked_connections_per_layer'] == [count*(1+(0 if i < 22 else len(spec['roles']['R']))) for i in range(36)]
            assert a['decode_blocked_connections_per_layer'] == [count]*36
            assert set(a['open_R_image_probability_max']) == {str(i) for i in range(22)}
            assert all(v > 0 for v in a['open_R_image_probability_max'].values())
            assert a['non_target_mask_exact'] and a['Image_Query_native'] and a['BOS_Self_legal'] and a['hook_cleanup']
            assert all(length == n+len(g['token_ids'])-1 for length in g['cache_lengths'])
            for t, e in enumerate(a['actual_positions']):
                assert e['step'] == t and e['cache_position'] == ([n+t-1] if t else [0, n-1])
                assert e['shape'] == ([3, 1, 1] if t else [3, 1, n])
            records[identifier] = r
            hashes.append(dict(path=str(path.relative_to(OUT)), sha256=sha256(path)))
            maximum_row_error = max(maximum_row_error, a['row_sum_max_error'])
    assert len(records) == 3000
    seal(OUT/'checkpoint_hashes.json', hashes)
    seal(OUT/'validation.json', dict(passed=True, new_generations=3000, reused_baselines=6000, historical_control_records=3000,
        new_answers_independently_decoded_rescored=3000, mask_probability_max=0., maximum_row_sum_error=maximum_row_error,
        all_input_model_source_bindings_verified=True, both_GPUs_strict_Smoke_passed=True,
        no_tolerance_relaxation=True, image_cluster_shards_disjoint=True, NaN_Inf_OOM=False))
    return samples, records, baselines


def figures(summary, intervals, task_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 220})
    fig, ax = plt.subplots(figsize=(6.2, 3.7), layout='constrained')
    y = np.array([r['accuracy_pct'] for r in summary])
    ax.bar(np.arange(3), y, width=.6, color=['#386cb0', '#087e8b', '#9a9a9a'])
    ax.errorbar(np.arange(3), y, yerr=[y-intervals[0], intervals[1]-y], fmt='none', color='#303030', capsize=4)
    for i, value in enumerate(y):
        ax.text(i, intervals[1, i]+1.5, f'{value:.2f}%', ha='center')
    ax.set(xticks=np.arange(3), xticklabels=['All-Writer\nHistorical baseline', 'Keep L0-L21\nJoint Drop L22-L35', 'No-Writer\nHistorical reference only'],
        ylabel='GQA Accuracy (%)', ylim=(0, 100), title='Qwen3-VL: Joint Visual Writing Permission')
    ax.grid(axis='y', alpha=.18)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'figure1_accuracy_comparison.{ext}')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 3.7), layout='constrained')
    y = np.array([r['delta_accuracy_pp'] for r in task_rows])
    lo = np.array([r['ci_95_two_sided_low'] for r in task_rows])
    hi = np.array([r['ci_95_two_sided_high'] for r in task_rows])
    ax.bar(np.arange(6), y, width=.6, color='#087e8b')
    ax.errorbar(np.arange(6), y, yerr=[y-lo, hi-y], color='#303030', capsize=4, fmt='none')
    ax.axhline(0, color='.3', lw=.8)
    ax.axhline(-1, color='#c73e48', ls='--', lw=1, label='Overall prespecified margin (-1 pp)')
    ax.set(xticks=np.arange(6), xticklabels=[r['subset'] for r in task_rows],
        ylabel='Accuracy change vs All-Writer (pp)', title='Task-level Effects (Exploratory Paired 95% CI)')
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=.18)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'figure2_task_delta_accuracy.{ext}')
    plt.close(fig)


def main():
    start = time.time()
    p = verify()
    samples, records, baselines = audit()
    main_result, values, intervals = compare(samples, records, baselines, 'overall')
    selected = set(read(OUT/'image_disjoint_manifest.json')['keys'])
    independent, _, _ = compare([s for s in samples if key(s) in selected], records, baselines, 'image_disjoint')
    tasks = [compare([s for s in samples if s['task'] == task], records, baselines, task)[0]
        for task in sorted({s['task'] for s in samples})]
    summary = []
    for i, condition in enumerate(['All-Writer', CONDITION, 'No-Writer']):
        gs = [(records[key(s)] if i == 1 else baselines[key(s)][condition])['generation'] for s in samples]
        summary.append(dict(condition=condition, questions=3000, correct=int(values[:, i].sum()),
            accuracy_pct=float(values[:, i].mean()*100), ci_low=float(intervals[0, i]), ci_high=float(intervals[1, i]),
            source='NEW' if i == 1 else 'HISTORICAL', reference_only=i == 2,
            eos=sum(g['eos'] for g in gs), truncated=sum(g['truncated'] for g in gs),
            empty=sum(g['empty'] for g in gs), repeated=sum(g['repeated_trigram'] for g in gs)))
    paired, harms, rescues = [], [], []
    for s in samples:
        full, keep = baselines[key(s)]['All-Writer']['generation'], records[key(s)]['generation']
        row = dict(task=s['task'], question_id=s['question_id'], image_id=s['image_id'], image_sha256=s['image_sha256'],
            gold=s['answer'], All_Writer_text=full['text'], Keep_text=keep['text'],
            All_Writer_token_ids=json.dumps(full['token_ids']), Keep_token_ids=json.dumps(keep['token_ids']),
            All_Writer_correct=full['correct'], Keep_correct=keep['correct'],
            correct_to_wrong=int(full['correct'] and not keep['correct']), wrong_to_correct=int(not full['correct'] and keep['correct']),
            first_token_flip=keep['first_token_flip'], text_changed=int(full['text'] != keep['text']),
            All_Writer_eos=full['eos'], Keep_eos=keep['eos'], All_Writer_truncated=full['truncated'], Keep_truncated=keep['truncated'],
            Keep_empty=keep['empty'], Keep_repeated=keep['repeated_trigram'], input_binding=records[key(s)]['input_binding'])
        paired.append(row)
        if row['correct_to_wrong']:
            harms.append(row)
        if row['wrong_to_correct']:
            rescues.append(row)
    table('accuracy_summary.csv', summary)
    table('paired_comparison.csv', [main_result])
    table('task_accuracy.csv', tasks)
    table('image_disjoint_summary.csv', [independent])
    table('per_question.csv', paired)
    table('harm_cases.csv', harms, list(paired[0]))
    table('recovered_cases.csv', rescues, list(paired[0]))
    figures(summary, intervals, tasks)
    labels = {'SUPPORTED': '支持预设−1pp功能性非劣效', 'INSUFFICIENT': '证据不足，不能宣布−1pp非劣效', 'BELOW_MARGIN': '存在不满足−1pp标准的证据'}
    text = ['# Qwen3-VL联合关闭L22–L35视觉写入权限', '',
        '**COMPLETE：3,000条新条件完整生成，6,000条历史基线复用，全部逐题绑定和评分验证通过。**', '',
        '## 主要结果', '', '| 条件 | Accuracy | 正确题数 | 来源 |', '| --- | ---: | ---: | --- |']
    for r in summary:
        text.append(f"| {r['condition']} | {r['accuracy_pct']:.4f}% | {r['correct']}/3000 | {'本轮新增' if r['source']=='NEW' else '历史参考、不参与主非劣效检验' if r['reference_only'] else '历史配对基线'} |")
    r = main_result
    text += ['', f"Keep−All：**{r['delta_accuracy_pp']:+.4f} pp**；图片SHA整簇配对双侧95%CI：[{r['ci_95_two_sided_low']:.4f}, {r['ci_95_two_sided_high']:.4f}] pp。",
        f"单侧95%置信下限：**{r['ci_95_one_sided_lower']:.4f} pp**；主判定：**{labels[r['primary_NI_decision']]}**。判据为下限严格大于−1，不使用p值不显著或双侧CI包含0证明无损。",
        f"−0.5pp描述性敏感性标准：{'下限满足' if r['stricter_minus_0_5_pp_sensitivity'] else '下限未满足'}；不替换预设−1pp主要标准。",
        f"原对→错 **{r['correct_to_wrong']}** 题，原错→对 **{r['wrong_to_correct']}** 题；文本改变 {r['text_changes']} 题，首Token翻转 {r['first_token_flips']} 题。反向变化完整保留，不把净分接近解释为逐题等价。", '',
        '## 六任务与内部图片隔离', '', '| 任务 | All-Writer | Keep-L0-L21 | ΔAcc pp | 损害/纠正 |', '| --- | ---: | ---: | ---: | ---: |']
    for t in tasks:
        text.append(f"| {t['subset']} | {t['All_Writer_accuracy_pct']:.3f}% | {t['Keep_L0_L21_accuracy_pct']:.3f}% | {t['delta_accuracy_pp']:+.3f} | {t['correct_to_wrong']}/{t['wrong_to_correct']} |")
    sensitive = min(tasks, key=lambda x: x['delta_accuracy_pp'])
    text += ['', f"最大任务级点估计损害为 {sensitive['subset']}（{sensitive['delta_accuracy_pp']:+.3f} pp）。任务结果及其区间仅作探索性描述，不用最有利任务重定义成功。",
        f"历史Semantic240图像不重叠内部子集：{independent['questions']}题、{independent['image_clusters']}个图片簇；All {independent['All_Writer_accuracy_pct']:.4f}%，Keep {independent['Keep_L0_L21_accuracy_pct']:.4f}%；ΔAcc {independent['delta_accuracy_pp']:+.4f} pp，探索性双侧95%CI [{independent['ci_95_two_sided_low']:.4f}, {independent['ci_95_two_sided_high']:.4f}] pp。{'与总体点估计方向一致' if independent['delta_accuracy_pp']*r['delta_accuracy_pp']>=0 else '与总体点估计方向不同'}。该2640题集合在GPU运行前冻结，不是独立外部验证。", '',
        '## 研究判断与边界', '',
        ('本轮支持：在当前冻结GQA3000、Qwen模型及协议下，保留L0–L21、联合关闭后14层R→Image写入权限达到预设−1pp功能性非劣效，可作为层维度联合功能稀疏性的限定证据。'
        if r['primary_NI_decision']=='SUPPORTED' else '本轮未支持预设−1pp功能性非劣效；不能仅凭单层低敏感宣称后14层写入权限可以联合省略。'),
        '本轮仅改变R→Image的Softmax前−inf权限：R仍包含vision_end和模板特殊Token；P−1与所有真实Decode在36层均不直接读图。其他Prefill Query、Image Query、QKV、MLP、Residual、DeepStack、mRoPE与真实KV更新原生。没有删除视觉Token、删除Transformer层、o_proj清零、物理KV剪枝或测量加速。即使非劣效成立，也不是严格无损、物理删除安全或跨任务保证。', '',
        '## 审计、耗时与输出', '',
        f"历史来源Commit：`{BASE}`；对样本、原输入、原分片、模型版本、权重文件、逐题Checkpoint SHA、Token IDs和官方评分逐条核验，All-Writer为2637/3000（87.9%），未用原生Full替代。",
        'CPU回归、双GPU各3题真实Smoke通过：Native/No-op全部有效Logits严格torch.equal；历史All-Writer/Drop-L35全部有效Logits哈希精确复现；禁止概率为0；非目标Query、22层开放路径、独立Cache、36层MLP、DeepStack/mRoPE、输入/权重/视觉特征及正常/异常Hook清理通过。未放宽容差。',
        f"本轮Keep：EOS {r['eos']}/3000，截断 {r['truncated']}，空答案 {r['empty']}，重复三元组 {r['repeated']}；不存在NaN、Inf或OOM。",
        '两GPU正式运行耗时：'+'、'.join(f"{read(OUT/f'shard{i}/complete.json')['elapsed_seconds']/60:.2f}分钟" for i in (0,1))+'。GPU退出与释放见gpu_release_audit.json。',
        f"统计为20,000次Image SHA整簇配对Bootstrap、问题等权、种子{SEED}；仅一个预声明主要比较，不追加36项校正。运行源码Commit：`{p['code_commit']}`。",
        '原始新条件Checkpoint归档及SHA索引、全部损害/纠正案例、六任务与2640题统计、两张PNG/PDF均已保存。到此停止，不追加预算搜索或其他实验。']
    (OUT/'report_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    for shard in (0, 1):
        with tarfile.open(OUT/f'raw_checkpoints_shard{shard}.tar.gz', 'w:gz') as archive:
            for path in sorted((OUT/f'shard{shard}/checkpoints').glob('*.json')):
                archive.add(path, arcname=str(path.relative_to(OUT)), recursive=False)
        with tarfile.open(OUT/f'raw_checkpoints_shard{shard}.tar.gz') as archive:
            members = archive.getmembers()
            assert len(members) == 1500
            for member in members:
                assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == sha256(OUT/member.name)
    seal(OUT/'artifact_hashes.json', {str(f.relative_to(OUT)): sha256(f) for f in sorted(OUT.glob('*'))
        if f.is_file() and f.name not in ('artifact_hashes.json', 'completion.json', 'analysis.log', 'launcher.log')})
    seal(OUT/'completion.json', dict(status='COMPLETE', new_generations=3000, paired_questions=3000,
        reused_baseline_generations=6000, questions_in_image_disjoint_subset=2640, image_clusters=2409,
        analysis_seconds=time.time()-start, primary_NI_decision=r['primary_NI_decision'], code_commit=p['code_commit']))
    print('COMPLETE', json.dumps(main_result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

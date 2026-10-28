"""Single final, image-paired analysis of the frozen C2 confirmation."""
import math
import numpy as np
import torch
from knockout_probe import write_csv
from fixed_k_exit import atomic_json
from physical_exit_pilot import sequence_score
from prepare_fixed_k_data import digest
from prepare_physical_exit_data import sha256
from phase38b_c2_confirm import OUTPUT, PLAN, read, protocol, checkpoint


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Nonempty finite one-dimensional paired data required')
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std(ddof=1)) if len(a)>1 else 0., minimum=float(a.min()), maximum=float(a.max()),
                p5=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
                positive=int((a>0).sum()), negative=int((a<0).sum()), zero=int((a==0).sum()),
                positive_fraction=float((a>0).mean()), negative_fraction=float((a<0).mean()))


def paired(a, b):
    """Rows/columns: baseline false,true / treatment false,true; exact two-sided."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError('Paired vectors required')
    harm, rescue = int((a & ~b).sum()), int((~a & b).sum())
    n = harm+rescue
    p = min(1., 2.*sum(math.comb(n, k) for k in range(min(harm, rescue)+1))/2**n) if n else 1.
    return dict(table=[[int((~a & ~b).sum()), rescue], [harm, int((a & b).sum())]],
                table_axes='baseline incorrect/correct rows; treatment incorrect/correct columns',
                correct_to_wrong=harm, wrong_to_correct=rescue, discordant=n,
                net_corrections=rescue-harm, accuracy_difference=float(b.mean()-a.mean()),
                exact_mcnemar_two_sided_p=p)


def bias_row(full):
    d = full['right_logp']-full['left_logp']
    margin = (d+PLAN['offline_right_bias'])*(1 if full['gold']=='right' else -1)
    return dict(question_id=full['question_id'], img_id=full['img_id'], gold=full['gold'],
                cohort=full['cohort'], condition='B1_offline_right_plus_1', d_full=d, d_bias=d+1.,
                margin=margin, full_margin=full['margin'], gain=margin-full['margin'], correct=margin>0,
                correct_to_wrong=full['correct'] and margin<=0, wrong_to_correct=not full['correct'] and margin>0,
                absolute_logp_status='undefined: offline candidate score bias, not model full-vocabulary LogP')


def statistics(items):
    n = len(items)
    rng = np.random.default_rng(PLAN['bootstrap_seed'])
    indices = rng.integers(0, n, size=(PLAN['bootstrap_draws'], n))
    def ci(x):
        values = np.asarray(x, dtype=float)
        return [float(v) for v in np.percentile(values[indices].mean(axis=1), [2.5, 97.5])]
    full = np.array([i['C0']['correct'] for i in items])
    c2 = np.array([i['C2']['correct'] for i in items])
    bias = np.array([i['B1']['correct'] for i in items])
    gains = [i['C2']['gain'] for i in items]
    direction = [i['delta_d'] for i in items]
    out = dict(n=n, accuracy={name:dict(correct=int(x.sum()), total=n, fraction=float(x.mean()), percent=float(x.mean()*100))
              for name, x in [('C0', full), ('C2', c2), ('B1_offline', bias)]},
        c2_vs_full=paired(full, c2), c2_vs_bias=paired(bias, c2), bias_vs_full=paired(full, bias),
        gain=describe(gains), delta_d=describe(direction), delta_d_minus_fixed_1=describe(np.array(direction)-1.),
        ci95=dict(mean_gain=ci(gains), c2_minus_full_accuracy=ci(c2.astype(float)-full),
                  c2_minus_bias_accuracy=ci(c2.astype(float)-bias), mean_delta_d=ci(direction)),
        absolute_logp={}, bootstrap=dict(seed=PLAN['bootstrap_seed'], draws=PLAN['bootstrap_draws'],
            unit='one unique image paired across all conditions', method='percentile 2.5/97.5 of paired mean; no layer/head/token replication',
            inferential_scope='confirmation only; discovery and pooled intervals descriptive, unadjusted'))
    for label in ['gold', 'wrong', 'left', 'right']:
        values = [i['C2'][label+'_logp']-i['C0'][label+'_logp'] for i in items]
        out['absolute_logp'][label] = dict(**describe(values), mean_ci95=ci(values))
    both = [i['C2']['left_logp']>i['C0']['left_logp'] and i['C2']['right_logp']>i['C0']['right_logp'] for i in items]
    out['both_candidates_absolute_logp_increase'] = dict(count=sum(both), total=n, fraction=float(np.mean(both)))
    out['rescue_overlap_vs_full'] = dict(
        both_c2_and_bias=int((~full & c2 & bias).sum()),
        c2_only=int((~full & c2 & ~bias).sum()), bias_only=int((~full & ~c2 & bias).sum()),
        neither=int((~full & ~c2 & ~bias).sum()),
        caution='Descriptive overlap, not a causal mediation or fraction of mechanism explained')
    out['ties'] = {name:sum(i[name]['margin']==0 for i in items) for name in ['C0', 'C2', 'B1']}
    out['worst_gain_samples'] = [dict(question_id=i['C2']['question_id'], img_id=i['C2']['img_id'],
        question=i['C2']['question'], gold=i['C2']['gold'], full_margin=i['C0']['margin'],
        c2_margin=i['C2']['margin'], gain=i['C2']['gain'], full_correct=i['C0']['correct'],
        c2_correct=i['C2']['correct']) for i in sorted(items, key=lambda x:(x['C2']['gain'], x['C2']['question_id']))[:10]]
    return out


def audit(c, m):
    gate = read(OUTPUT/'run/validation.json')
    if not gate['passed'] or gate['binding'] != m['binding'] or gate['count'] != 240 or not gate['full_state_dict_exact']:
        raise ValueError('Formal run incomplete; never aggregate partial data')
    samples = m['samples']; expected = {s['question_id'] for s in samples}
    if set(gate['question_ids']) != expected or set(gate['samples']) != expected:
        raise ValueError('Missing or duplicate sample coverage')
    if set(p.name for p in (OUTPUT/'run').glob('sample_*')) != {'sample_'+q for q in expected}:
        raise ValueError('Unexpected checkpoint directory')
    items, raw_checks, edits = [], 0, 0
    for s in samples:
        q = s['question_id']; folder = OUTPUT/'run'/('sample_'+q)
        data = checkpoint(folder, s, m['binding'], gate['execution'])
        if sha256(folder/'sample.json') != gate['samples'][q] or not data['native_restoration_exact']:
            raise ValueError('Final checkpoint validation failed')
        rows = {r['condition']:r for r in data['scores']}
        if set(rows) != {'C0', 'C2'} or len(data['scores']) != 2:
            raise ValueError('GPU condition coverage changed')
        with np.load(folder/'logits.npz') as raw:
            if set(raw.files) != {'C0_prediction_logits', 'C2_prediction_logits'}:
                raise ValueError('Foreign raw condition')
            for name in ['C0', 'C2']:
                array = raw[name+'_prediction_logits']
                if array.shape != (2, 32064) or not np.isfinite(array).all() or not np.array_equal(array[0], array[1]):
                    raise ValueError('Bad logits shape/finiteness/future leakage')
                logs = [sequence_score(torch.from_numpy(array[j])[None], [r['token_id'] for r in rec['answer_records']])
                        for j, rec in enumerate(data['mapping']['candidate_mapping'])]
                r = rows[name]; gold = s['candidates'].index(s['answer'])
                if any(r[k]!=v for k, v in [('left_logp', logs[s['candidates'].index('left')]),
                    ('right_logp', logs[s['candidates'].index('right')]), ('gold_logp', logs[gold]),
                    ('wrong_logp', logs[1-gold]), ('margin', logs[gold]-logs[1-gold]), ('correct', logs[gold]>logs[1-gold])]):
                    raise ValueError('Independent original full-vocabulary rescoring differs')
                raw_checks += 2
        for r in rows.values():
            f = rows['C0']
            if r['question_id']!=q or r['img_id']!=s['img_id'] or r['gold']!=s['answer'] or r['cohort']!=s['cohort']:
                raise ValueError('Wrong sample/label/cohort')
            for key, value in dict(gain=r['margin']-f['margin'], gold_logp_gain=r['gold_logp']-f['gold_logp'],
                                  wrong_logp_gain=r['wrong_logp']-f['wrong_logp'],
                                  correct_to_wrong=f['correct'] and not r['correct'], wrong_to_correct=not f['correct'] and r['correct']).items():
                if r[key] != value:
                    raise ValueError('Derived score/flip algebra failed')
        if len(data['audit']) != 4:
            raise ValueError('Missing forward audits')
        for j in range(2):
            f, t = data['audit'][j], data['audit'][j+2]
            if f['condition']!='C0' or t['condition']!='C2' or f['edits'] or [e['layer'] for e in t['edits']] != [2,3,4]:
                raise ValueError('Intervention coverage failed')
            for key in ['prefix_all_tokens_layer_sha256', 'question_sha256', 'input_sha256']:
                if f[key] != t[key]:
                    raise ValueError('Input/causal invariant failed')
            if f['prediction_hidden_layer_sha256']['1'] != t['prediction_hidden_layer_sha256']['1']:
                raise ValueError('L1 changed')
            for edit in t['edits']:
                if edit['query']!=data['mapping']['answer_query'] or not all(edit[k] for k in ['target_zero', 'other_tokens_bitwise_exact', 'original_tensor_unchanged']):
                    raise ValueError('Local MLP edit escaped target')
                edits += 1
        rows['B1'] = bias_row(rows['C0'])
        rows['delta_d'] = (rows['C2']['right_logp']-rows['C2']['left_logp'])-(rows['C0']['right_logp']-rows['C0']['left_logp'])
        rows['C2']['delta_d'] = rows['delta_d']
        rows['C0']['delta_d'] = 0.
        items.append(rows)
    return items, dict(passed=True, independently_rescored_candidates=raw_checks, local_mlp_edits=edits,
        count=240, cpu=read(OUTPUT/'tests/validation.json'), smoke=read(OUTPUT/'smoke/validation.json'),
        full_state_dict_exact=True, raw_logits_exact_rescoring=True,
        unique_question_ids=len(expected), unique_image_ids=len({s['img_id'] for s in samples}),
        discovery_regressions_bitwise_exact=16, causal_prefix_includes_visual_tokens=True,
        no_attention_mask_bos_sink_roi_changes=True, partial_failures=[str(p.relative_to(OUTPUT)) for p in OUTPUT.rglob('failure_*.json')])


def plots(items):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [i for i in items if i['C2']['cohort']=='Confirmation224']
    folder=OUTPUT/'plots'; folder.mkdir()
    colors={'left':'#19776d', 'right':'#b64460'}
    fig, ax = plt.subplots(figsize=(8,5))
    for answer in ['left','right']:
        values=[i for i in selected if i['C2']['gold']==answer]
        ax.scatter([i['C0']['right_logp']-i['C0']['left_logp'] for i in values], [i['delta_d'] for i in values], s=20, alpha=.7, label=answer, color=colors[answer])
    ax.axhline(1., color='black', linestyle='--', label='fixed offline +1'); ax.axhline(0,color='grey',lw=.7)
    ax.set(xlabel='Native logP(right) - logP(left)', ylabel='C2 change in right - left LogP', title='Independent confirmation: all 224 images')
    ax.legend(); fig.tight_layout(); fig.savefig(folder/'direction_shift.png',dpi=160); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,4))
    for answer in ['left','right']:
        group=[i for i in selected if i['C2']['gold']==answer]
        axes[0].hist([i['C2']['gain'] for i in group], bins=24, alpha=.5, label=answer,color=colors[answer])
        axes[1].scatter([i['C2']['gold_logp_gain'] for i in group], [i['C2']['wrong_logp_gain'] for i in group],s=20,alpha=.7,label=answer,color=colors[answer])
    axes[0].set(xlabel='Margin gain (all negative extremes retained)',ylabel='Images')
    axes[1].set(xlabel='Gold absolute LogP gain',ylabel='Wrong absolute LogP gain')
    for ax in axes: ax.axvline(0,color='grey',lw=.6); ax.legend()
    axes[1].axhline(0,color='grey',lw=.6)
    fig.tight_layout();fig.savefig(folder/'margin_and_absolute_logp.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(3)
    for j, group in enumerate(['all','left','right']):
        sub=[i for i in selected if group=='all' or i['C2']['gold']==group]
        ax.bar(x+(j-1)*.25,[np.mean([i[name]['correct'] for i in sub])*100 for name in ['C0','C2','B1']],.25,label=group)
    ax.set(xticks=x,xticklabels=['Native Full','C2 MLP bypass','Offline right +1'],ylabel='Accuracy (%)',ylim=(0,100));ax.legend()
    fig.tight_layout();fig.savefig(folder/'accuracy.png',dpi=160);plt.close(fig)


def report(summaries, config):
    c=summaries['224']; a=c['accuracy']; left,right=c['groups']['left'],c['groups']['right']
    accuracy_ci=c['ci95']['c2_minus_full_accuracy']
    verdict=('观察到准确率提高，配对95%区间不含0' if accuracy_ci[0]>0 else
             '观察到准确率下降，配对95%区间不含0' if accuracy_ci[1]<0 else
             '配对95%区间包含0，不能确认准确率有稳定改善')
    bias_ci=c['ci95']['c2_minus_bias_accuracy']
    bias_verdict=('配对95%区间支持C2准确率高于该固定偏置' if bias_ci[0]>0 else
                  '配对95%区间支持C2准确率低于该固定偏置' if bias_ci[1]<0 else
                  '配对95%区间包含0，不能确认C2准确率优于该固定偏置')
    lines=['# Phase 3.8b：C2 MLP Bypass 独立确认实验', '',
      '**状态：完成，240/240；主要确认集224图，发现集16图单独报告。**', '',
      f"冻结代码：`{config['code_commit']}`；基线 `{config['base_commit']}`。物理 GPU1 RTX3090；FP16/eager；Transformers4.49.0；原模型 revision、图片、Prompt、候选全词表 LogP 协议不变。", '',
      '## 集合与准确率', '', '| 集合 | Native Full | C2 | 离线 right +1 | Full对→C2错 / 错→对 |', '|---|---:|---:|---:|---:|']
    for key, title in [('224','Confirmation224（主要）'),('16','Discovery16（复现）'),('240','Full240（描述性合并，含发现集）')]:
        s=summaries[key]
        cells=[f"{s['accuracy'][name]['correct']}/{s['n']} ({s['accuracy'][name]['percent']:.2f}%)" for name in ['C0','C2','B1_offline']]
        lines.append(f"| {title} | {' | '.join(cells)} | {s['c2_vs_full']['correct_to_wrong']} / {s['c2_vs_full']['wrong_to_correct']} |")
    lines += ['', '## 224图确认统计', '',
      f"C2 − Full 配对准确率差 {100*c['c2_vs_full']['accuracy_difference']:+.2f} 个百分点；95% 图级配对 Bootstrap CI（比例单位）{c['ci95']['c2_minus_full_accuracy']}；Exact McNemar 双侧 p={c['c2_vs_full']['exact_mcnemar_two_sided_p']:.6g}。",
      f"C2 − 固定偏置准确率差 {100*c['c2_vs_bias']['accuracy_difference']:+.2f} 个百分点；CI {c['ci95']['c2_minus_bias_accuracy']}；Exact McNemar p={c['c2_vs_bias']['exact_mcnemar_two_sided_p']:.6g}。",
      f"**确认结论：{verdict}；{bias_verdict}。**",
      f"平均 Margin Gain {c['gain']['mean']:+.6f}；95% CI {c['ci95']['mean_gain']}；中位数 {c['gain']['median']:+.6f}；标准差 {c['gain']['std']:.6f}；P5/P95 {c['gain']['p5']:+.6f}/{c['gain']['p95']:+.6f}；范围 [{c['gain']['minimum']:+.6f},{c['gain']['maximum']:+.6f}]。",
      f"Gain 正/负/零：{c['gain']['positive']}/{c['gain']['negative']}/{c['gain']['zero']}。平均 DeltaD={c['delta_d']['mean']:+.6f}；中位数 {c['delta_d']['median']:+.6f}；正/负/零 {c['delta_d']['positive']}/{c['delta_d']['negative']}/{c['delta_d']['zero']}。", '',
      '| 分组 | Full | C2 | 离线偏置 | 平均 Gain [95% CI] | 平均 DeltaD |', '|---|---:|---:|---:|---|---:|']
    for label,s in [('left',left),('right',right)]:
        lines.append(f"| {label} | {s['accuracy']['C0']['correct']}/112 | {s['accuracy']['C2']['correct']}/112 | {s['accuracy']['B1_offline']['correct']}/112 | {s['gain']['mean']:+.6f} {s['ci95']['mean_gain']} | {s['delta_d']['mean']:+.6f} |")
    both=c['both_candidates_absolute_logp_increase']
    lines += ['', '## 研究问题逐条回答', '',
      f"1. **独立准确率是否提高？** 本次观察 Full {a['C0']['correct']}/224 → C2 {a['C2']['correct']}/224；{verdict}。不把点估计改善等同普遍能力提高。",
      f"2. **纠错与损害？** 原错→对 {c['c2_vs_full']['wrong_to_correct']}，原对→错 {c['c2_vs_full']['correct_to_wrong']}，净纠错 {c['c2_vs_full']['net_corrections']}；全部损害和极端点保留。",
      f"3. **左右不对称？** left 平均 Gain {left['gain']['mean']:+.6f}，right {right['gain']['mean']:+.6f}；组内准确率、翻转、CI 见 JSON，不能用合并结果掩盖方向性。",
      f"4. **right 相对优势？** {c['delta_d']['positive']}/224 张 DeltaD>0，平均 {c['delta_d']['mean']:+.6f}。这是候选方向变化，不直接等同视觉证据改善。",
      f"5. **两个绝对 LogP 同时提高？** {both['count']}/224 ({100*both['fraction']:.2f}%)；Gold 平均增量 {c['absolute_logp']['gold']['mean']:+.6f}，Wrong {c['absolute_logp']['wrong']['mean']:+.6f}。两候选可同时增大，因此排序改善与绝对概率改善须分开。",
      f"6. **优于固定方向偏置？** C2 {a['C2']['correct']}/224，离线 +1 {a['B1_offline']['correct']}/224；{bias_verdict}。相对Full共同纠错 {c['rescue_overlap_vs_full']['both_c2_and_bias']}，仅C2纠错 {c['rescue_overlap_vs_full']['c2_only']}，仅偏置纠错 {c['rescue_overlap_vs_full']['bias_only']}。这是描述性重叠，不是机制解释比例。+1预先固定，不在确认集上拟合；离线 B1 没有真实全词表绝对 LogP 增量。",
      f"7. **平均 Gain 与退化尾部？** 平均 {c['gain']['mean']:+.6f}，最小 {c['gain']['minimum']:+.6f}；不得遗漏负尾或因发现集全对而假定无损。",
      '8. **机制判断？** DeltaD和固定偏置对照检验简单方向校准的解释力；C2是因果干预，B1是离线数学反事实。即使C2优于B1，也不能排除更复杂的非视觉校准；若相近也不证明机制等价。未执行视觉反事实，不能宣称真实视觉关系理解提升。',
      '9. **对L2–L4功能的影响？** 本轮仅确认P−1 MLP旁路的决策效应，限制对MLP必需性或有益性的普遍解释；没有全量C1/C3/C4，不能比较Attention与MLP的重要性，也不能推出被动层结论。',
      '10. **支持提前删除视觉Token吗？** 不支持。仅修改单个预测Query的三层MLP输出，保留全部Token、Attention和32层计算；没有测试删除、裁剪、物理退出或Decode。', '',
      '## 最严重负向样本', '', '| Question ID | Gold | Full Margin | C2 Margin | Gain | 原问题 |', '|---|---|---:|---:|---:|---|']
    for row in c['worst_gain_samples']:
        lines.append(f"| {row['question_id']} | {row['gold']} | {row['full_margin']:+.6f} | {row['c2_margin']:+.6f} | {row['gain']:+.6f} | {row['question']} |")
    lines += ['', '## 审计与范围', '',
      'CPU全套测试和预定一左一右GPU Smoke先于正式运行；全部16发现图的C0/C2原始logits与原Phase3.8逐元素精确一致。240图每图两候选、同GPU同权重配对，全部960候选从保存的原始全词表logits独立重算。1440条MLP局部编辑逐条审计。每层全部因果前缀（含视觉Token）、Question隐藏、L1、输入、恢复Full、Hook清理、全权重前后哈希均校验。未要求未来Token隐藏不变。',
      'Bootstrap固定种子、10000次，以唯一图片为配对单位。主要推断为224图；分组和多个终点的CI/p为未做多重比较校正的补充。发现集和合并集只作描述。冻结上游源文件名称含correct_question，数据本身是预选GQA子集，不代表无偏总体。',
      '同机Native与历史RTX4090数值差异另存historical_full_differences.csv，不混作配对基线，不放宽容差。未新增条件或根据中间分数调整参数。完整数值、配对列联表、绝对LogP、方向变化、退化问题文本、审计和图表均在本目录。', '']
    (OUTPUT/'report.md').write_text('\n'.join(lines))


def main():
    if (OUTPUT/'validation.json').exists():
        raise FileExistsError('Final summary already exists; never silently overwrite')
    config, manifest = protocol()
    items, validation = audit(config, manifest)
    write_csv(OUTPUT/'per_sample_scores.csv', [i[name] for i in items for name in ['C0','C2']])
    write_csv(OUTPUT/'offline_bias_baseline.csv', [i['B1'] for i in items])
    write_csv(OUTPUT/'direction_and_absolute_changes.csv', [dict(question_id=i['C2']['question_id'], gold=i['C2']['gold'],
        cohort=i['C2']['cohort'], delta_d=i['delta_d'], delta_d_minus_fixed_1=i['delta_d']-1.,
        **{'delta_'+k+'_logp':i['C2'][k+'_logp']-i['C0'][k+'_logp'] for k in ['gold','wrong','left','right']}) for i in items])
    differences=[]
    for s in manifest['samples']:
        data=read(OUTPUT/'run'/f"sample_{s['question_id']}"/'sample.json')
        differences.append(dict(question_id=s['question_id'], candidate0_logp_delta=data['historical_logp_deltas'][0],
                                candidate1_logp_delta=data['historical_logp_deltas'][1]))
    write_csv(OUTPUT/'historical_full_differences.csv', differences)
    summaries={}
    for key, cohort in [('224','Confirmation224'), ('16','Discovery16'), ('240',None)]:
        selected=[i for i in items if cohort is None or i['C2']['cohort']==cohort]
        summary=statistics(selected)
        summary['cohort']=cohort or 'Full240_descriptive_includes_discovery'
        summary['groups']={answer:statistics([i for i in selected if i['C2']['gold']==answer]) for answer in ['left','right']}
        summaries[key]=summary
        atomic_json(OUTPUT/f'summary_{key}.json', summary)
    plots(items); report(summaries,config)
    validation.update(binding=manifest['binding'], code_commit=config['code_commit'], score_rows=480,
                      offline_rows=240, scientific_aggregations=1, historical_full_difference_max_abs=max(abs(r[k]) for r in differences for k in ['candidate0_logp_delta','candidate1_logp_delta']))
    atomic_json(OUTPUT/'validation.json', validation)
    # Live stdout is deliberately outside this directory and is never self-hashed.
    atomic_json(OUTPUT/'publication_validation.json', dict(passed=True, files={str(p.relative_to(OUTPUT)):sha256(p)
        for p in sorted(OUTPUT.rglob('*')) if p.is_file() and p.name not in ['publication_validation.json','.lock']}))
    print('COMPLETE SUMMARY 224 confirmation / 16 discovery / 240 total', flush=True)


if __name__=='__main__':
    main()

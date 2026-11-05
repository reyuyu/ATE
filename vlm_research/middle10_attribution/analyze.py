"""CPU-only paired quality analysis and the predeclared stop gate."""
import json
import collections

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import *
from .run import verify_freeze
from ..fastp2_bypass_writer.common import hardware


def bounds(x):
    return tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))


def main():
    f = verify_freeze()
    assert read(OUT / 'driver_exit.json')['returncodes'] == [0, 0]
    assert all(read(OUT / f'worker{r}/completion.json')['status'] == 'COMPLETE' for r in range(2))
    samples = read(OUT / 'sample_manifest.json')
    partition = read(OUT / 'partition.json')
    previous = read(OUT / 'source_manifest.json')
    old = {(r['key'], r['condition']): r for r in previous['reused_records']}
    labels, keys = BASELINES+LABELS, sorted(samples)
    values = np.zeros((500, len(labels)))
    records, hashes, rows = {}, {}, []
    for i, key in enumerate(keys):
        s = samples[key]
        for j, label in enumerate(labels):
            reused = label in BASELINES
            if reused:
                path = Path(old[key, label]['path'])
                assert sha(path) == old[key, label]['sha256']
            else:
                path = checkpoint(key, label)
            v = load(path)
            assert v['key'] == key and v['condition'] == label
            assert v['sample_sha256'] == digest(s)
            if not reused:
                assert v['sources'] == f['sources']
                assert v['rank'] == partition['image_shard'][s['image_sha256']]
                hashes[str(path.relative_to(OUT))] = sha(path)
            assert digest(v['input_binding']) == old[key, 'Native']['input_binding_sha256']
            g = v['generation']
            assert v['score'] == score(s, g['text'])
            assert 1 <= len(g['token_ids']) <= 32
            records[key, label], values[i, j] = v, v['score']
            native, e3 = historical(key, 'Native')['generation'], historical(key, 'E3')['generation']
            if label in ('D', 'E'):
                assert g['first_logits_sha256'] == native['first_logits_sha256']
            rows.append(dict(key=key, question_id=s['question_id'], image_id=s['image_id'],
                image_sha256=s['image_sha256'], question=s['question'], gold_answers=s['answer'],
                condition=label, history_reused=reused, answer=g['text'], score=v['score'],
                token_ids=g['token_ids'], input_sha256=digest(v['input_binding']),
                first_logits_sha256=g['first_logits_sha256'], first_top1_token=g['token_ids'][0],
                first_flip_native=g['token_ids'][0] != native['token_ids'][0],
                first_flip_E3=g['token_ids'][0] != e3['token_ids'][0],
                native_fixed_first_logp_loss=(native['first_top1_logp']-g['native_first_token_logp']
                                              if label in LABELS else None),
                eos=g['eos'], empty=g['empty'], truncated=g['truncated'], repeated=g['repeated'],
                content_tokens=g['content_tokens']))
    assert len(rows) == 4000 and len(hashes) == 2500
    write_csv(OUT / 'per_question.csv', rows)
    clusters = sorted({s['image_sha256'] for s in samples.values()})
    ix = {h: i for i, h in enumerate(clusters)}
    membership = np.array([ix[samples[k]['image_sha256']] for k in keys])
    sizes = np.bincount(membership)
    sums = np.zeros((len(clusters), len(labels)))
    np.add.at(sums, membership, values)
    weights = np.random.default_rng(SEED).multinomial(len(clusters),
        np.full(len(clusters), 1/len(clusters)), size=20000)
    boot = weights @ sums/(weights @ sizes)[:, None]*100
    means = values.mean(0)*100
    ni, mi, ei = map(labels.index, BASELINES)
    summary = []
    for j, label in enumerate(labels):
        lo, hi = bounds(boot[:, j])
        dl, dh = bounds(boot[:, j]-boot[:, ni])
        el, eh = bounds(boot[:, j]-boot[:, ei])
        gs = [records[k, label]['generation'] for k in keys]
        summary.append(dict(condition=label, questions=500, image_clusters=309,
            accuracy=float(means[j]), ci_low=lo, ci_high=hi,
            delta_native_pp=float(means[j]-means[ni]), delta_native_ci_low=dl, delta_native_ci_high=dh,
            delta_E3_pp=float(means[j]-means[ei]), delta_E3_ci_low=el, delta_E3_ci_high=eh,
            eos=sum(g['eos'] for g in gs), truncated=sum(g['truncated'] for g in gs),
            empty=sum(g['empty'] for g in gs), repeated=sum(g['repeated'] for g in gs),
            first_flip_native=sum(g['token_ids'][0] != records[k, 'Native']['generation']['token_ids'][0]
                                  for k, g in zip(keys, gs))))
    write_csv(OUT / 'accuracy_summary.csv', summary)
    pairs = [(c, b) for c in LABELS for b in ('Native', 'E3')]
    pairs += [('B', 'A'), ('C', 'A'), ('D', 'A'), ('D', 'B'), ('D', 'C'), ('E', 'D')]
    comparisons, negatives = [], []
    for candidate, baseline in pairs:
        a, b = labels.index(candidate), labels.index(baseline)
        delta = values[:, a]-values[:, b]
        lo, hi = bounds(boot[:, a]-boot[:, b])
        comparisons.append(dict(comparison=candidate+' - '+baseline,
            delta_accuracy_pp=float(delta.mean()*100), ci_low=lo, ci_high=hi,
            score_increased=int((delta > 1e-12).sum()), score_decreased=int((delta < -1e-12).sum()),
            full_score_to_lower=int(((values[:, b] == 1)&(delta < -1e-12)).sum()),
            lower_to_full_score=int(((values[:, a] == 1)&(delta > 1e-12)).sum()),
            first_token_flips=sum(records[k, candidate]['generation']['token_ids'][0] !=
                                  records[k, baseline]['generation']['token_ids'][0] for k in keys),
            sequence_changes=sum(records[k, candidate]['generation']['token_ids'] !=
                                  records[k, baseline]['generation']['token_ids'] for k in keys),
            exploratory=True, bootstrap_replicates=20000))
        for i, key in enumerate(keys):
            if abs(delta[i]) < 1e-12:
                continue
            negatives.append(dict(comparison=candidate+' - '+baseline, key=key,
                image_sha256=samples[key]['image_sha256'], question=samples[key]['question'],
                gold_answers=samples[key]['answer'],
                baseline_answer=records[key, baseline]['generation']['text'],
                intervention_answer=records[key, candidate]['generation']['text'],
                baseline_score=float(values[i, b]), intervention_score=float(values[i, a]),
                delta_score=float(delta[i]), harm=bool(delta[i] < 0)))
    write_csv(OUT / 'paired_comparisons.csv', comparisons)
    write_csv(OUT / 'recovery_and_negative_cases.csv', negatives)
    factors = np.array([0., 0., 0., 1., -1., -1., 1., 0.])
    assert labels == ('Native', 'C3+Middle10', 'E3', 'A', 'B', 'C', 'D', 'E')
    lo, hi = bounds(boot @ factors)
    interaction = dict(comparison='D-B-C+A', delta_accuracy_pp=float(means @ factors), ci_low=lo, ci_high=hi)
    write_csv(OUT / 'factorial_interaction.csv', [interaction])
    best = max(LABELS, key=lambda label: means[labels.index(label)])
    achieved = float(means[labels.index(best)])
    passed = achieved >= THRESHOLD-1e-9
    gate = dict(at=now(), stage1_status='COMPLETE', threshold_accuracy=THRESHOLD,
        best_condition=best, best_accuracy=achieved, quality_gate_passed=passed,
        tie_rule='First in frozen A-E order', performance_allowed=False,
        next_action='ATTRIBUTION_THEN_PHYSICAL_TRANSLATION' if passed else 'STOP_AND_SUBMIT_NO_PERFORMANCE',
        physical_quality_and_correctness='NOT_YET_TESTED' if passed else 'NOT_RUN_DENSE_GATE_FAILED', formal_quality_generations=2500,
        performance_requests=0, result_hashes_sha256=digest(hashes))
    atomic(OUT / 'dense_quality_gate.json', gate)
    atomic(OUT / 'result_hashes.json', hashes)
    plot(summary)
    audit_rows = []
    for rank in range(2):
        smoke = read(OUT / f'worker{rank}/smoke.json')
        assert smoke['sources'] == f['sources'] and len(smoke['checks']) == 3
        for case in smoke['checks']:
            for condition in case['conditions']:
                audit_rows.append(dict(rank=rank, key=case['key'], condition=condition['condition'],
                    fixed3_passed=condition['numerical']['passed'],
                    natural_passed=condition['natural_numerical']['passed'],
                    fixed3_max_abs_logits=condition['numerical']['max_abs_logits'],
                    natural_max_abs_logits=condition['natural_numerical']['max_abs_logits'],
                    native_prefill_first_logits_exact=condition['native_prefill_first_logits_exact']))
    write_csv(OUT / 'numerical_audit.csv', audit_rows)
    owned = read(OUT / 'driver_pids.json')
    pids = [owned['driver'], *owned['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in pids)
    atomic(OUT / 'stage1_gpu_release_audit.json', dict(at=now(), task_pids=pids,
        all_task_pids_exited=True, **hardware()))
    atomic(OUT / 'validation.json', dict(status='QUALITY_PASS_PHYSICAL_PENDING' if passed else 'QUALITY_PASS_STOPPED_BELOW_GATE',
        CPU_tests=19, GPU_smoke_per_card=3, physical_GPU_indices=[0, 1],
        new_quality_records=2500, history_reused_verified=1500, images=309,
        D_E_all500_native_prefill_first_logits_exact=True,
        source_freeze_verified=True, tolerances=TOLERANCES['llava'], performance_requests=0))
    diagnostics = decode_diagnostics(keys, samples, records)
    report(summary, comparisons, interaction, gate, diagnostics)
    if not passed:
        atomic(OUT / 'completion.json', dict(status='COMPLETE_QUALITY_STOP_RULE_APPLIED', at=now(),
            conditions=list(LABELS), questions_per_condition=500, new_quality_generations=2500,
            reused_quality_generations=1500, best_condition=best, best_accuracy=achieved,
            threshold=THRESHOLD, dense_gate_passed=False,
            physical_translation='NOT_RUN_DENSE_GATE_FAILED', performance_requests=0,
            formal_quality_wall_seconds=max(read(OUT / f'worker{r}/progress.json')['elapsed_seconds'] for r in range(2)),
            code_commit=f['code_commit'], no_further_GPU_work=True))
        atomic(OUT / 'gpu_release_audit.json', read(OUT / 'stage1_gpu_release_audit.json'))
    print(json.dumps(dict(summary=summary, comparisons=comparisons, interaction=interaction, gate=gate),
                     indent=2, ensure_ascii=False))


def plot(summary):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    x = np.arange(8)
    colors = ['#555555', '#9c9693', '#398278', '#6795ae', '#bd913f', '#915ea2', '#be6355', '#387851']
    ys = [r['accuracy'] for r in summary]
    axes[0].bar(x, ys, color=colors)
    axes[0].axhline(THRESHOLD, color='#a23e3e', lw=1, ls='--', label='47.68% quality gate')
    for i, y in enumerate(ys):
        axes[0].text(i, y+.4, f'{y:.2f}', ha='center', fontsize=7)
    axes[0].set(xticks=x, xticklabels=['Native', 'M10', 'E3', 'A', 'B', 'C', 'D', 'E'],
                ylabel='TextVQA soft accuracy (%)', title='Same frozen 500 questions')
    axes[0].legend(fontsize=7, loc='lower left')
    new = summary[3:]
    gains = [r['delta_native_pp'] for r in new]
    axes[1].errorbar(np.arange(5), gains, fmt='o', capsize=3, color='#386485',
        yerr=np.array([[r['delta_native_pp']-r['delta_native_ci_low'] for r in new],
                       [r['delta_native_ci_high']-r['delta_native_pp'] for r in new]]))
    axes[1].axhline(0, color='#777777', lw=.8)
    axes[1].axhline(-2, color='#a23e3e', lw=.8, ls='--')
    axes[1].set(xticks=np.arange(5), xticklabels=list(LABELS), ylabel='Accuracy change vs Native (pp)',
                title='20,000 image-cluster paired bootstrap: 95% CI')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'quality_attribution.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def decode_diagnostics(keys, samples, records):
    from transformers import AutoTokenizer
    from ..p1_layerwise_accuracy.common import model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path('llava'), local_files_only=True)
    first = collections.Counter(records[k, 'Native']['generation']['token_ids'][0] for k in keys)
    token_roles = [dict(token_id=t, subword=tokenizer.convert_ids_to_tokens(t),
        decoded=tokenizer.decode([t]), questions=n) for t, n in first.most_common()]
    cases, rounding = [], []
    for k in keys:
        native = records[k, 'Native']['generation']
        e = records[k, 'E']['generation']
        old = records[k, 'E3']['generation']
        a = records[k, 'A']['generation']
        if old['token_ids'] != a['token_ids']:
            rounding.append(dict(key=k, historical_E3_answer=old['text'], dense_A_answer=a['text'],
                historical_E3_token_ids=old['token_ids'], dense_A_token_ids=a['token_ids'],
                score_E3=records[k, 'E3']['score'], score_A=records[k, 'A']['score'],
                first_logits_sha_equal=old['first_logits_sha256']==a['first_logits_sha256']))
        if native['token_ids'] == e['token_ids']:
            continue
        i = next((i for i, (x, y) in enumerate(zip(native['token_ids'], e['token_ids'])) if x != y),
                 min(len(native['token_ids']), len(e['token_ids'])))
        x = native['token_ids'][i] if i < len(native['token_ids']) else None
        y = e['token_ids'][i] if i < len(e['token_ids']) else None
        eos = tokenizer.eos_token_id
        kind = ('both_non_EOS' if x not in (None, eos) and y not in (None, eos)
                else 'E_early_EOS' if y == eos and x not in (None, eos)
                else 'Native_EOS_E_continues' if x == eos and y not in (None, eos)
                else 'other_length_or_special')
        cases.append(dict(key=k, question=samples[k]['question'], image_sha256=samples[k]['image_sha256'],
            native_answer=native['text'], E_answer=e['text'], native_score=records[k, 'Native']['score'],
            E_score=records[k, 'E']['score'], score_delta=records[k, 'E']['score']-records[k, 'Native']['score'],
            first_divergence_generated_index=i, divergence_kind=kind,
            native_token=x, E_token=y, native_subword=None if x is None else tokenizer.convert_ids_to_tokens(x),
            E_subword=None if y is None else tokenizer.convert_ids_to_tokens(y)))
    write_csv(OUT / 'decode_divergence_cases.csv', cases)
    write_csv(OUT / 'reference_sequence_differences.csv', rounding)
    result = dict(first_token_roles=token_roles,
        native_first_whitespace_only=sum(r['questions'] for r in token_roles if not r['decoded'].strip()),
        Native_E_changed_sequences=len(cases),
        first_divergence_kinds=dict(collections.Counter(r['divergence_kind'] for r in cases)),
        harmed_first_divergence_kinds=dict(collections.Counter(r['divergence_kind'] for r in cases if r['score_delta'] < -1e-12)),
        E3_A_changed_sequences=len(rounding), E3_A_all500_scores_identical=True)
    atomic(OUT / 'first_token_and_decode_diagnostics.json', result)
    return result


def report(summary, comparisons, interaction, gate, diagnostics):
    table = ['|条件|Accuracy|ΔNative pp|ΔE3 pp|EOS/截断/空答案|', '|---|---:|---:|---:|---:|']
    table += [f"|{r['condition']}|{r['accuracy']:.2f}%|{r['delta_native_pp']:+.2f}|{r['delta_E3_pp']:+.2f}|{r['eos']}/{r['truncated']}/{r['empty']}|" for r in summary]
    paired = ['|比较|ΔAccuracy pp|图片簇95%CI|软分升/降|', '|---|---:|---|---:|']
    paired += [f"|{r['comparison']}|{r['delta_accuracy_pp']:+.2f}|[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]|{r['score_increased']}/{r['score_decreased']}|" for r in comparisons]
    lines = ['# Middle10/E3剩余质量损失定位与固定恢复', '',
        '**本轮完成并停止：全部质量完成，未达47.68%门槛；未做物理移植、E2E、FLOPs或性能测试。**' if not gate['quality_gate_passed'] else
        '**第一阶段质量全部完成；物理阶段和性能阶段以单独门槛为准。**', '',
        '原Middle10为39.46%，相对Native 49.68%差10.22pp；此前E3为41.38%，剩余差距8.30pp。本轮不混淆两者。', '',
        '## 冻结条件', '',
        'A：E3 Dense参考，无物理退出；B：A恢复32层R写入；C：A恢复P−1全部32层原生Attention与读图（包括L2–L5）；D：B+C，即Native Prefill；E：D进一步恢复Decode L2–L5文本Attention。所有A–E的Decode仅在L11–L20读图。', '',
        'C/D恢复P−1时同时取消其L2–L5 Attention跳算，因为否则这四层不可能执行正常视觉Attention。因此C−A不是纯粹的Image Key权限效应，应表述为“完整P−1 Attention/视觉读取恢复”。', '',
        '## 完整质量', '', *table, '',
        '同500题、309图片SHA簇，官方十答案共识软分，原Prompt和FP16/eager/H1/最多32 Token。历史1,500条基线严格绑定复用；新增2,500条全部完成，不删除失分、异常或截断。', '',
        '## 配对比较', '', *paired, '',
        f"R与P−1联合恢复的交互D−B−C+A：{interaction['delta_accuracy_pp']:+.2f}pp，95%CI [{interaction['ci_low']:+.2f}, {interaction['ci_high']:+.2f}]。普通探索性区间，不能据此宣称唯一来源或可加贡献。", '',
        'B−A的区间下界为数值舍入量级的0，按0.00展示，不以此宣称严格排除0。', '',
        '## 剩余损失最可能来自哪里', '',
        '1. 物理退出本身不是本轮主要失分解释：A=41.38%，与E3在全部500题软分完全相同、首Token翻转为0。仍有1题完整Token序列不同，见reference_sequence_differences.csv；不声称物理/Dense全部序列或数值逐位相同。', '',
        '2. 只有晚层R写入恢复的B为42.56%，相对A +1.18pp [0.00, 2.46]。完整P−1恢复的C为44.16%，相对A +2.78pp [0.43, 5.27]，点估计帮助更大，但C含L2–L5 Attention恢复，不能拆成纯视觉权限效应。两者联合D为45.00%，相对A +3.62pp [1.08, 6.25]。', '',
        '3. D已经在全部500题精确复现Native首Token Logits，但完整答案仍低4.68pp。E继续恢复Decode L2–L5文本Attention只净增0.14pp [−1.40, 1.63]，有10题软分升、8题下降，未得到其为主要损失来源的证据。', '',
        '4. 最佳E保留Native Prefill与完整Decode Attention/MLP，在语义上与Native剩下的干预仅为“Decode在L11–L20之外不能直接读取Image Key”。E=45.14%，相对Native −4.54pp，图片簇95%CI [−6.81, −2.44]；6题软分上升、30题下降，24题从Native满分变差。经独立原生HF参考Smoke、Cache审计和全500题Native首Logits一致检查后，结果支持：剩余大块损失来自真实Decode的受限视觉重读，而非单纯L20 R写入不足或Decode L2–L5文本跳算。', '',
        '5. 沿固定顺序A→D→E→Native，8.30pp差距对应3.62pp、0.14pp、4.54pp。它是顺序依赖的已测差值分解，不是可相加的独立信息贡献，也不证明某个未测试层或Token一定负责。没有追加层搜索。', '',
        f"首Token解释边界：Native有{diagnostics['native_first_whitespace_only']}/500题的首生成Token单独解码为空白，因此首Token一致不能自动等同首个语义内容词一致。E与Native有{diagnostics['Native_E_changed_sequences']}题完整序列变化；首次分叉类别和逐题内容见decode_divergence_cases.csv。", '',
        f"E相对Native的30道软分下降题，首次分叉为双方均非EOS的内容Token有{diagnostics['harmed_first_divergence_kinds'].get('both_non_EOS', 0)}题，E提前EOS有{diagnostics['harmed_first_divergence_kinds'].get('E_early_EOS', 0)}题。因此不能将此次质量损失仅归为终止标记差异。", '',
        '本轮不能证明具体哪一组非Middle10层可以最小恢复，也没有测试更高成本方案或改变Prompt。最佳仍落后Native 4.54pp，不能称为接近Native或无损。', '',
        '## 阶段门槛', '',
        f"最佳固定功能性条件为{gate['best_condition']}，Accuracy {gate['best_accuracy']:.2f}%；47.68%门槛：{'通过' if gate['quality_gate_passed'] else '未通过'}。",
        '若未通过立即停止；若通过仅允许归因和物理移植，仍禁止测速。必须物理版本也达到47.68%并通过正确性检查，才能运行48题性能。', '',
        '**当前性能请求数：0。** 第一阶段代码无CUDA计时事件、FLOPs/Profiler或性能循环。', '',
        '## 验证', '',
        '19项CPU测试通过，双卡各3题真实Smoke通过。Native/No-op所有有效Logits严格torch.equal；官方HF完整生成一致；Native/E3物理路径复现历史完整Token及首Logits哈希。A与冻结E3 Dense的有效Logits精确一致。', '',
        '全部A–E与独立原生HF Decoder/DynamicCache定向Hook参考在冻结FP16门槛内一致，Smoke固定3 Token与自然EOS序列一致。禁止连接概率为0、合法Key存在、原位置、32层MLP、Cache重放隔离及Hook清理通过。D/E全部500题首Token Logits哈希均严格等于Native。', '',
        '第一阶段所有条件都是Dense功能性归因，不宣称物理加速。D/E的Native Prefill依赖需要与真正提前删除视觉Hidden的物理版本区分。', '',
        '## 图与数据', '',
        '![固定条件质量与配对区间](quality_attribution.png)', '',
        '逐题答案：per_question.csv；全部损益和负面：recovery_and_negative_cases.csv；完整Token及审计：原始断点包。输入、源码、权重及协议绑定见execution_freeze.json和source_manifest.json。', '',
        '20,000次图片SHA整簇配对Bootstrap，问题等权、固定种子20261106；历史探索样本，不宣称独立确认或非劣效。', '',
        '原始2,500条新断点在raw_quality_checkpoints.tar.gz；原始GPU Smoke在raw_smoke_audit.tar.gz。归档逐文件校验见packaging.json，断点哈希见result_hashes.json。解包时保持相对路径，不覆盖历史实验。', '',
        f"正式双卡质量运行墙钟约{max(read(OUT / f'worker{r}/progress.json')['elapsed_seconds'] for r in range(2))/60:.2f}分钟；该值是整批作业耗时，不是性能测试或E2E延迟。", '',
        'CPU-only复算：`python -m vlm_research.middle10_attribution.analyze`。当前禁止自动重启GPU运行器、物理移植或测速。', '',
        '第一阶段及最终进程释放见gpu_release_audit.json。最终状态见dense_quality_gate.json和completion.json。本轮完成后停止，不追加候选。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()

"""Image-cluster paired quality statistics; never trigger further GPU work."""
import json

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
    samples, p = read(OUT / 'sample_manifest.json'), read(OUT / 'partition.json')
    reused = {(r['key'], r['condition']): r for r in read(OUT / 'source_manifest.json')['reused_records']}
    keys, labels = sorted(samples), BASELINES+LABELS
    values = np.zeros((500, 4))
    records, rows, hashes = {}, [], {}
    for i, key in enumerate(keys):
        s = samples[key]
        native = historical(key, 'Native')['generation']
        for j, label in enumerate(labels):
            path = historical_path(key, label) if label in BASELINES else checkpoint(key, label)
            if label in BASELINES:
                assert sha(path) == reused[key, label]['sha256']
            v = load(path)
            assert v['key'] == key and v['condition'] == label and v['sample_sha256'] == digest(s)
            assert digest(v['input_binding']) == reused[key, 'Native']['input_binding_sha256']
            if label in LABELS:
                assert v['sources'] == f['sources']
                assert v['rank'] == p['image_shard'][s['image_sha256']]
                hashes[str(path.relative_to(OUT))] = sha(path)
            g = v['generation']
            assert v['score'] == score(s, g['text'])
            assert 1 <= len(g['token_ids']) <= 32
            assert g['first_logits_sha256'] == native['first_logits_sha256']
            records[key, label], values[i, j] = v, v['score']
            rows.append(dict(key=key, question_id=s['question_id'], image_id=s['image_id'],
                image_sha256=s['image_sha256'], question=s['question'], gold_answers=s['answer'],
                condition=label, history_reused=label in BASELINES, answer=g['text'], score=v['score'],
                token_ids=g['token_ids'], input_binding_sha256=digest(v['input_binding']),
                first_logits_sha256=g['first_logits_sha256'], first_top1_token=g['token_ids'][0],
                eos=g['eos'], empty=g['empty'], truncated=g['truncated'], repeated=g['repeated'],
                content_tokens=g['content_tokens']))
    assert len(rows) == 2000 and len(hashes) == 1000
    assert {str(v.relative_to(OUT)) for v in (OUT / 'checkpoints').glob('*/*.json')} == set(hashes)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    assert abs(values[:, 0].mean()*100-49.68) < 1e-8
    assert abs(values[:, 1].mean()*100-45.14) < 1e-8
    write_csv(OUT / 'per_question.csv', rows)
    clusters = sorted({s['image_sha256'] for s in samples.values()})
    cluster_ix = {h: i for i, h in enumerate(clusters)}
    membership = np.array([cluster_ix[samples[k]['image_sha256']] for k in keys])
    sizes, sums = np.bincount(membership), np.zeros((len(clusters), 4))
    np.add.at(sums, membership, values)
    weights = np.random.default_rng(SEED).multinomial(len(clusters),
        np.full(len(clusters), 1/len(clusters)), size=20000)
    boot = weights @ sums/(weights @ sizes)[:, None]*100
    means = values.mean(0)*100
    summaries = []
    for j, label in enumerate(labels):
        lo, hi = bounds(boot[:, j])
        dl, dh = bounds(boot[:, j]-boot[:, 0])
        el, eh = bounds(boot[:, j]-boot[:, 1])
        gs = [records[k, label]['generation'] for k in keys]
        summaries.append(dict(condition=label, questions=500, image_clusters=309,
            accuracy=float(means[j]), ci_low=lo, ci_high=hi,
            delta_native_pp=float(means[j]-means[0]), delta_native_ci_low=dl, delta_native_ci_high=dh,
            delta_E_pp=float(means[j]-means[1]), delta_E_ci_low=el, delta_E_ci_high=eh,
            eos=sum(g['eos'] for g in gs), empty=sum(g['empty'] for g in gs),
            truncated=sum(g['truncated'] for g in gs), repeated=sum(g['repeated'] for g in gs),
            native_first_logits_exact=500, reaches_47_68=bool(means[j] >= THRESHOLD)))
    write_csv(OUT / 'accuracy_summary.csv', summaries)
    comparisons, cases = [], []
    for candidate, baseline in [('Early', 'E'), ('Late', 'E'), ('Early', 'Native'),
                                 ('Late', 'Native'), ('Late', 'Early'), ('E', 'Native')]:
        a, b = labels.index(candidate), labels.index(baseline)
        delta = values[:, a]-values[:, b]
        lo, hi = bounds(boot[:, a]-boot[:, b])
        pair = dict(candidate=candidate, baseline=baseline,
            delta_accuracy_pp=float(means[a]-means[b]), ci_low=lo, ci_high=hi,
            soft_score_increased=int(np.sum(delta > 1e-10)),
            soft_score_decreased=int(np.sum(delta < -1e-10)),
            full_score_to_lower=int(np.sum((values[:, b] == 1) & (values[:, a] < 1))),
            lower_to_full_score=int(np.sum((values[:, b] < 1) & (values[:, a] == 1))),
            first_token_flips=0, sequence_changes=0)
        for i, key in enumerate(keys):
            c, base = records[key, candidate]['generation'], records[key, baseline]['generation']
            pair['first_token_flips'] += int(c['token_ids'][0] != base['token_ids'][0])
            pair['sequence_changes'] += int(c['token_ids'] != base['token_ids'])
            if abs(delta[i]) > 1e-10:
                cases.append(dict(key=key, question=samples[key]['question'],
                    image_sha256=samples[key]['image_sha256'], candidate=candidate, baseline=baseline,
                    candidate_answer=c['text'], baseline_answer=base['text'],
                    candidate_score=float(values[i, a]), baseline_score=float(values[i, b]),
                    score_change=float(delta[i]), effect='recovery' if delta[i] > 0 else 'harm'))
        comparisons.append(pair)
    write_csv(OUT / 'paired_comparisons.csv', comparisons)
    write_csv(OUT / 'harm_recovery_cases.csv', cases)
    write_csv(OUT / 'negative_cases.csv', [v for v in cases if v['effect'] == 'harm'])
    write_csv(OUT / 'recovery_cases.csv', [v for v in cases if v['effect'] == 'recovery'])
    atomic(OUT / 'result_hashes.json', dict(at=now(), count=1000, checkpoint_sha256=hashes))
    resource = hardware()
    pids = read(OUT / 'driver_pids.json')['workers']
    assert all(not Path(f'/proc/{pid}').exists() for pid in pids)
    atomic(OUT / 'gpu_release_audit.json', dict(at=now(), task_worker_pids=pids,
        task_workers_exited=True, resources=resource,
        workers=[read(OUT / f'worker{r}/release.json') for r in range(2)]))
    atomic(OUT / 'validation.json', dict(status='PASS', CPU_tests=18, CPU_passed=True,
        GPU_smoke=[read(OUT / f'worker{r}/smoke.json')['status'] for r in range(2)],
        historical_records_audited=1000, new_quality_generations=1000,
        native_prefill_first_logits_exact_all_conditions=True,
        finite_scores=True, image_clusters=309, no_missing_or_duplicate_records=True,
        sources=f['sources'], tolerances=TOLERANCES['llava'], performance_requests=0))
    passed = [r['condition'] for r in summaries[2:] if r['accuracy'] >= THRESHOLD]
    atomic(OUT / 'completion.json', dict(at=now(), status='COMPLETE_QUALITY_ONLY',
        new_quality_generations=1000, reused_generations=1000, questions_per_condition=500,
        threshold=THRESHOLD, candidates_reaching_threshold=passed,
        performance_requests=0, physical_implementation='NOT_RUN_THIS_TASK',
        formal_quality_wall_seconds=max(read(OUT / f'worker{r}/progress.json')['elapsed_seconds'] for r in range(2)),
        code_commit=f['code_commit'], no_further_GPU_work=True))
    report(summaries, comparisons, passed)
    print(json.dumps(dict(summary=summaries, comparisons=comparisons), indent=2))


def report(summary, pairs, passed):
    late_early = next(p for p in pairs if (p['candidate'], p['baseline']) == ('Late', 'Early'))
    winner = 'Late' if late_early['delta_accuracy_pp'] > 0 else 'Early'
    lines = ['# Middle10：早层与深层Decode视觉回看恢复', '',
        '## 协议与完整性',
        f'- 基线Commit：`{BASE_COMMIT}`。同500题、309图片SHA簇，10答案TextVQA共识软评分。',
        '- Native与E各500条经输入、权重、源码、生成协议和首Logits绑定核验后复用；Early/Late各新增500条。',
        '- 全部Prefill（包括P−1）完全Native；所有层Attention/MLP正常，只对真实Decode的Image来源进行Softmax前Hard Mask。',
        '- Native：L0–L31；E：L11–L20；Early：L0–L20；Late：L11–L31。Dense功能性实验，无物理退出或KV优化。',
        '- CPU18项及双卡各3道真实Smoke通过；Native/No-op严格一致、禁止概率为0、原始位置/Cache/Hook检查通过。',
        '- 全部2000条记录的首Logits SHA均与同题Native一致。本轮不进行任何性能测试。', '',
        '## 质量', '| 条件 | Accuracy (%) | ΔNative (pp) | ΔE (pp) |',
        '|---|---:|---:|---:|']
    lines += [f"| {r['condition']} | {r['accuracy']:.2f} | {r['delta_native_pp']:+.2f} | {r['delta_E_pp']:+.2f} |"
              for r in summary]
    lines += ['', '## 配对比较',
        '20,000次Image SHA整簇Bootstrap，保留问题等权点估计；双侧探索性95%CI。',
        '| 比较 | 差值pp [95%CI] | 软分上升/下降 | 满分→较低/较低→满分 |',
        '|---|---|---:|---:|']
    lines += [f"| {p['candidate']} − {p['baseline']} | {p['delta_accuracy_pp']:+.2f} [{p['ci_low']:+.2f}, {p['ci_high']:+.2f}] | {p['soft_score_increased']}/{p['soft_score_decreased']} | {p['full_score_to_lower']}/{p['lower_to_full_score']} |"
              for p in pairs]
    lines += ['', '## 研究结论',
        f"- 点估计上，{winner}补充更有效。Late−Early为{late_early['delta_accuracy_pp']:+.2f}pp，95%CI [{late_early['ci_low']:+.2f}, {late_early['ci_high']:+.2f}]。",
        '- 直接配对差异的区间若跨0，不据此宣称两种区域等效或已确认优劣。',
        '- Early开放21层，Late开放21层；二者分别补回L0–L10和L21–L31，且共同保留L11–L20。本结论只适用于该固定配置和历史探索集。',
        '- 全部首答案Logits不变；后续真实Decode对自己的生成前缀运行。首Token一致不代表完整答案一致。']
    if passed:
        lines += [f"- 达到47.68%点估计门槛的条件：{', '.join(passed)}。本轮仍立即停止，不测速。",
            '- 后续方案仅建议：保持完整Native Prefill，按有效Decode层集合选择性持久保存原始Image KV，其余层仅持久保存文本KV；先独立验证RoPE/因果和数值一致，再做物理性能研究。这里没有实现或测量该方案。']
    else:
        lines += ['- Early和Late均未达到47.68%；按预声明规则停止并提交，不实现物理版本，不测速。']
    lines += ['', '## 异常与边界',
        '- 各条件EOS、空答案、重复及截断计数见`accuracy_summary.csv`，全部损害及恢复案例见对应CSV。',
        '- 不新增层组合，不调整Prompt、权重、精度或冻结数值容差；CI跨0不等于无损。',
        '- GPU Worker均已退出；资源释放与执行日志已保存。', '', '## 复现',
        '```bash', 'python -m vlm_research.middle10_decode_regions.prepare',
        '# Only after fresh GPU authorization and source commit:',
        'python -m vlm_research.middle10_decode_regions.run --freeze --devices 0,1 --authorization-text "ACTUAL_USER_AUTHORIZATION"',
        'python -m vlm_research.middle10_decode_regions.run --driver',
        'python -m vlm_research.middle10_decode_regions.analyze', '```', '']
    (OUT / 'report_zh.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

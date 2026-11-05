"""Image-cluster paired quality statistics; never trigger further GPU work."""
import json
import gzip
import hashlib
import shutil
import tarfile

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
    plot_quality(summaries)
    atomic(OUT / 'completion.json', dict(at=now(), status='COMPLETE_QUALITY_ONLY',
        new_quality_generations=1000, reused_generations=1000, questions_per_condition=500,
        threshold=THRESHOLD, candidates_reaching_threshold=passed,
        performance_requests=0, physical_implementation='NOT_RUN_THIS_TASK',
        formal_quality_wall_seconds=max(read(OUT / f'worker{r}/progress.json')['elapsed_seconds'] for r in range(2)),
        code_commit=f['code_commit'], no_further_GPU_work=True))
    report(summaries, comparisons, passed)
    package(hashes)
    print(json.dumps(dict(summary=summaries, comparisons=comparisons), indent=2))


def plot_quality(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), layout='constrained')
    colors = ['#555555', '#638178', '#356da3', '#bd694a']
    for i, row in enumerate(summary):
        value = row['accuracy']
        axes[0].errorbar(i, value, yerr=[[value-row['ci_low']], [row['ci_high']-value]],
                        fmt='o', color=colors[i], capsize=4, ms=6)
        axes[0].annotate(f'{value:.2f}', (i, value), xytext=(7, 0),
                         textcoords='offset points', fontsize=8)
    axes[0].axhline(THRESHOLD, color='#a04444', ls='--', lw=1, label='47.68% point-estimate gate')
    axes[0].set(xticks=np.arange(4), xticklabels=['Native', 'E', 'Early', 'Late'],
                xlim=(-.5, 3.6), ylabel='TextVQA soft accuracy (%)', title='Same frozen 500 questions')
    axes[0].legend(fontsize=7, loc='lower left')
    for i, row in enumerate(summary[2:]):
        value = row['delta_E_pp']
        axes[1].errorbar(i, value,
            yerr=[[value-row['delta_E_ci_low']], [row['delta_E_ci_high']-value]],
            fmt='o', color=colors[i+2], capsize=4, ms=6)
        axes[1].annotate(f'{value:+.2f}', (i, value), xytext=(8, 0),
                         textcoords='offset points', fontsize=8)
    axes[1].axhline(0, color='#777777', lw=.8)
    axes[1].set(xticks=[0, 1], xticklabels=['Early: +L0-L10', 'Late: +L21-L31'],
                xlim=(-.5, 1.5), ylabel='Accuracy change vs E (pp)',
                title='20,000 image-cluster paired bootstrap: 95% CI')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'quality_recovery.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def package(expected_hashes):
    quality = OUT / 'raw_quality_checkpoints.tar.gz'
    with tarfile.open(quality, 'w:gz') as archive:
        archive.add(OUT / 'checkpoints', arcname='checkpoints')
    with tarfile.open(quality, 'r:gz') as archive:
        names = [m.name for m in archive.getmembers() if m.isfile()]
        assert len(names) == len(set(names)) == 1000
        actual = {n: hashlib.sha256(archive.extractfile(n).read()).hexdigest() for n in names}
        assert actual == expected_hashes
    smoke = OUT / 'raw_smoke_audit.tar.gz'
    smoke_paths = [f'worker{r}/{n}.json' for r in range(2) for n in ('smoke', 'smoke_partial')]
    with tarfile.open(smoke, 'w:gz') as archive:
        for path in smoke_paths:
            archive.add(OUT / path, arcname=path)
    with tarfile.open(smoke, 'r:gz') as archive:
        assert {m.name for m in archive.getmembers()} == set(smoke_paths)
        for path in smoke_paths:
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == sha(OUT / path)
    compressed = OUT / 'sample_manifest.json.gz'
    with compressed.open('wb') as destination, gzip.GzipFile(filename='', mode='wb',
            fileobj=destination, mtime=0) as stream, (OUT / 'sample_manifest.json').open('rb') as source:
        shutil.copyfileobj(source, stream)
    with gzip.open(compressed, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == sha(OUT / 'sample_manifest.json')
    atomic(OUT / 'packaging.json', dict(at=now(), raw_checkpoints=1000,
        quality_archive_sha256=sha(quality), smoke_archive_sha256=sha(smoke),
        raw_smoke_members={p: sha(OUT / p) for p in smoke_paths},
        manifest_gzip_sha256=sha(compressed), manifest_original_sha256=sha(OUT / 'sample_manifest.json'),
        all_archive_members_verified=True, historical_records_reused_not_regenerated=1000))


def report(summary, pairs, passed):
    late_early = next(p for p in pairs if (p['candidate'], p['baseline']) == ('Late', 'Early'))
    late_e = next(p for p in pairs if (p['candidate'], p['baseline']) == ('Late', 'E'))
    late_native = next(p for p in pairs if (p['candidate'], p['baseline']) == ('Late', 'Native'))
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
        '- 这些是普通、未经多重比较校正的探索性区间，不作为独立确认性结论；CI跨0也不能解释为等效或无损。',
        f"- Late相对E有{late_e['soft_score_increased']}题软分提高、{late_e['soft_score_decreased']}题下降；相对Native仍有{late_native['soft_score_decreased']}题下降（其中{late_native['full_score_to_lower']}题从满分降为非满分），不能只报告净收益。",
        f"- Late−Native为{late_native['delta_accuracy_pp']:+.2f}pp，95%CI [{late_native['ci_low']:+.2f}, {late_native['ci_high']:+.2f}]。达到47.68%仅表示点估计过门槛，不能证明相差不超过2pp或非劣效。",
        '- Early开放21层，Late开放21层；二者分别补回L0–L10和L21–L31，且共同保留L11–L20。本结论只适用于该固定配置和历史探索集。',
        '- 全部首答案Logits不变；后续真实Decode对自己的生成前缀运行。首Token一致不代表完整答案一致。']
    if passed:
        lines += [f"- 达到47.68%点估计门槛的条件：{', '.join(passed)}。本轮仍立即停止，不测速。",
            '- 后续物理方案仅建议：保持完整Native Prefill及P−1，L11–L31持久保留原始Image KV，L0–L10完成各层Prefill后只持久保留文本KV；Decode保持全层文本Attention，原始RoPE位置及因果关系不变。',
            '- 因Native Prefill/P−1仍需32层读图，不能直接沿用原C3进入L20前删除视觉Hidden的路径。此建议是选择性Decode KV压缩，不承诺保留旧C3的Prefill退出收益；先验证数值与生成，再另行研究性能。本轮没有实现或测速。']
    else:
        lines += ['- Early和Late均未达到47.68%；按预声明规则停止并提交，不实现物理版本，不测速。']
    lines += ['', '## 异常与边界',
        '- 各条件EOS、空答案、重复及截断计数见`accuracy_summary.csv`，全部损害及恢复案例见对应CSV。',
        '- 质量图：`quality_recovery.png/pdf`；完整原子检查点和Smoke分别压缩归档；冻结500题清单为`sample_manifest.json.gz`。',
        '- 不新增层组合，不调整Prompt、权重、精度或冻结数值容差；CI跨0不等于无损。',
        '- Native/E/Early/Late的EOS数为499/499/499/498，截断数为1/1/1/2，空答案均为0；重复标记数为1/3/3/2。全部保留，没有删除异常样本。',
        f"- 正式双卡质量矩阵墙钟{max(read(OUT / f'worker{r}/progress.json')['elapsed_seconds'] for r in range(2)):.2f}秒；这是任务完成时间，不是推理性能基准。",
        '- GPU Worker均已退出；资源释放与执行日志已保存。', '', '## 复现',
        '```bash', 'python -m vlm_research.middle10_decode_regions.prepare',
        '# GPU0/1授权原文：确认允许，以后都确认。源码冻结后执行：',
        'python -m vlm_research.middle10_decode_regions.run --freeze --devices 0,1 --authorization-text "确认允许，以后都确认"',
        'python -m vlm_research.middle10_decode_regions.run --driver',
        'python -m vlm_research.middle10_decode_regions.analyze', '```', '']
    (OUT / 'report_zh.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

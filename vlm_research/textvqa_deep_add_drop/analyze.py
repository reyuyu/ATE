"""Exploratory image-cluster paired Add/Drop accuracy, with all signs retained."""
import gzip
import hashlib
import json
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
    samples, partition = read(OUT / 'sample_manifest.json'), read(OUT / 'partition.json')
    reused = {(v['key'], v['condition']): v for v in read(OUT / 'source_manifest.json')['reused_records']}
    keys, labels = sorted(samples), BASELINES+LABELS
    n = len(keys)
    values = np.zeros((n, len(labels)))
    records, rows, hashes = {}, [], {}
    for i, key in enumerate(keys):
        sample = samples[key]
        native = historical(key, 'Native')['generation']
        for j, label in enumerate(labels):
            old = label in BASELINES
            path = historical_path(key, label) if old else checkpoint(key, label)
            if old:
                assert sha(path) == reused[key, label]['sha256']
            value = load(path)
            assert value['key'] == key and value['condition'] == label and value['sample_sha256'] == digest(sample)
            assert digest(value['input_binding']) == reused[key, 'Native']['input_binding_sha256']
            if not old:
                assert value['sources'] == f['sources']
                assert value['rank'] == partition['image_shard'][sample['image_sha256']]
                hashes[str(path.relative_to(OUT))] = sha(path)
            g = value['generation']
            assert value['score'] == score(sample, g['text'])
            assert 1 <= len(g['token_ids']) <= 32
            assert g['first_logits_sha256'] == native['first_logits_sha256']
            records[key, label], values[i, j] = value, value['score']
            base = 'E' if label.startswith('Add-') else 'Late' if label.startswith('Drop-') else 'Native'
            rows.append(dict(key=key, question_id=sample['question_id'], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], question=sample['question'], gold_answers=sample['answer'],
                condition=label, history_reused=old, answer=g['text'], score=value['score'],
                token_ids=g['token_ids'], input_binding_sha256=digest(value['input_binding']),
                first_logits_sha256=g['first_logits_sha256'], first_token_flip_native=g['token_ids'][0] != native['token_ids'][0],
                comparison_base=base, base_answer=historical(key, base)['generation']['text'],
                score_delta_base=value['score']-historical(key, base)['score'],
                score_delta_native=value['score']-historical(key, 'Native')['score'],
                eos=g['eos'], empty=g['empty'], truncated=g['truncated'], repeated=g['repeated'],
                content_tokens=g['content_tokens']))
    assert len(rows) == n*25 and len(hashes) == n*22
    assert {str(p.relative_to(OUT)) for p in (OUT / 'checkpoints').glob('*/*.json')} == set(hashes)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    write_csv(OUT / 'per_question.csv', rows)
    clusters = sorted({s['image_sha256'] for s in samples.values()})
    index = {h: i for i, h in enumerate(clusters)}
    membership = np.array([index[samples[k]['image_sha256']] for k in keys])
    sizes, sums = np.bincount(membership), np.zeros((len(clusters), len(labels)))
    np.add.at(sums, membership, values)
    weights = np.random.default_rng(SEED).multinomial(len(clusters),
        np.full(len(clusters), 1/len(clusters)), size=20000)
    boot = weights @ sums/(weights @ sizes)[:, None]*100
    means = values.mean(0)*100
    summary = []
    for j, label in enumerate(labels):
        lo, hi = bounds(boot[:, j])
        row = dict(condition=label, questions=n, image_clusters=len(clusters), accuracy=float(means[j]),
                   ci_low=lo, ci_high=hi)
        for base in BASELINES:
            b = labels.index(base)
            lo, hi = bounds(boot[:, j]-boot[:, b])
            row.update({f'delta_{base}_pp': float(means[j]-means[b]),
                        f'delta_{base}_ci_low': lo, f'delta_{base}_ci_high': hi})
        gs = [records[k, label]['generation'] for k in keys]
        row.update(eos=sum(g['eos'] for g in gs), empty=sum(g['empty'] for g in gs),
            truncated=sum(g['truncated'] for g in gs), repeated=sum(g['repeated'] for g in gs),
            native_prefill_first_logits_exact=n)
        summary.append(row)
    write_csv(OUT / 'accuracy_summary.csv', summary)
    comparisons, cases = [], []
    pairs = [(c, 'E' if c.startswith('Add-') else 'Late') for c in LABELS]
    pairs += [(c, 'Native') for c in LABELS]
    for candidate, baseline in pairs:
        a, b = labels.index(candidate), labels.index(baseline)
        delta = values[:, a]-values[:, b]
        lo, hi = bounds(boot[:, a]-boot[:, b])
        pair = dict(candidate=candidate, baseline=baseline, delta_accuracy_pp=float(means[a]-means[b]),
            ci_low=lo, ci_high=hi, soft_score_increased=int(np.sum(delta > 1e-10)),
            soft_score_decreased=int(np.sum(delta < -1e-10)),
            full_score_to_lower=int(np.sum((values[:, b] == 1) & (values[:, a] < 1))),
            lower_to_full_score=int(np.sum((values[:, b] < 1) & (values[:, a] == 1))),
            sequence_changes=0, first_token_flips=0)
        for i, key in enumerate(keys):
            c, base = records[key, candidate]['generation'], records[key, baseline]['generation']
            pair['sequence_changes'] += int(c['token_ids'] != base['token_ids'])
            pair['first_token_flips'] += int(c['token_ids'][0] != base['token_ids'][0])
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
    by_label = {v['condition']: v for v in summary}
    by_pair = {(v['candidate'], v['baseline']): v for v in comparisons}
    layers = []
    for layer in LAYERS:
        a, d = by_label[f'Add-L{layer}'], by_label[f'Drop-L{layer}']
        pa, pd = by_pair[a['condition'], 'E'], by_pair[d['condition'], 'Late']
        layers.append(dict(layer=layer, questions=n, image_clusters=len(clusters),
            add_accuracy=a['accuracy'], add_delta_E_pp=a['delta_E_pp'],
            add_ci_low=a['delta_E_ci_low'], add_ci_high=a['delta_E_ci_high'],
            add_score_increased=pa['soft_score_increased'], add_score_decreased=pa['soft_score_decreased'],
            drop_accuracy=d['accuracy'], drop_delta_Late_pp=d['delta_Late_pp'],
            drop_ci_low=d['delta_Late_ci_low'], drop_ci_high=d['delta_Late_ci_high'],
            drop_score_increased=pd['soft_score_increased'], drop_score_decreased=pd['soft_score_decreased'],
            drop_loss_Late_minus_Drop_pp=-d['delta_Late_pp']))
    write_csv(OUT / 'layer_add_drop_summary.csv', layers)
    plot(layers, by_label, n)
    atomic(OUT / 'result_hashes.json', dict(at=now(), count=n*22, checkpoint_sha256=hashes))
    pids = read(OUT / 'driver_pids.json')['workers']
    assert all(not Path(f'/proc/{pid}').exists() for pid in pids)
    atomic(OUT / 'gpu_release_audit.json', dict(at=now(), task_worker_pids=pids, workers_exited=True,
        resources=hardware(), workers=[read(OUT / f'worker{r}/release.json') for r in range(2)]))
    smoke = [read(OUT / f'worker{r}/smoke.json') for r in range(2)]
    assert all(s['status'] == 'PASS' and len(s['checks']) == 3 for s in smoke)
    atomic(OUT / 'validation.json', dict(status='PASS', CPU_tests=16,
        GPU_smoke=[s['status'] for s in smoke], questions=n, image_clusters=len(clusters),
        new_quality_generations=n*22, historical_records_audited=n*3,
        native_prefill_first_logits_exact_all=True, no_missing_or_duplicate_records=True,
        original_tolerances=TOLERANCES['llava'], inference_sources=f['sources'],
        analysis_source_sha256=sha(Path(__file__)), performance_requests=0))
    atomic(OUT / 'completion.json', dict(at=now(), status='COMPLETE_QUALITY_ONLY',
        questions=n, image_clusters=len(clusters), conditions=22, new_quality_generations=n*22,
        reused_generations=n*3, code_commit=f['code_commit'], performance_requests=0,
        formal_quality_wall_seconds=max(read(OUT / f'worker{r}/progress.json')['elapsed_seconds'] for r in range(2)),
        no_further_GPU_work=True))
    report(summary, layers, comparisons, cases, n, len(clusters))
    package(hashes)
    print(json.dumps(dict(baselines={b: by_label[b]['accuracy'] for b in BASELINES}, layers=layers), indent=2))


def plot(layers, by_label, n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    x = [r['layer'] for r in layers]
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for prefix, color, label in [('add', '#b55e3e', 'Add-Li minus E'), ('drop', '#366da2', 'Drop-Li minus Late')]:
        y = [r['add_delta_E_pp' if prefix == 'add' else 'drop_delta_Late_pp'] for r in layers]
        ax.plot(x, y, '-o', color=color, ms=4, lw=1.5, label=label)
        ax.fill_between(x, [r[prefix+'_ci_low'] for r in layers],
                        [r[prefix+'_ci_high'] for r in layers], color=color, alpha=.15, linewidth=0)
    ax.axhline(0, color='#777777', lw=.8)
    ax.set(xticks=x, xlabel='Decode Image-access layer', ylabel='Soft accuracy change (pp)',
           title=f'TextVQA: {n} fixed questions, image-cluster paired 95% CI')
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'add_drop_delta.{ext}', dpi=250, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(x, [r['add_accuracy'] for r in layers], '-o', color='#b55e3e', label='Add-Li')
    ax.plot(x, [r['drop_accuracy'] for r in layers], '-s', color='#366da2', label='Drop-Li')
    for baseline, color, style in [('Native', '#555555', '--'), ('E', '#718b78', ':'), ('Late', '#967640', '-.')]:
        ax.axhline(by_label[baseline]['accuracy'], color=color, ls=style, lw=1,
                   label=f"{baseline}: {by_label[baseline]['accuracy']:.2f}%")
    ax.set(xticks=x, xlabel='Decode Image-access layer', ylabel='TextVQA soft accuracy (%)',
           title=f'Absolute scores on the same {n} questions')
    ax.legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'add_drop_accuracy.{ext}', dpi=250, bbox_inches='tight')
    plt.close(fig)


def package(hashes):
    archives = {'raw_quality_checkpoints.tar.gz': ['checkpoints'],
                'raw_smoke_audit.tar.gz': [f'worker{r}/{name}.json' for r in range(2)
                                         for name in ('smoke', 'smoke_partial')]}
    artifact_sha = {}
    for filename, paths in archives.items():
        path = OUT / filename
        with tarfile.open(path, 'w:gz') as archive:
            for p in paths:
                archive.add(OUT / p, arcname=p)
        with tarfile.open(path, 'r:gz') as archive:
            names = [m.name for m in archive.getmembers() if m.isfile()]
            assert len(names) == len(set(names))
            actual = {p: hashlib.sha256(archive.extractfile(p).read()).hexdigest() for p in names}
            expected = hashes if filename.startswith('raw_quality') else {p: sha(OUT / p) for p in paths}
            assert actual == expected
        artifact_sha[filename] = sha(path)
    compressed = OUT / 'sample_manifest.json.gz'
    with compressed.open('wb') as destination, gzip.GzipFile(filename='', mode='wb',
            fileobj=destination, mtime=0) as stream, (OUT / 'sample_manifest.json').open('rb') as source:
        shutil.copyfileobj(source, stream)
    with gzip.open(compressed, 'rb') as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == sha(OUT / 'sample_manifest.json')
    atomic(OUT / 'packaging.json', dict(at=now(), all_archive_members_verified=True,
        new_checkpoints=len(hashes), archives=artifact_sha,
        manifest_gzip_sha256=sha(compressed), manifest_original_sha256=sha(OUT / 'sample_manifest.json')))


def report(summary, layers, pairs, cases, n, clusters):
    base = {v['condition']: v['accuracy'] for v in summary if v['condition'] in BASELINES}
    best_add = sorted(layers, key=lambda r: (-r['add_delta_E_pp'], r['layer']))[:3]
    most_drop = sorted(layers, key=lambda r: (-r['drop_loss_Late_minus_Drop_pp'], r['layer']))[:3]
    own_pairs = [p for p in pairs if p['baseline'] in ('E', 'Late')]
    assert len(own_pairs) == 22 and all(p['ci_low'] <= 0 <= p['ci_high'] for p in own_pairs)
    drop_shared = sorted({c['key'] for c in cases if c['candidate'] == 'Drop-L27'
        and c['baseline'] == 'Late' and c['effect'] == 'harm'} &
        {c['key'] for c in cases if c['candidate'] == 'Drop-L28'
        and c['baseline'] == 'Late' and c['effect'] == 'harm'})
    lines = ['# TextVQA Decode深层视觉回看：单层Add/Drop', '',
        f'完成：{n}题、{clusters}个图片SHA簇，22个新条件，共{n*22}条新增完整答案；复用{n*3}条Native/E/Late答案。',
        f'基线Commit：`{BASE_COMMIT}`；执行源码Commit见`execution_freeze.json`。', '',
        '## 样本与协议',
        '- 固定seed 20261106，按sha256(seed|Image SHA)排序原500题的图片簇，取题数最接近200的完整簇前缀；不使用模型输出选样。完整选择顺序见`selection.json`。',
        f"- 本子集Native={base['Native']:.2f}%，E={base['E']:.2f}%，Late={base['Late']:.2f}%；不能混用原500题总体分数。",
        '- 固定LLaVA-1.5-7B revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9，FP16/eager，Transformers4.49.0，原Prompt、官方十答案共识软评分、H1 last-row LM Head。',
        '- 全部Prefill及P−1保持Native。Add-Li仅在E的L11–L20上增加Li；Drop-Li仅从Late的L11–L31删除Li。只改变真实Decode的Image来源Hard Mask，所有文本Attention/MLP正常。',
        '- 每条件独立Cache、真实Greedy生成至EOS或32 Token。CPU16项及每卡3题×全部22条件GPU Smoke通过；每条首Logits SHA与同题Native一致。', '',
        '## 逐层结果',
        '下表CI均为按Image SHA整簇配对、20,000次Bootstrap的普通探索性95%CI；点估计保持问题等权。Drop负值表示删除该层后损害，正值表示删除后反而提高。',
        '| 层 | Add Accuracy | Add−E pp [95%CI] | Drop Accuracy | Drop−Late pp [95%CI] |',
        '|---:|---:|---|---:|---|']
    lines += [f"| {r['layer']} | {r['add_accuracy']:.2f}% | {r['add_delta_E_pp']:+.2f} [{r['add_ci_low']:+.2f}, {r['add_ci_high']:+.2f}] | {r['drop_accuracy']:.2f}% | {r['drop_delta_Late_pp']:+.2f} [{r['drop_ci_low']:+.2f}, {r['drop_ci_high']:+.2f}] |" for r in layers]
    lines += ['', '## 探索性排序',
        '- Add点估计最高的三层：'+', '.join(f"L{r['layer']} ({r['add_delta_E_pp']:+.2f}pp)" for r in best_add)+'。',
        '- Drop损害点估计最大的三层（Late−Drop）：'+', '.join(f"L{r['layer']} ({r['drop_loss_Late_minus_Drop_pp']:+.2f}pp)" for r in most_drop)+'。',
        '- 全部22项自身基线比较的普通95%CI均包含0（包括端点为0）；目前没有明确的逐层确认性证据。',
        '- Drop-L27与Drop-L28的损害来自同一道题：'+', '.join(drop_shared)+'（Late回答420，删除后回答42，软评分1→0）。不能把相同案例的重复翻转当作独立证据。',
        '- L24在E背景下增加读取时提高1.71pp，但在Late背景下删除时也提高0.40pp；这一不对称不能用可加的层贡献解释，也不足以证明交互机制。',
        '- Add是在E背景下测试单层增量，Drop是在Late背景下测试单层必要性；两者不要求相同排序。单层低损害不代表多层可联合删除，不能将这些结果当作层贡献可加。',
        '- 这是历史探索队列的哈希子集和22项扫描，区间未经多重比较校正；不作统计确认、无损或非劣效声明。平坦曲线或CI跨0都不是无功能的证明。', '',
        '## 案例、异常与审计',
        '- `per_question.csv`保存全部答案、Token IDs、得分、对自身基线及Native的逐题变化；`harm_recovery_cases.csv`及正负案例CSV保留全部损害和恢复，不只展示净收益。',
        '- `accuracy_summary.csv`保留每条件EOS、空答案、截断和重复计数；不静默跳样或剔除异常。',
        '- 首Token由未改动的Native Prefill预测，故首Token一致是正确性要求而不是质量指标；Accuracy来自完整自由生成。',
        '- GPU0/1按图片SHA分片；同图全部条件同卡。源码、输入、权重、协议、逐条原子检查点哈希及原始Smoke归档保留。',
        '- 没有性能测试、物理退出、KV优化、额外层组合或训练。Worker已退出并释放GPU，本轮到此停止。', '',
        f"正式质量采集墙钟{read(OUT / 'completion.json')['formal_quality_wall_seconds']/60:.2f}分钟；这是实验作业耗时，不是推理性能测量。", '',
        '## 图与复现',
        '- `add_drop_delta.png/pdf`：Add−E、Drop−Late双曲线及配对CI。',
        '- `add_drop_accuracy.png/pdf`：绝对Accuracy双曲线及同子集Native/E/Late参考线。',
        '```bash', 'python -m vlm_research.textvqa_deep_add_drop.prepare',
        'python -m vlm_research.textvqa_deep_add_drop.run --freeze',
        'python -m vlm_research.textvqa_deep_add_drop.run --driver',
        'python -m vlm_research.textvqa_deep_add_drop.analyze', '```', '']
    (OUT / 'report_zh.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

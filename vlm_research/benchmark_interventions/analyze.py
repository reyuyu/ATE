"""Require complete paired coverage; cluster bootstrap by actual image SHA256."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.dataset_registry import digest, freeze_json, load_manifest, sha256
from benchmarks.scoring import pope_metrics
from .run import CONDITIONS, OUTPUT, SUITE, load_config


def paired_interval(values, clusters, iterations=10000):
    values = np.asarray(values, dtype=float)
    unique = sorted(set(clusters))
    totals = np.asarray([values[np.asarray(clusters) == key].sum() for key in unique])
    counts = np.asarray([sum(x == key for x in clusters) for key in unique])
    rng = np.random.default_rng(20261029)
    boot = []
    for _ in range(iterations):
        sample = rng.integers(0, len(unique), len(unique))
        boot.append(float(totals[sample].sum()/counts[sample].sum()))
    return {'mean': float(values.mean()), 'median': float(np.median(values)),
        'p5': float(np.percentile(values, 5)), 'p95': float(np.percentile(values, 95)),
        'paired_image_cluster_bootstrap_95ci': np.percentile(boot, [2.5, 97.5]).tolist(),
        'images': len(unique), 'questions': len(values), 'bootstrap_replicates': iterations}


def mcnemar(b, c):
    n = b+c
    return min(1., 2*sum(math.comb(n, k) for k in range(min(b, c)+1))/2**n) if n else 1.


def metric_value(row, condition):
    metrics = row['conditions'][condition]['metrics']
    return float(next(metrics[k] for k in ('official_generation_correct', 'official_generation_vqa_soft_accuracy',
                                         'mechanism_generation_literal_exact_match') if k in metrics))


def verify_logits(row, archive):
    errors = []
    for condition in CONDITIONS:
        result = row['conditions'][condition]
        for i, candidate in enumerate(result['candidate_scores']):
            x = archive[f'{condition}_candidate_{i}'].astype(np.float64)
            normalizer = x.max(-1)+np.log(np.exp(x-x.max(-1, keepdims=True)).sum(-1))
            value = sum(x[j, token['token_id']]-normalizer[j] for j, token in enumerate(candidate['records']))
            errors.append(abs(float(value)-candidate['logp']))
        for i, tid in enumerate(result['generated_token_ids']):
            x = archive[f'{condition}_generation_{i}'].astype(np.float64)
            logp = x[tid]-x.max()-np.log(np.exp(x-x.max()).sum())
            errors.append(abs(float(logp)-result['generated_token_logp'][i]))
    maximum = max(errors, default=0.)
    if maximum > 1e-10:
        raise AssertionError('Independent raw-logit full-vocabulary LogP check failed')
    return maximum


def analyze(output):
    config = load_config(output)
    binding = digest(config)
    rows = {}
    environments = []
    for shard in range(4):
        folder = output/f'shard{shard}'
        gate = json.loads((folder/'validation.json').read_text())
        expected = sum(j['shard'] == shard for j in config['jobs'])
        if not gate['passed'] or gate['completed'] != expected or gate['config_sha256'] != binding:
            raise ValueError('Incomplete shard; do not produce full-suite conclusions')
        environments.append(json.loads((folder/'environment.json').read_text()))
        for job in [j for j in config['jobs'] if j['shard'] == shard]:
            key = (job['unit'], job['question_id'])
            path = folder/job['unit']/f'{job["question_id"]}.json'
            row = json.loads(path.read_text())
            npz = path.with_suffix('.npz')
            if row['config_sha256'] != binding or row['npz_sha256'] != sha256(npz):
                raise ValueError('Checkpoint integrity failure')
            with np.load(npz) as archive:
                row['independent_logp_max_error'] = verify_logits(row, archive)
            if key in rows:
                raise ValueError('Duplicate question counted')
            rows[key] = row
    if len({e['code_commit'] for e in environments}) != 1 or len({e['model_sha256'] for e in environments}) != 1:
        raise ValueError('Worker code/model disagreement')
    summaries = {}
    flat = []
    for unit in config['manifest_sha256']:
        unit_rows = [r for (name, qid), r in sorted(rows.items()) if name == unit]
        clusters = [r['cluster_id'] for r in unit_rows]
        summaries[unit] = {'questions': len(unit_rows), 'unique_image_sha256': len(set(clusters)), 'conditions': {}}
        native = np.asarray([metric_value(r, 'Full') for r in unit_rows])
        labels = [r['answer'] for r in unit_rows]
        for condition in CONDITIONS:
            current = np.asarray([metric_value(r, condition) for r in unit_rows])
            entry = {'generated_score_mean': float(current.mean()),
                     'generated_score_gain': paired_interval(current-native, clusters),
                     'eos_count': sum(r['conditions'][condition]['eos_observed'] for r in unit_rows),
                     'length_limit_count': sum(r['conditions'][condition]['length_limit_reached'] for r in unit_rows)}
            if unit != 'textvqa':
                b = int(((native == 1) & (current == 0)).sum())
                c = int(((native == 0) & (current == 1)).sum())
                entry.update(full_correct_to_wrong=b, full_wrong_to_correct=c,
                    exact_mcnemar_p_question_level=mcnemar(b, c),
                    mcnemar_caveat='Question-level descriptive only when multiple questions share an image; '
                                   'primary uncertainty is image-cluster paired bootstrap.')
            if unit.startswith('pope_'):
                entry['pope_generated_metrics'] = pope_metrics(
                    [r['conditions'][condition]['generated_text'] for r in unit_rows], labels)
            if unit_rows[0]['conditions'][condition]['candidate_scores']:
                margins = np.asarray([r['conditions'][condition]['metrics']['diagnostic_candidate_margin'] for r in unit_rows])
                full_margin = np.asarray([r['conditions']['Full']['metrics']['diagnostic_candidate_margin'] for r in unit_rows])
                gains = margins-full_margin
                entry.update(candidate_accuracy=float((margins > 0).mean()),
                    margin_gain=paired_interval(gains, clusters), positive_gain_count=int((gains > 0).sum()),
                    gold_absolute_logp_gain=paired_interval([
                        r['conditions'][condition]['metrics']['diagnostic_gold_sequence_logp']-
                        r['conditions']['Full']['metrics']['diagnostic_gold_sequence_logp'] for r in unit_rows], clusters),
                    worst_margin_cases=[{'question_id': unit_rows[i]['question_id'], 'gain': float(gains[i])}
                                        for i in np.argsort(gains)[:10]])
                entry['by_gold_label'] = {}
                for label in sorted(set(labels)):
                    ix = [i for i, v in enumerate(labels) if v == label]
                    entry['by_gold_label'][label] = {'n': len(ix), 'generated_accuracy': float(current[ix].mean()),
                        'candidate_accuracy': float((margins[ix] > 0).mean()), 'mean_margin_gain': float(gains[ix].mean())}
            summaries[unit]['conditions'][condition] = entry
        for row in unit_rows:
            for condition in CONDITIONS:
                r = row['conditions'][condition]
                full = row['conditions']['Full']
                flat.append({'unit': unit, 'question_id': row['question_id'], 'image_sha256': row['image_sha256'],
                    'condition': condition, 'gold': json.dumps(row['answer'], ensure_ascii=False),
                    'generated_text': r['generated_text'], 'generated_score': metric_value(row, condition),
                    'generated_score_gain': metric_value(row, condition)-metric_value(row, 'Full'),
                    'candidate_margin': r['metrics'].get('diagnostic_candidate_margin'),
                    'margin_gain': (r['metrics']['diagnostic_candidate_margin']-full['metrics']['diagnostic_candidate_margin'])
                                   if 'diagnostic_candidate_margin' in r['metrics'] else None,
                    'gold_absolute_logp': r['metrics'].get('diagnostic_gold_sequence_logp')})
    freeze_json(output/'aggregate.json', {'config_sha256': binding, 'complete_questions': len(rows), 'units': summaries,
        'raw_logits_independently_verified': True, 'max_logp_error': max(r['independent_logp_max_error'] for r in rows.values()),
        'code_commit': environments[0]['code_commit'], 'model_sha256': environments[0]['model_sha256']})
    with (output/'per_sample.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plots(output, flat, summaries)
    lines = ['# Full / B4 / C2 跨数据集配对评测', '', f'全部 {len(rows)} 题已完成，三个条件逐题同机配对。',
        '', 'B4：L0–L1 原 Prompt P−1→真实 Question 阻断；C2：L2–L4 原 P−1 MLP 输出置零。',
        '生成仅在首次 Prefill 干预，随后使用修改后前缀的原生 KV Cache；Decode 不重新施加干预。',
        '模型、Prompt、冻结样本和评分器均沿用 Phase 4.0。没有调参或重新采样。', '',
        '| 单元 | 题数 | Full 生成评分 | B4 | C2 |', '|---|---:|---:|---:|---:|']
    for unit, summary in summaries.items():
        values = [100*summary['conditions'][c]['generated_score_mean'] for c in CONDITIONS]
        lines.append(f'| {unit} | {summary["questions"]} | '+ ' | '.join(f'{x:.2f}%' for x in values)+' |')
    lines += ['', '## 解读边界', '',
        '- GQA 三任务为作者 correct_question 子集，生成分数是 literal exact match，不冒称完整官方 GQA 结果。',
        '- POPE 使用真实生成的 Accuracy/Precision/Recall/F1/Yes ratio；候选排序另列，不能混用。',
        '- ScienceQA 使用选项字母生成评分，保留全部原选项；TextVQA 为无 OCR 输入的十答案软评分。',
        '- 所有区间按实际图片 SHA256 聚类配对 Bootstrap，固定 10,000 次。多题同图的 McNemar 仅作题级描述。',
        '- 多数据集、多条件的区间与 p 值未作多重检验修正，不凭单个名义显著结果宣称普遍机制。',
        '- 开放任务不构造错误候选或 Margin；封闭任务保存完整候选全词表 LogP，平局算错。',
        '- 准确率、Margin、绝对 Gold LogP 分别解读；MLP 阻断收益不等于证明增强视觉信息。',
        '- 可选官方 GQA Balanced 仍阻塞，不计入本批，也没有替换数据。', '',
        '详见 `aggregate.json`、`per_sample.csv`、逐题原始 Logits 和各 Worker 审计。', '',
        f'执行代码：`{environments[0]["code_commit"]}`。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def plots(output, rows, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = output/'figures'
    directory.mkdir(exist_ok=True)
    units = list(summaries)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    x = np.arange(len(units))
    for i, condition in enumerate(CONDITIONS):
        axes[0].bar(x+(i-1)*.24, [100*summaries[u]['conditions'][condition]['generated_score_mean'] for u in units],
                    width=.24, label=condition)
    axes[0].set_xticks(x, units, rotation=20, ha='right')
    axes[0].set_ylabel('Generated task score (%)')
    axes[0].legend()
    for i, condition in enumerate(CONDITIONS[1:]):
        data = [[r['margin_gain'] for r in rows if r['unit'] == u and r['condition'] == condition and r['margin_gain'] is not None]
                for u in units]
        active = [(u, d) for u, d in zip(units, data) if d]
        axes[1].boxplot([d for u, d in active], positions=np.arange(len(active))+(i-.5)*.3, widths=.24,
                        patch_artist=True, boxprops=dict(facecolor=['#65a9a1', '#b381af'][i]), showfliers=True)
    labels = [u for u in units if any(r['margin_gain'] is not None and r['unit'] == u for r in rows)]
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=20, ha='right')
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set_ylabel('Candidate Margin gain (all extreme points retained)')
    fig.tight_layout()
    fig.savefig(directory/'generated_scores_and_margin.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUTPUT)
    a = p.parse_args()
    analyze(a.output)


if __name__ == '__main__':
    main()

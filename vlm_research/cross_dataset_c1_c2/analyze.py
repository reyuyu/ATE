"""Complete-coverage C1/C2 comparison with actual-image-cluster paired intervals."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_interventions.analyze import metric_value
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.scoring import pope_metrics
from .run import OUTPUT, LOCAL_HISTORY, history_row, load_config

CONDITIONS = ('Full', 'C1', 'C2')


def clustered_stats(matrix, clusters, iterations=20000):
    values = np.asarray(matrix, dtype=float)
    unique, inverse = np.unique(clusters, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.zeros((len(unique), values.shape[1]))
    np.add.at(sums, inverse, values)
    rng = np.random.default_rng(20261029)
    boot = []
    for start in range(0, iterations, 128):
        indices = rng.integers(0, len(unique), size=(min(128, iterations-start), len(unique)))
        boot.append(sums[indices].sum(1)/counts[indices].sum(1)[:, None])
    samples = np.concatenate(boot)
    result = []
    for column in range(values.shape[1]):
        result.append({'mean': float(values[:, column].mean()),
            'paired_image_cluster_95ci': np.percentile(samples[:, column], [2.5, 97.5]).tolist(),
            'bonferroni8_familywise95ci': np.percentile(samples[:, column], [.3125, 99.6875]).tolist(),
            'images': len(unique), 'questions': len(values), 'bootstrap_replicates': iterations})
    return result


def verify_raw(row, archive):
    maximum = 0.
    for condition, result in row['conditions'].items():
        for i, candidate in enumerate(result['candidate_scores']):
            x = archive[f'{condition}_candidate_{i}'].astype(np.float64)
            z = x.max(-1)+np.log(np.exp(x-x.max(-1, keepdims=True)).sum(-1))
            value = sum(x[j, record['token_id']]-z[j] for j, record in enumerate(candidate['records']))
            maximum = max(maximum, abs(float(value)-candidate['logp']))
        for i, tid in enumerate(result['generated_token_ids']):
            x = archive[f'{condition}_generation_{i}'].astype(np.float64)
            value = x[tid]-x.max()-np.log(np.exp(x-x.max()).sum())
            maximum = max(maximum, abs(float(value)-result['generated_token_logp'][i]))
    if maximum > 1e-10:
        raise AssertionError('Independent saved-logit recomputation failed')
    return maximum


def analyze(output):
    config = load_config(output)
    decision = json.loads((output/'decision.json').read_text())
    binding = digest(config)
    rows, environments = [], []
    max_error = 0.
    for shard in range(4):
        directory = output/f'shard{shard}'
        gate = json.loads((directory/'validation.json').read_text())
        weight = json.loads((directory/'weight_audit.json').read_text())
        jobs = [j for j in config['jobs'] if j['shard'] == shard]
        if not gate['passed'] or not weight['passed'] or gate['completed'] != len(jobs) or gate['config_sha256'] != binding:
            raise ValueError('Incomplete/failed worker: no full-suite conclusions')
        environments.append(json.loads((directory/'environment.json').read_text()))
        for job in jobs:
            path = directory/job['unit']/f'{job["question_id"]}.json'
            row = json.loads(path.read_text())
            if row['config_sha256'] != binding or sha256(path.with_suffix('.npz')) != row['npz_sha256']:
                raise ValueError('Checkpoint binding/hash failed')
            with np.load(path.with_suffix('.npz')) as archive:
                max_error = max(max_error, verify_raw(row, archive))
            if decision['reuse_history']:
                parent = history_row(job)
                if sha256(parent) != row['history_checkpoint_sha256']:
                    raise ValueError('Historical source mismatch')
                historical = json.loads(parent.read_text())
                row['conditions']['Full'] = historical['conditions']['Full']
                row['conditions']['C2'] = historical['conditions']['C2']
            rows.append(row)
    if len(rows) != 2620 or len({(r['unit'], r['question_id']) for r in rows}) != 2620:
        raise ValueError('Coverage/duplicates error')
    if len({e['code_commit'] for e in environments}) != 1 or len({e['model_sha256'] for e in environments}) != 1:
        raise ValueError('Worker code/model differs')
    aggregate, flat = {}, []
    for unit in config['manifest_sha256']:
        group = sorted([r for r in rows if r['unit'] == unit], key=lambda r: r['question_id'])
        clusters = [r['cluster_id'] for r in group]
        score = {c: np.asarray([metric_value(r, c) for r in group]) for c in CONDITIONS}
        names = ['C1_gain', 'C2_gain', 'C1_minus_C2', 'C1_absolute_change', 'C2_absolute_change', 'C1_minus_C2_absolute_change']
        columns = [score['C1']-score['Full'], score['C2']-score['Full'], score['C1']-score['C2'],
                   abs(score['C1']-score['Full']), abs(score['C2']-score['Full']),
                   abs(score['C1']-score['Full'])-abs(score['C2']-score['Full'])]
        closed = bool(group[0]['conditions']['Full']['candidate_scores'])
        if closed:
            for condition in ('C1', 'C2'):
                for label, key in [('margin', 'diagnostic_candidate_margin'), ('gold_logp', 'diagnostic_gold_sequence_logp')]:
                    names.append(condition+'_'+label+'_gain')
                    columns.append(np.asarray([r['conditions'][condition]['metrics'][key]-r['conditions']['Full']['metrics'][key] for r in group]))
        stats = dict(zip(names, clustered_stats(np.stack(columns, axis=1), clusters)))
        summary = {'n': len(group), 'unique_image_sha256': len(set(clusters)), 'statistics': stats, 'conditions': {}}
        for condition in CONDITIONS:
            current, full = score[condition], score['Full']
            entry = {'generated_score': float(current.mean()),
                'score_increased': int((current > full).sum()), 'score_decreased': int((current < full).sum()),
                'score_unchanged': int((current == full).sum()),
                'generated_text_changed': sum(r['conditions'][condition]['generated_text'] != r['conditions']['Full']['generated_text'] for r in group),
                'eos_count': sum(r['conditions'][condition]['eos_observed'] for r in group),
                'length_limit_count': sum(r['conditions'][condition]['length_limit_reached'] for r in group)}
            if unit != 'textvqa':
                entry.update(full_correct_to_wrong=int(((full == 1) & (current == 0)).sum()),
                             full_wrong_to_correct=int(((full == 0) & (current == 1)).sum()))
            if closed:
                entry['candidate_accuracy'] = float(np.mean([r['conditions'][condition]['metrics']['diagnostic_candidate_margin'] > 0 for r in group]))
            if unit.startswith('pope_'):
                entry['pope_metrics'] = pope_metrics([r['conditions'][condition]['generated_text'] for r in group],
                                                   [r['answer'] for r in group])
            summary['conditions'][condition] = entry
        aggregate[unit] = summary
        for row in group:
            for condition in CONDITIONS:
                result = row['conditions'][condition]
                full = row['conditions']['Full']
                flat.append({'unit': unit, 'question_id': row['question_id'], 'image_sha256': row['image_sha256'],
                    'condition': condition, 'generated_text': result['generated_text'], 'generated_score': metric_value(row, condition),
                    'generated_score_gain': metric_value(row, condition)-metric_value(row, 'Full'),
                    'gold': json.dumps(row['answer'], ensure_ascii=False),
                    'candidate_margin': result['metrics'].get('diagnostic_candidate_margin'),
                    'candidate_margin_gain': result['metrics']['diagnostic_candidate_margin']-full['metrics']['diagnostic_candidate_margin'] if closed else None,
                    'gold_absolute_logp': result['metrics'].get('diagnostic_gold_sequence_logp'),
                    'gold_absolute_logp_gain': result['metrics']['diagnostic_gold_sequence_logp']-full['metrics']['diagnostic_gold_sequence_logp'] if closed else None})
    freeze_json(output/'aggregate.json', {'status': 'complete', 'n': len(rows), 'config_sha256': binding,
        'reused_historical_full_c2': decision['reuse_history'], 'history_result_commit': config['history_result_commit'],
        'code_commit': environments[0]['code_commit'], 'all_raw_c1_logits_independently_verified': True,
        'max_float64_logp_recompute_error': max_error, 'units': aggregate})
    with (output/'per_sample.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    report(output, aggregate, environments[0]['code_commit'], decision)


def report(output, aggregate, commit, decision):
    lines = ['# C1 / C2 跨数据集横向对比', '', '完整覆盖2620题，无重采样、训练或层搜索。',
        'C1：L2–L4 原Prompt P−1 的 Attention o_proj 输出置零；C2：同层同位置MLP输出置零。',
        '仅首次Prefill干预，保留原生KV Cache，后续Decode无Hook；候选全序列LogP只干预固定原P−1。',
        f'历史Full/C2复用：{decision["reuse_history"]}。四卡保持原分片与GPU映射；逐卡验证环境、模型哈希和小样本完整Logits一致。', '',
        '| 数据集 | 题数 | Full | C1 | C2 |', '|---|---:|---:|---:|---:|']
    for unit, data in aggregate.items():
        lines.append(f'| {unit} | {data["n"]} | '+' | '.join(f'{data["conditions"][c]["generated_score"]*100:.2f}%' for c in CONDITIONS)+' |')
    lines += ['', '## 相对Full的生成评分变化及配对95%区间', '',
              '| 单元 | C1变化pp [95% CI] | C2变化pp [95% CI] | C1−C2 pp [95% CI] |', '|---|---|---|---|']
    for unit, data in aggregate.items():
        entries = []
        for name in ('C1_gain', 'C2_gain', 'C1_minus_C2'):
            s = data['statistics'][name]
            lo, hi = s['paired_image_cluster_95ci']
            entries.append(f'{s["mean"]*100:+.3f} [{lo*100:+.3f}, {hi*100:+.3f}]')
        lines.append('| '+unit+' | '+' | '.join(entries)+' |')
    lines += ['', '## 纠正、损害与预测稳定性', '',
        '| 单元 | 条件 | 原错→对/软评分上升 | 原对→错/软评分下降 | 生成文本变化 |', '|---|---|---:|---:|---:|']
    for unit, data in aggregate.items():
        for c in ('C1', 'C2'):
            entry = data['conditions'][c]
            lines.append(f'| {unit} | {c} | {entry["score_increased"]} | {entry["score_decreased"]} | {entry["generated_text_changed"]} |')
    lines += ['', 'TextVQA为十答案软评分，上升/下降不能冒称完整纠错/致错；二值任务按实际生成评分翻转。', '',
        '## 候选诊断平均变化（nats）', '',
        '| 单元 | C1 Margin | C2 Margin | C1 Gold LogP | C2 Gold LogP |', '|---|---:|---:|---:|---:|']
    for unit, data in aggregate.items():
        stats = data['statistics']
        if 'C1_margin_gain' not in stats:
            lines.append(f'| {unit} | N/A | N/A | N/A | N/A |')
        else:
            values = [stats[name]['mean'] for name in ('C1_margin_gain', 'C2_margin_gain', 'C1_gold_logp_gain', 'C2_gold_logp_gain')]
            lines.append('| '+unit+' | '+' | '.join(f'{v:+.6f}' for v in values)+' |')
    lines += ['', '## 哪种更新更容易省略', '',
        '以下以生成评分绝对变化的均值衡量“影响大小”，同时保留变化方向和纠正/损害数量，不把获益误报为无影响。',
        '直接C1−C2置信区间为同图配对，另预先规定8单元Bonferroni家族95%区间。', '',
        '| 单元 | C1平均绝对评分变化pp | C2平均绝对评分变化pp | C1−C2生成分差家族95%CI pp |', '|---|---:|---:|---|']
    for unit, data in aggregate.items():
        stats = data['statistics']
        lo, hi = stats['C1_minus_C2']['bonferroni8_familywise95ci']
        lines.append(f'| {unit} | {stats["C1_absolute_change"]["mean"]*100:.3f} | '
                     f'{stats["C2_absolute_change"]["mean"]*100:.3f} | [{lo*100:+.3f}, {hi*100:+.3f}] |')
    lines += ['', '## 验证与解读边界', '',
        '- CPU测试、逐卡原始C1精确等价/无操作/Cache/正常和异常Hook清理检查均通过；未放宽容差。',
        '- 每题核对全部32层P−1之前的隐藏状态哈希，与Full完全一致；非目标Token的局部o_proj输出保持精确一致。',
        '- 每题保存生成答案、完整候选序列LogP及原始全词表Logits；独立NumPy重算新增条件LogP通过。',
        '- 区间固定20,000次图片SHA256聚类配对Bootstrap，种子20261029，不把同图多问当独立图片。',
        '- 95%区间包含零不是等价性或无害性证明；未预设非劣效界值。整体结论不能超出固定模型/Prompt/这些子集。',
        '- GQA是作者correct_question子集的literal exact match，不是完整官方GQA；ScienceQA为字母生成；TextVQA无OCR。',
        '- POPE三策略分别报告Accuracy/Precision/Recall/F1/Yes ratio，完整指标见aggregate.json。',
        '- QueryAttr和TextVQA没有人工构造候选，Margin/Gold候选LogP诊断为N/A。候选排序与生成分数不得混用。',
        '- 不训练、不修改权重、不扫描层、不做额外消融；不修改main。', '',
        f'执行代码Commit：`{commit}`；历史结果Commit：`{config_history_commit()}`。',
        '逐题数据：per_sample.csv；原始结果及审计：shard*/；源码、数据和协议绑定：config.json。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def config_history_commit():
    from .run import HISTORY_COMMIT
    return HISTORY_COMMIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    analyze(parser.parse_args().output)


if __name__ == '__main__':
    main()

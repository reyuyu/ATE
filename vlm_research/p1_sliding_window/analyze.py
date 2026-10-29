"""Descriptive paired curves and top/bottom windows; no extra GPU work."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.dataset_registry import digest, freeze_json
from .hooks import KINDS
from .run import UNITS, verify

LABELS = {'gqa_chooserel': 'ChooseRel (32)', 'scienceqa_img': 'ScienceQA-IMG (16)', 'textvqa': 'TextVQA (16)'}
COLORS = {'A-Skip': '#1676a3', 'M-Skip': '#b44d35', 'Joint-Skip': '#26815d'}


def ordered(entries, descending):
    return sorted(entries, key=lambda r: ((-1 if descending else 1)*r['mean_absolute_change'], r['window_start']))[:3]


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def analyze(output):
    config, samples = verify(output)
    binding = digest(config)
    done = json.loads((output/'completion.json').read_text())
    if not done['passed'] or done['questions'] != 64 or done['interventions'] != 5760 or done['manifest_sha256'] != binding:
        raise ValueError('Incomplete scan: no full conclusions')
    flat, baselines, summaries, ranks = [], [], {}, []
    for unit in UNITS:
        grouped = {(kind, start): [] for kind in KINDS for start in range(30)}
        for sample in samples[unit]:
            directory = output/unit/sample['question_id']
            full = json.loads((directory/'Full.json').read_text())
            if full['manifest_sha256'] != binding or full['image_sha256'] != sample['image_sha256']:
                raise ValueError('Baseline binding mismatch')
            fm = full['metrics']
            baselines.append({'dataset': unit, 'question_id': sample['question_id'], 'gold': json.dumps(sample['answer'], ensure_ascii=False),
                              'full_metrics': json.dumps(fm, ensure_ascii=False), 'image_sha256': sample['image_sha256']})
            for start in range(30):
                for kind in KINDS:
                    row = json.loads((directory/f'{kind}_L{start:02d}.json').read_text())
                    if row['manifest_sha256'] != binding or row['p_minus_1'] != sample['metadata']['input_mapping']['p_minus_1']:
                        raise ValueError('Question/window binding mismatch')
                    if row['window_start'] != start or row['window_end'] != start+2 or row['condition'] != kind:
                        raise ValueError('Window labels changed')
                    current = row['metrics']
                    closed = unit != 'textvqa'
                    key = 'diagnostic_candidate_margin' if closed else 'fixed_full_token_logp'
                    change = current[key]-fm[key]
                    first_changed = current['first_token_id'] != fm['first_token_id']
                    if not closed and current['fixed_full_token_id'] != fm['first_token_id']:
                        raise AssertionError('TextVQA token was retargeted')
                    item = {'dataset': unit, 'question_id': sample['question_id'], 'image_sha256': sample['image_sha256'],
                            'gold': json.dumps(sample['answer'], ensure_ascii=False), 'condition': kind,
                            'window_start': start, 'window_end': start+2, 'p_minus_1': row['p_minus_1'],
                            'full_margin': fm[key] if closed else None, 'margin': current[key] if closed else None,
                            'metric_change': change, 'absolute_change': abs(change),
                            'full_correct': bool(fm['diagnostic_candidate_correct']) if closed else None,
                            'correct': bool(current['diagnostic_candidate_correct']) if closed else None,
                            'correct_to_wrong': bool(fm['diagnostic_candidate_correct'] and not current['diagnostic_candidate_correct']) if closed else None,
                            'wrong_to_correct': bool(not fm['diagnostic_candidate_correct'] and current['diagnostic_candidate_correct']) if closed else None,
                            'full_candidate_logp': json.dumps(fm.get('candidate_scores'), ensure_ascii=False),
                            'candidate_logp': json.dumps(current.get('candidate_scores'), ensure_ascii=False),
                            'full_first_token_id': fm['first_token_id'], 'current_first_token_id': current['first_token_id'],
                            'first_token_changed': first_changed,
                            'fixed_full_token_logp': current.get('fixed_full_token_logp'),
                            'full_fixed_token_logp': fm.get('fixed_full_token_logp'),
                            'window_end_hidden_cosine_distance': row['window_end_hidden_cosine_distance'],
                            'final_pre_norm_hidden_cosine_distance': row['final_pre_norm_hidden_cosine_distance'],
                            'final_normalized_hidden_cosine_distance': row['final_normalized_hidden_cosine_distance']}
                    flat.append(item)
                    grouped[kind, start].append(item)
        series = {}
        for kind in KINDS:
            curve = []
            for start in range(30):
                group = grouped[kind, start]
                end = float(np.mean([r['window_end_hidden_cosine_distance'] for r in group]))
                final = float(np.mean([r['final_pre_norm_hidden_cosine_distance'] for r in group]))
                closed = unit != 'textvqa'
                entry = {'condition': kind, 'window_start': start, 'window_end': start+2, 'n': len(group),
                         'mean_absolute_change': float(np.mean([r['absolute_change'] for r in group])),
                         'mean_signed_change': float(np.mean([r['metric_change'] for r in group])),
                         'correct_to_wrong': sum(r['correct_to_wrong'] for r in group) if closed else None,
                         'wrong_to_correct': sum(r['wrong_to_correct'] for r in group) if closed else None,
                         'full_candidate_accuracy': float(np.mean([r['full_correct'] for r in group])) if closed else None,
                         'candidate_accuracy': float(np.mean([r['correct'] for r in group])) if closed else None,
                         'first_token_changed': sum(r['first_token_changed'] for r in group),
                         'mean_window_end_cosine_distance': end, 'mean_final_pre_norm_cosine_distance': final,
                         'mean_final_normalized_cosine_distance': float(np.mean([r['final_normalized_hidden_cosine_distance'] for r in group])),
                         'final_pre_norm_over_end_mean_drift': final/end if end else None,
                         'samples_with_directional_drift_attenuation': sum(r['final_pre_norm_hidden_cosine_distance'] < r['window_end_hidden_cosine_distance'] for r in group)}
                curve.append(entry)
            series[kind] = {'curve': curve, 'most_sensitive': ordered(curve, True), 'least_sensitive': ordered(curve, False)}
            for label, entries in (('most', series[kind]['most_sensitive']), ('least', series[kind]['least_sensitive'])):
                for rank, entry in enumerate(entries, 1):
                    ranks.append({'dataset': unit, 'ranking': label, 'rank': rank, **entry})
        summaries[unit] = series
    if len(flat) != 5760 or len(baselines) != 64 or len({(r['dataset'], r['question_id'], r['condition'], r['window_start']) for r in flat}) != 5760:
        raise ValueError('Coverage/duplicate error')
    freeze_json(output/'aggregate.json', {'status': 'complete', 'questions': 64, 'interventions': 5760,
                'full_baselines': 64, 'manifest_sha256': binding, 'datasets': summaries})
    write_csv(output/'per_question.csv', flat)
    write_csv(output/'full_baselines.csv', baselines)
    write_csv(output/'top_bottom_windows.csv', ranks)
    all_windows = [{'dataset': unit, **entry} for unit in UNITS for kind in KINDS for entry in summaries[unit][kind]['curve']]
    write_csv(output/'window_summary.csv', all_windows)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelsize': 12, 'axes.titlesize': 14})
    for unit in UNITS:
        fig, ax = plt.subplots(figsize=(9, 4.7), dpi=180)
        for kind, marker in zip(KINDS, ('o', 's', '^')):
            curve = summaries[unit][kind]['curve']
            ax.plot(range(30), [r['mean_absolute_change'] for r in curve], label=kind, color=COLORS[kind],
                    linewidth=2, marker=marker, markersize=4, markevery=3)
        ax.set(xlim=(-.3, 29.3), ylim=(0, None), xlabel='Window start layer (window = Ls to Ls+2)',
               ylabel='Mean absolute LogP change (nats)\n(fixed Full first token)' if unit == 'textvqa' else 'Mean absolute Margin change (nats)',
               title=LABELS[unit]+' | Original prompt P-1 only')
        ax.set_xticks(list(range(0, 30, 3))+[29])
        ax.grid(axis='y', alpha=.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures/f'{unit}.png')
        plt.close(fig)
    lines = ['# GPU0 P-1 Sliding-Window Sensitivity', '',
             '完整复用Residual-only Pilot64题：ChooseRel32（left/right各16）、ScienceQA16、TextVQA16；未重新抽样。',
             'LLaVA-1.5-7B FP16/eager，Transformers4.49.0，GPU0同机Full配对。三层窗口、步长1，L0-L2至L29-L31，30窗口x3干预x64题=5760条结果。',
             '仅原Prompt P-1：A-Skip置零o_proj输出，M-Skip置零MLP输出，Joint-Skip同时置零；其他Token正常计算。无KV Cache、无自由生成、无训练、无额外干预。', '',
             '## 核心曲线', '']
    for unit in UNITS:
        lines += [f'![{LABELS[unit]}](figures/{unit}.png)', '']
    lines += ['## 最敏感/最不敏感窗口', '',
              '以平均绝对变化排序，不是平均有符号增益。表内记号：窗口起始层: 平均绝对变化nats [正确变错/错误变对]；TextVQA方括号为首Token改变题数，不是答案错误翻转。并列按较小起始层优先。', '',
              '| 数据集 | 干预 | 最敏感3个窗口 | 最不敏感3个窗口 |', '|---|---|---|---|']
    for unit in UNITS:
        for kind in KINDS:
            cells = []
            for field in ('most_sensitive', 'least_sensitive'):
                parts = []
                for e in summaries[unit][kind][field]:
                    flips = str(e['first_token_changed']) if unit == 'textvqa' else f'{e["correct_to_wrong"]}/{e["wrong_to_correct"]}'
                    parts.append(f'L{e["window_start"]}-{e["window_end"]}: {e["mean_absolute_change"]:.4f} [{flips}]')
                cells.append('; '.join(parts))
            lines.append(f'| {unit} | {kind} | '+ ' | '.join(cells)+' |')
    lines += ['', '## Attention与MLP相对敏感层', '',
              '以下仅比较同窗口A/M的平均绝对变化，描述差异，不作显著性或普遍必要性结论。', '',
              '| 数据集 | A影响大于M最多的3个窗口（差值nats） | M影响大于A最多的3个窗口（差值nats） |', '|---|---|---|']
    for unit in UNITS:
        a = summaries[unit]['A-Skip']['curve']
        m = summaries[unit]['M-Skip']['curve']
        pairs = [(i, a[i]['mean_absolute_change']-m[i]['mean_absolute_change']) for i in range(30)]
        cells = []
        for sign in (1, -1):
            selected = sorted([(i, sign*v) for i, v in pairs if sign*v > 0], key=lambda x: (-x[1], x[0]))[:3]
            cells.append('; '.join(f'L{i}-{i+2}: {v:.4f}' for i, v in selected) or '无')
        lines.append(f'| {unit} | '+' | '.join(cells)+' |')
    lines += ['', '## Hidden扰动衰减', '',
              '窗口末端与最终L31均用层残差输出（最终RMSNorm之前）与Full同深度比较；另保存RMSNorm后最终Hidden距离。下表选择最终/末端均值比最小的窗口（不含终止于L31的窗口）；比值小于1只表示方向漂移衰减，不证明信息恢复、答案改善或因果补偿机制。', '',
              '| 数据集 | 干预 | 最强衰减窗口 | 末端平均距离 | 最终平均距离 | 最终/末端 |', '|---|---|---|---:|---:|---:|']
    for unit in UNITS:
        for kind in KINDS:
            pool = [e for e in summaries[unit][kind]['curve'] if e['window_start'] < 29 and e['final_pre_norm_over_end_mean_drift'] is not None]
            e = min(pool, key=lambda e: (e['final_pre_norm_over_end_mean_drift'], e['window_start']))
            lines.append(f'| {unit} | {kind} | L{e["window_start"]}-{e["window_end"]} | {e["mean_window_end_cosine_distance"]:.6f} | {e["mean_final_pre_norm_cosine_distance"]:.6f} | {e["final_pre_norm_over_end_mean_drift"]:.3f} |')
    lines += ['', '## 口径与验证', '',
              '- ChooseRel与ScienceQA均以完整候选LogP比较，gold减最强wrong，平局算错；不使用自由生成的短答案匹配，避免上轮ChooseRel输出格式混淆。',
              '- 这些封闭候选全部为单Token；保留原N+1 teacher-forcing布局，Smoke已验证各候选在P-1的全词表Logits精确一致，故同一次前向即可得到全部完整候选LogP；遇多Token候选直接停止，不截断。',
              '- TextVQA固定每题Full首次argmax Token，测其LogP变化和首Token是否改变；不运行完整答案生成，不将这些数值当作VQA软评分或答案正确率。',
              '- 只做三数据集各1题Smoke：No-op=Full，局部非目标输出不变，L2-L4的A/M/Joint与原实现精确一致，Full/Joint与上轮原始Logits精确一致；不放宽容差。',
              '- 完整覆盖64个Full和5760条干预；参数版本不变、Hook恢复。遵从任务约束，未做模型全量权重哈希或额外耗时GPU审计。',
              '- 相邻窗口高度重叠，且仅64题Pilot；窗口排序是描述性结果，不能视为独立重复证据，不作统计显著性/等价性声明。',
              '- 只研究P-1局部更新的必要性；输出置零仍执行对应计算，绝不解释为整层可跳过或实际推理加速。', '',
              '## 文件', '',
              '- `per_question.csv`：5760条逐题变化、候选LogP、翻转、固定Token及Hidden漂移。',
              '- `full_baselines.csv`、`window_summary.csv`、`top_bottom_windows.csv`：原生基线、全部270个汇总点和54个Top/Bottom条目。',
              '- 各题目录JSON/NPZ：原始P-1全词表Logits、窗口末端/最终Hidden、完整候选LogP和冻结配置绑定；NPZ仅本机保留，不上传二进制。',
              '- `manifest.json`、`environment.json`、`cpu_tests.json`、`smoke.json`、`completion.json`：样本与执行版本、环境和最小验证记录。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')

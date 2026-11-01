"""Question-weighted whole-image descriptive intervals; never fit stage boundaries."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .common import OUT, RAW, ROOT, MODELS, TASKS, read, write, sha256, csv_write, source_hashes, verify_sources
from .metrics import metrics

NAMES = dict(llava='LLaVA-1.5-7B', qwen='Qwen3-VL-8B', internvl='InternVL3-8B')
MODEL_ORDER = ['llava', 'qwen', 'internvl']
FIELDS = ['signed_loss', 'absolute_change', 'top1_flip', 'js_divergence']


def bootstrap(values, image_hashes, repetitions=20000, seed=20261101):
    image_hashes = np.array(image_hashes)
    unique, inverse = np.unique(image_hashes, return_inverse=True)
    sums = np.zeros((len(unique), values.shape[1]))
    np.add.at(sums, inverse, values)
    counts = np.bincount(inverse)
    random = np.random.default_rng(seed)
    draws = np.empty((repetitions, values.shape[1]))
    for begin in range(0, repetitions, 128):
        end = min(begin + 128, repetitions)
        weights = random.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), end - begin).astype(np.float64)
        draws[begin:end] = (weights @ sums) / (weights @ counts)[:, None]
    return np.quantile(draws, [.025, .975], axis=0)


def collect():
    protocol = verify_sources()
    assert sha256(OUT / 'model_and_environment_bindings.json') == protocol['model_bindings_sha256']
    frozen_inputs = {}
    for name in MODELS:
        assert sha256(OUT / f'{name}_input_manifest.json') == protocol['input_manifests'][name]
        frozen_inputs[name] = {(s['task'], s['question_id']): s for s in read(OUT / f'{name}_input_manifest.json')['samples']}
    addon = read(OUT / 'internvl_native_addendum.json')
    assert addon['original_protocol_sha256'] == sha256(OUT / 'protocol.json')
    assert addon['qwen_completion_sha256'] == sha256(OUT / 'qwen_completion.json')
    for path, expected in addon['source_sha256'].items():
        assert sha256(path) == expected
    shared = read(OUT / 'shared_3000_manifest.json')['samples']
    for sample in shared:
        assert sha256(sample['image_path']) == sample['image_sha256'], 'Frozen source image changed'
    lookup = {(s['task'], s['question_id']): s for s in shared}
    llava = list(csv.DictReader((OUT / 'llava_per_question_layer.csv').open()))
    for row in llava:
        for key in ['layer', 'input_length', 'image_tokens', 'full_token_id', 'off_token_id', 'top1_flip']:
            row[key] = int(row[key])
        for key in ['relative_depth', 'full_top1_logp', 'off_full_top1_logp', 'signed_loss', 'absolute_change',
                    'js_divergence', 'full_top1_probability', 'full_top1_margin']:
            row[key] = float(row[key])
    data = {'llava': llava}
    coverage = dict(passed=False, expected_questions=3000, expected_per_task=500,
                    same_input_identity=['task', 'question_id', 'image_id', 'image_sha256', 'question', 'answer'], models={})
    assert read(OUT / 'llava_metric_compatibility.json')['passed']
    for name, config in MODELS.items():
        assert read(OUT / f'{name}_completion.json')['complete']
        assert read(OUT / f'{name}_gpu_smoke.json')['passed']
        checkpoint_paths = sorted((OUT / name / 'checkpoints').glob('*.json'))
        assert len(checkpoint_paths) == 3000
        rows, keys, full_vectors, max_errors = [], set(), 0, Counter()
        for number, path in enumerate(checkpoint_paths, 1):
            checkpoint = read(path)
            key = checkpoint['task'], checkpoint['question_id']
            assert key in lookup and key not in keys and checkpoint['complete']
            keys.add(key)
            assert checkpoint['image_sha256'] == lookup[key]['image_sha256']
            assert checkpoint['input_binding'] == frozen_inputs[name][key]['input_binding']
            assert checkpoint['input_length'] == frozen_inputs[name][key]['layout']['input_length']
            assert checkpoint['image_tokens'] == frozen_inputs[name][key]['layout']['image_tokens']
            assert checkpoint['protocol_sha256'] == sha256(OUT / 'protocol.json')
            assert checkpoint['code_commit'] == read(OUT / f'{name}_completion.json')['code_commit']
            assert [r['layer'] for r in checkpoint['rows']] == list(range(config['layers']))
            assert sha256(checkpoint['raw_path']) == checkpoint['raw_sha256']
            raw = load_file(checkpoint['raw_path'])
            full, off = raw['full'], raw['off']
            assert off.shape == (config['layers'], full.numel())
            assert full.dtype == off.dtype == torch.bfloat16
            assert torch.isfinite(full).all() and torch.isfinite(off).all()
            full_vectors += 1
            for old in checkpoint['rows']:
                layer = old['layer']
                new = metrics(full, off[layer])
                for column, value in new.items():
                    error = abs(value - old[column])
                    max_errors[column] = max(error, max_errors[column])
                    assert value == old[column], (name, key, layer, column, error)
                rows.append(dict(model=name, task=key[0], question_id=key[1], image_id=checkpoint['image_id'],
                    image_sha256=checkpoint['image_sha256'], input_length=checkpoint['input_length'],
                    image_tokens=checkpoint['image_tokens'], **old))
            if number % 250 == 0:
                print('POSTVALIDATE', name, number, '/3000', flush=True)
        assert keys == set(lookup)
        csv_write(OUT / f'{name}_per_question_layer.csv', rows)
        data[name] = rows
        coverage['models'][name] = dict(full_vectors=full_vectors, off_vectors=len(rows),
            raw_sha256_and_all_metrics_exact=True, max_replay_errors=dict(max_errors),
            failures_retained=len(list((OUT / name / 'failures').glob('*.json'))))
    base_keys = set(lookup)
    for name, frame in data.items():
        layers = 32 if name == 'llava' else MODELS[name]['layers']
        assert len(frame) == 3000 * layers
        assert len({(r['task'], r['question_id'], r['layer']) for r in frame}) == len(frame)
        pairs = {(r['task'], r['question_id']): r for r in frame}
        assert len(pairs) == 3000 and set(pairs) == base_keys
        assert Counter(r['task'] for r in pairs.values()) == Counter({t: 500 for t in TASKS})
        for item in pairs.values():
            shared_sample = lookup[item['task'], item['question_id']]
            assert item['image_sha256'] == shared_sample['image_sha256']
            assert str(item['image_id']) == shared_sample['image_id']
        coverage['models'].setdefault(name, {})
        coverage['models'][name].update(questions=3000, rows=len(frame), layers=layers,
            per_task={t: 500 for t in TASKS}, no_duplicate_question_layer=True, complete_common_ids=True)
    coverage.update(passed=True, common_complete_questions=3000,
        common_unique_images=len({s['image_sha256'] for s in shared}), no_silent_exclusions=True)
    write(OUT / 'coverage_audit.json', coverage)
    return data


def summarize(data):
    summaries, intervals = [], []
    for name in MODEL_ORDER:
        frame = data[name]
        layers = sorted({r['layer'] for r in frame})
        for group in ['Overall', *TASKS]:
            part = frame if group == 'Overall' else [r for r in frame if r['task'] == group]
            questions = {(r['task'], r['question_id']): r['image_sha256'] for r in part}
            identities = sorted(questions)
            images = [questions[key] for key in identities]
            order = {(t, q): i for i, (t, q) in enumerate(identities)}
            tensor = np.empty((len(questions), len(layers), len(FIELDS)), dtype=np.float64)
            for row in part:
                tensor[order[row['task'], row['question_id']], int(row['layer'])] = [row[f] for f in FIELDS]
            ci = bootstrap(tensor.reshape(len(questions), -1), images).reshape(2, len(layers), len(FIELDS))
            for layer in layers:
                values = tensor[:, layer]
                row = dict(model=name, group=group, layer=int(layer), relative_depth=float(layer / (len(layers) - 1)),
                    n_questions=len(questions), n_image_clusters=len(set(images)))
                for j, field in enumerate(FIELDS):
                    vector = values[:, j]
                    row.update({f'mean_{field}': float(vector.mean()), f'{field}_ci95_low': float(ci[0, layer, j]),
                        f'{field}_ci95_high': float(ci[1, layer, j]), f'median_{field}': float(np.median(vector)),
                        f'p90_{field}': float(np.quantile(vector, .9)), f'p95_{field}': float(np.quantile(vector, .95)),
                        f'std_{field}': float(vector.std(ddof=1))})
                    intervals.append(dict(model=name, group=group, layer=int(layer), metric=field,
                        estimate=float(vector.mean()), low=float(ci[0, layer, j]), high=float(ci[1, layer, j]),
                        seed=20261101, repetitions=20000, cluster='Image SHA256', estimator='Question-weighted'))
                row.update(top1_flip_count=int(values[:, 2].sum()), min_signed_loss=float(values[:, 0].min()),
                           max_signed_loss=float(values[:, 0].max()), positive_loss_count=int((values[:, 0] > 0).sum()),
                           negative_loss_count=int((values[:, 0] < 0).sum()))
                summaries.append(row)
            print('BOOTSTRAP', name, group, 'n=', len(questions), flush=True)
    csv_write(OUT / 'layer_sensitivity_summary.csv', [r for r in summaries if r['group'] == 'Overall'])
    csv_write(OUT / 'task_layer_sensitivity.csv', [r for r in summaries if r['group'] != 'Overall'])
    csv_write(OUT / 'bootstrap_intervals.csv', intervals)
    csv_write(OUT / 'relative_depth_comparison.csv', [r for r in summaries if r['group'] == 'Overall'])
    extremes = []
    for name, frame in data.items():
        for group in TASKS:
            part = [r for r in frame if r['task'] == group]
            extremes.extend(sorted(part, key=lambda r: r['absolute_change'], reverse=True)[:10])
    csv_write(OUT / 'extreme_cases.csv', extremes)
    comparisons = []
    for name in MODEL_ORDER:
        part = [r for r in summaries if r['model'] == name and r['group'] == 'Overall']
        peak = max(part, key=lambda r: r['mean_absolute_change'])
        signed = max(part, key=lambda r: r['mean_signed_loss'])
        candidate = [r for r in part if 2 <= r['layer'] <= 5]
        median_layer = float(np.median([r['mean_absolute_change'] for r in part]))
        comparisons.append(dict(model=name, n_questions=3000, unique_images=len({r['image_sha256'] for r in data[name]}),
            absolute_peak_layer=peak['layer'], absolute_peak_nat=peak['mean_absolute_change'],
            signed_peak_layer=signed['layer'], signed_peak_nat=signed['mean_signed_loss'],
            peak_relative_depth=peak['relative_depth'],
            l2_l5_mean_absolute_change=float(np.mean([r['mean_absolute_change'] for r in candidate])),
            all_layers_mean_absolute_change=float(np.mean([r['mean_absolute_change'] for r in part])),
            median_layer_mean_absolute_change=float(np.median([r['mean_absolute_change'] for r in part])),
            l2_l5_below_median_layer_mean=bool(np.mean([r['mean_absolute_change'] for r in candidate]) < np.median([r['mean_absolute_change'] for r in part])),
            l2_l5_individual_layers_below_median=sum(r['mean_absolute_change'] < median_layer for r in candidate),
            mean_full_top1_probability=float(np.mean([r['full_top1_probability'] for r in data[name] if r['layer'] == 0])),
            mean_full_top1_margin=float(np.mean([r['full_top1_margin'] for r in data[name] if r['layer'] == 0])),
            full_input_length_min=min(r['input_length'] for r in data[name]), full_input_length_max=max(r['input_length'] for r in data[name]),
            image_tokens_min=min(r['image_tokens'] for r in data[name]), image_tokens_max=max(r['image_tokens'] for r in data[name])))
    csv_write(OUT / 'model_comparison.csv', comparisons)
    return summaries, comparisons


def plots(summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.titlesize': 8.5, 'axes.labelsize': 7.5})
    captions = {}
    plot_rows = []
    def save(fig, name, caption):
        fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
        fig.savefig(OUT / (name + '.png'), dpi=400, bbox_inches='tight')
        plt.close(fig)
        captions[name] = caption

    def matrix(name, field):
        part = [r for r in summaries if r['model'] == name and r['group'] != 'Overall']
        layers = sorted({r['layer'] for r in part})
        return np.array([[next(r['mean_' + field] for r in part if r['group'] == task and r['layer'] == layer)
                          for layer in layers] for task in TASKS])

    def heat(ax, name, field, norm=False, labels=True):
        values = matrix(name, field)
        maximum = float(abs(values).max())
        show = values / maximum if norm and maximum else values
        signed = field == 'signed_loss'
        options = dict(cmap='RdBu_r', vmin=-maximum, vmax=maximum) if signed else dict(cmap='viridis', vmin=0, vmax=1 if norm else maximum)
        image = ax.pcolormesh(np.arange(values.shape[1] + 1) - .5, np.arange(7) - .5,
                              show, shading='flat', rasterized=False, antialiased=False, **options)
        ax.set_xlim(-.5, values.shape[1] - .5)
        ax.set_ylim(5.5, -.5)
        ax.set_yticks(range(6), TASKS if labels else [''] * 6)
        last = values.shape[1] - 1
        ticks = sorted(set([0, *range(5, last, 5), last]))
        if len(ticks) > 2 and ticks[-1] - ticks[-2] < 3:
            ticks.pop(-2)
        ax.set_xticks(ticks, ticks)
        ax.set_xlabel('Language decoder layer')
        ax.set_title(NAMES[name] + (f'\nM / max(M); max={maximum:.4g} nat' if norm else ''))
        for task_index, task in enumerate(TASKS):
            for layer in range(values.shape[1]):
                plot_rows.append(dict(model=name, task=task, layer=layer, metric=field,
                    raw_value=float(values[task_index, layer]), displayed_value=float(show[task_index, layer]),
                    normalization='M/max(M) per-model across all task-layer cells' if norm else 'none',
                    panel_max_abs_raw=maximum))
        return image

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    for col, (ax, name) in enumerate(zip(axes, MODEL_ORDER)):
        image = heat(ax, name, 'absolute_change', norm=True, labels=col == 0)
        fig.colorbar(image, ax=ax, shrink=.8, label='Model-normalized mean |change|')
    save(fig, 'figure1_cross_model_heatmap',
        'Cross-Model Layer-wise Attention-Update Sensitivity. Identical 3000 questions, 500/task. '
        'M=mean_question |logP_Full(t*)-logP_OFF(t*)|, t*=each-model Full first argmax. '
        'Each panel independently displays M/max(M), maximum raw nat and own colorbar; not equal absolute scales. '
        'Single-layer original P-1 o_proj OFF only; no Decode or generation Accuracy endpoint.')

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), constrained_layout=True)
    for col, name in enumerate(MODEL_ORDER):
        rows = sorted([r for r in summaries if r['model'] == name and r['group'] == 'Overall'], key=lambda r: r['layer'])
        x = [r['layer'] for r in rows]
        for row, (field, label, factor) in enumerate([('signed_loss', 'Signed loss (nat)', 1),
                                                    ('absolute_change', 'Mean absolute change (nat)', 1),
                                                    ('top1_flip', 'Top-1 flip (%)', 100)]):
            ax = axes[row, col]
            ax.plot(x, [r['mean_' + field] * factor for r in rows], color='#216c8c', lw=1.3)
            ax.fill_between(x, [r[field + '_ci95_low'] * factor for r in rows],
                            [r[field + '_ci95_high'] * factor for r in rows], color='#216c8c', alpha=.18)
            ax.axvspan(2, 5, color='#b55236', alpha=.10)
            ax.axhline(0, color='#777777', lw=.5)
            ax.set_ylabel(label)
            ax.set_xlabel('Language decoder layer')
            ax.set_title(NAMES[name] if row == 0 else '')
            ax.spines[['right', 'top']].set_visible(False)
    save(fig, 'figure2_raw_sensitivity_curves',
        'Real nat sensitivity and Top-1 flips; independent per-model y axes. No smoothing or interpolation. '
        'Shading is ordinary descriptive 95% whole-Image-SHA256 cluster bootstrap (20000, seed 20261101), question-weighted. '
        'Light band marks the fixed L2-L5 candidate, not an independently confirmed safe window.')

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0), constrained_layout=True)
    for col, name in enumerate(MODEL_ORDER):
        for row, field in enumerate(['signed_loss', 'absolute_change', 'top1_flip']):
            image = heat(axes[row, col], name, field, labels=col == 0)
            fig.colorbar(image, ax=axes[row, col], shrink=.75,
                         label='Flip rate' if field == 'top1_flip' else 'Signed nat' if field == 'signed_loss' else 'Absolute nat')
    save(fig, 'figure3_six_task_raw_heatmaps',
        'Six-task signed loss, mean absolute change and flip rate, each model 500/task. '
        'All raw numbers, separate colorbars. Signed red is decreased Full-reference first-token probability; negative values retained. '
        'These metrics are neither Gold cross entropy nor final-answer Accuracy.')

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), constrained_layout=True)
    colors = dict(llava='#216c8c', qwen='#a84061', internvl='#3d875c')
    for col, field in enumerate(['signed_loss', 'absolute_change', 'top1_flip']):
        for name in MODEL_ORDER:
            rows = sorted([r for r in summaries if r['model'] == name and r['group'] == 'Overall'], key=lambda r: r['layer'])
            x = [r['relative_depth'] for r in rows]
            y = np.array([r['mean_' + field] for r in rows])
            axes[0, col].plot(x, y, label=NAMES[name], color=colors[name], lw=1.3)
            scale = float(abs(y).max())
            axes[1, col].plot(x, y / scale if scale else y, label=NAMES[name], color=colors[name], lw=1.3)
        axes[0, col].set_title(field.replace('_', ' ').title())
        axes[0, col].set_ylabel('Rate' if field == 'top1_flip' else 'nat')
        axes[1, col].set_ylabel('Curve / max |curve|')
        for row in range(2):
            axes[row, col].set_xlabel('Relative depth: layer / (layers - 1)')
            axes[row, col].axhline(0, color='#777777', lw=.5)
            axes[row, col].spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=6)
    save(fig, 'figure4_relative_depth',
        'Relative-depth comparison without interpolation: layer/(L-1), denominators 31/35/27. '
        'Upper row raw units; lower row each complete curve divided by its maximum absolute ordinate. '
        'Normalized shape is not evidence of equal absolute sensitivity, stage boundaries or shared causal sink mechanism.')
    csv_write(OUT / 'figure_plot_data.csv', plot_rows)
    csv_write(OUT / 'figure1_plot_data.csv', [r for r in plot_rows if r['normalization'] != 'none'])
    csv_write(OUT / 'figure2_plot_data.csv', [r for r in summaries if r['group'] == 'Overall'])
    csv_write(OUT / 'figure3_plot_data.csv', [r for r in plot_rows if r['normalization'] == 'none'])
    depth_rows = []
    for name in MODEL_ORDER:
        part = [r for r in summaries if r['model'] == name and r['group'] == 'Overall']
        for field in ['signed_loss', 'absolute_change', 'top1_flip']:
            maximum = max(abs(r['mean_' + field]) for r in part)
            for r in part:
                value = r['mean_' + field]
                depth_rows.append(dict(model=name, layer=r['layer'], relative_depth=r['relative_depth'],
                    metric=field, raw_value=value, normalized_value=value / maximum if maximum else value,
                    normalization='raw / max(abs(raw)) across overall curve within each model and metric', scale=maximum))
    csv_write(OUT / 'figure4_plot_data.csv', depth_rows)
    write(OUT / 'figure_captions.json', captions)


def report(summaries, comparisons):
    shared = read(OUT / 'shared_3000_manifest.json')
    lines = ['# 三模型同题 Layer-Attention Sensitivity：3,000题', '',
        '## 首页结论', '',
        '未发现Qwen旧图与其Signed指标原始数据不一致。旧热图展示 Mean Signed Loss，不能当作本轮要求的 Mean Absolute Change；'
        '这属于图表用途/指标区分，不能据此反称旧Signed图数值错误。'
        '旧120题全部原始Logits及4,320条指标已复算，损失方向、目标Token、层号、任务标签正确；GPU回归Logits精确一致。'
        'CPU不同线程的JS末位求和差异，通过8线程重放精确复现历史结果；不据此断言旧进程实际线程数，未放宽GPU正确性容差。', '',
        f'三模型均覆盖相同3,000题、六任务各500题，{shared["unique_images"]}个Image SHA256簇。'
        'LLaVA严格复用并复算历史数据；Qwen和InternVL在物理GPU1顺序执行。', '',
        '| 模型 | 绝对变化峰层 | 峰值nat | Signed峰层 | Signed峰值nat | L2–L5平均绝对变化nat |',
        '|---|---:|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append(f'| {NAMES[row["model"]]} | L{row["absolute_peak_layer"]} | {row["absolute_peak_nat"]:.6f} | L{row["signed_peak_layer"]} | {row["signed_peak_nat"]:.6f} | {row["l2_l5_mean_absolute_change"]:.6f} |')
    if all(c['l2_l5_individual_layers_below_median'] == 4 for c in comparisons):
        lines += ['', '**跨模型复现的是局部早层相对低敏感，而不是共同的中层峰。** '
            '三模型的L2、L3、L4、L5单层平均绝对变化均低于各自全层层均值中位数；'
            '该中位数仅为描述性参照，不是预注册安全阈值。Qwen和InternVL并不复制LLaVA的全局峰位。']
    lines += ['', '本轮只衡量单层、单Query、首Token分布变化；不能单凭此证明四层联合关闭不损害完整答案。'
        '官方Prompt、分词和视觉预算不同，LLaVA为FP16、另两者为BF16，差异不能纯粹归因于架构。'
        '值得将固定L2–L5作为后续独立完整答案验证候选，但本轮没有执行额外生成，也未证明非劣效。', '',
        '![三模型六任务平均绝对变化热图](figure1_cross_model_heatmap.png)', '',
        '主图为每模型独立归一化；各面板标明实际最大nat。真实绝对单位和独立95%区间见Figure 2及CSV。', '',
        '## L2–L5逐层对照', '', '| 模型 | L2 | L3 | L4 | L5 | 全层均值 | 层均值中位数 |', '|---|---:|---:|---:|---:|---:|---:|']
    for comp in comparisons:
        name = comp['model']
        part = [r for r in summaries if r['model'] == name and r['group'] == 'Overall']
        values = [next(r['mean_absolute_change'] for r in part if r['layer'] == layer) for layer in range(2, 6)]
        lines.append('| ' + NAMES[name] + ' | ' + ' | '.join(f'{v:.6f}' for v in values)
            + f' | {comp["all_layers_mean_absolute_change"]:.6f} | {comp["median_layer_mean_absolute_change"]:.6f} |')
    lines += ['', '## 按任务峰位', '', '| 模型 | 任务 | 绝对变化峰层 | Mean Absolute nat | Signed nat（同层） |', '|---|---|---:|---:|---:|']
    for name in MODEL_ORDER:
        for task in TASKS:
            peak = max((r for r in summaries if r['model'] == name and r['group'] == task), key=lambda r: r['mean_absolute_change'])
            lines.append(f'| {NAMES[name]} | {task} | L{peak["layer"]} | {peak["mean_absolute_change"]:.6f} | {peak["mean_signed_loss"]:.6f} |')
    lines += ['', '## 研究问题与解释边界', '']
    for comp in comparisons:
        name = comp['model']
        lines.append(f'- {NAMES[name]}：绝对变化峰L{comp["absolute_peak_layer"]}（相对深度{comp["peak_relative_depth"]:.3f}）。'
            f'L2–L5均值{"低于" if comp["l2_l5_below_median_layer_mean"] else "不低于"}本模型层均值中位数。'
            f'这四层中{comp["l2_l5_individual_layers_below_median"]}/4层各自低于该参照。'
            '这是描述性参照，不是预注册低损害判据。')
    if all(c['l2_l5_below_median_layer_mean'] for c in comparisons):
        lines += ['- 三模型L2–L5平均单层绝对变化均低于各自层均值中位数，支持“局部早层P−1 Attention更新相对低敏感”的描述性跨模型现象。'
                  '英文表述宜用 low sensitivity；若使用 redundancy，必须限定为此单步、单Query、单层干预端点，不暗示已证明多层安全省略。']
    else:
        lines += ['- L2–L5在三个模型中并非一致低于各自层均值中位数，不能宣称该固定早层区域的统一低敏感规律；应保留架构差异。']
    smallest = min(comparisons, key=lambda r: r['all_layers_mean_absolute_change'])
    largest = max(comparisons, key=lambda r: r['all_layers_mean_absolute_change'])
    lines += [f'- 真实量级：全层平均绝对变化最大为{NAMES[largest["model"]]}的{largest["all_layers_mean_absolute_change"]:.6f}nat，'
              f'最小为{NAMES[smallest["model"]]}的{smallest["all_layers_mean_absolute_change"]:.6f}nat；'
              f'比值{largest["all_layers_mean_absolute_change"] / smallest["all_layers_mean_absolute_change"]:.2f}。模型归一化热图不能用于比较这种绝对量级。']
    qwen = next(r for r in comparisons if r['model'] == 'qwen')
    intern = next(r for r in comparisons if r['model'] == 'internvl')
    lines += [f'- Qwen此前末层峰：本轮绝对变化峰L{qwen["absolute_peak_layer"]}、Signed峰L{qwen["signed_peak_layer"]}；'
        f'{"末层总体峰复现" if qwen["absolute_peak_layer"] == qwen["signed_peak_layer"] == 35 else "未完整复现末层双指标总体峰"}。',
        f'- InternVL此前L19峰：本轮绝对变化峰L{intern["absolute_peak_layer"]}、Signed峰L{intern["signed_peak_layer"]}；'
        f'{"L19总体峰复现" if intern["absolute_peak_layer"] == intern["signed_peak_layer"] == 19 else "未完整复现L19双指标总体峰"}。',
        '- 峰位和量级不同应视为架构相关结果；不能为了复现LLaVA而平滑、重分段或删除极端题。极端题保存在extreme_cases.csv。',
        '- 本轮Question ID、Image ID/SHA256及Gold严格相同，但模型官方Prompt、分词、视觉预算和精度不同。'
        'LLaVA为FP16/576视觉Token；Qwen为BF16/动态视觉Token/DeepStack；InternVL为BF16/单448图块/256视觉Token。'
        '因此跨模型差异不能单独归因于网络架构，模型自身Full目标Token和原始预测置信度也不必跨模型相同。'
        '各模型的Full Top-1概率与Margin记录在model_comparison.csv，不能忽略这种基线差异。',
        '- 本轮不新增Attention Mass采集，也不把三个模型的Anchor/BOS解释成相同Token或相同机制。'
        'InternVL历史Start Anchor极低，不能据此构造共享BOS高Mass解释。',
        '- 单层LogP曲线不证明多层联合关闭安全，也不证明最终答案Accuracy、物理跳层、K/V删除或GPU加速。'
        '历史Qwen中层窗口净Accuracy为0但有2损害/2纠正；InternVL L2–L5窗口Accuracy零翻转但有首Token和文本变化，都是端点不等价的实例。',
        '- 是否支持“跨架构局部早层Attention更新低敏感”只能依据上述真实量级作描述性判断。'
        '即使三模型早层相对较低，也不使用本轮探索数据宣称L2–L5非劣效或普遍冗余。',
        '- 如进入下一阶段，应固定L2–L5，用未参与本轮的Image-disjoint完整答案生成和预注册非劣效边界验证；本轮未执行这些额外实验。', '',
        '## 任务差异与跨模型措辞', '']
    for name in MODEL_ORDER:
        peaks = {}
        for task in TASKS:
            peak = max((r for r in summaries if r['model'] == name and r['group'] == task), key=lambda r: r['mean_absolute_change'])
            peaks.setdefault(peak['layer'], []).append(task)
        lines.append('- ' + NAMES[name] + '：' + '；'.join(f'L{layer}为{", ".join(tasks)}的绝对变化峰' for layer, tasks in sorted(peaks.items()))
            + '。这些是探索性峰位，不是经过多重比较校正的任务差异检验。')
    if all(c['l2_l5_individual_layers_below_median'] == 4 for c in comparisons):
        lines += ['', '建议论文表述：**Localized early-layer attention updates show low single-step sensitivity across heterogeneous VLMs, '
            'although global sensitivity profiles differ.** '
            '不将本轮证据直接升级为已证明的redundancy；architecture-related形态差异仍与Prompt、精度和视觉预算等共同变化。']
    lines += ['', '## 原生置信度与实际单位', '',
        '| 模型 | Full Top-1平均概率 | Full平均Top-1/Top-2 Logit差 | 输入Token范围 | 视觉Token范围 |',
        '|---|---:|---:|---:|---:|']
    for comp in comparisons:
        lines.append(f'| {NAMES[comp["model"]]} | {comp["mean_full_top1_probability"]:.6f} | {comp["mean_full_top1_margin"]:.6f} | '
            f'{comp["full_input_length_min"]}–{comp["full_input_length_max"]} | {comp["image_tokens_min"]}–{comp["image_tokens_max"]} |')
    lines += ['', 'Full Top-1 Margin在此明确定义为前两名原始Logit差，而非候选答案Margin或最终Accuracy。'
        '主Loss始终固定各模型自己的Full首Token，不将不同模型的分词Token强行对齐。', '',
        '## 统计及正确性', '',
        '每模型每层：Mean Signed、Mean Absolute、Median、P90/P95、Std、Top-1 Flip和JS。'
        '20,000次整图SHA256簇Bootstrap，seed=20261101，重采样簇后保持题目加权；普通描述性95%区间。'
        '未将不同层、Head或问题时间步当作独立图片，未作未经校正的确认性显著性声明。', '',
        '所有主指标固定自身原生Full首Token argmax，FP32 LogSoftmax；绝对值逐题计算后再平均；负Signed保留。'
        '正式采集每题单独验证视觉重放精确一致，所有层采用独立空Cache。'
        '新增两模型的原始BF16完整词表Logits存资源目录且绑定SHA256；LLaVA复用历史FP16前向后以FP32保存的Logits。'
        '不提交权重或大型张量。', '',
        '实现阶段保留了两次Smoke失败记录，均发生在正式条件采集前：'
        '首次Qwen模型加载处的inference-mode参数版本计数审计已修正；'
        'InternVL首次Smoke历史Full最大差0.0625来自输出头矩阵形状：原生HF生成设置logits_to_keep=1，初版直接Prefill遗漏该设置。'
        '独立InternVL addendum恢复原生设置后，历史Full、No-op、视觉重放Logits最大差全部为0，历史单层Loss精确匹配。'
        '未放宽容差、未修改权重/环境/样本/视觉预算；Qwen已完成的冻结源码及结果未改变。'
        '见internvl_native_addendum.json、protocol_attempt0.json和对应失败日志。', '',
        '![真实单位曲线](figure2_raw_sensitivity_curves.png)', '',
        '![六任务真实单位热图](figure3_six_task_raw_heatmaps.png)', '',
        '![相对深度](figure4_relative_depth.png)', '']
    if (OUT / 'historical_window_context.json').exists():
        context = read(OUT / 'historical_window_context.json')
        lines += ['## 历史L2–L5完整答案端点（仅背景）', '',
                  '| 模型 | 历史题数 | Full Accuracy % | L2–L5 ΔAccuracy pp | 原对→错 | 原错→对 |',
                  '|---|---:|---:|---:|---:|---:|']
        for name in MODEL_ORDER:
            rows = context[name]['rows']
            rows = [r for r in rows if r.get('group', 'Overall') == 'Overall']
            full = next(r for r in rows if r['condition'] == 'Full')
            target = next(r for r in rows if r['condition'] == ('W02' if name == 'llava' else 'W2'))
            accuracy = full.get('accuracy_pct', full.get('accuracy_percent'))
            lines.append(f'| {NAMES[name]} | {full["n"]} | {float(accuracy):.3f} | {float(target["delta_pp"]):+.3f} | '
                         f'{target["correct_to_wrong"]} | {target["wrong_to_correct"]} |')
        lines += ['', '历史样本量、处理和预注册状态不同，不把这些结果并入本轮3,000题单层指标，'
                  '也不把三模型历史窗口当作同题、独立的跨模型非劣效检验。', '']
    for name in ['qwen', 'internvl']:
        completion = read(OUT / f'{name}_completion.json')
        memory = read(OUT / f'{name}_run_memory.json')
        lines.append(f'- {NAMES[name]}：正式执行{completion["seconds"] / 3600:.3f}小时；'
            f'进程显存峰值{memory["peak_process_memory_mib"]}MiB；'
            f'Allocated峰{memory["peak_allocated_mib"]:.1f}MiB；Reserved峰{memory["peak_reserved_mib"]:.1f}MiB。')
    lines += ['', 'GPU1释放审计见gpu1_release.json。本轮至此停止，不追加生成、来源Mask、训练或新数据。', '']
    lines += ['## 图表下载', '',
        '[Figure 1 PDF](figure1_cross_model_heatmap.pdf) / '
        '[Figure 2 PDF](figure2_raw_sensitivity_curves.pdf) / '
        '[Figure 3 PDF](figure3_six_task_raw_heatmaps.pdf) / '
        '[Figure 4 PDF](figure4_relative_depth.pdf)', '',
        '矢量热图单元与曲线使用真实CSV直接绘制；PNG为400 DPI。具体定义见figure_captions.json，'
        '逐图绘图数据为figure1_plot_data.csv至figure4_plot_data.csv。', '']
    (OUT / 'report_zh.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    torch.set_num_threads(1)
    data = collect()
    summaries, comparisons = summarize(data)
    plots(summaries)
    report(summaries, comparisons)
    write(OUT / 'validation.json', dict(passed=True, scope='complete same-question 3000 x three models',
        coverage_sha256=sha256(OUT / 'coverage_audit.json'), qwen_legacy_sha256=sha256(OUT / 'qwen_legacy_validation.json'),
        gpu_smokes={n: sha256(OUT / f'{n}_gpu_smoke.json') for n in MODELS}, no_new_generation=True,
        all_raw_logits_hashes_and_metrics_replayed=True, all_figures_real_data=True,
        gpu_released=read(OUT / 'gpu1_release.json')['released'], analysis_code_sha256=source_hashes()))
    execution = read(OUT / 'execution_summary.json')
    execution.update(status='complete', analysis_passed=True, validation_sha256=sha256(OUT / 'validation.json'),
        common_questions=3000, models=3, total_single_layer_records=288000)
    write(OUT / 'execution_summary.json', execution)
    hashes = {str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'}
    write(OUT / 'artifact_hashes.json', hashes)
    print('COMPLETE: all three models, 3000 matched questions, 288000 single-layer records', flush=True)


if __name__ == '__main__':
    main()

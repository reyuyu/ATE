#!/usr/bin/env python3
"""CPU-only replot of frozen, committed GQA3000 statistics; no inference dependencies.

python -m pip install numpy matplotlib
python vlm_research/analysis/layer_sensitivity_plot.py
Repeat without network: add --offline (uses hash-checked source_inputs cache).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

VERSION = '1.1.0'
COMMIT = '6a0dae9fed134624e732e80984080bfc17aa18b2'
LLAVA_HISTORY = '6eb8328c9652342343395cf19bf7149e7d7f3047'
QWEN_HISTORY = 'dfe0cbacd09de9dfeacff980b911e6aec22a7665'
REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / 'vlm_research/results/layer_sensitivity_comparison_3000'
BASE = 'vlm_research/results/gqa3000_p1_layerwise_accuracy'
ORDER = [('LLaVA', 'Visual Writing'), ('LLaVA', 'P-1 Update'),
         ('Qwen', 'Visual Writing'), ('Qwen', 'P-1 Update')]
COLORS = {'Visual Writing': '#007F86', 'P-1 Update': '#B5483A'}
TASKS = ('ChooseAttr', 'ChooseCat', 'ChooseRel', 'CompareAttr', 'LogicalObj', 'QueryAttr')
TASK_COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#332288')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inputs(offline):
    specs = [(f'{m}_{f.removesuffix(".csv")}', COMMIT, f'{BASE}/{m}/{f}')
             for m in ('llava', 'qwen') for f in
             ('historical_writing_reference.csv', 'layer_paired_comparisons.csv', 'accuracy_summary.csv',
              'task_layer_accuracy.csv')]
    specs += [
        ('llava_original', LLAVA_HISTORY, 'vlm_research/results/gqa3000_visual_writing_necessity/layer_necessity_summary.csv'),
        ('llava_writing_report', LLAVA_HISTORY, 'vlm_research/results/gqa3000_visual_writing_necessity/report_zh.md'),
        ('qwen_original', QWEN_HISTORY, 'vlm_research/results/qwen3vl_visual_writing_necessity_3000/layer_paired_comparisons.csv'),
        ('qwen_writing_baseline', QWEN_HISTORY, 'vlm_research/results/qwen3vl_visual_writing_necessity_3000/accuracy_summary.csv'),
        ('qwen_writing_report', QWEN_HISTORY, 'vlm_research/results/qwen3vl_visual_writing_necessity_3000/report_zh.md'),
        ('p1_report', COMMIT, f'{BASE}/cross_model/report_zh.md'),
    ]
    cache = OUTPUT / 'source_inputs'
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / 'source_manifest.json'
    old = json.loads(manifest_path.read_text('utf-8')) if manifest_path.exists() else {}
    prior = {s['key']: s for s in old.get('sources', [])}

    def obtain(spec):
        key, commit, path = spec
        url = f'https://raw.githubusercontent.com/reyuyu/ATE/{commit}/{path}'
        local = cache / (key + Path(path).suffix)
        if local.exists():
            content = local.read_bytes()
            check(key in prior and sha(content) == prior[key]['sha256'], f'Unverified source cache: {key}')
            check(prior[key]['commit'] == commit and prior[key]['path'] == path, f'Cache provenance: {key}')
        else:
            check(not offline, f'Missing offline input: {key}')
            try:
                content = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=REPO,
                                                  stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read()
            local.write_bytes(content)
        record = {'key': key, 'commit': commit, 'path': path, 'url': url,
                  'sha256': sha(content),
                  'git_blob_sha1': hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest(),
                  'cache': str(local.relative_to(OUTPUT)).replace('\\', '/')}
        return key, content, record

    with ThreadPoolExecutor(max_workers=6) as pool:
        fetched = list(pool.map(obtain, specs))
    manifest = {'repository': 'https://github.com/reyuyu/ATE', 'primary_commit': COMMIT,
                'script_version': VERSION, 'script_sha256': sha(Path(__file__).read_bytes()),
                'python': platform.python_version(), 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
                'sources': [x[2] for x in fetched]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', 'utf-8')
    return {key: content for key, content, _ in fetched}


def table(content):
    return list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))


def overall(row):
    return row.get('subset', row.get('task')) == 'overall'


def write_csv(name, rows):
    with (OUTPUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def convert(source):
    all_rows = []
    baselines = {}
    for model, intervention in ORDER:
        m = model.lower()
        n = 32 if m == 'llava' else 36
        writing = intervention == 'Visual Writing'
        key = f'{m}_historical_writing_reference' if writing else f'{m}_layer_paired_comparisons'
        rows = [r for r in table(source[key]) if overall(r)]
        if writing:
            baseline_table = table(source['qwen_writing_baseline']) if m == 'qwen' else rows
            baseline_name = 'All-Writer'
            prefix = 'Drop-L'
            fields = ('delta_accuracy_ci_low', 'delta_accuracy_ci_high',
                      'delta_accuracy_corrected_low', 'delta_accuracy_corrected_high') if m == 'llava' else (
                      'ci_low', 'ci_high', 'Bonferroni36_low', 'Bonferroni36_high')
        else:
            baseline_table = table(source[f'{m}_accuracy_summary'])
            baseline_name = 'Native Full'
            prefix = 'Drop-P1-L'
            fields = ('ci95_low', 'ci95_high', 'bonferroni95_low', 'bonferroni95_high')
        baseline_rows = [r for r in baseline_table if overall(r) and r['condition'] == baseline_name]
        check(len(baseline_rows) == 1, f'Unique baseline: {model}, {intervention}')
        b = baseline_rows[0]
        baseline = float(b['accuracy_pct'])
        check(int(b['questions']) == 3000 and int(b.get('image_clusters', b.get('images'))) == 2409,
              f'Baseline cohort: {key}')
        expected_base = {('llava', True): 87.03333333333333, ('qwen', True): 87.9,
                         ('llava', False): 87.56666666666667, ('qwen', False): 88.43333333333333}[m, writing]
        check(np.isclose(baseline, expected_base, atol=1e-8, rtol=0), f'Unexpected baseline: {key}')
        baselines[model, intervention] = baseline
        rows = [r for r in rows if re.fullmatch(re.escape(prefix) + r'\d+', r['condition'])]
        rows.sort(key=lambda r: int(r['condition'][len(prefix):]))
        check(len(rows) == n and [int(r['condition'][len(prefix):]) for r in rows] == list(range(n)),
              f'Layer completeness: {key}')
        if writing:
            original = {r['condition']: r for r in table(source[f'{m}_original']) if overall(r)}
        for layer, r in enumerate(rows):
            questions = int(r['questions'])
            clusters = int(r.get('image_clusters', r.get('images')))
            check(questions == 3000 and clusters == 2409, f'Cohort: {key} L{layer}')
            if writing:
                o = original[r['condition']]
                for f in ('accuracy_pct', 'delta_accuracy_pp', 'correct_to_wrong', 'wrong_to_correct', *fields):
                    check(np.isclose(float(r[f]), float(o[f]), rtol=0, atol=1e-10), f'Historical copy mismatch: {key} {f}')
            delta = float(r['delta_accuracy_pp'])
            acc = float(r['accuracy_pct'])
            check(np.isclose(acc - baseline, delta, atol=1e-10, rtol=0), f'Delta sign/baseline: {key}')
            ctw, wtc = int(r['correct_to_wrong']), int(r['wrong_to_correct'])
            check(np.isclose((wtc-ctw)*100/questions, delta, atol=1e-10, rtol=0), f'Flip counts: {key}')
            lo, hi, clo, chi = [float(r[f]) for f in fields]
            check(np.all(np.isfinite([delta, lo, hi, clo, chi])) and clo <= lo <= delta <= hi <= chi,
                  f'CI order: {key} L{layer}')
            s = -delta
            all_rows.append(dict(model=model, intervention=intervention, layer=layer, num_layers=n,
                relative_depth=layer/(n-1), baseline_name=baseline_name, baseline_accuracy_pct=baseline,
                ablated_accuracy_pct=acc, signed_sensitivity_pp=s, drop_only_sensitivity_pp=max(0., s),
                signed_ci95_low=-hi, signed_ci95_high=-lo,
                signed_corrected_ci_low=-chi, signed_corrected_ci_high=-clo,
                significant_accuracy_drop=-chi > 0, significant_accuracy_gain=-clo < 0,
                correct_to_wrong=ctw, wrong_to_correct=wtc, questions=questions, image_clusters=clusters,
                source_commit=COMMIT, source_csv=f'{BASE}/{m}/'+('historical_writing_reference.csv' if writing else 'layer_paired_comparisons.csv')))
    check(len(all_rows) == 136, 'Expected 136 rows')
    expected_peaks = [(11, 2.6), (15, 25/30), (18, 35/30), (24, 1.4)]
    summaries = []
    for (model, intervention), (peak_layer, peak_value) in zip(ORDER, expected_peaks):
        rows = [r for r in all_rows if (r['model'], r['intervention']) == (model, intervention)]
        peak = max(rows, key=lambda r: r['signed_sensitivity_pp'])
        check(peak['layer'] == peak_layer and np.isclose(peak['signed_sensitivity_pp'], peak_value, atol=1e-10),
              f'Peak: {model} {intervention}')
        drops = [r['layer'] for r in rows if r['significant_accuracy_drop']]
        gains = [r['layer'] for r in rows if r['significant_accuracy_gain']]
        if intervention == 'P-1 Update':
            check(drops == ([26] if model == 'LLaVA' else [24, 27, 34]), f'P-1 significant losses: {model}')
        if (model, intervention) == ('Qwen', 'Visual Writing'):
            check(drops == [8, 10, 11, 18], 'Qwen writing historical report agreement')
        summaries.append(dict(model=model, intervention=intervention,
            baseline_name=rows[0]['baseline_name'], baseline_accuracy_pct=rows[0]['baseline_accuracy_pct'],
            peak_layer=peak['layer'], peak_relative_depth=peak['relative_depth'],
            maximum_drop_pp=peak['signed_sensitivity_pp'],
            significant_drop_layers=';'.join(map(str, drops)), significant_gain_layers=';'.join(map(str, gains)),
            positive_layers=sum(r['signed_sensitivity_pp'] > 0 for r in rows),
            negative_layers=sum(r['signed_sensitivity_pp'] < 0 for r in rows),
            zero_layers=sum(r['signed_sensitivity_pp'] == 0 for r in rows),
            p1_minus_writing_peak_layers=4 if model == 'LLaVA' else 6))
    last = all_rows[-1]
    check(last['model'] == 'Qwen' and last['layer'] == 35 and last['significant_accuracy_gain']
          and np.isclose(last['signed_sensitivity_pp'], -19/30), 'Qwen P-1 L35 signed gain')
    return all_rows, summaries


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
        'ytick.labelsize': 8, 'legend.fontsize': 7.5, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6, 'grid.linewidth': .4,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 400})


def save(fig, stem):
    fig.savefig(OUTPUT / (stem+'.pdf'), metadata={'Creator': f'ATE layer_sensitivity_plot.py {VERSION}'})
    fig.savefig(OUTPUT / (stem+'.png'), dpi=400)
    plt.close(fig)


def subset(rows, model, intervention):
    return [r for r in rows if r['model'] == model and r['intervention'] == intervention]


def plot_line(ax, rows, intervention, annotate=True):
    x = np.array([r['layer'] for r in rows])
    y = np.array([r['signed_sensitivity_pp'] for r in rows])
    color = COLORS[intervention]
    ax.fill_between(x, [r['signed_ci95_low'] for r in rows], [r['signed_ci95_high'] for r in rows],
                    color=color, alpha=.13, linewidth=0)
    ax.plot(x, y, color=color, linewidth=1.2, marker='o' if intervention == 'Visual Writing' else 's',
            markersize=2.4, linestyle='-' if intervention == 'Visual Writing' else '--', label=intervention)
    for r in rows:
        if r['significant_accuracy_drop'] or r['significant_accuracy_gain']:
            if r['significant_accuracy_gain']:
                ax.scatter([r['layer']], [r['signed_sensitivity_pp']], marker='D', s=14,
                           facecolors='white', edgecolors='#2d69a3', linewidths=.8, zorder=5)
                ax.annotate('\u2020', (r['layer'], r['signed_sensitivity_pp']), xytext=(-8,-4),
                            textcoords='offset points', color='#2d69a3', fontsize=10, ha='center', va='center')
            else:
                ax.text(r['layer'], r['signed_sensitivity_pp']+.12, '*', color=color,
                        fontsize=10, ha='center', va='center')
    if annotate:
        peak = rows[int(np.argmax(y))]
        ax.annotate(f"L{peak['layer']}: +{peak['signed_sensitivity_pp']:.2f}",
                    (peak['layer'], peak['signed_sensitivity_pp']), xytext=(7, 15), textcoords='offset points',
                    fontsize=7.5, color=color, ha='left')


def curves(rows):
    ymin = min(r['signed_ci95_low'] for r in rows)-.30
    ymax = max(r['signed_ci95_high'] for r in rows)+.45
    fig, axes = plt.subplots(2, 2, figsize=(6.875, 4.7), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.16, top=.89, hspace=.61, wspace=.18)
    for ax, (model, intervention), label in zip(axes.flat, ORDER, 'abcd'):
        rs = subset(rows, model, intervention)
        plot_line(ax, rs, intervention)
        ax.set_title(f'({label}) {model} | '+('Visual Writing' if intervention == 'Visual Writing' else 'P\u22121 Attention Update'), loc='left', fontweight='bold', pad=22)
        ax.text(0, 1.015, f"{rs[0]['baseline_name']}: {rs[0]['baseline_accuracy_pct']:.4f}%", transform=ax.transAxes,
                va='bottom', fontsize=7.5, color='#555555')
        ax.axhline(0, color='#666666', linewidth=.65)
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(np.arange(np.ceil(ymin), np.floor(ymax)+1))
        ax.set_xlim(-.6, rs[-1]['layer']+.6)
        ax.set_xticks(sorted(set([0, 8, 16, 24, rs[-1]['layer']])))
        ax.grid(axis='y', color='#e5e5e5'); ax.set_axisbelow(True)
        ax.set_xlabel('Layer index')
    fig.supylabel('Signed Accuracy-Drop Sensitivity (pp)', x=.015, fontsize=9)
    fig.text(.5, .045, 'Shading: saved pointwise 95% CI   |   * corrected drop   \u2020 corrected gain', ha='center', fontsize=8)
    fig.text(.5, .012, 'Bonferroni families remain separate: 32 / 36 layers per model and intervention.', ha='center', fontsize=7.5)
    save(fig, 'figure_A_layer_sensitivity_curves')
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 3.25), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, top=.83, bottom=.30, wspace=.20)
    for ax, model in zip(axes, ('LLaVA', 'Qwen')):
        for intervention in COLORS:
            rs = subset(rows, model, intervention)
            plot_line(ax, rs, intervention, annotate=False)
            peak = max(rs, key=lambda r:r['signed_sensitivity_pp'])
            ax.annotate(f"L{peak['layer']}", (peak['layer'], peak['signed_sensitivity_pp']),
                        xytext=(3, 13), textcoords='offset points', color=COLORS[intervention], fontsize=8)
        ax.set_title(model, fontweight='bold', loc='left', pad=22)
        bs = [subset(rows, model, i)[0]['baseline_accuracy_pct'] for i in COLORS]
        ax.text(0, 1.01, f'All-Writer {bs[0]:.4f}% | Native Full {bs[1]:.4f}%', transform=ax.transAxes, fontsize=7)
        ax.set_ylim(ymin, ymax); ax.set_yticks(np.arange(np.ceil(ymin), np.floor(ymax)+1))
        ax.axhline(0, color='#666666', linewidth=.65)
        n = 32 if model == 'LLaVA' else 36
        ax.set_xlim(-.6, n-.4); ax.set_xticks([0, 8, 16, 24, n-1]); ax.set_xlabel('Layer index')
        ax.grid(axis='y', color='#e5e5e5'); ax.set_axisbelow(True)
    fig.supylabel('Signed sensitivity (pp)', x=.015, fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.10),
               ncol=2, frameon=False)
    fig.text(.5,.045,'Saved 95% CI; * corrected drop; \u2020 corrected gain. Baselines differ by intervention.',ha='center',fontsize=7.5)
    save(fig, 'figure_A2_model_overlay_curves')


def heatmap(rows, signed):
    matrix = np.full((4,36), np.nan)
    key = 'signed_sensitivity_pp' if signed else 'drop_only_sensitivity_pp'
    for ri, (model, intervention) in enumerate(ORDER):
        for r in subset(rows, model, intervention): matrix[ri,r['layer']] = r[key]
    vmax = max(abs(r['signed_sensitivity_pp']) for r in rows)
    cmap = plt.get_cmap('RdBu_r' if signed else 'YlOrRd').copy()
    cmap.set_bad('#d4d4d4')
    fig, ax = plt.subplots(figsize=(6.875, 2.65))
    fig.subplots_adjust(left=.247, right=.985, top=.77, bottom=.40)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if signed else plt.Normalize(0,vmax)
    # pcolormesh emits vector cells; the NaN cells use the explicit gray N/A color.
    mesh = ax.pcolormesh(np.arange(37)-.5, np.arange(5)-.5, np.ma.masked_invalid(matrix),
                         cmap=cmap, norm=norm, edgecolors='none', rasterized=False)
    ax.set_facecolor('#d4d4d4'); ax.set_ylim(3.5,-.5)
    ax.set_yticks(range(4), ['LLaVA \u2014 Visual Writing','LLaVA \u2014 P\u22121 Update',
                              'Qwen \u2014 Visual Writing','Qwen \u2014 P\u22121 Update'])
    ticks=list(range(0,36,2))+[31,35]
    ticks=sorted(set(ticks))
    ax.set_xticks(ticks, [f'L{x}' for x in ticks], rotation=60, ha='right')
    ax.tick_params(axis='both', length=0)
    for ri, (model, intervention) in enumerate(ORDER):
        for r in subset(rows, model, intervention):
            symbol = '*' if r['significant_accuracy_drop'] else ('\u2020' if signed and r['significant_accuracy_gain'] else '')
            if symbol:
                rgba=cmap(norm(r[key])); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                ax.text(r['layer'],ri,symbol,ha='center',va='center',fontsize=9,
                        color='white' if lum<.48 else '#202020')
    ax.text(33.5,.5,'N/A',ha='center',va='center',fontsize=8,color='#555555')
    ax.plot([31.5,31.5],[-.5,1.5],color='white',lw=.7)
    for y in (.5,1.5,2.5): ax.axhline(y,color='white',lw=.6)
    for spine in ax.spines.values(): spine.set_visible(False)
    title='Layer-wise Signed Sensitivity' if signed else 'Layer-wise Accuracy-Drop Sensitivity'
    fig.suptitle(title, y=.98, fontsize=10, fontweight='bold')
    subtitle='Values = Baseline ACC \u2212 Ablated ACC; positive = drop, negative = gain.' if signed else 'Values = max(0, Baseline ACC \u2212 Ablated ACC), in percentage points.'
    fig.text(.5,.86,subtitle,ha='center',fontsize=7.5)
    cax=fig.add_axes([.43,.20,.37,.045])
    cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=([-vmax,0,vmax] if signed else [0,1,2,vmax]))
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor('face')  # prevent PDF viewer hairlines between vector colorbar cells
    cb.ax.tick_params(labelsize=7.5,length=2); cb.outline.set_linewidth(.4)
    cb.set_label('Signed sensitivity (pp)' if signed else 'Accuracy drop (pp)',fontsize=8,labelpad=1)
    legend='* Bonferroni-corrected drop'+('   \u2020 Bonferroni-corrected gain' if signed else '')+'   |   Gray: nonexistent layers'
    fig.text(.5,.015,legend,ha='center',fontsize=7)
    save(fig, 'figure_C_signed_sensitivity_heatmap' if signed else 'figure_B_accuracy_drop_heatmap')


def report(rows, summaries):
    lines=['# 跨模型层敏感性：仅使用 GitHub 历史结果', '',
        f'主数据冻结 commit：`{COMMIT}`。全部 136 行通过校验；四组均为 3000 题、2409 个图片 SHA256 簇。', '',
        '## 定义与基线', '',
        'Visual Writing 衡量在既有 All-Writer 协议内关闭单层视觉写入后的完整答案 Accuracy 变化。'
        'P−1 Update 衡量在原始 Prompt 最后位置单层整个 Attention o_proj 输出更新置零后的变化；后续 Decode 原生。'
        '它不是仅屏蔽 P−1→Image Key，也不是纯视觉读取敏感性。', '',
        '统一 Signed S = ACC_base − ACC_ablated = −delta_accuracy_pp，单位 pp；'
        'Drop-only = max(0, S)。正值为掉分，负值为涨分。普通与校正区间均由 [low,high] 转为 [−high,−low]，没有新估计 CI 或 p 值。', '',
        'All-Writer 的写入隔离协议与 Native Full 的原生推理不同，必须分别配对，不能混用基线。', '',
        '| 模型 | 干预 | 基线 | 基线 ACC (%) | 最大损害层 | S (pp) | 校正后掉分层 | 校正后涨分层 |',
        '|---|---|---|---:|---|---:|---|---|']
    for s in summaries:
        layers=lambda key: ', '.join('L'+x for x in s[key].split(';') if x) or '无'
        lines.append(f"| {s['model']} | {s['intervention']} | {s['baseline_name']} | {s['baseline_accuracy_pct']:.6f} | L{s['peak_layer']} | +{s['maximum_drop_pp']:.4f} | {layers('significant_drop_layers')} | {layers('significant_gain_layers')} |")
    last=rows[-1]
    lines += ['', '## 数值和历史核验', '',
        'LLaVA Writing / P−1 峰分别为 L11 / L15（相差 4 层）；Qwen 分别为 L18 / L24（相差 6 层）。'
        '峰是最大点估计，不自动等于校正后显著层。例如 LLaVA P−1 峰在 L15，校正后掉分证据只在 L26。', '',
        f"Qwen P−1 L35：S={last['signed_sensitivity_pp']:.6f} pp；校正后区间"
        f"[{last['signed_corrected_ci_low']:.6f}, {last['signed_corrected_ci_high']:.6f}] pp，"
        f"原对→错 {last['correct_to_wrong']}，原错→对 {last['wrong_to_correct']}。区间全负，因此标记为显著 Accuracy 增益，绝不取绝对值。", '',
        'Writing 历史 CSV 副本与原始冻结 CSV 的 Accuracy、差值、全部区间及翻转计数逐字段核对一致。'
        'Qwen Writing 显著掉分层与历史报告的 L8、L10、L11、L18 一致。'
        'LLaVA Writing 历史报告明确分列 NLL 和 Accuracy，但文字列出的显著层属于 NLL；本报告只使用 Accuracy 校正区间自动判断，未挪用 NLL 显著性。'
        '两模型 P−1 显著掉分层与历史跨模型报告一致。', '',
        '历史置信区间采用图片 SHA256 整簇配对 Bootstrap（20,000 次），问题等权；'
        'Bonferroni 家族按原始模型和干预保持 32/36 层分别校正，不合并四组为新的家族。有限重采样使极端尾部分位数存在 Monte Carlo 误差。', '',
        '## 图的读法与限制', '',
        'Figure A：2×2 曲线，全部使用同一纵轴范围；浅色带为普通 95% CI，星号和 dagger 分别依据历史 Bonferroni 区间表示掉分和涨分。'
        'Figure A2：同模型叠加两类干预，线型和颜色同时区分，明确注明各自基线。', '',
        'Figure B：四行共用绝对色标；仅此主图使用 Drop-only。Figure C：保留全部负值，色标以 0 为中心且正负范围对称。'
        '热图实际层号为 L0–L35，LLaVA 的 L32–L35 是 NaN 并显示灰色 N/A，不是 0。热图使用矢量网格，PDF 坐标轴、线条、文字保留矢量。', '',
        '两类敏感性峰不重合，支持“在各自协议和基线下，完整答案 Accuracy 对两类干预的层依赖不同”的描述。'
        '不证明存在两个独立神经模块，不将模型差异单独归因于架构；原始 Prompt、精度、视觉布局等也可能不同。', '',
        'Drop-only 为 0 或 CI 跨 0 不证明没有逐题损害或等价；对→错和错→对可能抵消，统一 CSV 保留两种计数。'
        '单层低敏感不证明联合关闭安全。Hard Mask 和 o_proj 输出旁路均不是物理加速实验，也不证明层可删除。'
        '本图比较完整答案 Accuracy，不是 Gold NLL 或首 Token 概率。', '',
        '## CPU 复现', '', '```bash', 'python -m pip install numpy matplotlib',
        'python vlm_research/analysis/layer_sensitivity_plot.py',
        '# 缓存齐全后完全离线：', 'python vlm_research/analysis/layer_sensitivity_plot.py --offline', '```', '',
        '源码只导入标准库、NumPy、Matplotlib；不连接服务器，不加载模型，不使用 GPU，不读取服务器 logits、checkpoint 或 NPZ。'
        '所有输入均为 GitHub 已提交 CSV/报告；source_manifest.json 保存固定 commit、原始 SHA256、Git blob SHA1、脚本版本及 SHA256。'
        'source_inputs 保留原始字节供离线复现；历史实验文件不改写。', '']
    (OUTPUT/'report_zh.md').write_text('\n'.join(lines),'utf-8')


def convert_tasks(source, overall_rows):
    """Keep native per-task baselines and saved CIs; do not assign significance labels."""
    converted, baselines = [], []
    for model, n in (('LLaVA', 32), ('Qwen', 36)):
        raw = table(source[f'{model.lower()}_task_layer_accuracy'])
        check({r['task'] for r in raw} == set(TASKS), f'Six tasks: {model}')
        for task in TASKS:
            group = [r for r in raw if r['task'] == task]
            bs = [r for r in group if r['condition'] == 'Native Full']
            check(len(bs) == 1 and int(bs[0]['questions']) == 500, f'Task baseline: {model} {task}')
            b = bs[0]; baseline = float(b['accuracy_pct'])
            baselines.append(dict(model=model, task=task, baseline_name='Native Full',
                                  baseline_accuracy_pct=baseline, correct=int(b['correct']),
                                  questions=500, image_clusters=int(b['images'])))
            drops = [r for r in group if r['condition'].startswith('Drop-P1-L')]
            drops.sort(key=lambda r:int(r['layer']))
            check(len(drops) == n and [int(r['layer']) for r in drops] == list(range(n)),
                  f'Task layers: {model} {task}')
            for layer, r in enumerate(drops):
                acc, delta = float(r['accuracy_pct']), float(r['delta_accuracy_pp'])
                ctw, wtc = int(r['correct_to_wrong']), int(r['wrong_to_correct'])
                check(int(r['questions']) == 500 and int(r['images']) == int(b['images']), 'Task cohort')
                check(np.isclose(acc-baseline, delta, atol=1e-10, rtol=0), 'Task baseline/delta identity')
                check(np.isclose((wtc-ctw)/5, delta, atol=1e-10, rtol=0), 'Task flip identity')
                lo, hi, clo, chi = [float(r[k]) for k in ('ci95_low','ci95_high','bonferroni95_low','bonferroni95_high')]
                check(np.all(np.isfinite([acc,delta,lo,hi,clo,chi])) and clo<=lo<=delta<=hi<=chi, 'Task CI order')
                converted.append(dict(model=model,task=task,intervention='P-1 Attention Update',layer=layer,
                    num_layers=n,relative_depth=layer/(n-1),baseline_name='Native Full',baseline_accuracy_pct=baseline,
                    ablated_accuracy_pct=acc,signed_sensitivity_pp=-delta,drop_only_sensitivity_pp=max(0.,-delta),
                    signed_ci95_low=-hi,signed_ci95_high=-lo,signed_corrected_ci_low=-chi,signed_corrected_ci_high=-clo,
                    correct_to_wrong=ctw,wrong_to_correct=wtc,questions=500,image_clusters=int(r['images']),
                    interpretation='descriptive task analysis; no confirmatory significance labels',
                    source_commit=COMMIT,source_csv=f'{BASE}/{model.lower()}/task_layer_accuracy.csv'))
        for layer in range(n):
            task_rows=[r for r in converted if r['model']==model and r['layer']==layer]
            ov=next(r for r in overall_rows if r['model']==model and r['intervention']=='P-1 Update' and r['layer']==layer)
            check(np.isclose(np.mean([r['signed_sensitivity_pp'] for r in task_rows]),ov['signed_sensitivity_pp'],atol=1e-10),
                  'Task mean equals overall (six equal-size tasks)')
            check(sum(r['correct_to_wrong'] for r in task_rows)==ov['correct_to_wrong'] and
                  sum(r['wrong_to_correct'] for r in task_rows)==ov['wrong_to_correct'], 'Task flip sum equals overall')
    check(len(converted)==408 and len(baselines)==12,'408 task-layer rows and 12 baselines')
    write_csv('p1_task_sensitivity_unified.csv',converted)
    write_csv('p1_task_baselines.csv',baselines)
    return converted


def task_figures(rows):
    low=min(r['signed_sensitivity_pp'] for r in rows)
    high=max(r['signed_sensitivity_pp'] for r in rows)
    pad=(high-low)*.13
    fig, axes=plt.subplots(2,1,figsize=(6.875,4.95),sharey=True)
    fig.subplots_adjust(left=.11,right=.98,top=.81,bottom=.14,hspace=.48)
    for ax,model,label in zip(axes,('LLaVA','Qwen'),'ab'):
        for ti,(task,color) in enumerate(zip(TASKS,TASK_COLORS)):
            rs=[r for r in rows if r['model']==model and r['task']==task]
            ax.plot([r['layer'] for r in rs],[r['signed_sensitivity_pp'] for r in rs],
                    color=color,lw=1.15,marker=('o','s','^','D','v','P')[ti],markersize=2.5,
                    linestyle=('-','--','-.',':','-','--')[ti],label=task)
        ax.set_title(f'({label}) {model}: P\u22121 Attention Update',loc='left',fontweight='bold',pad=7)
        ax.axhline(0,color='#666666',lw=.65)
        ax.set_ylim(low-pad,high+pad); ax.set_xlim(-.5,31.5 if model=='LLaVA' else 35.5)
        ax.set_xticks([0,4,8,12,16,20,24,28,31] if model=='LLaVA' else [0,4,8,12,16,20,24,28,32,35])
        ax.grid(axis='y',color='#e5e5e5');ax.set_axisbelow(True);ax.set_xlabel('Layer index')
    fig.supylabel('Signed accuracy-drop sensitivity (pp)',x=.012,fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.99),
               ncol=3,frameon=False,columnspacing=1.5,handlelength=2.5)
    fig.text(.5,.045,'Task-specific Native Full baseline; 500 questions per task. Descriptive results.',ha='center',fontsize=7.5)
    save(fig,'figure_D_p1_task_sensitivity_curves')
    fig,axes=plt.subplots(2,1,figsize=(6.875,4.25),sharex=True)
    fig.subplots_adjust(left=.165,right=.98,top=.88,bottom=.25,hspace=.37)
    vmax=max(abs(low),abs(high));norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d4d4d4')
    for ax,model,label in zip(axes,('LLaVA','Qwen'),'ab'):
        matrix=np.full((6,36),np.nan)
        for r in rows:
            if r['model']==model:matrix[TASKS.index(r['task']),r['layer']]=r['signed_sensitivity_pp']
        mesh=ax.pcolormesh(np.arange(37)-.5,np.arange(7)-.5,np.ma.masked_invalid(matrix),
                           cmap=cmap,norm=norm,edgecolors='none',rasterized=False)
        ax.set_facecolor('#d4d4d4');ax.set_ylim(5.5,-.5);ax.set_yticks(range(6),TASKS)
        ax.tick_params(axis='both',length=0);ax.set_title(f'({label}) {model}',loc='left',fontweight='bold',pad=5)
        if model=='LLaVA':
            ax.text(33.5,2.5,'N/A',ha='center',va='center',color='#555555',fontsize=8)
            ax.plot([31.5,31.5],[-.5,5.5],color='white',lw=.7)
        for y in np.arange(.5,5.5):ax.axhline(y,color='white',lw=.4)
        for spine in ax.spines.values():spine.set_visible(False)
    ticks=sorted(set(list(range(0,36,2))+[31,35]));axes[1].set_xticks(ticks,[f'L{x}' for x in ticks],rotation=55,ha='right')
    fig.suptitle('P\u22121 Attention Update: Task-wise Sensitivity',y=.98,fontsize=10,fontweight='bold')
    cax=fig.add_axes([.36,.13,.44,.032]);cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[-vmax,0,vmax])
    cb.solids.set_rasterized(False);cb.solids.set_edgecolor('face');cb.outline.set_linewidth(.4)
    cb.ax.tick_params(labelsize=7.5,length=2);cb.set_label('Native Full ACC \u2212 Ablated ACC (pp)',fontsize=8,labelpad=2)
    fig.text(.5,.015,'Shared signed scale; positive = drop, negative = gain. Task effects are descriptive.',ha='center',fontsize=7)
    save(fig,'figure_E_p1_task_sensitivity_heatmap')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    try:
        source=inputs(args.offline)
        rows,summaries=convert(source)
        task_rows=convert_tasks(source,rows)
    except Exception as exc:
        (OUTPUT/'validation.json').write_text(json.dumps({'passed':False,'error':str(exc)},indent=2)+'\n','utf-8')
        raise
    write_csv('layer_sensitivity_unified.csv',rows)
    write_csv('layer_sensitivity_summary.csv',summaries)
    style(); curves(rows); heatmap(rows,False); heatmap(rows,True); report(rows,summaries)
    task_figures(task_rows)
    with (OUTPUT/'report_zh.md').open('a',encoding='utf-8') as stream:
        stream.write('\n## P−1 六任务补充图\n\n'
            'Figure D 将六类 GQA 的 Signed Accuracy-drop Sensitivity 叠加在同一坐标系，LLaVA/Qwen 各一面板，'
            '颜色、线型和点形区分任务，共享纵轴。Figure E 显示各模型六任务×层的 Signed 热图，正负对称、两模型共用同一 pp 色标。'
            'LLaVA L32–L35 为 NaN/灰色 N/A。每条任务曲线使用自己的 Native Full（每类 500 题），'
            '12 个基线见 p1_task_baselines.csv；408 个任务层值及历史区间见 p1_task_sensitivity_unified.csv。'
            '任务均值与总体敏感性、翻转计数求和均经自动核验。任务结果仅作描述，不添加总体显著性星号；'
            '曲线不叠加六组 CI 以保持可读，原始区间已完整保留在 CSV。\n')
    validation={'passed':True,'unified_rows':len(rows),'counts':[32,32,36,36],
        'checks':['unique complete layers','3000 questions and 2409 image clusters on all rows and baselines',
        'four separate baselines','signed delta and flip count identities','ordinary and corrected CI sign/order',
        'historical writing copies match frozen originals','four known peaks','P-1 significant drops',
        'Qwen writing report significant drops','Qwen P-1 L35 negative significant gain'],
        'llava_writing_report_note':'Historical prose significance list is NLL, not Accuracy; Accuracy CI verified against original CSV.',
        'llava_missing_layers':'L32-L35 are NaN in both heatmaps',
        'corrected_families':'Original model/intervention-specific Bonferroni 32/36; not pooled',
        'new_experiments':False,'new_ci_or_pvalues':False,
        'outputs':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
                   for p in sorted(OUTPUT.glob('figure_*')) if p.suffix in ('.pdf','.png')}}
    validation['task_rows']=len(task_rows)
    validation['task_checks']=['six tasks per model','32/36 complete layers per task','500 questions each',
        '12 distinct task baselines','task means equal overall','task flip sums equal overall','signed CI conversion']
    check(len(validation['outputs'])==12 and all(v['bytes']>1000 for v in validation['outputs'].values()),'Twelve nonempty figure outputs')
    (OUTPUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n','utf-8')
    print(json.dumps({'validation':'passed','rows':len(rows),'summaries':summaries,'output':str(OUTPUT)},indent=2))


if __name__=='__main__':
    main()

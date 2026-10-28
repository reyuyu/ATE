"""Six CPU-only figures for two existing GQA cases; no model/framework imports."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image

from exit_boundary_analysis import ROOT, P32, P34, ALL_K, attention_metrics, choose_best, read_csv, sha

OUTPUT = ROOT/'results/phase3_5_two_case_visualization'
CASES = ['04132006', '15133706']
GROUPS = ['question', 'answer']
ROI_COLORS = ['#00FFFF', '#FF4DFF']


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def load_cases():
    files = [P34/'manifest.json', P34/'sink_manifest.json', P34/'artifact_sha256.json',
             P34/'sample_layer_attention.csv', P32/'manifest.json', P32/'per_sample_results.csv']
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in files}
    m = json.loads(files[0].read_text()); old = json.loads(files[4].read_text())
    if m['parent_manifest_sha256'] != sha(files[4]):
        raise ValueError('Parent manifest changed')
    inventory = json.loads(files[2].read_text())['files']
    for path in [files[0], files[1], files[3]]:
        if sha(path) != inventory[path.name]['sha256']:
            raise ValueError('Historical artifact changed')
    ip = m['processor_config']
    if ip['size'] != {'shortest_edge': 336} or ip['crop_size'] != {'height': 336, 'width': 336} or ip['resample'] != 3:
        raise ValueError('Unexpected true Processor geometry')
    sample_index = {s['question_id']: s for s in m['samples']}
    old_index = {s['question_id']: s for s in old['samples']}
    entries = {s['question_id']: s for s in json.loads(files[1].read_text())['samples']}
    score_rows = read_csv(files[5]); layers = read_csv(files[3]); result = {}
    for q in CASES:
        s = sample_index[q]; entry = entries[q]
        for field in ['img_id', 'question', 'answer', 'image_sha256', 'candidates']:
            if s[field] != old_index[q][field]:
                raise ValueError('Case identity mismatch')
        path = P34/'attention_rows'/f'{q}_task.npz'
        if sha(path) != inventory[str(path.relative_to(P34))]['sha256']:
            raise ValueError('Cached attention changed')
        inputs[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path) as saved:
            a = saved['attention'].astype(np.float64)
        legacy = attention_metrics(a, entry['roi_indices'], entry['sink_nonroi_indices'])
        # Required order for this visualization: average heads FIRST.
        mean_attention = a.mean(1)  # layer, query group, visual patch
        mass = mean_attention.sum(-1)
        if not np.isfinite(mean_attention).all() or (mass <= 0).any():
            raise ValueError('Nonfinite or undefined cached visual distribution')
        p = mean_attention/mass[..., None]
        enrichment = p[..., entry['roi_indices']].sum(-1)/(len(entry['roi_indices'])/576)
        entropy = -(p*np.log(np.maximum(p, 1e-300))).sum(-1)/np.log(576)
        if not np.allclose(p.sum(-1), 1, atol=1e-14, rtol=0):
            raise ValueError('Conditional probabilities do not sum to one')
        if entry['roi_indices'] != s['roi_indices']:
            raise ValueError('ROI mapping changed')
        for l in range(32):
            for g, group in enumerate(GROUPS):
                r = next(r for r in layers if r['question_id'] == q and int(r['layer']) == l and r['query_group'] == group)
                for metric in ['roi_enrichment', 'normalized_entropy', 'total_visual_mass']:
                    if abs(float(r[metric])-legacy[metric][l, :, g].mean()) > 1e-12:
                        raise ValueError('Historical metric recheck failed')
        selected = [r for r in score_rows if r['question_id'] == q]
        if len(selected) != 8 or {int(r['cutoff']) for r in selected} != set(ALL_K):
            raise ValueError('Incomplete/duplicate K grid')
        scores = {int(r['cutoff']): float(r['answer_margin']) for r in selected}
        for r in selected:
            if abs(float(r['full_margin'])-scores[32]) > 1e-12 or r['candidate_correct'] != str(float(r['answer_margin']) > 0):
                raise ValueError('Score baseline or classification mismatch')
        if not scores[32] < 0 < scores[24]:
            raise ValueError('These designated cases do not exhibit the stated historical flip')
        image_path = Path(s['image_path'])
        if sha(image_path) != s['image_sha256']:
            raise ValueError('Case image changed')
        inputs[str(image_path)] = sha(image_path)
        geometry = s['geometry']
        with Image.open(image_path) as im:
            if list(im.size) != geometry['original_size'] or not geometry['pixel_values_exact']:
                raise ValueError('Unverified image geometry')
            crop = im.convert('RGB').resize(tuple(geometry['resized_size']), Image.Resampling.BICUBIC)
            x, y = geometry['crop_left'], geometry['crop_top']
            crop = crop.crop((x, y, x+336, y+336))
        # Check that each cached geometric object covers exactly its saved patches.
        for obj in s['objects']:
            x0, y0, x1, y1 = obj['crop_xyxy']
            patches = [r*24+c for r in range(24) for c in range(24)
                       if min((c+1)*14, x1) > max(c*14, x0) and min((r+1)*14, y1) > max(r*14, y0)]
            if patches != obj['patch_indices']:
                raise ValueError('Cached box-to-patch mapping mismatch')
        result[q] = dict(sample=s, crop=crop, attention=p, mass=mass, enrichment=enrichment, entropy=entropy,
                         legacy=legacy, scores=scores, zero_conditional_patches=int((p == 0).sum()))
    return result, inputs


def case_header(q, d):
    return (f'{q} | '+d['sample']['question']+'\n'
            f'Correct answer: {d["sample"]["answer"]} | Full Margin={d["scores"][32]:+.7f} | K24 Margin={d["scores"][24]:+.7f}')


def heatmap_grid(q, d, group, norm):
    gi = GROUPS.index(group)
    fig = plt.figure(figsize=(24, 14.5))
    grid = fig.add_gridspec(4, 8, left=.012, right=.945, bottom=.065, top=.875, wspace=.025, hspace=.20)
    for layer in range(32):
        ax = fig.add_subplot(grid[layer//8, layer % 8])
        ax.imshow(d['crop'], extent=(0, 336, 336, 0))
        # Nearest-neighbor display preserves the actual 24x24 patch values.
        ax.imshow(d['attention'][layer, gi].reshape(24, 24), extent=(0, 336, 336, 0),
                  cmap='inferno', norm=norm, interpolation='nearest', alpha=.62)
        for oi, obj in enumerate(d['sample']['objects']):
            x0, y0, x1, y1 = obj['crop_xyxy']
            rect = Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor=ROI_COLORS[oi], lw=1.7)
            rect.set_path_effects([pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()])
            ax.add_patch(rect)
        color = '#A45300' if layer == 23 else '#C02020' if layer == 24 else 'black'
        ax.set_title(f'L{layer}  V={d["mass"][layer, gi]:.4f}  E={d["enrichment"][layer, gi]:.3f}', fontsize=10, color=color, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2 if layer in [23, 24] else .25); spine.set_color(color if layer in [23, 24] else 'gray')
    title = 'Question -> Image' if group == 'question' else 'Answer-prediction (P-1) -> Image'
    fig.suptitle(case_header(q, d)+'\n'+title+' | all 32 heads averaged BEFORE visual conditioning', fontsize=16, y=.985)
    objtext = ' | '.join(f'{"CYAN" if i == 0 else "MAGENTA"}: {o["name"]}' for i, o in enumerate(d['sample']['objects']))
    fig.text(.48, .9, objtext+' | outlined actual object boxes after verified CLIP crop', ha='center', fontsize=12)
    cax = fig.add_axes([.958, .19, .009, .56])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='inferno'), cax=cax).set_label('Visual-conditional patch probability (shared LOG scale)', fontsize=11)
    fig.text(.48, .026,
             f'One common color range for ALL 32 layers, BOTH query groups and BOTH cases: [{norm.vmin:.3g}, {norm.vmax:.3g}] (logarithmic).\n'
             'V = raw attention mass on all 576 visual keys; E = ROI patch-union enrichment of the shown head-mean map. No per-layer maximum rescaling.\n'
             'K24 exits BEFORE L24: L23 is the last retained visual-reading layer; these panels show the cached FULL model, not an exit run.',
             ha='center', va='center', fontsize=11)
    fig.savefig(OUTPUT/f'{q}_{group}_32_layers.png', dpi=160)
    plt.close(fig)


def alignment_plot(q, d):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for gi, group in enumerate(GROUPS):
        axes[0].plot(range(32), d['enrichment'][:, gi], '-o', ms=3, label=group)
    axes[0].axhline(1, color='gray', ls='--', label='Uniform ROI baseline')
    axes[0].set_ylabel('ROI enrichment of head-mean attention')
    axes[0].legend(loc='upper right', fontsize=9)
    top = axes[0].secondary_xaxis('top')
    top.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31], [f'L{i}' for i in [0, 4, 8, 12, 16, 20, 24, 28, 31]])
    axes[0].tick_params(labelbottom=False)
    for ax in axes:
        for k in ALL_K:
            ax.axvline(k-.5, color='#C02020' if k == 24 else '#A45300' if k == 16 else 'gray',
                       ls='--' if k in [16, 24] else ':', alpha=.7 if k in [16, 24] else .35, lw=1)
        ax.set_xlim(-1, 32); ax.grid(alpha=.15)
    for k in [16, 24]:
        axes[0].text(k-.5, .98, f' K{k}: L{k-1}|L{k}', transform=axes[0].get_xaxis_transform(), va='top', fontsize=8)
    ks = np.array(ALL_K); ys = np.array([d['scores'][k] for k in ALL_K]); correct = ys > 0
    ax = axes[1]
    ax.plot(ks-.5, ys, color='black', lw=1.1)
    ax.scatter(ks[~correct]-.5, ys[~correct], marker='s', color='gray', label='Still wrong (Margin<=0)', zorder=3)
    ax.scatter(ks[correct]-.5, ys[correct], marker='^', color='tab:green', label='Full wrong -> Exit correct', zorder=3)
    ax.axhline(d['scores'][32], color='purple', ls='--', label='Full Margin')
    ax.axhline(0, color='black', lw=.8, label='Decision boundary: Margin=0')
    _, ties, _ = choose_best(d['scores'], ALL_K)
    ax.scatter(np.array(ties)-.5, [d['scores'][k] for k in ties], marker='*', facecolors='none', edgecolors='tab:blue', s=220, label=f'Best tested K: {ties}', zorder=4)
    ax.set_xticks(ks-.5, [str(k) for k in ALL_K]); ax.set_xlabel('Exit depth K; shared plot coordinate K-0.5 (K32 = Full)')
    ax.set_ylabel('Correct minus wrong logP (nats)'); ax.legend(fontsize=8, loc='lower left')
    # Keep all eight K on the main axis; an inset makes weak-margin flips legible.
    inset = ax.inset_axes([.42, .12, .56, .43]); zoomk = np.array([16, 20, 24, 28, 32]); zoomy = np.array([d['scores'][k] for k in zoomk])
    inset.plot(zoomk, zoomy, '-o', ms=4, color='black')
    inset.scatter(zoomk[zoomy > 0], zoomy[zoomy > 0], marker='^', c='tab:green', s=45, zorder=4)
    inset.axhline(0, color='black', lw=.6); inset.axhline(d['scores'][32], color='purple', ls='--', lw=.7)
    for k in zoomk:
        inset.annotate(f'{d["scores"][k]:+.5f}', (k, d['scores'][k]), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7)
    span = max(float(np.ptp(zoomy)), .06); inset.set_ylim(min(zoomy.min(), 0)-span*.25, max(zoomy.max(), 0)+span*.4)
    inset.set_xticks(zoomk); inset.tick_params(labelsize=7); inset.grid(alpha=.15); inset.set_title('K16–32: near-zero Margin detail', fontsize=8)
    fig.suptitle(case_header(q, d)+'\nK24 boundary is BETWEEN L23 and L24; L23 is the last retained visual-reading layer.', fontsize=11)
    fig.tight_layout(h_pad=2)
    fig.savefig(OUTPUT/f'{q}_roi_exit_margin.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    if args.verify_existing:
        record = json.loads((OUTPUT/'validation.json').read_text())
        for section in ['input_sha256', 'output_sha256']:
            for name, expected in record[section].items():
                if sha(ROOT/name) != expected:
                    raise ValueError('Changed input/output: '+name)
        print('All cached inputs and six figures verified; no GPU/Forward.')
        return
    if OUTPUT.exists():
        raise ValueError('Output exists; refusing overwrite. Use --verify-existing.')
    started = time.perf_counter(); cases, inputs = load_cases()
    positives = np.concatenate([d['attention'][d['attention'] > 0] for d in cases.values()])
    norm = LogNorm(vmin=float(positives.min()), vmax=float(positives.max()))
    OUTPUT.mkdir(parents=True)
    rows = []
    for q, d in cases.items():
        for group in GROUPS:
            heatmap_grid(q, d, group, norm)
        alignment_plot(q, d)
        for layer in range(32):
            for gi, group in enumerate(GROUPS):
                rows.append(dict(question_id=q, query_group=group, layer=layer,
                                 visual_mass=d['mass'][layer, gi], roi_enrichment=d['enrichment'][layer, gi], normalized_entropy=d['entropy'][layer, gi],
                                 legacy_mean_head_enrichment=d['legacy']['roi_enrichment'][layer, :, gi].mean(),
                                 legacy_mean_head_entropy=d['legacy']['normalized_entropy'][layer, :, gi].mean()))
    with (OUTPUT/'layer_metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    description = ['# 两案例：32层视觉 Attention 与 Physical Exit','',
                   '仅CPU读取既有Phase3.4所选Attention行/已验证Processor几何和Phase3.2完整8K Margin；没有GPU、Forward、训练或额外样本。','',
                   '## 图像与颜色口径','',
                   '- 每张热图4×8，行优先L0–L31。真实CLIP shortest-edge336/BICUBIC/中心裁剪；复用实际整数缩放、裁剪偏移与裁后对象框，蓝绿色/品红边框分别对应官方对象1/2。ROI富集使用原576Patch正面积交集并集，而非框内语义分割。',
                   '- 先平均全部32Head，再对576视觉Patch归一化。四张热图、两类Query和两图共用同一对数色标，没有按层最大值拉伸；V是未归一化的视觉总Attention Mass，E是显示的Head均值地图的面积富集。对数颜色不能按线性强度阅读；小质量层的明亮条件热点不等于原始视觉总质量更高。',
                   '- E=(显示地图的ROI条件概率)/(ROI Patch数/576)，熵为显示地图的H/log576。此前Phase3.5统计先逐Head条件化再平均，属于不同非线性顺序；layer_metrics.csv同时保存旧口径以便核对，本次曲线与热图使用同一新口径。Question行是缓存中真实问题Token行均值，Answer行是P−1。',
                   '- 曲线上方为真实L0–L31，下方K点在共享坐标K−0.5；K24在L23和L24之间。主曲线保留完整8K，内嵌图放大K16–32弱Margin，Margin>0才正确，平局判错。原生Full=K32，没有将Full Attention当作退出后Attention。','',
                   f'全图公共对数色标：[{norm.vmin:.9g}, {norm.vmax:.9g}]，正值无截断；零值数量={sum(d["zero_conditional_patches"] for d in cases.values())}。','',
                   '## 图表与实际观察','']
    for q, d in cases.items():
        description += [f'### {q}', '', d['sample']['question'], '',
                        f'标准答案：**{d["sample"]["answer"]}**。Full Margin={d["scores"][32]:+.7f}；K24 Margin={d["scores"][24]:+.7f}。', '',
                        f'- [Question→Image全部32层]({q}_question_32_layers.png)', f'- [Answer P−1→Image全部32层]({q}_answer_32_layers.png)',
                        f'- [ROI曲线与Physical Margin对照]({q}_roi_exit_margin.png)', '',
                        '| Query | L23→L24 ROI富集 | L23→L24条件熵 | L23→L24原始视觉Mass |', '|---|---|---|---|']
        for gi, group in enumerate(GROUPS):
            description.append(f'| {group} | {d["enrichment"][23, gi]:.4f}→{d["enrichment"][24, gi]:.4f} | {d["entropy"][23, gi]:.4f}→{d["entropy"][24, gi]:.4f} | {d["mass"][23, gi]:.4f}→{d["mass"][24, gi]:.4f} |')
        description += ['', '全8K Margin：'+', '.join(f'K{k}={d["scores"][k]:+.7f}' for k in ALL_K)+'。', '']
    description += ['04132006在K24恢复时Question ROI富集下降（此Head均值地图中约1.029→0.745），但Answer略升；两者条件熵都上升。15133706恢复时Question和Answer ROI富集都上升、熵也上升，说明更分散与ROI占比上升可以共存。两例并无一致的“ROI恰好下降”模式，不能把条件熵升高自动归因Sink，也不能解释为确定因果。', '',
                    '04132006的K24/K28并列全网格最高，15133706的K24最高。两例均为事后案例，Full和K24 Margin接近零，原FP16数值敏感性与有限候选协议的限制保留；本轮没有新FP32或模型验证。','',
                    '## 复现','', '```bash', 'cd /home/zsy/ATE/vlm_research', "CUDA_VISIBLE_DEVICES='' python two_case_attention_viz.py", "CUDA_VISIBLE_DEVICES='' python two_case_attention_viz.py --verify-existing", '```', '',
                    '已有目录拒绝覆盖。validation.json保存缓存/图像/脚本/六图SHA256、条件概率归一化、映射/历史指标复核结果和颜色口径。历史阶段及此前Phase3.5结果不变。']
    (OUTPUT/'README.md').write_text('\n'.join(description)+'\n')
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise RuntimeError('Unexpected model framework import')
    if len(list(OUTPUT.glob('*.png'))) != 6:
        raise AssertionError('Expected exactly six PNGs')
    inputs['two_case_attention_viz.py'] = sha(__file__)
    inputs['exit_boundary_analysis.py'] = sha(ROOT/'exit_boundary_analysis.py')
    dump(OUTPUT/'validation.json', dict(passed=True, cases=CASES, model_forward_count=0, gpu_used=False,
         selected_attention_shape=[32, 32, 2, 576], png_count=6, layer_metric_rows=len(rows),
         cached_roi_and_image_hash_verified=True, legacy_metrics_match_historical=True,
         head_mean_then_conditional=True, shared_color_norm=dict(type='log', vmin=norm.vmin, vmax=norm.vmax),
         conditional_probability_max_sum_error=max(float(np.abs(d['attention'].sum(-1)-1).max()) for d in cases.values()),
         nonpositive_color_values=sum(d['zero_conditional_patches'] for d in cases.values()),
         wall_seconds=time.perf_counter()-started, input_sha256=inputs,
         output_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file()}))
    print('Created six figures and README; no GPU/Forward; all source images and cache hashes verified.')


if __name__ == '__main__':
    main()

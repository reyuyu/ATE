"""Phase 3.5: CPU-only analysis of cached Physical Exit scores and Full attention.

No torch/transformers imports, model Forward, network requests or CUDA access.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent
P32 = ROOT / 'results/phase3_2_fixed_k_exit'
P34 = ROOT / 'results/phase3_4_roi_sink_diagnostic'
OUTPUT = ROOT / 'results/phase3_5_exit_boundary_viz'
ALL_K = [0, 8, 12, 16, 20, 24, 28, 32]
INTERIOR_K = [8, 12, 16, 20, 24, 28]
BEST_K = [12, 16, 20, 24, 28]
CASES = ['04132006', '15133706', '0216546', '02310173', '05480434']
GROUPS = ['question', 'answer']
COLORS = {'question': 'tab:blue', 'answer': 'tab:orange'}
PROTOCOL = dict(
    phase='3.5', bootstrap_seed=20261031, bootstrap_replicates=5000,
    primary_boundaries=[16, 24], exploratory_best_candidates=BEST_K,
    metrics='Per-head visual conditioning, then equal-head mean. Question rows already averaged over real Question tokens. Entropy divided by log(576). Sink is frozen NON-ROI candidate share.',
    layer_alignment='Exit K removes tokens before L_K: before=L(K-1), after=L_K. Shared plot coordinate for boundary K is K-0.5; K0/32 have no two-sided Full boundary metric.',
    low_mass='Flag if either endpoint head-mean visual mass <1e-3 OR any endpoint head visual mass <1e-4. Keep all 48 in primary analysis.',
    sensitivity_head_mass_thresholds=[1e-4, 1e-3, 1e-2],
    sensitivity='Also exclude flagged images; separately recompute using only identical heads whose mass is >= threshold at BOTH endpoints. Keep image weighting equal; report no-head images. Raw Gain>0 primary; sensitivity Gain>1e-10 and >0.01 nats.',
    best_ties='All K within absolute 1e-10 nats of grid maximum are tied; choose smallest tied K deterministically and save all tied K. Strict classification remains margin>0, ties incorrect.',
    statistics='Image paired bootstrap percentile 95% CI, resampling whole images. Pearson and average-tie-rank Spearman; undefined constant replicates excluded and counted. Exploratory, no multiplicity correction or causal claim.',
    comparison='Gain>0 versus Gain<=0 Delta_ROI distributions; mean and median differences from image bootstrap. Do not pool five historical cases into main estimates.',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows to write')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def ranks(values):
    """Average ranks, including ties; supports vectors or a batch of vectors."""
    a = np.asarray(values, dtype=float)
    one = a.ndim == 1
    a = np.atleast_2d(a)
    order = np.argsort(a, axis=1, kind='stable')
    sorted_a = np.take_along_axis(a, order, axis=1)
    n = a.shape[1]
    idx = np.broadcast_to(np.arange(n), a.shape)
    first = np.concatenate([np.ones((len(a), 1), bool), np.diff(sorted_a, axis=1) != 0], axis=1)
    last = np.concatenate([np.diff(sorted_a, axis=1) != 0, np.ones((len(a), 1), bool)], axis=1)
    starts = np.maximum.accumulate(np.where(first, idx, 0), axis=1)
    ends = np.minimum.accumulate(np.where(last, idx, n-1)[:, ::-1], axis=1)[:, ::-1]
    result = np.empty_like(a)
    np.put_along_axis(result, order, (starts + ends) / 2 + 1, axis=1)
    return result[0] if one else result


def row_correlation(x, y):
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    x, y = x-x.mean(axis=1, keepdims=True), y-y.mean(axis=1, keepdims=True)
    den = np.sqrt((x*x).sum(1) * (y*y).sum(1))
    return np.divide((x*y).sum(1), den, out=np.full(len(x), np.nan), where=den > 0)


def ci(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return [float(x) for x in np.percentile(valid, [2.5, 97.5])] if len(valid) else [None, None]


def paired_summary(rows, label):
    if not rows:
        return dict(analysis=label, query_group='', cutoff='', n=0)
    x = np.array([r['delta_roi'] for r in rows])
    entropy = np.array([r['delta_entropy'] for r in rows])
    gain = np.array([r['gain'] for r in rows])
    idx = np.random.default_rng(PROTOCOL['bootstrap_seed']).integers(0, len(rows), size=(PROTOCOL['bootstrap_replicates'], len(rows)))
    xb, gb = x[idx], gain[idx]
    positive = gb > 0
    np_, nn = positive.sum(1), (~positive).sum(1)
    valid = (np_ > 0) & (nn > 0)
    mean_diff = (np.where(positive, xb, 0).sum(1) / np.maximum(np_, 1)
                 - np.where(~positive, xb, 0).sum(1) / np.maximum(nn, 1))
    median_diff = np.full(len(idx), np.nan)
    if valid.any():
        median_diff[valid] = (np.nanmedian(np.where(positive[valid], xb[valid], np.nan), axis=1)
                              - np.nanmedian(np.where(~positive[valid], xb[valid], np.nan), axis=1))
    pos = gain > 0
    r = dict(analysis=label, query_group=rows[0]['query_group'], cutoff='selected' if label.startswith('best_') else rows[0]['cutoff'], n=len(rows),
             roi_down=int((x < 0).sum()), entropy_up=int((entropy > 0).sum()), roi_down_entropy_up=int(((x < 0) & (entropy > 0)).sum()),
             focused_before_down=int(np.sum([(a['roi_before'] > 1 and a['delta_roi'] < 0) for a in rows])),
             crosses_roi_uniform=int(np.sum([(a['roi_before'] > 1 and a['roi_after'] <= 1) for a in rows])),
             gain_positive=int(pos.sum()), gain_nonpositive=int((~pos).sum()),
             roi_down_gain_positive=int(((x < 0) & pos).sum()),
             roi_down_entropy_up_gain_positive=int(((x < 0) & (entropy > 0) & pos).sum()),
             gain_positive_focused_before_down=int(sum(a['gain'] > 0 and a['roi_before'] > 1 and a['delta_roi'] < 0 for a in rows)),
             gain_gt_1e_minus10=int((gain > 1e-10).sum()), gain_gt_0_01=int((gain > .01).sum()),
             gain_near_zero=int((np.abs(gain) <= 1e-10).sum()),
             mean_delta_roi=float(x.mean()), median_delta_roi=float(np.median(x)),
             mean_gain=float(gain.mean()), mean_delta_entropy=float(entropy.mean()),
             low_mass_images=sum(a['low_mass_flag'] for a in rows),
             full_correct=sum(a['full_correct'] for a in rows), exit_correct=sum(a['exit_correct'] for a in rows),
             correct_to_wrong=sum(a['flip_category'] == 'correct_to_wrong' for a in rows),
             wrong_to_correct=sum(a['flip_category'] == 'wrong_to_correct' for a in rows))
    for name, values in [('delta_roi', x), ('delta_entropy', entropy), ('delta_visual_mass', np.array([a['delta_visual_mass'] for a in rows]))]:
        for method in ['pearson', 'spearman']:
            a, b = (values, gain) if method == 'pearson' else (ranks(values), ranks(gain))
            observed = row_correlation(a, b)[0]
            aa, bb = values[idx], gb
            if method == 'spearman':
                aa, bb = ranks(aa), ranks(bb)
            bs = row_correlation(aa, bb)
            lo, hi = ci(bs)
            prefix = f'{name}_gain_{method}'
            r[prefix] = float(observed) if np.isfinite(observed) else None
            r[prefix+'_ci_low'], r[prefix+'_ci_high'] = lo, hi
            r[prefix+'_undefined_bootstrap'] = int((~np.isfinite(bs)).sum())
    for mask, group in [(pos, 'gain_positive'), (~pos, 'gain_nonpositive')]:
        data = x[mask]
        for statistic, value in [('mean', np.mean(data) if len(data) else None), ('median', np.median(data) if len(data) else None),
                                 ('q25', np.percentile(data, 25) if len(data) else None), ('q75', np.percentile(data, 75) if len(data) else None)]:
            r[f'{group}_delta_roi_{statistic}'] = float(value) if value is not None else None
    r['delta_roi_group_mean_difference'] = float(x[pos].mean()-x[~pos].mean()) if pos.any() and (~pos).any() else None
    r['delta_roi_group_median_difference'] = float(np.median(x[pos])-np.median(x[~pos])) if pos.any() and (~pos).any() else None
    for name, values in [('mean', mean_diff[valid]), ('median', median_diff[valid])]:
        r[f'group_{name}_difference_ci_low'], r[f'group_{name}_difference_ci_high'] = ci(values)
    return r


def attention_metrics(a, roi, sink):
    if a.shape != (32, 32, 2, 576) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Bad cached attention')
    mass = a.sum(-1)
    if (mass <= 0).any():
        raise ValueError('Zero visual mass: conditional metrics undefined, investigate missing data')
    p = a / mass[..., None]
    return dict(roi_enrichment=p[..., roi].sum(-1)/(len(roi)/576), total_visual_mass=mass,
                normalized_entropy=-(p*np.log(np.maximum(p, 1e-300))).sum(-1)/np.log(576),
                sink_nonroi_conditional_share=p[..., sink].sum(-1))


def choose_best(scores, candidates):
    best = max(scores[k] for k in candidates)
    ties = [k for k in candidates if abs(scores[k]-best) <= 1e-10]
    return min(ties), ties, best


def flip_category(full, exit_margin):
    return ('correct' if full > 0 else 'wrong') + '_to_' + ('correct' if exit_margin > 0 else 'wrong')


def boundary_row(q, group, k, metrics, score, main, head_mask=None):
    g = GROUPS.index(group)
    mask = np.ones(32, bool) if head_mask is None else head_mask
    if not mask.any():
        return None
    before = {name: float(values[k-1, mask, g].mean()) for name, values in metrics.items()}
    after = {name: float(values[k, mask, g].mean()) for name, values in metrics.items()}
    mass = metrics['total_visual_mass'][[k-1, k], :, g]
    low = bool(mass.mean(1).min() < 1e-3 or mass.min() < 1e-4)
    full, margin = float(score[32]), float(score[k])
    return dict(question_id=q, cohort='main' if main else 'historical_case', query_group=group, cutoff=k,
                before_layer=k-1, after_layer=k, roi_before=before['roi_enrichment'], roi_after=after['roi_enrichment'],
                delta_roi=after['roi_enrichment']-before['roi_enrichment'],
                entropy_before=before['normalized_entropy'], entropy_after=after['normalized_entropy'],
                delta_entropy=after['normalized_entropy']-before['normalized_entropy'],
                visual_mass_before=before['total_visual_mass'], visual_mass_after=after['total_visual_mass'],
                delta_visual_mass=after['total_visual_mass']-before['total_visual_mass'],
                sink_share_before=before['sink_nonroi_conditional_share'], sink_share_after=after['sink_nonroi_conditional_share'],
                delta_sink_share=after['sink_nonroi_conditional_share']-before['sink_nonroi_conditional_share'],
                full_margin=full, physical_margin=margin, gain=margin-full,
                full_correct=full > 0, exit_correct=margin > 0, flip_category=flip_category(full, margin),
                low_mass_flag=low, minimum_endpoint_head_mass=float(mass.min()),
                endpoint_heads_below_1e_minus4=int((mass < 1e-4).sum()), heads_used=int(mask.sum()))


def load_cache():
    paths = [P32/'per_sample_results.csv', P32/'manifest.json', P34/'sample_layer_attention.csv',
             P34/'layer_attention_summary.csv', P34/'sink_manifest.json', P34/'manifest.json', P34/'artifact_sha256.json']
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required cache; no GPU fallback: '+repr(missing))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    m32, m34 = json.loads(paths[1].read_text()), json.loads(paths[5].read_text())
    if m34['parent_manifest_sha256'] != sha(paths[1]):
        raise ValueError('Parent manifest changed')
    main, cases = m34['main_question_ids'], m34['case_question_ids']
    if len(main) != 48 or set(cases) != set(CASES) or len(cases) != 5 or set(main)&set(cases):
        raise ValueError('Wrong or overlapping cohorts')
    sm32 = {s['question_id']: s for s in m32['samples']}
    sm34 = {s['question_id']: s for s in m34['samples']}
    entries = {s['question_id']: s for s in json.loads(paths[4].read_text())['samples']}
    source_integrity = json.loads(paths[6].read_text())['files']
    for p in paths[2:6]:
        if sha(p) != source_integrity[p.name]['sha256']:
            raise ValueError('Historical attention artifact changed: '+p.name)
    physical = read_csv(paths[0]); scores = {}; physical_index = {}
    for r in physical:
        q, k = r['question_id'], int(r['cutoff'])
        if (q, k) in physical_index:
            raise ValueError('Duplicate Physical Exit score')
        physical_index[q, k] = r
        scores.setdefault(q, {})[k] = float(r['answer_margin'])
        if r['candidate_correct'] != str(float(r['answer_margin']) > 0) or r['full_correct'] != str(float(r['full_margin']) > 0):
            raise ValueError('Historical classification mismatch')
    if len(physical) != 240*8 or any(set(s) != set(ALL_K) for s in scores.values()):
        raise ValueError('Incomplete Phase3.2 grid')
    source_rows = read_csv(paths[2]); attention_index = {(r['question_id'], int(r['layer']), r['query_group']): r for r in source_rows}
    if len(attention_index) != len(source_rows) or len(source_rows) != 53*32*2:
        raise ValueError('Incomplete/duplicate attention table')
    data = {}; maxdiff = 0.
    for q in main+CASES:
        sample = sm34[q]
        for field in ['img_id', 'question', 'answer', 'image_sha256', 'candidates']:
            if sample[field] != sm32[q][field]:
                raise ValueError('Image/question/candidate identity mismatch')
        e = entries[q]
        if e['roi_indices'] != sample['roi_indices'] or bool(e['main']) != (q in main):
            raise ValueError('ROI/cohort mapping mismatch')
        path = P34/'attention_rows'/f'{q}_task.npz'
        if not path.exists():
            raise FileNotFoundError('Missing cached attention; no GPU fallback: '+str(path))
        if sha(path) != source_integrity[str(path.relative_to(P34))]['sha256']:
            raise ValueError('Raw attention changed')
        hashes[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path) as saved:
            a = saved['attention'].astype(np.float64)
        metrics = attention_metrics(a, e['roi_indices'], e['sink_nonroi_indices'])
        for layer in range(32):
            for g, group in enumerate(GROUPS):
                r = attention_index[q, layer, group]
                if r['main'] != str(q in main) or abs(float(r['full_margin'])-scores[q][32]) > 1e-12:
                    raise ValueError('Full score/cohort mismatch between phases')
                for name, values in metrics.items():
                    diff = abs(float(r[name])-values[layer, :, g].mean())
                    maxdiff = max(maxdiff, diff)
                    if diff > 1e-12:
                        raise ValueError('Recomputed attention metric differs: '+repr((q, layer, group, name, diff)))
        for k in ALL_K:
            if abs(float(physical_index[q, k]['full_margin'])-scores[q][32]) > 1e-12:
                raise ValueError('Full grid inconsistent')
        data[q] = dict(sample=sample, entry=e, metrics=metrics, attention=a, scores=scores[q])
    summary = read_csv(paths[3])
    if len(summary) != 64:
        raise ValueError('Layer summary incomplete')
    for r in summary:
        for metric in ['roi_enrichment', 'total_visual_mass', 'normalized_entropy', 'sink_nonroi_conditional_share']:
            actual = np.mean([data[q]['metrics'][metric][int(r['layer']), :, GROUPS.index(r['query_group'])].mean() for q in main])
            if abs(actual-float(r[metric])) > 1e-12:
                raise ValueError('Main48 layer summary mismatch')
    return main, data, hashes, maxdiff


def savefig(fig, name, h_pad=1.08):
    fig.tight_layout(h_pad=h_pad)
    fig.savefig(OUTPUT/name, dpi=155, bbox_inches='tight')
    plt.close(fig)


def case_plots(q, d):
    metrics, scores = d['metrics'], d['scores']
    best, ties, value = choose_best(scores, ALL_K)
    interior_best, interior_ties, _ = choose_best(scores, BEST_K)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for group in GROUPS:
        g = GROUPS.index(group)
        axes[0].plot(range(32), metrics['roi_enrichment'][:, :, g].mean(1), color=COLORS[group], label=group)
    axes[0].axhline(1, color='gray', ls='--', lw=.8)
    axes[0].set_ylabel('ROI enrichment (mean of 32 heads)')
    axes[0].legend()
    for ax in axes:
        for k in ALL_K:
            ax.axvline(k-.5, color='gray', alpha=.4, ls=':', lw=.8)
        ax.set_xlim(-1, 32)
        ax.grid(alpha=.15)
    axes[0].tick_params(labelbottom=False)
    layer_axis = axes[0].secondary_xaxis('top')
    layer_axis.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31], [f'L{i}' for i in [0, 4, 8, 12, 16, 20, 24, 28, 31]])
    # Shared coordinates use layer indices; bottom labels express K=layer-boundary+0.5.
    axes[0].text(.01, .98, 'Full layers L0...L31; dotted boundaries K=0/8/12/16/20/24/28/32', transform=axes[0].transAxes, va='top', fontsize=8)
    x = np.array(ALL_K)-.5
    y = np.array([scores[k] for k in ALL_K])
    axes[1].plot(x, y, color='black', lw=1)
    styles = {'correct_to_correct': ('o', 'tab:green'), 'correct_to_wrong': ('X', 'tab:red'),
              'wrong_to_correct': ('^', 'tab:blue'), 'wrong_to_wrong': ('s', 'gray')}
    for category, (marker, color) in styles.items():
        mask = [flip_category(scores[32], scores[k]) == category for k in ALL_K]
        if any(mask):
            axes[1].scatter(x[mask], y[mask], marker=marker, c=color, label=category.replace('_', ' '), zorder=4)
    axes[1].scatter(np.array(ties)-.5, [scores[k] for k in ties], marker='*', s=200, facecolors='none', edgecolors='purple', label='best in all 8 tested K', zorder=5)
    axes[1].axhline(scores[32], color='purple', ls='--', label=f'Full={scores[32]:+.5f}')
    axes[1].axhline(0, color='black', lw=.8)
    axes[1].set_xticks(np.array(ALL_K)-.5, [str(k) for k in ALL_K])
    axes[1].set_xlabel('Exit depth K (point at boundary coordinate K-0.5); K32 = Full')
    axes[1].set_ylabel('Correct minus wrong logP (nats)')
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle(f'{q}: {d["sample"]["question"]}\nK24 exits BEFORE L24: last visual-reading layer L23; best tested K={ties}; restricted grid best={interior_ties}', fontsize=10)
    savefig(fig, f'case_{q}_alignment.png')
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for col, (start, end, k) in enumerate([(12, 18, 16), (20, 27, 24)]):
        for row, (name, title) in enumerate([('roi_enrichment', 'ROI enrichment'), ('total_visual_mass', 'Raw visual attention mass'), ('normalized_entropy', 'Visual-conditional entropy / log(576)')]):
            ax = axes[row, col]
            for group in GROUPS:
                a = metrics[name][start:end+1, :, GROUPS.index(group)]
                ax.plot(range(start, end+1), a.mean(1), '-o', ms=3, color=COLORS[group], label=group)
            ax.axvline(k-.5, color='red', ls='--', label=f'K{k}: L{k-1} -> L{k}')
            ax.set_xticks(range(start, end+1))
            ax.set_xlabel('Full model layer index')
            ax.set_ylabel(title)
            ax.grid(alpha=.2)
            if name == 'roi_enrichment':
                ax.axhline(1, color='gray', ls=':', lw=.7)
            if row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f'{q}: all 32 heads equally weighted; neither heads nor layers selected by effect', fontsize=11)
    savefig(fig, f'case_{q}_local_boundaries.png')
    return dict(question_id=q, best_all_grid=best, ties_all_grid=';'.join(map(str, ties)), best_margin=value,
                best_restricted_grid=interior_best, ties_restricted_grid=';'.join(map(str, interior_ties)))


def case_spatial(q, d):
    """Use saved true-Processor geometry; never call a model/processor."""
    sample = d['sample']
    path = Path(sample['image_path'])
    if not path.exists():
        return dict(question_id=q, available=False, reason='Local source image missing; no network fallback')
    if sha(path) != sample['image_sha256']:
        raise ValueError('Case source image changed')
    g = sample['geometry']
    with Image.open(path) as im:
        image = im.convert('RGB').resize(tuple(g['resized_size']), resample=Image.Resampling.BICUBIC)
        left, top = g['crop_left'], g['crop_top']
        image = image.crop((left, top, left+336, top+336))
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    # Equal image/group scaling across before/after; raw means annotated.
    for row, k in enumerate([16, 24]):
        for group_index, group in enumerate(GROUPS):
            maps = d['attention'][[k-1, k], :, group_index].mean(1)
            conditional = maps / maps.sum(1, keepdims=True)
            vmax = conditional.max()
            for endpoint, layer in enumerate([k-1, k]):
                ax = axes[row, group_index*2+endpoint]
                ax.imshow(image)
                heat = Image.fromarray(conditional[endpoint].reshape(24, 24).astype(np.float32)).resize((336, 336), resample=Image.Resampling.BILINEAR)
                ax.imshow(heat, cmap='inferno', vmin=0, vmax=vmax, alpha=.6)
                for patch in d['entry']['roi_indices']:
                    r, c = divmod(patch, 24)
                    ax.add_patch(Rectangle((c*14, r*14), 14, 14, fill=False, edgecolor='cyan', lw=.25))
                gg = GROUPS.index(group)
                E = d['metrics']['roi_enrichment'][layer, :, gg].mean()
                H = d['metrics']['normalized_entropy'][layer, :, gg].mean()
                ax.set_title(f'{group} L{layer}: mass={maps[endpoint].sum():.4f}\nhead-mean E={E:.3f}; H={H:.3f}', fontsize=9)
                ax.axis('off')
    fig.suptitle(f'{q}: K16/K24 before/after; cyan=ROI; common visual-conditional scale within each pair\nSpatial image uses conditioned HEAD-MEAN map; scalar E/H uses MEAN of per-head metrics (different operation).', fontsize=10)
    savefig(fig, f'case_{q}_spatial.png', h_pad=3.5)
    return dict(question_id=q, available=True, image_sha256=sha(path), reason='')


def scatter_plots(rows, summaries):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    styles = {'correct_to_correct': ('o', 'tab:green'), 'correct_to_wrong': ('X', 'tab:red'),
              'wrong_to_correct': ('^', 'tab:blue'), 'wrong_to_wrong': ('s', 'gray')}
    for row, k in enumerate([16, 24]):
        for col, group in enumerate(GROUPS):
            ax = axes[row, col]
            values = [r for r in rows if r['cohort'] == 'main' and r['cutoff'] == k and r['query_group'] == group]
            s = next(s for s in summaries if s['cutoff'] == k and s['query_group'] == group)
            for category, (marker, color) in styles.items():
                selected = [r for r in values if r['flip_category'] == category]
                ax.scatter([r['delta_roi'] for r in selected], [r['gain'] for r in selected], marker=marker, c=color,
                           label=f'{category.replace("_", " ")} (n={len(selected)})', alpha=.8)
            flagged = [r for r in values if r['low_mass_flag']]
            ax.scatter([r['delta_roi'] for r in flagged], [r['gain'] for r in flagged], facecolors='none', edgecolors='purple', s=110, label=f'low mass flag ({len(flagged)})')
            ax.axhline(0, color='black', lw=.7); ax.axvline(0, color='black', lw=.7)
            ax.set_xlabel(f'Delta ROI enrichment: L{k} - L{k-1}')
            ax.set_ylabel(f'Gain(K{k}) = Physical Margin - Full Margin')
            ax.set_title(f'{group}, K{k}, all 48 images\nPearson={s["delta_roi_gain_pearson"]:.3f}; Spearman={s["delta_roi_gain_spearman"]:.3f}')
            ax.legend(fontsize=7, loc='best'); ax.grid(alpha=.15)
    fig.suptitle('Fixed boundaries; all extremes retained; only strict margin>0 counts as correct')
    savefig(fig, 'main48_fixed_boundary_scatter.png')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, k in enumerate([16, 24]):
        for col, group in enumerate(GROUPS):
            ax = axes[row, col]
            values = [r for r in rows if r['cohort'] == 'main' and r['cutoff'] == k and r['query_group'] == group]
            for j, flag in enumerate([False, True]):
                data = np.array([r['delta_roi'] for r in values if (r['gain'] > 0) == flag])
                if len(data):
                    ax.boxplot([data], positions=[j], widths=.4, showfliers=False)
                    jitter = np.random.default_rng(20261031).uniform(-.1, .1, len(data))
                    ax.scatter(j+jitter, data, s=16, alpha=.65, label=f'n={len(data)}')
            ax.set_xticks([0, 1], ['Gain<=0', 'Gain>0']); ax.set_ylabel('Delta ROI enrichment')
            ax.set_title(f'{group}, K{k}'); ax.axhline(0, color='gray', lw=.7); ax.legend(fontsize=8)
    fig.suptitle('All points retained; box=IQR, center=median; gain is not accuracy improvement')
    savefig(fig, 'main48_delta_roi_by_gain.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-existing', action='store_true', help='Verify input/output checksums; never rerun or overwrite')
    args = parser.parse_args()
    if args.verify_existing:
        inventory = json.loads((OUTPUT/'checksums.json').read_text())
        for category in ['inputs', 'outputs']:
            for path, expected in inventory[category].items():
                if sha(ROOT/path) != expected:
                    raise ValueError('Changed artifact: '+path)
        print('Verified', len(inventory['inputs']), 'inputs and', len(inventory['outputs']), 'outputs; no GPU/model access')
        return
    if OUTPUT.exists():
        raise ValueError('Output exists: refuse overwrite. Use --verify-existing.')
    OUTPUT.mkdir(parents=True)
    dump(OUTPUT/'analysis_protocol.json', PROTOCOL)
    started = time.perf_counter()
    main_ids, data, inputs, maxdiff = load_cache()
    rows = [boundary_row(q, group, k, data[q]['metrics'], data[q]['scores'], q in main_ids)
            for q in main_ids+CASES for k in INTERIOR_K for group in GROUPS]
    write_csv(OUTPUT/'boundary_changes.csv', rows)
    scores = [dict(question_id=q, cohort='main' if q in main_ids else 'historical_case', answer=data[q]['sample']['answer'],
                   cutoff=k, margin=data[q]['scores'][k], full_margin=data[q]['scores'][32], gain=data[q]['scores'][k]-data[q]['scores'][32],
                   flip_category=flip_category(data[q]['scores'][32], data[q]['scores'][k])) for q in main_ids+CASES for k in ALL_K]
    write_csv(OUTPUT/'all_eight_k_scores.csv', scores)
    main_rows = [r for r in rows if r['cohort'] == 'main']
    fixed = [paired_summary([r for r in main_rows if r['cutoff'] == k and r['query_group'] == g], 'fixed_boundary') for k in [16, 24] for g in GROUPS]
    write_csv(OUTPUT/'fixed_boundary_summary.csv', fixed)
    best_rows, best_info = [], []
    for q in main_ids:
        best, ties, margin = choose_best(data[q]['scores'], BEST_K)
        best_info.append(dict(question_id=q, best_k_observed=best, tied_k=';'.join(map(str, ties)), tie_count=len(ties),
                              best_margin=margin, full_margin=data[q]['scores'][32], gain=margin-data[q]['scores'][32]))
        for g in GROUPS:
            best_rows.append(dict(next(r for r in main_rows if r['question_id'] == q and r['cutoff'] == best and r['query_group'] == g), best_tied_k=';'.join(map(str, ties))))
    best_summary = [paired_summary([r for r in best_rows if r['query_group'] == g], 'best_observed') for g in GROUPS]
    write_csv(OUTPUT/'best_observed_per_image.csv', best_info)
    write_csv(OUTPUT/'best_observed_boundary_changes.csv', best_rows)
    write_csv(OUTPUT/'best_observed_summary.csv', best_summary)
    # Tie rule sensitivity: average each image's tied-boundary metrics, then bootstrap images.
    tie_averaged = []
    for info in best_info:
        q = info['question_id']; ties = list(map(int, info['tied_k'].split(';')))
        for g in GROUPS:
            v = [r for r in main_rows if r['question_id'] == q and r['cutoff'] in ties and r['query_group'] == g]
            base = dict(v[0])
            for field in ['roi_before', 'roi_after', 'delta_roi', 'delta_entropy', 'delta_visual_mass', 'gain']:
                base[field] = float(np.mean([r[field] for r in v]))
            base['low_mass_flag'] = any(r['low_mass_flag'] for r in v)
            tie_averaged.append(base)
    tie_summaries = [paired_summary([r for r in tie_averaged if r['query_group'] == g], 'best_tie_average_sensitivity') for g in GROUPS]
    write_csv(OUTPUT/'best_tie_sensitivity_summary.csv', tie_summaries)
    sensitivity = []
    for k in [16, 24]:
        for g in GROUPS:
            original = [r for r in main_rows if r['cutoff'] == k and r['query_group'] == g]
            s = paired_summary([r for r in original if not r['low_mass_flag']], 'exclude_low_mass_flag')
            s.update(cutoff=k, query_group=g, unavailable_images=48-s['n']); sensitivity.append(s)
            for threshold in PROTOCOL['sensitivity_head_mass_thresholds']:
                restricted = []
                for q in main_ids:
                    m = data[q]['metrics']['total_visual_mass'][[k-1, k], :, GROUPS.index(g)]
                    mask = (m >= threshold).all(0)
                    r = boundary_row(q, g, k, data[q]['metrics'], data[q]['scores'], True, mask)
                    if r is not None:
                        restricted.append(r)
                s = paired_summary(restricted, f'paired_heads_mass_ge_{threshold:g}')
                s.update(cutoff=k, query_group=g, unavailable_images=48-len(restricted)); sensitivity.append(s)
    # Apply the same predefined mass rule at each post-hoc selected boundary.
    best_lookup = {v['question_id']: v['best_k_observed'] for v in best_info}
    for g in GROUPS:
        original = [r for r in best_rows if r['query_group'] == g]
        s = paired_summary([r for r in original if not r['low_mass_flag']], 'best_exclude_low_mass_flag')
        s.update(cutoff='selected', query_group=g, unavailable_images=48-s['n']); sensitivity.append(s)
        for threshold in PROTOCOL['sensitivity_head_mass_thresholds']:
            restricted = []
            for q in main_ids:
                k = best_lookup[q]
                m = data[q]['metrics']['total_visual_mass'][[k-1, k], :, GROUPS.index(g)]
                r = boundary_row(q, g, k, data[q]['metrics'], data[q]['scores'], True, (m >= threshold).all(0))
                if r is not None:
                    restricted.append(r)
            s = paired_summary(restricted, f'best_paired_heads_mass_ge_{threshold:g}')
            s.update(cutoff='selected', query_group=g, unavailable_images=48-len(restricted)); sensitivity.append(s)
    # Dicts have the same columns unless a sensitivity cohort is empty.
    fields = list(dict.fromkeys(key for s in sensitivity for key in s))
    write_csv(OUTPUT/'low_mass_sensitivity.csv', [{key: s.get(key) for key in fields} for s in sensitivity])
    case_info = [case_plots(q, data[q]) for q in CASES]
    write_csv(OUTPUT/'case_grid_best.csv', case_info)
    spatial = [case_spatial(q, data[q]) for q in CASES]
    dump(OUTPUT/'spatial_cache_validation.json', spatial)
    for q in CASES:
        if Path(data[q]['sample']['image_path']).exists():
            inputs[str(Path(data[q]['sample']['image_path']))] = sha(data[q]['sample']['image_path'])
    scatter_plots(rows, fixed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = [sum(r['best_k_observed'] == k for r in best_info) for k in BEST_K]
    axes[0].bar([str(k) for k in BEST_K], counts)
    axes[0].set_xlabel('Observed best K (restricted sparse grid)'); axes[0].set_ylabel('Images / 48')
    for g in GROUPS:
        s = next(s for s in best_summary if s['query_group'] == g)
        axes[1].plot(['ROI down', 'Entropy up', 'Both'], [s['roi_down'], s['entropy_up'], s['roi_down_entropy_up']], '-o', label=g)
    axes[1].set_ylabel('Images / 48 at their selected K'); axes[1].legend()
    fig.suptitle('Post-hoc best K: exploratory selection, not independent confirmation')
    savefig(fig, 'main48_best_observed_k.png')
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    if 'torch' in sys.modules or 'transformers' in sys.modules:
        raise RuntimeError('Offline analysis unexpectedly imported model framework')
    validation = dict(passed=True, gpu_used=False, model_forward_count=0, imported_torch='torch' in sys.modules,
                      source_phase32_rows=1920, matched_main_images=48, independent_cases=5, fixed_primary_rows=192,
                      boundary_rows=len(rows), physical_score_rows=len(scores), max_recomputed_metric_csv_difference=maxdiff,
                      main_full_correct=sum(data[q]['scores'][32] > 0 for q in main_ids),
                      all_cached_attention_finite=True, bootstrap_replicates=PROTOCOL['bootstrap_replicates'],
                      all_source_hashes_verified=True, no_case_in_main_statistics=True,
                      wall_seconds=time.perf_counter()-started,
                      python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ['numpy', 'matplotlib', 'pillow']})
    dump(OUTPUT/'validation.json', validation)
    dump(OUTPUT/'summary.json', dict(protocol=PROTOCOL, fixed=fixed, best=best_summary, tie_sensitivity=tie_summaries, sensitivity=sensitivity, case_best=case_info))
    dump(OUTPUT/'checksums.json', dict(inputs=inputs, outputs={str(p.relative_to(ROOT)): sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != 'checksums.json'}))
    print(json.dumps(validation, indent=2)); print('Analysis complete; inputs untouched; no GPU/model/network access.')


if __name__ == '__main__':
    main()

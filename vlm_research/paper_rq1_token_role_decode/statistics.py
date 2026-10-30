"""Image-cluster statistics; no layer/token pseudo-replication."""
import numpy as np
from .protocol import SEED

BOOT = 10000
FAMILY = 22
CONTROLS = ['full_token_logp', 'full_token_entropy', 'generation_step', 'relative_generation_position', 'token_length']


def cluster_mean_ci(rows, values, family=1):
    values = np.asarray(values, dtype=float)
    if len(values) != len(rows) or not len(rows) or not np.isfinite(values).all():
        raise ValueError('Require finite aligned observations')
    ids, inverse = np.unique([r['image_sha256'] for r in rows], return_inverse=True)
    n = len(ids); sums = np.bincount(inverse, weights=values); counts = np.bincount(inverse)
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(n, np.full(n, 1/n), size=BOOT)
    samples = (weights @ sums) / (weights @ counts)
    lo, hi = np.quantile(samples, [.025, .975])
    fl, fh = np.quantile(samples, [.05/(2*family), 1-.05/(2*family)])
    return dict(mean=float(values.mean()), n_observations=len(rows), n_images=n,
        ci95_low=float(lo), ci95_high=float(hi), familywise95_low=float(fl), familywise95_high=float(fh),
        family=family, bootstrap_replicates=BOOT,
        interval_note='Ordinary/Bonferroni image-cluster percentile bootstrap; conditional observed-token sample, not a safety guarantee.')


def aggregate_positions(rows):
    grouped = {}
    for row in rows:
        key = row['image_id'], row['generation_step'], row['mode']
        grouped.setdefault(key, []).append(row)
    result = []
    for group in grouped.values():
        first = dict(group[0]); first['layer'] = 'mean9' if first['mode'] in ['Whole', 'Image'] else -1
        first['measured_layers'] = len(group)
        for name in ['loss', 'abs_delta_logp', 'relative_probability_change', 'top1_flip', 'js_divergence']:
            first[name] = float(np.mean([r[name] for r in group]))
        result.append(first)
    return result


def fixed_effects(rows):
    """Visual/Relation vs Function, within image, on nine-layer mean loss."""
    if not rows:
        return dict(status='no_samples'), []
    image_types = {}
    for row in rows: image_types.setdefault(row['image_sha256'], set()).add(row['word_type'])
    retained = {key for key, types in image_types.items() if {'Function', 'Visual'} <= types}
    rows = [r for r in rows if r['image_sha256'] in retained]
    if not rows: return dict(status='no_function_visual_pairs'), []
    ids, group = np.unique([r['image_sha256'] for r in rows], return_inverse=True)
    columns = ['Visual_vs_Function', 'Relation_vs_Function'] + CONTROLS
    x = np.array([[float(r['word_type'] == 'Visual'), float(r['word_type'] == 'Relation')] + [r[c] for c in CONTROLS] for r in rows], float)
    y = np.array([r['loss'] for r in rows], float)
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('Nonfinite controlled analysis')
    for i in range(len(ids)):
        mask = group == i; x[mask] -= x[mask].mean(0); y[mask] -= y[mask].mean()
    scale = np.sqrt((x*x).mean(0)); keep = scale > 1e-12
    dropped = [name for name, valid in zip(columns, keep) if not valid]
    if not keep[0] or not keep[1]:
        return dict(status='role_design_not_identifiable', dropped=dropped, n_images=len(ids)), []
    columns = [name for name, valid in zip(columns, keep) if valid]
    scale = scale[keep]; x = x[:, keep] / scale
    rank = np.linalg.matrix_rank(x, tol=np.linalg.svd(x, compute_uv=False)[0]*1e-10)
    meta = dict(n_positions=len(rows), n_images=len(ids), controls=CONTROLS, dropped_constant_columns=dropped,
        columns=columns, design_rank=int(rank), design_columns=x.shape[1], condition_number=float(np.linalg.cond(x)),
        bootstrap_replicates=BOOT, family=FAMILY, image_fixed_effects=True,
        interpretation='Adjusted conditional association in same-image Function/Visual-pair cohort, not causal effect of word role.')
    if rank != x.shape[1]:
        return dict(meta, status='rank_deficient_no_adjusted_claim'), []
    p = x.shape[1]; grams = np.zeros((len(ids), p, p)); rhs = np.zeros((len(ids), p))
    for i in range(len(ids)):
        xx = x[group == i]; yy = y[group == i]
        grams[i] = xx.T @ xx; rhs[i] = xx.T @ yy
    beta = np.linalg.solve(grams.sum(0), rhs.sum(0)) / scale
    rng = np.random.default_rng(SEED); weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=BOOT)
    matrices = np.einsum('bi,ijk->bjk', weights, grams); vectors = weights @ rhs
    eig = np.linalg.eigvalsh(matrices); valid = eig[:, 0] > eig[:, -1]*1e-10
    estimates = (np.linalg.pinv(matrices[valid], rcond=1e-10) @ vectors[valid, :, None]).squeeze(-1) / scale
    if not len(estimates): return dict(meta, status='no_full_rank_bootstrap_replicates'), []
    output = []
    for j, name in enumerate(columns):
        lo, hi = np.quantile(estimates[:, j], [.025, .975]); fl, fh = np.quantile(estimates[:, j], [.05/(2*FAMILY), 1-.05/(2*FAMILY)])
        output.append(dict(variable=name, effect=float(beta[j]), ci95_low=float(lo), ci95_high=float(hi),
            familywise95_low=float(fl), familywise95_high=float(fh), bootstrap_valid=int(valid.sum()),
            n_images=len(ids), n_positions=len(rows), family=FAMILY))
    return dict(meta, status='estimated', bootstrap_valid=int(valid.sum())), output

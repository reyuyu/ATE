"""Image-level descriptive intervals and equally weighted caption fixed effects."""
from functools import lru_cache
import numpy as np
from .common import SEED

BOOT = 10000


@lru_cache(maxsize=20)
def weights(ids):
    n = len(ids)
    return np.random.default_rng(SEED).multinomial(n, np.full(n, 1/n), size=BOOT)


def ci(values):
    assert values and all(np.isfinite(v) for v in values.values())
    ids = tuple(sorted(values))
    data = np.array([values[k] for k in ids])
    draws = weights(ids) @ data / len(ids)
    low, high = np.quantile(draws, [.025, .975])
    return dict(mean=float(data.mean()), ci95_low=float(low), ci95_high=float(high), n_images=len(ids))


def adjusted(rows, outcome, extra=()):
    groups = {}
    for r in rows:
        groups.setdefault(r['image_sha256'], []).append(r)
    groups = {k: v for k, v in groups.items() if {'Function', 'Visual'} <= {r['role'] for r in v}}
    if not groups:
        return dict(status='no_within_image_role_pairs'), []
    columns = ['Visual_vs_Function', 'Relation_vs_Function', 't', 'relative_t', 'logp', 'entropy', 'word_token_length', *extra]
    xs, ys, group_ids = [], [], []
    for image, group in sorted(groups.items()):
        x = np.array([[float(r['role'] == 'Visual'), float(r['role'] == 'Relation'),
                       r['t'], r['relative_t'], r['logp'], r['entropy'], r['word_token_length'],
                       *[r[c] for c in extra]] for r in group], float)
        y = np.array([r[outcome] for r in group], float)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        xs.append((x-x.mean(0))/np.sqrt(len(group)))
        ys.append((y-y.mean())/np.sqrt(len(group)))
        group_ids.append(image)
    x, y = np.concatenate(xs), np.concatenate(ys)
    scale = np.sqrt((x*x).mean(0))
    keep = scale > 1e-12
    dropped = [c for c, k in zip(columns, keep) if not k]
    columns, scale = [c for c, k in zip(columns, keep) if k], scale[keep]
    x = x[:, keep]/scale
    rank = np.linalg.matrix_rank(x, tol=np.linalg.svd(x, compute_uv=False)[0]*1e-10)
    meta = dict(images=len(group_ids), positions=len(y), dropped=dropped, rank=int(rank), columns=columns,
                condition_number=float(np.linalg.cond(x)), caption_equal_weight=True,
                history_key_count_collinear_with_t=True, controls='position, relative length, LogP, entropy, word length')
    if rank != x.shape[1]:
        return dict(meta, status='rank_deficient_no_inference'), []
    grams = np.stack([(a[:, keep]/scale).T@(a[:, keep]/scale) for a in xs])
    rhs = np.stack([(a[:, keep]/scale).T@b for a, b in zip(xs, ys)])
    estimate = np.linalg.solve(grams.sum(0), rhs.sum(0))/scale
    w = weights(tuple(group_ids))
    matrices = np.einsum('bi,ijk->bjk', w, grams)
    vectors = w@rhs
    eigen = np.linalg.eigvalsh(matrices)
    valid = eigen[:, 0] > eigen[:, -1]*1e-10
    assert valid.any(), 'No identifiable cluster resamples'
    samples = (np.linalg.pinv(matrices[valid], rcond=1e-10)@vectors[valid, :, None]).squeeze(-1)/scale
    low, high = np.quantile(samples, [.025, .975], axis=0)
    result = [dict(variable=c, effect=float(estimate[i]), ci95_low=float(low[i]), ci95_high=float(high[i]),
                   n_images=len(group_ids), positions=len(y), valid_bootstrap=int(valid.sum())) for i, c in enumerate(columns)]
    return dict(meta, status='estimated', valid_bootstrap=int(valid.sum())), result

"""Paired metrics with complete MME pairs and MMBench circular groups."""
from collections import defaultdict

import numpy as np


def complete_keys(task, planned, available):
    if task not in ('MME', 'MMBench'):
        return set(available)
    field = 'pair_id' if task == 'MME' else 'circular_group'
    groups = defaultdict(set)
    for row in planned:
        groups[row[field]].add(row['key'])
    return set().union(*(members for members in groups.values() if members <= available)) if groups else set()


def contributions(task, rows, mapping):
    clusters = sorted({mapping[r['key']]['cluster'] for r in rows})
    ix = {c: i for i, c in enumerate(clusters)}
    if task == 'MME':
        categories = sorted({r['mme_task'] for r in rows})
        if len(categories) != 14:
            return None, None
        values = np.zeros((len(clusters), 4*len(categories)))
        catix = {c: i for i, c in enumerate(categories)}
        pairs = defaultdict(list)
        for r in rows:
            i = ix[mapping[r['key']]['cluster']]
            j = 4*catix[r['mme_task']]
            values[i, j] += r['scoring']['score']
            values[i, j+1] += 1
            pairs[r['pair_id']].append(r)
        for pair in pairs.values():
            assert len(pair) == 2
            c = mapping[pair[0]['key']]['cluster']
            assert all(mapping[r['key']]['cluster'] == c for r in pair)
            i, j = ix[c], 4*catix[pair[0]['mme_task']]
            values[i, j+2] += all(r['scoring']['score'] == 1 for r in pair)
            values[i, j+3] += 1
        def evaluate(x):
            return 100*np.sum(x[..., 0::4]/x[..., 1::4] + x[..., 2::4]/x[..., 3::4], axis=-1)
    elif task == 'MMBench':
        groups = defaultdict(list)
        for r in rows:
            groups[r['circular_group']].append(r)
        values = np.zeros((len(clusters), 2))
        for group in groups.values():
            c = mapping[group[0]['key']]['cluster']
            assert all(mapping[r['key']]['cluster'] == c for r in group)
            values[ix[c]] += [all(r['scoring']['score'] == 1 for r in group), 1]
        def evaluate(x):
            return 100*x[..., 0]/x[..., 1]
    else:
        values = np.zeros((len(clusters), 2))
        for r in rows:
            values[ix[mapping[r['key']]['cluster']]] += [r['scoring']['score'], 1]
        def evaluate(x):
            return 100*x[..., 0]/x[..., 1]
    return (clusters, values), evaluate


def metric(task, rows, mapping):
    if not rows:
        return None
    pair, evaluate = contributions(task, rows, mapping)
    if pair is None:
        return None
    return float(evaluate(pair[1].sum(axis=0)))


def paired(task, a, b, mapping, repeats=20000, seed=20261111):
    if not a:
        return dict(delta=None, low=None, high=None, clusters=0)
    assert {r['key'] for r in a} == {r['key'] for r in b}
    aa, evaluate = contributions(task, a, mapping)
    bb, _ = contributions(task, b, mapping)
    if aa is None:
        return dict(delta=None, low=None, high=None, clusters=0)
    assert aa[0] == bb[0]
    x, y = aa[1], bb[1]
    delta = float(evaluate(y.sum(axis=0)) - evaluate(x.sum(axis=0)))
    if repeats == 0:
        return dict(delta=delta, low=None, high=None, clusters=len(x))
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repeats, 64):
        weights = rng.multinomial(len(x), np.full(len(x), 1/len(x)), size=min(64, repeats-start))
        with np.errstate(divide='ignore', invalid='ignore'):
            draws.extend((evaluate(weights @ y) - evaluate(weights @ x)).tolist())
    finite = np.asarray(draws)[np.isfinite(draws)]
    if len(finite) < .99*repeats:
        return dict(delta=delta, low=None, high=None, clusters=len(x), bootstrap_replicates=repeats,
            valid_bootstrap_replicates=len(finite), seed=seed,
            ci_status='UNAVAILABLE_MISSING_CATEGORY_IN_BOOTSTRAP; no imputation or altered resampling')
    low, high = np.quantile(finite, [.025, .975])
    return dict(delta=delta, low=float(low), high=float(high), clusters=len(x), bootstrap_replicates=repeats,
        valid_bootstrap_replicates=len(finite), seed=seed, ci_status='AVAILABLE',
        excluded_undefined_replicates=repeats-len(finite))


def pope(rows):
    tp = sum(r['gold'] == 'yes' and r['scoring']['parsed'] == 'yes' for r in rows)
    fp = sum(r['gold'] == 'no' and r['scoring']['parsed'] == 'yes' for r in rows)
    fn = sum(r['gold'] == 'yes' and r['scoring']['parsed'] == 'no' for r in rows)
    precision = tp/max(tp+fp, 1)
    recall = tp/max(tp+fn, 1)
    return dict(n=len(rows), accuracy=100*sum(r['scoring']['score'] for r in rows)/max(len(rows), 1),
        precision=100*precision, recall=100*recall, f1=100*2*precision*recall/max(precision+recall, 1e-30),
        yes_ratio=100*sum(r['scoring']['parsed']=='yes' for r in rows)/max(len(rows), 1))

"""Predeclared image-paired intervals and exact discordant-pair inference."""
import hashlib
import math

import numpy as np

from .protocol import RULES, GROUPS


def exact_mcnemar(harmed, recovered):
    if type(harmed) is not int or type(recovered) is not int or min(harmed, recovered) < 0:
        raise ValueError('Discordance counts must be nonnegative integers')
    n = harmed + recovered
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(harmed, recovered) + 1)) / 2**n) if n else 1.


def metrics(rows):
    full = np.array([r['full_margin'] for r in rows])
    b4 = np.array([r['b4_margin'] for r in rows])
    gain = b4 - full
    values = dict(full_accuracy=float(np.mean(full > 0)), b4_accuracy=float(np.mean(b4 > 0)),
                  accuracy_difference=float(np.mean((b4 > 0).astype(float) - (full > 0))),
                  mean_gain=float(gain.mean()), median_gain=float(np.median(gain)),
                  mean_absolute_gain=float(np.abs(gain).mean()), p5_gain=float(np.quantile(gain, .05)),
                  p95_gain=float(np.quantile(gain, .95)),
                  mean_gold_logp_change=float(np.mean([r['gold_logp_change'] for r in rows])),
                  mean_wrong_logp_change=float(np.mean([r['wrong_logp_change'] for r in rows])),
                  mean_right_minus_left_change=float(np.mean([r['right_minus_left_change'] for r in rows])))
    for layer in (0, 1):
        for group in GROUPS:
            for condition in ('full', 'b4'):
                key = f'l{layer}_{condition}_{group}_mass'
                values[key] = float(np.mean([r[key] for r in rows]))
            values[f'l{layer}_{group}_mass_change'] = values[f'l{layer}_b4_{group}_mass'] - values[f'l{layer}_full_{group}_mass']
            values[f'l{layer}_local_{group}_redistribution'] = float(np.mean([r[f'l{layer}_local_{group}_redistribution'] for r in rows]))
    return values


def paired_intervals(rows, label, resamples=None):
    settings = RULES['bootstrap']
    resamples = settings['resamples'] if resamples is None else resamples
    seed = settings['seed'] + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], 'big')
    rng = np.random.default_rng(seed)
    gain = np.array([r['gain'] for r in rows])
    full = np.array([r['full_margin'] > 0 for r in rows], dtype=float)
    b4 = np.array([r['b4_margin'] > 0 for r in rows], dtype=float)
    linear = dict(full_accuracy=full, b4_accuracy=b4, accuracy_difference=b4-full,
                  mean_gain=gain, mean_absolute_gain=np.abs(gain),
                  mean_gold_logp_change=np.array([r['gold_logp_change'] for r in rows]),
                  mean_wrong_logp_change=np.array([r['wrong_logp_change'] for r in rows]),
                  mean_right_minus_left_change=np.array([r['right_minus_left_change'] for r in rows]))
    for layer in (0, 1):
        for group in GROUPS:
            for condition in ('full', 'b4'):
                key = f'l{layer}_{condition}_{group}_mass'
                linear[key] = np.array([r[key] for r in rows])
            linear[f'l{layer}_{group}_mass_change'] = linear[f'l{layer}_b4_{group}_mass'] - linear[f'l{layer}_full_{group}_mass']
            key = f'l{layer}_local_{group}_redistribution'
            linear[key] = np.array([r[key] for r in rows])
    samples = {name: [] for name in list(linear) + ['median_gain', 'p5_gain', 'p95_gain']}
    for start in range(0, resamples, 500):
        indices = rng.integers(0, len(rows), (min(500, resamples-start), len(rows)))
        for name, vector in linear.items():
            samples[name].extend(vector[indices].mean(1).tolist())
        sampled_gain = gain[indices]
        for name, q in [('median_gain', .5), ('p5_gain', .05), ('p95_gain', .95)]:
            samples[name].extend(np.quantile(sampled_gain, q, axis=1).tolist())
    return {name: np.quantile(vector, [.025, .975]).tolist() for name, vector in samples.items()}


def summarize(rows, label, resamples=None):
    if not rows:
        return dict(n=0, status='empty', bootstrap_ci95=None)
    if len({r['img_id'] for r in rows}) != len(rows):
        raise ValueError('Image, not head/layer/question occurrence, is the sampling unit')
    harmed = sum(r['full_margin'] > 0 and r['b4_margin'] <= 0 for r in rows)
    recovered = sum(r['full_margin'] <= 0 and r['b4_margin'] > 0 for r in rows)
    return dict(n=len(rows), full_correct=sum(r['full_margin'] > 0 for r in rows),
                b4_correct=sum(r['b4_margin'] > 0 for r in rows), harmed=harmed, recovered=recovered,
                full_ties=sum(r['full_margin'] == 0 for r in rows), b4_ties=sum(r['b4_margin'] == 0 for r in rows),
                positive_gain=sum(r['gain'] > 0 for r in rows), negative_gain=sum(r['gain'] < 0 for r in rows),
                zero_gain=sum(r['gain'] == 0 for r in rows), exact_mcnemar_p=exact_mcnemar(harmed, recovered),
                **metrics(rows), bootstrap_ci95=paired_intervals(rows, label, resamples),
                worst_samples=sorted(rows, key=lambda r: (r['gain'], r['question_id']))[:10])


def summarize_cohorts(rows):
    result = {}
    for cohort in ['confirmation224', 'discovery16', 'all240']:
        rr = [r for r in rows if cohort == 'all240' or r['cohort'] == cohort]
        subsets = dict(all=rr, left=[r for r in rr if r['answer'] == 'left'],
                       right=[r for r in rr if r['answer'] == 'right'],
                       full_correct=[r for r in rr if r['full_margin'] > 0],
                       full_wrong=[r for r in rr if r['full_margin'] <= 0])
        result[cohort] = {name: summarize(values, cohort + '/' + name) for name, values in subsets.items()}
    return result

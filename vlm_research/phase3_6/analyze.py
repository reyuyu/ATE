"""Single master aggregation, only after four complete compatible shards."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import math
from pathlib import Path

import numpy as np
from .protocol import (CONDITIONS, DISCOVERY, RESEARCH, STATISTICS, atomic, checkpoint,
                       digest, protocol, read, sha)


def rank(values):
    x = np.asarray(values)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), float)
    start = 0
    while start < len(x):
        end = start+1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start+end-1)/2
        start = end
    return ranks


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def interval(values, operation='mean', other=None):
    x = np.asarray(values, float)
    y = np.asarray(other, float) if other is not None else None
    rng = np.random.default_rng(STATISTICS['bootstrap_seed'])
    points = []
    for indices in rng.integers(0, len(x), size=(STATISTICS['bootstrap_repetitions'], len(x))):
        if operation == 'pearson':
            value = correlation(x[indices], y[indices])
        elif operation == 'spearman':
            value = correlation(rank(x[indices]), rank(y[indices]))
        else:
            value = float(np.mean(x[indices]))
        if value is not None:
            points.append(value)
    return dict(low=float(np.quantile(points, .025)) if points else None,
                high=float(np.quantile(points, .975)) if points else None,
                valid_repetitions=len(points))


def paired_accuracy(full, current):
    a, b = np.asarray(full) > 0, np.asarray(current) > 0
    harmed = int(np.sum(a & ~b))
    recovered = int(np.sum(~a & b))
    n = harmed+recovered
    p = min(1., 2*sum(math.comb(n, k) for k in range(min(harmed, recovered)+1))/2**n) if n else 1.
    return dict(full_correct_to_wrong=harmed, full_wrong_to_correct=recovered,
                discordant=n, exact_mcnemar_two_sided_p=p,
                accuracy_difference=float(np.mean(b.astype(float)-a)),
                accuracy_difference_ci=interval(b.astype(float)-a.astype(float)))


def describe(rows):
    margins = np.array([r['margin'] for r in rows])
    full = np.array([r['full_margin'] for r in rows])
    gains = margins-full
    absolute = np.array([r['absolute_correct_logp_gain'] for r in rows])
    return dict(n=len(rows), accuracy=float(np.mean(margins > 0)), full_accuracy=float(np.mean(full > 0)),
                mean_correct_logp=float(np.mean([r['correct_logp'] for r in rows])),
                mean_full_correct_logp=float(np.mean([r['full_correct_logp'] for r in rows])),
                full_correct_retained=float(np.sum((full > 0) & (margins > 0))/np.sum(full > 0)) if np.any(full > 0) else None,
                tie_count=int(np.sum(margins == 0)), full_positive_to_negative=int(np.sum((full > 0) & (margins < 0))),
                full_positive_to_tie=int(np.sum((full > 0) & (margins == 0))),
                mean_gain=float(gains.mean()), median_gain=float(np.median(gains)),
                p5_gain=float(np.quantile(gains, .05)), p95_gain=float(np.quantile(gains, .95)),
                min_gain=float(gains.min()), max_gain=float(gains.max()),
                positive_gain_count=int(np.sum(gains > 0)), negative_gain_count=int(np.sum(gains < 0)),
                zero_gain_count=int(np.sum(gains == 0)), mean_gain_ci=interval(gains),
                severe_degradation_count=int(np.sum(gains <= STATISTICS['severe_gain_at_most'])),
                mean_absolute_correct_logp_gain=float(absolute.mean()),
                absolute_correct_logp_increased_count=int(np.sum(absolute > 0)),
                mean_absolute_correct_logp_gain_ci=interval(absolute),
                **paired_accuracy(full, margins))


def sample_acceptance(body, root):
    import json
    audit_name = next(name for name in body['artifacts'] if name.endswith('audit.json.gz'))
    with gzip.open(root / audit_name, 'rt') as stream:
        audit = json.load(stream)
    if not audit['passed'] or not audit['post_cleanup_native_exact'] or not audit['third_party_source_unchanged']:
        raise ValueError('Sample cleanup/library acceptance failed')
    if audit['input_contamination'] or audit['answer_query'] != len(audit['prompt_ids'])-1 or len(audit['visual_indices']) != 576:
        raise ValueError('Sample mapping/input acceptance failed')
    total = 0
    for condition in CONDITIONS[1:]:
        items = audit['conditions'][condition['name']]
        if len(items) != 2:
            raise ValueError('Candidate audit coverage mismatch')
        for i, candidate in enumerate(items):
            if candidate['question_hidden_sha256'] != audit['question_hidden_sha256'][i]:
                raise ValueError('Question hidden hash mismatch')
            if condition['strength'] == 0 and candidate['effective_text_logits_sha256'] != audit['original_full_text_sha256'][i]:
                raise ValueError('No-op effective-text hash mismatch')
            layers = candidate['layer_audits']
            if [a['layer'] for a in layers] != list(range(condition['start'], 32)):
                raise ValueError('Local layer audit coverage mismatch')
            for a in layers:
                if not all(a[key] for key in ['passed', 'other_queries_bitwise_exact', 'nonvisual_bitwise_exact',
                                              'finite_nonnegative', 'formula_rounded_target_bitwise_exact']):
                    raise ValueError('Local intervention audit failure')
                if a['query'] != audit['answer_query'] or a['visual_keys'] != 576 or a['row_sum_max_abs_error'] > .002:
                    raise ValueError('Query/visual/row acceptance failed')
                if not np.allclose(a['visual_mass_after'], a['visual_mass_before'], atol=2e-5, rtol=.001):
                    raise ValueError('Visual mass acceptance failed')
                if condition['strength'] != 0 and not a['conditional_rounding_bound_passed']:
                    raise ValueError('FP16 derived rounding bound failed')
                total += 1
    if total != 124 or audit['layer_audits'] != total:
        raise ValueError('Missing local audits')
    for row in body['rows']:
        numeric = [row[k] for k in ['margin', 'gain', 'correct_logp', 'wrong_logp', 'full_margin',
                                    'correct_logit', 'wrong_logit', 'full_vocabulary_logsumexp']]
        if not np.isfinite(numeric).all() or abs(row['margin']-(row['correct_logp']-row['wrong_logp'])) > 2e-12:
            raise ValueError('Invalid/nonfinite candidate scores')
        if abs(row['correct_logp']-(row['correct_logit']-row['full_vocabulary_logsumexp'])) > 2e-12:
            raise ValueError('Full-vocabulary absolute logP not consistent with normalizer')
    return True


def aggregate(output):
    if (output / 'summary_validation.json').exists():
        raise FileExistsError('Master aggregation already completed; do not aggregate twice')
    config, binding = protocol(output)
    cohort = {s['question_id']: s for s in config['samples']}
    bodies = {}
    for i in range(4):
        folder = output / f'shards/shard_{i}'
        gate = read(folder / 'validation.json')
        ids = config['shards'][i]
        if not gate['passed'] or gate['binding'] != binding or gate['code_commit'] != config['code_commit'] or gate['question_ids'] != ids or gate['count'] != 60:
            raise ValueError('All four frozen shard gates must pass before aggregation')
        if {p.stem for p in (folder / 'checkpoints').glob('*.json')} != set(ids):
            raise ValueError('Shard has missing, duplicate or foreign checkpoints')
        for qid in ids:
            path = folder / f'checkpoints/{qid}.json'
            if qid in bodies or sha(path) != gate['checkpoint_sha256'][qid]:
                raise ValueError('Duplicate/mutated completed checkpoint')
            body = checkpoint(path, binding, cohort[qid], folder)
            sample_acceptance(body, folder)
            bodies[qid] = body
    if set(bodies) != set(cohort):
        raise ValueError('Incomplete cohort; no population conclusions allowed')
    rows = [row for q in sorted(bodies) for row in bodies[q]['rows']]
    for name in [c['name'] for c in CONDITIONS]:
        if len([r for r in rows if r['condition'] == name]) != 240:
            raise ValueError('Condition denominator mismatch')
    cached = set(config['cached_question_ids'])
    cohorts = dict(primary_239=[q for q in sorted(cohort) if q != DISCOVERY],
                   uncached_187=[q for q in sorted(cohort) if q not in cached],
                   all_240=sorted(cohort), discovery_1=[DISCOVERY])
    if [len(v) for v in cohorts.values()] != [239, 187, 240, 1]:
        raise ValueError('Analysis cohort cardinality mismatch')
    lookup = {(r['question_id'], r['condition']): r for r in rows}
    summary, contrasts, correlations, directions = {}, {}, {}, {}
    for cohort_name, ids in cohorts.items():
        summary[cohort_name] = {}
        contrasts[cohort_name], correlations[cohort_name], directions[cohort_name] = {}, {}, {}
        for name in RESEARCH:
            rr = [lookup[q, name] for q in ids]
            groups = {'overall': rr, 'left': [r for r in rr if r['answer'] == 'left'],
                      'right': [r for r in rr if r['answer'] == 'right'],
                      'full_correct': [r for r in rr if r['full_margin'] > 0],
                      'full_wrong': [r for r in rr if r['full_margin'] <= 0],
                      'weak_margin': [r for r in rr if abs(r['full_margin']) <= .15]}
            summary[cohort_name][name] = {g: describe(v) for g, v in groups.items() if v}
            x = [r['gain'] for r in rr]
            y = [r['physical_k24_gain'] for r in rr]
            correlations[cohort_name][name] = dict(pearson=correlation(x, y), spearman=correlation(rank(x), rank(y)),
                         pearson_ci=interval(x, 'pearson', y), spearman_ci=interval(x, 'spearman', y),
                         note='Physical gain uses historical Full; inheritance gain uses current same-machine Full')
        for start, strength in [(24, .5), (24, 1), (25, 1)]:
            a, b = f'L23_from{start}_lambda{strength:g}', f'L24_from{start}_lambda{strength:g}'
            difference = np.array([lookup[q, a]['gain']-lookup[q, b]['gain'] for q in ids])
            contrasts[cohort_name][f'from{start}_lambda{strength:g}'] = dict(
                mean_paired_gain_difference=float(difference.mean()), median=float(np.median(difference)),
                positive_count=int(np.sum(difference > 0)), negative_count=int(np.sum(difference < 0)),
                zero_count=int(np.sum(difference == 0)), paired_mean_ci=interval(difference))
        for source in (23, 24):
            a = np.array([lookup[q, f'L{source}_from24_lambda0.5']['gain'] for q in ids])
            b = np.array([lookup[q, f'L{source}_from24_lambda1']['gain'] for q in ids])
            directions[cohort_name][f'L{source}'] = dict(n=len(ids),
                both_positive=int(np.sum((a > 0) & (b > 0))), both_negative=int(np.sum((a < 0) & (b < 0))),
                both_zero=int(np.sum((a == 0) & (b == 0))), opposite_sign=int(np.sum(a*b < 0)),
                monotone_nondecreasing=int(np.sum((a >= 0) & (b >= a))),
                monotone_nonincreasing=int(np.sum((a <= 0) & (b <= a))))
    with (output / 'per_sample.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(binding=binding, code_commit=config['code_commit'], statistics=STATISTICS,
                  cohort_question_ids=cohorts, summary=summary, paired_source_contrasts=contrasts,
                  physical_correlations=correlations, lambda_directions=directions,
                  severe_and_worst_samples={cohort_name: {name: sorted(
                      [lookup[q, name] for q in ids], key=lambda r: (r['gain'], r['question_id']))[:10]
                      for name in RESEARCH} for cohort_name, ids in cohorts.items()},
                  historical_full_differences={q: bodies[q]['historical_comparison'] for q in sorted(bodies)})
    comparisons = list(result['historical_full_differences'].values())
    cache_comparisons = [c['cache'] for c in comparisons if c['cache'] is not None]
    result['historical_comparison_summary'] = dict(
        candidate_logp_exact_images=sum(c['candidate_logp_exact'] for c in comparisons),
        full_margin_max_abs_difference=max(abs(c['full_margin_difference']) for c in comparisons),
        historical_full_correctness_changed=sum((c['historical_full_margin'] > 0) != (c['same_machine_full_margin'] > 0) for c in comparisons),
        cached_images_compared=len(cache_comparisons), cache_bitwise_exact_images=sum(c['bitwise_exact'] for c in cache_comparisons),
        cache_max_abs_difference=max(c['max_abs_difference'] for c in cache_comparisons))
    atomic(output / 'aggregate.json', result)
    plots(output, lookup, cohorts['primary_239'])
    report(output, result)
    atomic(output / 'summary_validation.json', dict(passed=True, status='completed', binding=binding,
           code_commit=config['code_commit'], images=240, conditions_per_image=9, research_conditions=6,
           rows=len(rows), layer_audits=sum(b['layer_audits'] for b in bodies.values()),
           timestamp=datetime.now(timezone.utc).isoformat(),
           artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'summary_validation.json'}))


def plots(output, lookup, ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'left': '#286eaa', 'right': '#c44862'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (start, strength) in zip(axes, [(24, .5), (24, 1), (25, 1)]):
        for label, color in colors.items():
            ii = [q for q in ids if lookup[q, 'full']['answer'] == label]
            x = [lookup[q, f'L24_from{start}_lambda{strength:g}']['gain'] for q in ii]
            y = [lookup[q, f'L23_from{start}_lambda{strength:g}']['gain'] for q in ii]
            ax.scatter(x, y, s=12, alpha=.7, c=color, label=label)
        lo, hi = min(*ax.get_xlim(), *ax.get_ylim()), max(*ax.get_xlim(), *ax.get_ylim())
        ax.plot([lo, hi], [lo, hi], '--', color='gray', lw=1)
        ax.axhline(0, lw=.5, color='black'); ax.axvline(0, lw=.5, color='black')
        ax.set(xlabel='L24 gain [nats]', ylabel='L23 gain [nats]', title=f'L{start}-L31; lambda={strength:g}')
        ax.legend()
    fig.suptitle('Paired source comparison: 239 images, discovery case excluded')
    fig.tight_layout(); fig.savefig(output / 'source_gain_scatter.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = ['full']+RESEARCH
    acc, harmed, restored = [], [], []
    for name in names:
        m = np.array([lookup[q, name]['margin'] for q in ids]); full = np.array([lookup[q, 'full']['margin'] for q in ids])
        acc.append(np.mean(m > 0)); harmed.append(np.sum((full > 0) & (m <= 0))); restored.append(np.sum((full <= 0) & (m > 0)))
    for ax, values, title in zip(axes, [acc, harmed, restored], ['Accuracy', 'Full correct to wrong', 'Full wrong to correct']):
        ax.bar(np.arange(len(names)), values, color=['#777777']+['#286eaa']*2+['#c44862']*2+['#39845b']*2)
        ax.set_xticks(np.arange(len(names)), names, rotation=65, ha='right', fontsize=7); ax.set_title(title)
    fig.tight_layout(); fig.savefig(output / 'accuracy_and_flips.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.flat, RESEARCH):
        for label, color in colors.items():
            ii = [q for q in ids if lookup[q, name]['answer'] == label]
            ax.scatter([lookup[q, name]['physical_k24_gain'] for q in ii],
                       [lookup[q, name]['gain'] for q in ii], s=12, alpha=.7, color=color, label=label)
        ax.axhline(0, color='gray', lw=.6); ax.axvline(0, color='gray', lw=.6)
        ax.set(title=name, xlabel='Historical Physical K24 gain', ylabel='Inheritance gain'); ax.legend()
    fig.tight_layout(); fig.savefig(output / 'physical_gain_scatter.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5))
    gains = [[lookup[q, name]['gain'] for q in ids] for name in RESEARCH]
    ax.boxplot(gains, tick_labels=RESEARCH, showfliers=True)
    rng = np.random.default_rng(36)
    for i, values in enumerate(gains, 1):
        ax.scatter(rng.uniform(i-.13, i+.13, len(values)), values, s=7, alpha=.4, color='#286eaa')
    ax.axhline(0, color='black', lw=.6); ax.set_ylabel('Gain [nats]; all negative extremes retained')
    ax.tick_params(axis='x', rotation=25); fig.tight_layout()
    fig.savefig(output / 'gain_distributions.png', dpi=160); plt.close(fig)


def report(output, result):
    rr = result['summary']['primary_239']
    lines = ['# Phase 3.6: 240-image Attention inheritance validation', '',
             'Status: completed. Four frozen 60-image shards passed; one master aggregation.',
             f"Frozen implementation: `{result['code_commit']}`. Primary analysis excludes 15133706 (n=239).",
             'The separate 187-image subgroup had no Phase3.4 task Attention cache. The other cached images are not automatically previously intervened examples.',
             'All 240 fresh Full references are from their own image; historical differences are retained. No training, decoding or physical rerun.', '',
             '| Condition | Accuracy | Harmed | Recovered | Mean gain | 95% paired CI | Positive gain | Absolute correct logP gain |',
             '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    for name in RESEARCH:
        r = rr[name]['overall']; ci = r['mean_gain_ci']
        lines.append(f"| {name} | {r['accuracy']:.4f} | {r['full_correct_to_wrong']} | {r['full_wrong_to_correct']} | {r['mean_gain']:+.6f} | [{ci['low']:+.6f}, {ci['high']:+.6f}] | {r['positive_gain_count']}/{r['n']} | {r['mean_absolute_correct_logp_gain']:+.6f} |")
    lines += ['', '## Research Questions', '']
    for name in RESEARCH:
        r = rr[name]['overall']; left, right = rr[name].get('left'), rr[name].get('right')
        corr = result['physical_correlations']['primary_239'][name]
        lines.append(f"- {name}: positive margin gain in {r['positive_gain_count']}/239, harmed {r['full_correct_to_wrong']} Full-correct images; left/right mean gain {left['mean_gain']:+.6f}/{right['mean_gain']:+.6f}. Physical Pearson/Spearman {corr['pearson']}/{corr['spearman']} (bootstrap intervals in aggregate.json).")
    for name, r in result['paired_source_contrasts']['primary_239'].items():
        ci = r['paired_mean_ci']
        lines.append(f"- Matched {name}: L23 minus L24 mean gain {r['mean_paired_gain_difference']:+.6f}, 95% paired CI [{ci['low']:+.6f}, {ci['high']:+.6f}], wins/losses/ties {r['positive_count']}/{r['negative_count']}/{r['zero_count']}.")
    supported = [name for name in RESEARCH if rr[name]['overall']['mean_gain_ci']['low'] > 0 and rr[name]['overall']['positive_gain_count'] > 239/2]
    lines += ['', ('Evidence is not confined to the discovery case for: '+', '.join(supported)+'. This does not establish a universal mechanism.' if supported else
                   'No frozen condition simultaneously has a majority of positive gains and a strictly positive paired mean-gain interval. These data do not support generalizing the discovery-case improvement.'),
              'Any continuation requires a new explicit protocol. Do not choose an after-the-fact best lambda, claim equivalence to Physical Exit, or interpret relative ranking gain as absolute correct-answer probability gain.', '',
              'All 240 supplemental results, the uncached 187 subgroup, Full-correct/Full-wrong/weak-margin strata, lambda directions, paired flips and exact paired statistics, tail losses and bootstrap correlations are in aggregate.json and per_sample.csv.',
              'Bootstrap resamples images with all conditions paired (5000 fixed-seed repetitions). Intervals are exploratory and not multiplicity adjusted. Ties are incorrect. No head/layer is an independent sample.', '',
              '![Paired sources](source_gain_scatter.png)', '![Accuracy and flips](accuracy_and_flips.png)',
              '![Physical comparison](physical_gain_scatter.png)', '![All gains](gain_distributions.png)']
    (output / 'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    aggregate(args.output_root.resolve())

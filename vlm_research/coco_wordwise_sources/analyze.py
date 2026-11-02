"""All subwords retained; word analyses explicitly use first subword positions."""
from collections import defaultdict, Counter
import json
import numpy as np
from scipy.stats import rankdata

from .common import *
from .statistics import ci, adjusted


def js(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a/a.sum(), b/b.sum()
    m = (a+b)/2
    return float(.5*(sum(a[a > 0]*np.log(a[a > 0]/m[a > 0]))+sum(b[b > 0]*np.log(b[b > 0]/m[b > 0]))))


def primary_mass(row):
    return np.array([row[0], row[1], row[2], row[3]+row[4]], float)


def mean_by(rows, value):
    groups = defaultdict(list)
    for r in rows:
        groups[r['image_sha256']].append(float(r[value]))
    return {k: float(np.mean(v)) for k, v in groups.items()}


def main():
    config = verify()
    assert read(OUT / 'collection_completion.json')['complete'] and read(OUT / 'gpu_smoke.json')['passed']
    samples = read(OUT / 'sample_manifest.json')['samples']
    tokens, layer_rows, contribution_rows, image_variation = [], [], [], []
    checkpoints, raw_by_image = {}, {}
    for sample in samples:
        r = read(OUT / 'checkpoints' / (sample['image_identity'].replace(':', '_')+'.json'))
        assert r['binding'] == digest(config) and r['sample_sha256'] == digest(sample)
        assert sha256(r['raw_path']) == r['raw_sha256'] and sha256(OUT / r['head_csv']) == r['head_sha256']
        assert r['native_observer_logits_exact']
        checkpoints[r['image_id']] = r
        with np.load(r['raw_path']) as data:
            masses = data['mass'].mean(2).astype(float)
            counts = data['key_counts']
            assert len(masses) == len(r['token_ids'])
        raw_by_image[r['image_id']] = masses
        base = dict(image_id=r['image_id'], image_sha256=r['image_sha256'])
        per_t = []
        for t, label in enumerate(r['labels']):
            consumed = r['labels'][t-1] if t else None
            token = dict(base, t=t, phase='decode' if t else 'prefill', token_id=r['token_ids'][t],
                predicted_word=label['lexical_word'], predicted_piece=label['decoded_piece'],
                role=label['role'], raw_role=label['raw_role'], word_first=label['word_first'],
                subword_index=label['subword_index'], word_token_length=label['word_token_length'],
                word_first_step=label['word_first_step'], rule_source=label['rule_source'],
                consumed_token_id=r['token_ids'][t-1] if t else None,
                consumed_word=consumed['lexical_word'] if consumed else None,
                consumed_role=consumed['role'] if consumed else None,
                consumed_piece=consumed['decoded_piece'] if consumed else None,
                logp=r['token_logp'][t], entropy=r['token_entropy'][t],
                relative_t=t/max(len(r['token_ids'])-1, 1),
                caption_tokens=len(r['token_ids']), eos=label['role'] == 'EOS')
            tokens.append(token)
            per_t.append(token)
            for layer in range(32):
                mass = masses[t, layer]
                primary = primary_mass(mass)
                nonbos = primary[1:].sum()
                assert nonbos > 0
                record = dict(token, layer=layer,
                    **{name: float(mass[j]) for j, name in enumerate(SOURCES)},
                    GENERATED_HISTORY=float(mass[3]+mass[4]),
                    **{'n_'+name: int(counts[t, j]) for j, name in enumerate(SOURCES)},
                    n_GENERATED_HISTORY=int(counts[t, 3]+counts[t, 4]),
                    **{'nonbos_'+name: float(primary[j]/nonbos) for j, name in enumerate(MAIN) if j},
                    image_history_ratio=float(primary[1]/primary[3]) if primary[3] > 0 else None,
                    image_per_key=float(primary[1]/counts[t, 1]),
                    history_per_key=float(primary[3]/sum(counts[t, 3:])) if t else None,
                    adjacent_js=js(primary, primary_mass(masses[t-1, layer])) if t >= 2 else None,
                    adjacent_nonbos_js=js(primary[1:], primary_mass(masses[t-1, layer])[1:]) if t >= 2 else None)
                layer_rows.append(record)
        for c in r['contributions']:
            record = dict(per_t[c['t']], **{k: v for k, v in c.items() if k != 't'})
            layer = c['layer']
            current = primary_mass(masses[c['t'], layer])
            previous = primary_mass(masses[c['t']-1, layer])
            near = c['t'] >= 2 and js(current, previous) <= 1e-4 and js(current[1:], previous[1:]) <= 1e-3
            norm = c['adjacent_relative_norm_change']
            cosine = c['adjacent_source_cosine']
            record.update(near_mass=bool(near),
                contribution_changed=bool((norm is not None and norm >= .2) or (cosine is not None and cosine <= .9)))
            contribution_rows.append(record)
        for mode in ['all_decode', 'non_eos_decode']:
            positions = [t for t in range(1, len(masses)) if mode == 'all_decode' or not per_t[t]['eos']]
            assert positions
            for layer in range(32):
                series = np.stack([primary_mass(masses[t, layer]) for t in positions])
                for j, source in enumerate(MAIN):
                    values = series[:, j]
                    image_variation.append(dict(base, analysis=mode, layer=layer, source=source,
                        positions=len(values), minimum=float(values.min()), maximum=float(values.max()),
                        mean=float(values.mean()), range=float(np.ptp(values)), sd=float(values.std())))
    csv_write(OUT / 'per_token.csv', tokens)
    csv_write(OUT / 'per_token_layer_mass.csv', layer_rows)
    csv_write(OUT / 'per_token_contributions.csv', contribution_rows)
    csv_write(OUT / 'within_caption_variation.csv', image_variation)
    write(OUT / 'token_coverage.json', dict(images=len(samples), tokens=len(tokens),
        decode_tokens=sum(r['t'] > 0 for r in tokens), roles=dict(Counter(r['role'] for r in tokens)),
        multi_token_words=sum(w['token_length'] > 1 for r in checkpoints.values() for w in r['words']),
        nonfirst_lexical_subwords=sum(r['subword_index'] is not None and r['subword_index'] > 0 for r in tokens),
        alignment_failures=0, ambiguous_retained_as_unknown=sum(r['raw_role'] == 'Ambiguous' for r in tokens),
        prefill_separate=True, historical_caption_matches=sum(r['historical_tokens_exact'] for r in checkpoints.values())))

    variation_summary = []
    for mode in ['all_decode', 'non_eos_decode']:
        for layer in range(32):
            for source in MAIN:
                selected = [r for r in image_variation if r['analysis'] == mode and r['layer'] == layer and r['source'] == source]
                for metric in ['minimum', 'maximum', 'mean', 'range', 'sd']:
                    variation_summary.append(dict(analysis=mode, layer=layer, source=source, metric=metric,
                                                  **ci({r['image_sha256']: r[metric] for r in selected})))
    csv_write(OUT / 'layer_variation_summary.csv', variation_summary)
    primary = [r for r in layer_rows if r['layer'] == 31 and r['t'] >= 1 and r['word_first'] and r['role'] in ['Function', 'Visual', 'Relation']]
    role_summary, paired, adjusted_rows, adjusted_audit = [], [], [], []
    common = set.intersection(*[{r['image_sha256'] for r in primary if r['role'] == role} for role in ['Function', 'Visual', 'Relation']])
    outcomes = MAIN + ['nonbos_IMAGE', 'nonbos_PROMPT_TEXT', 'nonbos_GENERATED_HISTORY', 'image_history_ratio', 'image_per_key', 'history_per_key']
    for outcome in outcomes:
        by_role = {role: mean_by([r for r in primary if r['role'] == role], outcome) for role in ['Function', 'Visual', 'Relation']}
        for role, values in by_role.items():
            role_summary.append(dict(role=role, outcome=outcome, **ci(values)))
            restricted = {k: v for k, v in values.items() if k in common}
            if restricted:
                role_summary.append(dict(role=role+'_common3', outcome=outcome, **ci(restricted)))
        for left, right in [('Visual', 'Function'), ('Relation', 'Function'), ('Visual', 'Relation')]:
            available = sorted(set(by_role[left]) & set(by_role[right]))
            if available:
                paired.append(dict(contrast=left+'-'+right, outcome=outcome,
                    **ci({k: by_role[left][k]-by_role[right][k] for k in available})))
        audit, estimates = adjusted(primary, outcome)
        adjusted_audit.append(dict(outcome=outcome, **audit))
        adjusted_rows.extend(dict(outcome=outcome, **r) for r in estimates)
    csv_write(OUT / 'word_role_summary.csv', role_summary)
    csv_write(OUT / 'word_role_paired.csv', paired)
    if adjusted_rows:
        csv_write(OUT / 'word_role_adjusted.csv', adjusted_rows)
    write(OUT / 'regression_audit.json', adjusted_audit)

    consistency, correlations, contribution_summary = [], [], []
    for layer in SELECTED:
        for source in MAIN:
            selected = [r for r in contribution_rows if r['layer'] == layer and r['source'] == source]
            for value in ['mass', 'contribution_norm', 'relative_residual_ratio', 'cosine_with_full_attention', 'cosine_with_residual']:
                valid = [r for r in selected if r[value] is not None]
                contribution_summary.append(dict(layer=layer, source=source, metric=value, **ci(mean_by(valid, value))))
            groups = defaultdict(list)
            for row in selected:
                groups[row['image_sha256']].append(row)
            pearson, spearman = {}, {}
            for image, group in groups.items():
                x = np.array([r['mass'] for r in group])
                y = np.array([r['relative_residual_ratio'] for r in group])
                if x.std() > 0 and y.std() > 0 and len(x) >= 3:
                    pearson[image] = float(np.corrcoef(x, y)[0, 1])
                    spearman[image] = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
            for kind, values in [('Pearson', pearson), ('Spearman', spearman)]:
                if values:
                    correlations.append(dict(layer=layer, source=source, correlation=kind,
                        undefined_images=200-len(values), **ci(values)))
            for eos in [False, True]:
                eligible = [r for r in selected if r['t'] >= 2 and r['near_mass'] and (eos or not r['eos'])]
                if eligible:
                    rates = mean_by(eligible, 'contribution_changed')
                    consistency.append(dict(layer=layer, source=source, include_eos=eos, eligible_pairs=len(eligible),
                        changed_pairs=sum(r['contribution_changed'] for r in eligible),
                        images_with_change=len({r['image_sha256'] for r in eligible if r['contribution_changed']}), **ci(rates)))
    csv_write(OUT / 'mass_contribution_consistency.csv', consistency)
    csv_write(OUT / 'mass_contribution_correlations.csv', correlations)
    csv_write(OUT / 'contribution_summary.csv', contribution_summary)
    # All word tails are retained in the main files; this table makes their difference explicit.
    tails = []
    for kind in ['first_subword', 'later_subword', 'Punctuation', 'EOS', 'Unknown']:
        selected = [r for r in layer_rows if r['layer'] == 31 and r['t'] >= 1 and
            ((kind == 'first_subword' and r['word_first']) or
             (kind == 'later_subword' and r['subword_index'] is not None and r['subword_index'] > 0) or
             (kind in ['Punctuation', 'EOS', 'Unknown'] and r['role'] == kind))]
        for source in MAIN:
            if selected:
                tails.append(dict(subset=kind, source=source, tokens=len(selected), **ci(mean_by(selected, source))))
    csv_write(OUT / 'subword_and_eos_summary.csv', tails)
    from .figures import figures
    figures(checkpoints, raw_by_image, role_summary, paired, contribution_rows, consistency)
    from .report import report
    report(checkpoints, variation_summary, paired, adjusted_rows, consistency, correlations)


if __name__ == '__main__':
    main()

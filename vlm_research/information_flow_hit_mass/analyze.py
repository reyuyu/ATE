"""Descriptive, question-paired summaries; no VQA accuracy or layer-level inference."""
from collections import Counter, defaultdict
import csv
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .run import OUTPUT, ROOT, load_config, write_csv
from .attention import FIELDS
from information_flow_window9.analyze import NAMES, LABELS, COLORS
from information_flow_window9.masks import PATHS
from information_flow_window9 import run as parent
from benchmarks.dataset_registry import sha256


def rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return None
    a, b = (rank(values) for values in zip(*pairs))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def hit_stats(rows):
    n = len(rows)
    full = sum(r['full_first_token_hit'] for r in rows)
    hit = sum(r['masked_first_token_hit'] for r in rows)
    harmed = sum(r['hit_to_miss'] for r in rows)
    rescued = sum(r['miss_to_hit'] for r in rows)
    if hit - full != rescued - harmed:
        raise ValueError('Flip accounting mismatch')
    return {'n': n, 'full_hits': full, 'masked_hits': hit,
            'full_first_token_hit_rate_pct': 100 * full / n,
            'masked_first_token_hit_rate_pct': 100 * hit / n,
            'first_token_hit_delta_pp': 100 * (hit - full) / n,
            'hit_to_miss': harmed, 'miss_to_hit': rescued}


def collect():
    _, manifest = load_config()
    binding = json.loads((OUTPUT / 'execution_binding.json').read_text())
    completion = json.loads((OUTPUT / 'completion.json').read_text())
    if completion['status'] != 'scan_complete' or completion['binding'] != binding:
        raise ValueError('Partial or mixed-code run')
    data = []
    expected = {(p, c) for p in PATHS for c in range(32)}
    for sample in manifest['samples']:
        key = f'{sample["dataset"]}__{sample["question_id"]}'
        item = json.loads((OUTPUT / 'checkpoints' / f'{key}.json').read_text())
        rows = item['rows']
        if item['binding'] != binding or not item['native_restored_bitwise'] or len(rows) != 96:
            raise ValueError('Invalid checkpoint')
        if {(r['path'], r['center']) for r in rows} != expected:
            raise ValueError('Missing or duplicate path/center')
        if any((r['dataset'], r['question_id']) != (sample['dataset'], sample['question_id']) for r in rows):
            raise ValueError('Input/output sample mismatch')
        if sha256(ROOT / item['head_mass_path']) != item['head_mass_sha256']:
            raise ValueError('Head mass artifact changed')
        if [r['layer'] for r in item['native_mass']] != list(range(32)) or [r['center'] for r in item['window_mass']] != list(range(32)):
            raise ValueError('Missing mass layers/windows')
        data.append(item)
    if len(data) != 96 or Counter(d['baseline']['dataset'] for d in data) != Counter({t: 24 for t in parent.TASKS}):
        raise ValueError('Incomplete frozen 96-question cohort')
    return data, binding


def figure(task, curves, masses):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True, constrained_layout=True)
    native = [r for r in masses if r['dataset'] == task]
    for field, label, color in [('last_bos_mass', 'P-1 -> BOS', '#76599b'),
                                 ('last_image_mass', 'P-1 -> Image', '#3575b5'),
                                 ('last_sink_mass', 'P-1 -> Visual Sink (subset of Image)', '#248458')]:
        axes[0].plot([r['layer'] for r in native], [100*r[field] for r in native], lw=2, label=label, color=color)
    axes[0].set_ylabel('Native raw attention mass (%)')
    axes[0].set_title('Native Full: per-layer P-1 mass, mean over questions and heads', fontsize=11)
    axes[0].legend(loc='best', fontsize=9, frameon=False)
    for path in PATHS:
        series = [r for r in curves if r['dataset'] == task and r['path'] == path]
        x = [r['center'] for r in series]
        axes[1].plot(x, [r['mean_probability_change_pct'] for r in series], color=COLORS[path], lw=2, label=LABELS[path])
        axes[2].plot(x, [r['masked_first_token_hit_rate_pct'] for r in series], color=COLORS[path], lw=2, label=LABELS[path])
    full = next(r['full_first_token_hit_rate_pct'] for r in curves if r['dataset'] == task)
    axes[1].axhline(0, color='#666666', lw=.7)
    axes[1].set_ylabel('Fixed Full-token probability change (%)')
    axes[1].set_title('Original frozen probability curves; rerun verified exactly', fontsize=11)
    axes[1].legend(loc='best', fontsize=9, frameon=False)
    axes[2].axhline(full, color='#222222', linestyle='--', lw=1.3, label=f'Native Full ({full:.2f}%)')
    axes[2].set_ylim(0, 103)
    axes[2].set_ylabel('First-token hit rate (%)')
    axes[2].set_title('First-token text hit only, NOT complete-answer / VQA accuracy', fontsize=11)
    axes[2].legend(loc='best', fontsize=9, frameon=False)
    axes[2].set_xlabel('Layer (top) / clipped 9-layer window center (middle and bottom)')
    for ax in axes:
        ax.set_xlim(0, 31)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
        ax.grid(alpha=.18)
        ax.tick_params(labelbottom=True)
    fig.suptitle(f'{NAMES[task]} | frozen n=24 | LLaVA-1.5-7B', fontsize=15)
    for extension in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'{task}_mass_probability_hits.{extension}', dpi=180)
    plt.close(fig)


def main():
    data, binding = collect()
    rows = [r for item in data for r in item['rows']]
    native, window_mass, baselines = [], [], []
    groups = defaultdict(list)
    for item in data:
        b = item['baseline']
        identity = {k: b[k] for k in ('dataset', 'question_id', 'image_id')}
        native.extend({**identity, **r, 'sink_count': len(item['sink_indices'])} for r in item['native_mass'])
        window_mass.extend({**identity, **r, 'sink_count': len(item['sink_indices'])} for r in item['window_mass'])
        baselines.append({**identity, 'native_token_id': b['target_token_id'], 'native_token_text': b['target_token_text'],
                          'gold_answer': b['gold_answer'], 'full_first_token_hit': b['full_first_token_hit'],
                          'sink_count': len(item['sink_indices']), 'sink_source': item['sink_source']})
    for r in rows:
        groups[r['dataset'], r['path'], r['center']].append(r)
    old_curves = {(r['dataset'], r['path'], int(r['center'])): float(r['mean_probability_change_pct'])
                  for r in csv.DictReader((parent.OUTPUT / 'curves.csv').open()) if r['subset'] == 'all'}
    curves, summary = [], []
    for task in parent.TASKS:
        for path in PATHS:
            series = []
            for center in range(32):
                group = groups[task, path, center]
                mean = statistics.mean(r['probability_change_pct'] for r in group)
                if mean != old_curves[task, path, center]:
                    raise ValueError('Probability aggregate differs from original curve')
                row = {'dataset': task, 'path': path, 'center': center,
                       'mean_probability_change_pct': old_curves[task, path, center], **hit_stats(group)}
                curves.append(row)
                series.append(row)
            peak = min(series, key=lambda r: r['mean_probability_change_pct'])
            worst_hit = min(series, key=lambda r: r['masked_first_token_hit_rate_pct'])
            summary.append({**peak, 'peak_window_start': max(0, peak['center']-4),
                            'peak_window_end': min(31, peak['center']+4),
                            'minimum_hit_center': worst_hit['center'],
                            'minimum_first_token_hit_rate_pct': worst_hit['masked_first_token_hit_rate_pct'],
                            'minimum_hit_delta_pp': worst_hit['first_token_hit_delta_pp']})
    mean_native = []
    mass_fields = list(FIELDS) + ['last_sink_over_image', 'question_sink_over_image']
    for task in parent.TASKS:
        for layer in range(32):
            group = [r for r in native if r['dataset'] == task and r['layer'] == layer]
            mean_native.append({'dataset': task, 'layer': layer, 'n': len(group),
                **{f: statistics.mean(r[f] for r in group if r[f] is not None) if any(r[f] is not None for r in group) else None
                   for f in mass_fields}})
    correlations = []
    for item in data:
        b = item['baseline']
        for path in PATHS:
            series = sorted([r for r in item['rows'] if r['path'] == path], key=lambda r: r['center'])
            mass = item['window_mass']
            group = 'question' if path == 'image_to_question' else 'last'
            predictors = {'p_minus_1_bos_mass': 'last_bos_mass',
                          'path_sink_over_image': f'{group}_sink_over_image',
                          'path_raw_sink_mass': f'{group}_sink_mass'}
            drop = [-r['probability_change_pct'] for r in series]
            for name, field in predictors.items():
                correlations.append({'dataset': b['dataset'], 'question_id': b['question_id'], 'path': path,
                    'predictor': name, 'mass_field': field, 'windows': 32,
                    'spearman_with_signed_probability_drop': spearman([r[field] for r in mass], drop)})
    association = []
    for task in parent.TASKS:
        for path in PATHS:
            for predictor in ('p_minus_1_bos_mass', 'path_sink_over_image', 'path_raw_sink_mass'):
                values = [r['spearman_with_signed_probability_drop'] for r in correlations
                          if r['dataset'] == task and r['path'] == path and r['predictor'] == predictor
                          and r['spearman_with_signed_probability_drop'] is not None]
                association.append({'dataset': task, 'path': path, 'predictor': predictor,
                    'valid_questions': len(values), 'median_within_question_spearman': statistics.median(values) if values else None,
                    'negative_rho_questions': sum(v < 0 for v in values), 'positive_rho_questions': sum(v > 0 for v in values)})
    for filename, content in [('per_question.csv', rows), ('baselines.csv', baselines), ('native_layer_mass.csv', native),
                               ('native_window_mean_mass.csv', window_mass), ('mean_native_layer_mass.csv', mean_native),
                               ('curves.csv', curves), ('sensitivity_hit_summary.csv', summary),
                               ('association_per_question.csv', correlations), ('association_summary.csv', association)]:
        write_csv(OUTPUT / filename, content)
    for task in parent.TASKS:
        figure(task, curves, mean_native)
    table = ['| Task | Path | Probability peak center/window | Probability change (%) | Full first-token hit (%) | Masked hit (%) | Delta (pp) | Hit->miss / Miss->hit |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        table.append(f'| {NAMES[r["dataset"]]} | {LABELS[r["path"]]} | L{r["center"]} / L{r["peak_window_start"]}-L{r["peak_window_end"]} | '
                     f'{r["mean_probability_change_pct"]:.2f} | {r["full_first_token_hit_rate_pct"]:.2f} | '
                     f'{r["masked_first_token_hit_rate_pct"]:.2f} | {r["first_token_hit_delta_pp"]:+.2f} | '
                     f'{r["hit_to_miss"]} / {r["miss_to_hit"]} |')
    assoc_table = ['| Task | Path | Median rho(BOS, drop) | Median rho(Sink/Image share, drop) |', '|---|---|---:|---:|']
    for task in parent.TASKS:
        for path in PATHS:
            matched = {r['predictor']: r for r in association if r['dataset'] == task and r['path'] == path}
            def fmt(name):
                row = matched[name]
                value = row['median_within_question_spearman']
                return f'{value:+.3f} (n={row["valid_questions"]})' if value is not None else 'undefined'
            assoc_table.append(f'| {NAMES[task]} | {LABELS[path]} | {fmt("p_minus_1_bos_mass")} | {fmt("path_sink_over_image")} |')
    figures = '\n\n'.join(f'### {NAMES[t]}\n\n![{NAMES[t]}](./{t}_mass_probability_hits.png)' for t in parent.TASKS)
    config = json.loads((OUTPUT / 'config.json').read_text())
    declined = sum(r['first_token_hit_delta_pp'] < 0 for r in summary)
    report = f'''# GPU1 window9 continuation: first-token hits and Native attention mass

Status: complete scan and aggregation, **96 original questions / 9,216 masked forwards / 3,072 Native layer rows**. No resampling, training, full Decode or other window lengths. Old source/results and main were not edited.

**Every hit rate below is FIRST-TOKEN TEXT HIT, NOT full-answer exact match or VQA accuracy.** The parent strip/lower comparison with the first Gold token is unchanged, including its capitalization/BPE and multi-token-prefix limitations.

## Probability-sensitive windows and hit changes

{chr(10).join(table)}

The peak is selected only by the **original frozen probability curve**, not by the new hit rate. At these 12 path/task peaks, {declined}/12 have lower first-token hit rate than Native. One question is 4.17 percentage points (n=24); probability changes are relative percentages, while hit-rate deltas are percentage points. `sensitivity_hit_summary.csv` also retains each path's minimum-hit center, without replacing the original peak.

## Figures

{figures}

Top: Native raw P-1 attention masses, summed over the named keys and mean over heads/questions. Sink is a subset of Image, not a separate partition. Middle: immutable original all-question probability curves, reproduced exactly by the rerun. Bottom: the three new masked first-token-hit curves with Native baseline. Top x is a single layer; middle/bottom x is the centered nine-layer window, clipped at boundaries.

## BOS/Sink association (descriptive only)

{chr(10).join(assoc_table)}

Each rho is computed **within a question** across the same 32 overlapping windows, comparing mean Native window occupancy against signed probability drop (-relative change); the table takes the median rho over questions. Negative rho means higher occupancy accompanies lower drop. Constant/undefined sequences are excluded and valid question counts are explicit. These are not independent layer samples, significance tests or evidence that Sink causes robustness. Shared layer depth, overlapping windows and prompt routing can confound the association. BOS always refers to P-1; Sink/Image share uses Question queries for Image->Question, P-1 for the two Last paths. Raw-Sink-mass associations and per-question rhos are also saved.

## Sink and mass protocol

- Exactly the original manifest, sample order, prompts, batch4, FP16/eager LLaVA-1.5-7B, Transformers 4.49.0, no-cache prefill, GPU1 UUID {parent.GPU}.
- Reused Phase 3.4 `sink_candidates` unchanged: two discovery neutral prompts x Question/P-1 views; visual-conditional probability >=5/576 and top58, persistence in >=8 layers, candidate in >=3/4 views. Third neutral prompt is holdout only. All Sink candidates used, with no ROI exclusion or answer/hit-based selection. Empty sets remain empty.
- {config['historical_sink_images']} unique images reuse hash-verified Phase 3.4 labels/maps; {config['new_neutral_images']} unique images use the identical three neutral prompts and the same two-stage FP16 rounding. All 95 unique-image Sink definitions frozen before masked scoring. One cross-task shared image keeps its own single definition.
- Native raw mass: sum key weights, then equal head mean; Question mass additionally averages true Question queries, excluding template/instruction tokens. Five measurements per head: P-1->BOS/Image/Sink, Question->Image/Sink. Stored head summaries are [32 layers,32 heads,5] FP32, NOT full attention matrices.
- Sink/Image share = aggregate raw Sink mass / aggregate Image mass. No division by BOS or total text mass. Zero Image denominator -> null. For window means, average raw masses over exactly the same clipped window, then take their ratio (not mean per-head/layer ratios).
- Every Native prediction ID/LogP and every masked fixed-token LogP matches the old checkpoint exactly. Original probabilities/curves were not overwritten. Native vocabulary logits restore bitwise after intervention for all 96 questions. Observer-on/off equality and mask-direction checks passed in four real smoke questions; CPU tests cover mass/ratio/flip/rank calculations and cleanup. No OOM/nonfinite scores.

## Provenance and files

- Parent result commit: `{config['parent_commit']}`.
- Frozen execution commit: `{binding['code_commit']}`; full source SHA256/config bindings saved.
- `per_question.csv`: masked argmax ID/text, Native token, first-token hit/flip flags, original probability and LogP changes (9,216 rows).
- `baselines.csv`, `curves.csv`, `sensitivity_hit_summary.csv`: Native hits, every window's hit rates and flips, frozen-peak comparison.
- `native_layer_mass.csv`, `native_window_mean_mass.csv`: all five raw masses plus both Sink/Image ratios per question/layer or clipped window (3,072 rows each).
- `native_head_mass/*.npz`: per-head raw masses, field order in `attention.py:FIELDS`.
- `association_per_question.csv`, `association_summary.csv`: descriptive question-level associations, no p-values.
- `sink_manifest.json`, `sinks/`, `neutral_maps/`: old/new Sink provenance and neutral-only definitions.
- `config.json`, source bindings, environment, CPU/GPU logs, `smoke.json`, `completion.json`, checkpoints: validation/provenance. Parent hashes rechecked during aggregation.
'''
    (OUTPUT / 'report.md').write_text(report)
    print('\n'.join(table), flush=True)
    print('\n'.join(assoc_table), flush=True)


if __name__ == '__main__':
    main()

"""CPU-only same-sample alignment; historical A-Skip scores are never recomputed."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from exit_boundary_analysis import ranks

from benchmarks.dataset_registry import freeze_json
from .run import OUTPUT, PARENT, UNITS, verify

MASS = ('p_bos', 'p_image', 'q_bos', 'q_image')
SPATIAL = ('p_sink_mass', 'p_sink_share', 'p_sink_nonroi_mass', 'p_sink_nonroi_share',
           'p_roi_mass', 'p_roi_share', 'p_roi_enrichment', 'q_roi_mass', 'q_roi_share', 'q_roi_enrichment')


def write_csv(path, rows):
    if path.exists():
        raise ValueError(f'Never overwrite completed analysis: {path}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError(f'Invalid frozen boolean {value!r}')
    return value == 'True'


def window_values(rows, start):
    selected = [r for r in rows if start <= r['layer'] <= start+2]
    if len(selected) != 3:
        raise ValueError('Missing or duplicate layer in window')
    return {key: float(np.mean([r[key] for r in selected]))
            for key in MASS+SPATIAL if all(r.get(key) is not None for r in selected)}


def summarize(rows, cohort, start):
    units = {r['dataset'] for r in rows}
    if len(units) != 1 or len({r['question_id'] for r in rows}) != len(rows):
        raise ValueError('Mixed dataset or duplicate image/question denominator')
    closed = next(iter(units)) != 'textvqa'
    result = dict(cohort=cohort, dataset=rows[0]['dataset'], window_start=start, window_end=start+2, n=len(rows),
                  mean_absolute_margin_change=float(np.mean([float(r['absolute_change']) for r in rows])),
                  mean_signed_margin_change=float(np.mean([float(r['metric_change']) for r in rows])))
    if closed:
        result.update(full_candidate_accuracy=float(np.mean([boolean(r['full_correct']) for r in rows])),
                      candidate_accuracy=float(np.mean([boolean(r['correct']) for r in rows])),
                      correct_to_wrong=sum(boolean(r['correct_to_wrong']) for r in rows),
                      wrong_to_correct=sum(boolean(r['wrong_to_correct']) for r in rows))
    else:
        result['first_token_changed'] = sum(boolean(r['first_token_changed']) for r in rows)
        result['metric_note'] = 'Fixed Full first-token LogP change, NOT Margin or VQA accuracy'
    for key in MASS+SPATIAL:
        if all(r.get(key) is not None for r in rows):
            result[key] = float(np.mean([r[key] for r in rows]))
    return result


def figure(rows, title, path):
    x = [r['window_start'] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout='constrained')
    ax = axes[0]
    ax.plot(x, [100*r['full_candidate_accuracy'] for r in rows], '--', color='#333333', label='Native Full')
    ax.plot(x, [100*r['candidate_accuracy'] for r in rows], 'o-', ms=3, color='#087f8c', label='A-Skip')
    ax.set_ylabel('Candidate accuracy (%)')
    ax.set_ylim(0, 102)
    ax.legend(loc='lower left', ncol=2)
    flips = ax.twinx()
    flips.bar(x, [r['correct_to_wrong'] for r in rows], color='#c84b31', alpha=.23, label='Full correct -> wrong')
    flips.set_ylabel('Correct -> wrong (count)', color='#9d341e')
    flips.set_ylim(0, rows[0]['n'])
    flips.legend(loc='lower right')
    for field, label, color in [('p_bos', 'P-1 -> BOS', '#74509f'), ('p_image', 'P-1 -> Image', '#087f8c'),
                                ('p_sink_mass', 'P-1 -> Visual Sink (subset of Image)', '#c84b31')]:
        axes[1].plot(x, [r[field] for r in rows], label=label, color=color, lw=2)
    axes[1].set_ylabel('Native Full attention mass')
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc='best')
    axes[2].plot(x, [r['q_roi_enrichment'] for r in rows], label='Question -> ROI', color='#087f8c', lw=2)
    axes[2].plot(x, [r['p_roi_enrichment'] for r in rows], label='P-1 -> ROI', color='#c84b31', lw=2)
    axes[2].axhline(1, color='#555555', ls='--', lw=1, label='Uniform visual allocation')
    axes[2].set_ylabel('ROI enrichment (share / patch area)')
    axes[2].set_xlabel('Three-layer window start s (attention averaged over Ls, Ls+1, Ls+2)')
    axes[2].legend(loc='best')
    for ax in axes:
        ax.set_xlim(-.5, 29.5)
        ax.set_xticks(range(30))
        ax.grid(axis='y', alpha=.2)
        for lo, hi in [(-.5, 2.5), (19.5, 29.5)]:
            ax.axvspan(lo, hi, color='#737373', alpha=.06, zorder=0)
        ax.axvspan(8.5, 13.5, color='#cb9d32', alpha=.08, zorder=0)
    fig.suptitle(title+'\nFrozen A-Skip outcomes; Native Full attention only; overlapping windows, descriptive', fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    config = verify()
    done = json.loads((OUTPUT/'completion.json').read_text())
    if done['status'] != 'complete' or done['questions'] != 64:
        raise ValueError('All64 Native Full observations required')
    layers, lookup, sinks = [], {}, []
    for unit in UNITS:
        for sample in config['samples'][unit]:
            qid = sample['question_id']
            checkpoint = json.loads((OUTPUT/'checkpoints'/unit/f'{qid}.json').read_text())
            records = checkpoint['records']
            if len(records) != 32 or [r['layer'] for r in records] != list(range(32)):
                raise ValueError('Incomplete per-layer checkpoint')
            layers.extend(records)
            lookup[unit, qid] = records
            if checkpoint['sink'] is not None:
                sinks.append(dict(question_id=qid, **checkpoint['sink']))
    prior = list(csv.DictReader((PARENT/'per_question.csv').open()))
    a_skip = [r for r in prior if r['condition'] == 'A-Skip']
    if len(a_skip) != 1920 or len({(r['dataset'], r['question_id'], r['window_start']) for r in a_skip}) != 1920:
        raise ValueError('Frozen A-Skip coverage incorrect')
    joined = [dict(r, **window_values(lookup[r['dataset'], r['question_id']], int(r['window_start']))) for r in a_skip]
    cohorts = {'chooserel_roi16': ('gqa_chooserel', set(config['primary_ids'])),
               'chooserel_all32': ('gqa_chooserel', {s['question_id'] for s in config['samples']['gqa_chooserel']}),
               'scienceqa16': ('scienceqa_img', {s['question_id'] for s in config['samples']['scienceqa_img']}),
               'textvqa16': ('textvqa', {s['question_id'] for s in config['samples']['textvqa']})}
    summaries = []
    for cohort, (unit, ids) in cohorts.items():
        for start in range(30):
            rows = [r for r in joined if r['dataset'] == unit and r['question_id'] in ids and int(r['window_start']) == start]
            if len(rows) != len(ids):
                raise ValueError('Same-sample denominator mismatch')
            summaries.append(summarize(rows, cohort, start))
    # Verify full-cohort reductions reproduce the already-published window_summary.csv.
    old_summary = list(csv.DictReader((PARENT/'window_summary.csv').open()))
    validated = 0
    for row in summaries:
        if row['cohort'] == 'chooserel_roi16':
            continue
        old = next(r for r in old_summary if r['dataset'] == row['dataset'] and r['condition'] == 'A-Skip'
                   and int(r['window_start']) == row['window_start'])
        for new, original in [('mean_absolute_margin_change', 'mean_absolute_change'),
                              ('full_candidate_accuracy', 'full_candidate_accuracy'),
                              ('candidate_accuracy', 'candidate_accuracy'),
                              ('correct_to_wrong', 'correct_to_wrong'), ('wrong_to_correct', 'wrong_to_correct')]:
            if new in row and not np.isclose(row[new], float(old[original]), rtol=0, atol=1e-12):
                raise AssertionError(f'Historical summary changed: {row["cohort"]}/{new}')
        validated += 1
    primary = [r for r in summaries if r['cohort'] == 'chooserel_roi16']
    correlations = []
    for cohort in ('chooserel_roi16', 'chooserel_all32'):
        rows = [r for r in summaries if r['cohort'] == cohort]
        for feature in ('p_bos', 'p_image', 'p_sink_mass', 'p_sink_share', 'q_roi_enrichment', 'p_roi_enrichment'):
            if not all(feature in r for r in rows):
                continue
            x = [r[feature] for r in rows]
            for outcome in ('mean_absolute_margin_change', 'candidate_accuracy', 'correct_to_wrong'):
                y = [r[outcome] for r in rows]
                correlations.append(dict(cohort=cohort, feature=feature, outcome=outcome, windows=30,
                    pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(np.corrcoef(ranks(x), ranks(y))[0, 1]),
                    interpretation='Descriptive across30 overlapping windows, no independent-sample p-value or causal inference'))
    bands = []
    for label, starts in [('early_L0_L4', range(3)), ('middle_L9_L15', range(9, 14)), ('deep_L20_L31', range(20, 30))]:
        rows = [r for r in primary if r['window_start'] in starts]
        bands.append(dict(band=label, window_starts=list(starts), n_images=16, n_overlapping_windows=len(rows),
                          **{k: float(np.mean([r[k] for r in rows])) for k in
                             ('full_candidate_accuracy', 'candidate_accuracy', 'correct_to_wrong',
                              'wrong_to_correct', 'mean_absolute_margin_change', 'p_bos', 'p_image',
                              'p_sink_mass', 'p_sink_share', 'q_roi_enrichment', 'p_roi_enrichment')}))
    coverage = dict(total=64, datasets={unit: len(config['samples'][unit]) for unit in UNITS},
        primary_roi_n=16, primary_label_counts={label: sum(s['question_id'] in config['primary_ids'] and s['answer'] == label
            for s in config['samples']['gqa_chooserel']) for label in ('left', 'right')},
        roi_excluded={s['question_id']: config['roi'][s['question_id']]['geometry_exclusion_reasons']
            for s in config['samples']['gqa_chooserel'] if s['question_id'] not in config['primary_ids']},
        frozen_sink_n=len(config['frozen_sinks']), new_neutral_sink_n=len(config['missing_sink_ids']),
        scienceqa_textvqa_roi_sink='Not defined; only BOS/Image masses',
        per_layer_rows=len(layers), reused_A_skip_rows=len(joined), verified_historical_summary_rows=validated)
    write_csv(OUTPUT/'per_layer.csv', layers)
    write_csv(OUTPUT/'per_question_window_alignment.csv', joined)
    write_csv(OUTPUT/'window_alignment_summary.csv', summaries)
    write_csv(OUTPUT/'descriptive_correlations.csv', correlations)
    write_csv(OUTPUT/'band_summary.csv', [dict(r, window_starts=json.dumps(r['window_starts'])) for r in bands])
    freeze_json(OUTPUT/'sink_manifest.json', dict(rule=config['sink_rule'], samples=sinks))
    freeze_json(OUTPUT/'coverage.json', coverage)
    freeze_json(OUTPUT/'summary.json', dict(coverage=coverage, bands=bands, correlations=correlations,
        validation='64/64 same-GPU historical Full logits exact; observer/no-observer smoke exact; '
        'weights unchanged; validated32-layer rows;90 frozen window_summary rows reproduced.'))
    figures = OUTPUT/'figures'
    figures.mkdir(exist_ok=True)
    figure(primary, 'ChooseRel: same16 geometry-eligible questions', figures/'chooserel_roi16_alignment.png')
    # Do not plot aggregate ROI across geometry-ineligible32; keep full32 accuracy separately.
    ref = [r for r in summaries if r['cohort'] == 'chooserel_all32']
    fig, ax = plt.subplots(figsize=(11, 3.8), layout='constrained')
    ax.plot(range(30), [100*r['full_candidate_accuracy'] for r in ref], '--', label='Full (32 questions)')
    ax.plot(range(30), [100*r['candidate_accuracy'] for r in ref], 'o-', label='A-Skip (32 questions)', ms=3)
    ax.set(xlabel='Window start (three layers)', ylabel='Candidate accuracy (%)', xticks=range(30),
           title='Separate reference: all32 ChooseRel; not the16-question ROI denominator')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(figures/'chooserel_all32_reference.png', dpi=180)
    plt.close(fig)
    report(coverage, bands, correlations, primary, done)
    print(json.dumps(dict(coverage=coverage, bands=bands), indent=2))


def report(coverage, bands, correlations, primary, done):
    lines = ['# Native Full attention vs frozen A-Skip sensitivity', '',
        'Complete. No A-Skip reruns. Base: `2360834fae22cdb28dd66758893948066e84e69b`.', '',
        '![Same16 aligned comparison](figures/chooserel_roi16_alignment.png)', '',
        '## Coverage and definitions', '',
        f"- Original64 unchanged: ChooseRel32, ScienceQA16, TextVQA16. Primary same-image comparison:16 ROI-eligible ChooseRel "
        f"({coverage['primary_label_counts']['left']} left / {coverage['primary_label_counts']['right']} right), chosen only by frozen Phase3.4 geometry.",
        '- Phase3.4 actual processor resize/crop and official object boxes are reused, not remapped heuristically. '
        'Both objects >=2 intersecting patches; union4..192; area ratio<=16. ROI is the union of both objects, not segmentation. '
        'All32 accuracy is reported separately; geometry-ineligible ROI is never pooled into the primary figure.',
        '-6 frozen Sink lists reused,26 identified with exactly the Phase3.4 two-discovery/one-holdout neutral prompts. '
        'Sink selection never uses answers, margins, or A-Skip results. Sink includes ROI overlap; non-ROI Sink columns are also available. '
        'A Sink is a persistent neutral hotspot, not proven task-irrelevant noise.',
        '- ScienceQA/TextVQA: BOS and Image masses only; no invented ROI or Sink. '
        'TextVQA retains fixed-Full-first-token diagnostics, not VQA accuracy.',
        '- Raw masses are head means. Question rows average actual question tokens within each head first. '
        'ROI share is ROI mass/visual mass per head, then heads averaged; enrichment is share/(ROI patches/576), '
        'matching Phase3.4. Area denotes positive-intersection patch area, not continuous box pixel area. '
        'Question and P-1 use their real frozen input positions.',
        '- Every x=s combines Ls..Ls+2 with equal layer weights within image, then equal image weights. '
        'It is Native Full attention, not attention after A-Skip. Sink mass is a subset of Image mass; do not add both.', '',
        '## Same16 comparison', '',
        '| Layer band (windows wholly inside) | Mean A-Skip accuracy | Full accuracy | Mean correct->wrong/window | Mean abs Margin change | BOS | Image | Sink | Q ROI enrichment | P-1 ROI enrichment |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in bands:
        lines.append(f"| {b['band']} | {100*b['candidate_accuracy']:.2f}% | {100*b['full_candidate_accuracy']:.2f}% | "
                     f"{b['correct_to_wrong']:.2f} | {b['mean_absolute_margin_change']:.4f} | {b['p_bos']:.4f} | "
                     f"{b['p_image']:.4f} | {b['p_sink_mass']:.4f} | {b['q_roi_enrichment']:.3f} | {b['p_roi_enrichment']:.3f} |")
    lines += ['', 'Band values average overlapping windows, NOT extra independent samples or a simultaneous multi-layer ablation.', '',
              '## Descriptive association', '',
              '| Native Full feature | Pearson with mean abs Margin change | Spearman |', '|---|---:|---:|']
    for row in correlations:
        if row['cohort'] == 'chooserel_roi16' and row['outcome'] == 'mean_absolute_margin_change':
            lines.append(f"| {row['feature']} | {row['pearson']:.3f} | {row['spearman']:.3f} |")
    minimum = min(primary, key=lambda r: r['mean_absolute_margin_change'])
    maximum = max(primary, key=lambda r: r['mean_absolute_margin_change'])
    lines += ['', f"Within the same16, smallest absolute Margin effect is L{minimum['window_start']}–L{minimum['window_end']} "
        f"({minimum['mean_absolute_margin_change']:.4f}); largest is L{maximum['window_start']}–L{maximum['window_end']} "
        f"({maximum['mean_absolute_margin_change']:.4f}).",
        'These layer profiles can be consistent with competing BOS/Sink allocation or weaker ROI alignment, '
        'but cannot establish either as the cause of local Attention-update sensitivity. '
        'The30 windows overlap, layer depth confounds all profiles, heads/layers are not independent image samples; '
        'no significance test or causal claim is made. Attention mass alone does not measure update size, useful '
        'information, or later-layer compensation. High Sink mass does not imply low ROI enrichment because Sink may overlap ROI.', '',
        '## Validation and cost', '',
        '- Small CPU tests passed (see cpu_tests.txt); one observer/native GPU smoke exact. All64 new Full P-1 logits '
        'are bitwise identical to the historical same-GPU Full with the exact N+1 candidate layout (TextVQA N). '
        'No changed tolerance, no new baselines mixed with historical intervention scores.',
        '-32 layers/sample, actual BOS at token0 validated, finite logits/attention and original row-sum tolerance checked; '
        'validated observer hooks clean up and model parameter versions remain unchanged. '
        '90 all-cohort A-Skip summary rows reproduce the original window_summary.csv within1e-12 reduction rounding.',
        f"-64 task forwards +78 neutral forwards +1 observer-off smoke,0 intervention reruns. GPU wall time including model load: "
        f"{done['wall_seconds']:.2f}s. No generation, training, additional datasets, or speedup claim.", '',
        '## Files', '',
        '- `per_layer.csv`:2048 original layer rows (64x32).',
        '- `per_question_window_alignment.csv`:1920 frozen A-Skip rows joined with three-layer mean Full attention.',
        '- `window_alignment_summary.csv`:four explicitly separated cohorts,30 windows each.',
        '- `coverage.json`, `manifest.json`, `sink_manifest.json`:sample definitions, input/source hashes, exclusion reasons and Sink provenance.',
        '- `descriptive_correlations.csv`, `band_summary.csv`, `summary.json`:descriptive statistics.',
        '- `checkpoints/`:per-question statistics, exact-baseline validation and real input mappings.',
        '- `neutral_maps/`:small FP16 neutral maps for26 new Sink definitions; no full attention matrices retained.', '']
    (OUTPUT/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

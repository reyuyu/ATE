"""Unique CPU aggregation, paired image uncertainty and real-data paper figures."""
from collections import defaultdict
import csv
import fcntl
import io
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.dataset_registry import digest, freeze_json
from benchmarks.scoring import pope_metrics
from .inventory import OUTPUT
from .protocol import COUNTS, SEED, TIMES
from .run import gate, samples, verify
from .scorers import caption_scores

MODES = ('Prefill-only', 'Decode-only', 'Combined')
COLORS = dict(zip(MODES, ('#087f8c', '#c84b31', '#74509f')))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Empty table is not a result')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Never overwrite a different completed table')
        return
    with path.open('xb') as f:
        f.write(content)


def save_figure(fig, name):
    folder = OUTPUT/'figures'
    folder.mkdir(exist_ok=True)
    for extension in ('png', 'pdf'):
        path = folder/f'{name}.{extension}'
        if not path.exists():
            fig.savefig(path, dpi=200)
    plt.close(fig)


def load_stage(stage):
    gate(stage)
    config = verify()
    collected, seen = [], set()
    for rank in range(4):
        expected = {(s['dataset'], s['question_id']): s for s in samples(rank, stage)}
        for path in sorted((OUTPUT/f'{stage}/shard{rank}/checkpoints').glob('*.json')):
            value = json.loads(path.read_text())
            key = value['dataset'], value['question_id']
            if key in seen or key not in expected or value['binding'] != digest(config):
                raise ValueError('Duplicate, unknown or mismatched checkpoint')
            if value['sample_sha256'] != digest(expected[key]):
                raise ValueError('Sample content differs')
            seen.add(key)
            collected.append((expected[key], value['result'], rank))
        if not set(expected) <= seen:
            raise ValueError('Missing shard questions')
    target = 2000 if stage == 'discovery' else 5700
    if len(collected) != target:
        raise ValueError('Refuse full-study claim with partial sample coverage')
    return collected


def paired_bootstrap(matrix, seed=SEED, repetitions=50000):
    """Rows must already be complete image clusters with equal question counts."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or len(matrix) < 2:
        raise ValueError('Require finite paired image matrix')
    rng = np.random.default_rng(seed)
    output = np.empty((repetitions, matrix.shape[1]))
    for lo in range(0, repetitions, 128):
        n = min(128, repetitions-lo)
        indices = rng.integers(0, len(matrix), (n, len(matrix)))
        output[lo:lo+n] = matrix[indices].mean(1)
    return output


def zero_effect_harm_bound(n_images, score_range=1., family=144):
    return score_range*(1-(.05/(2*family))**(1/n_images))


def discovery_summary(collected):
    cells, joined = defaultdict(list), []
    for sample, result, rank in collected:
        expected_times = [t for t in TIMES if t < len(result['token_ids'])]
        entries = result['sensitivity']
        if len(entries) != 30*len(expected_times) or len({(r['time'], r['window_start']) for r in entries}) != len(entries):
            raise ValueError('Discovery window/time coverage invalid')
        mass = {(r['time'], r['layer']): r for r in result['mechanism']}
        for row in entries:
            current = dict(row, dataset=sample['dataset'], question_id=sample['question_id'], image_identity=sample['image_identity'])
            start, t = row['window_start'], row['time']
            for key in ('bos_mass', 'image_mass', 'question_mass', 'history_mass', 'attention_residual_ratio'):
                current[key] = float(np.mean([mass[t, layer][key] for layer in range(start, start+3)]))
            cells[sample['dataset'], t, start].append(current)
            joined.append(current)
    summaries = []
    for (unit, t, start), rows in sorted(cells.items()):
        summaries.append(dict(dataset=unit, time=t, window_start=start, n_questions=len(rows),
            n_images=len({r['image_identity'] for r in rows}), mean_js=float(np.mean([r['js'] for r in rows])),
            mean_full_token_delta_logp=float(np.mean([r['full_token_delta_logp'] for r in rows])),
            top1_flip_rate=float(np.mean([r['top1_changed'] for r in rows])),
            full_eos_fraction=float(np.mean([r['current_full_is_eos'] for r in rows])),
            **{key: float(np.mean([r[key] for r in rows])) for key in
               ('bos_mass', 'image_mass', 'question_mass', 'history_mass', 'attention_residual_ratio')}))
    write_csv(OUTPUT/'discovery_summary.csv', summaries)
    for rank in range(4):
        ids = {(s['dataset'], s['question_id']) for s, result, r in collected if r == rank}
        write_csv(OUTPUT/f'discovery_per_question_shard{rank}.csv', [r for r in joined if (r['dataset'], r['question_id']) in ids])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    vmax = max(r['mean_js'] for r in summaries)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#dddddd')
    for ax, unit in zip(axes.flat, COUNTS):
        data = np.full((len(TIMES), 30), np.nan)
        labels = []
        for y, t in enumerate(TIMES):
            rows = [r for r in summaries if r['dataset'] == unit and r['time'] == t]
            for r in rows:
                data[y, r['window_start']] = r['mean_js']
            n = rows[0]['n_questions'] if rows else 0
            images = rows[0]['n_images'] if rows else 0
            labels.append(f'{"P-1" if t == 0 else "t="+str(t)} (q={n}, img={images})')
        image = ax.imshow(data, aspect='auto', vmin=0, vmax=vmax, cmap=cmap)
        ax.set(title=unit, xlabel='Three-layer window start', yticks=range(5), yticklabels=labels,
               xticks=[0, 5, 10, 15, 20, 25, 29])
    fig.colorbar(image, ax=axes, label='Mean single-step JS divergence (nats)', shrink=.8)
    fig.suptitle('Figure 1. Native-prefix single-step Attention sensitivity; grey = not reached')
    save_figure(fig, 'figure1_layer_time_sensitivity')
    mechanism_regression(joined)
    return summaries


def mechanism_regression(rows):
    # Absorb task x time x window effects; all repeated observations cluster by image.
    keys = ('bos_mass', 'image_mass', 'question_mass', 'history_mass', 'attention_residual_ratio', 'full_confidence_logit')
    x, y, cells, images = [], [], defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        confidence = np.clip(row['full_confidence'], 1e-6, 1-1e-6)
        x.append([row[k] if k != 'full_confidence_logit' else np.log(confidence/(1-confidence)) for k in keys])
        y.append(row['js'])
        cells[row['dataset'], row['time'], row['window_start']].append(i)
        images[row['image_identity']].append(i)
    x, y = np.asarray(x), np.asarray(y)
    for indices in cells.values():
        x[indices] -= x[indices].mean(0)
        y[indices] -= y[indices].mean()
    scales = x.std(0)
    active = scales > 1e-12
    design = x[:, active]/scales[active]
    inverse = np.linalg.pinv(design.T@design)
    coefficient = inverse@(design.T@y)
    residual = y-design@coefficient
    meat = np.zeros_like(inverse)
    for indices in images.values():
        score = design[indices].T@residual[indices]
        meat += np.outer(score, score)
    covariance = inverse@meat@inverse
    clusters, n = len(images), len(rows)
    parameters = np.linalg.matrix_rank(design)+len(cells)-1
    covariance *= clusters/(clusters-1)*(n-1)/(n-parameters)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    output = [dict(feature=key, coefficient_js_nats_per_within_cell_SD=float(coefficient[i]),
                   cluster_robust_se=float(se[i]), ci95_low=float(coefficient[i]-1.96*se[i]),
                   ci95_high=float(coefficient[i]+1.96*se[i]), images=clusters, rows=n,
                   controls='Task x reached Decode time x window fixed effects and Full logit-confidence',
                   interpretation='Descriptive conditional association; image-cluster robust, not causal')
              for i, key in enumerate(np.array(keys)[active])]
    write_csv(OUTPUT/'mechanism_adjusted_associations.csv', output)


def confirmation_summary(collected):
    selected = json.loads((OUTPUT/'frozen_windows.json').read_text())['window_starts']
    plans = [(start, mode) for start in selected for mode in MODES]
    summaries, details = [], []
    for unit in COUNTS:
        items = [(s, r) for s, r, rank in collected if s['dataset'] == unit]
        # Stable identical image order for every condition; never score only successful cases.
        items.sort(key=lambda pair: (pair[0]['image_identity'], pair[0]['question_id']))
        if any(len(r['generations']) != 25 for s, r in items):
            raise ValueError('Incomplete confirmation generation plan')
        caption = unit in ('coco_caption', 'textcaps')
        all_metrics = ['CIDEr', 'SPICE'] if caption else ['official_generation_score']
        by_plan = {}
        for start, mode in [(None, 'Full')]+plans:
            generations = []
            for sample, result in items:
                matches = [g for g in result['generations'] if g['window_start'] == start and g['mode'] == mode]
                if len(matches) != 1:
                    raise ValueError('Missing or duplicate confirmation condition')
                generations.append(matches[0])
            if caption:
                references = {i: sample['answer'] for i, (sample, result) in enumerate(items)}
                predictions = {i: g['generated_text'] for i, g in enumerate(generations)}
                cache = OUTPUT/'caption_scores'/f'{unit}_{mode}_{start}.json'
                if cache.exists():
                    stored = json.loads(cache.read_text())
                    if stored['binding'] != digest(dict(references=references, predictions=predictions)):
                        raise ValueError('Caption score cache binding changed')
                    scores = {int(i): v for i, v in stored['per_image'].items()}
                else:
                    corpus, scores = caption_scores(references, predictions)
                    freeze_json(cache, dict(binding=digest(dict(references=references, predictions=predictions)),
                                            corpus=corpus, per_image=scores))
            else:
                scores = {i: {'official_generation_score': float(g['score'])} for i, g in enumerate(generations)}
            by_plan[start, mode] = (generations, scores)
        if unit == 'pope_adversarial':
            secondary = []
            for (start, mode), (generations, scores) in by_plan.items():
                values = pope_metrics([g['generated_text'] for g in generations], [s['answer'] for s, r in items])
                secondary.append(dict(window_start=start, mode=mode, **values))
            write_csv(OUTPUT/'pope_secondary_metrics.csv', secondary)
        for metric in all_metrics:
            full_gen, full_scores = by_plan[None, 'Full']
            full = np.array([full_scores[i][metric] for i in range(len(items))])
            values = np.array([[by_plan[plan][1][i][metric] for plan in plans] for i in range(len(items))])
            image_indices = defaultdict(list)
            for i, (sample, result) in enumerate(items):
                image_indices[sample['image_identity']].append(i)
            sizes = {len(v) for v in image_indices.values()}
            if len(sizes) != 1:
                raise ValueError('Unequal question counts per image need ratio-cluster bootstrap, not equal means')
            cluster_matrix = np.array([np.r_[full[indices].mean(), values[indices].mean(0)] for indices in image_indices.values()])
            bootstrap = paired_bootstrap(cluster_matrix)
            deltas = bootstrap[:, 1:]-bootstrap[:, :1]
            family = 48 if metric == 'SPICE' else 144
            alpha = .05/(2*family)
            margin = .02 if metric == 'CIDEr' else .01
            full_ci = np.quantile(bootstrap[:, 0], [.025, .975])
            for j, (start, mode) in enumerate(plans):
                gen, score = by_plan[start, mode]
                changes = values[:, j]-full
                marginal = np.quantile(deltas[:, j], [.025, .975])
                adjusted = np.quantile(deltas[:, j], [alpha, 1-alpha])
                absolute_ci = np.quantile(bootstrap[:, j+1], [.025, .975])
                degenerate = bool(np.all(changes == 0))
                guard_bound = zero_effect_harm_bound(len(image_indices), 10. if metric == 'CIDEr' else 1., family)
                resolved = not degenerate or guard_bound < margin
                binary = not caption and unit != 'textvqa'
                summary = dict(dataset=unit, metric=metric, n_questions=len(items), n_images=len(image_indices),
                    window_start=start, mode=mode, Full=float(full.mean()), score=float(values[:, j].mean()),
                    delta=float(changes.mean()), ci95_low=float(marginal[0]), ci95_high=float(marginal[1]),
                    familywise95_low=float(adjusted[0]), familywise95_high=float(adjusted[1]),
                    score_ci95_low=float(absolute_ci[0]), score_ci95_high=float(absolute_ci[1]),
                    full_ci95_low=float(full_ci[0]), full_ci95_high=float(full_ci[1]),
                    noninferiority_margin=margin, bootstrap_noninferiority_criterion=bool(adjusted[0] > -margin),
                    paired_deltas_all_zero=degenerate, zero_effect_rare_harm_upper_bound=float(guard_bound) if degenerate else None,
                    noninferiority_resolution_sufficient=resolved,
                    supports_low_harm=bool(adjusted[0] > -margin and resolved),
                    important_harm=bool(adjusted[1] < -margin), score_increased=int((changes > 0).sum()),
                    score_decreased=int((changes < 0).sum()),
                    correct_to_wrong=int(((full == 1) & (values[:, j] == 0)).sum()) if binary else None,
                    wrong_to_correct=int(((full == 0) & (values[:, j] == 1)).sum()) if binary else None,
                    generated_text_change_rate=float(np.mean([g['generated_text'] != f['generated_text'] for g, f in zip(gen, full_gen)])),
                    mean_length=float(np.mean([g['length'] for g in gen])), eos_rate=float(np.mean([g['eos'] for g in gen])),
                    mean_full_length=float(np.mean([g['length'] for g in full_gen])),
                    full_eos_rate=float(np.mean([g['eos'] for g in full_gen])))
                summaries.append(summary)
                for i, ((sample, result), g, f) in enumerate(zip(items, gen, full_gen)):
                    details.append(dict(dataset=unit, question_id=sample['question_id'], image_identity=sample['image_identity'],
                        metric=metric, window_start=start, mode=mode, full_text=f['generated_text'], text=g['generated_text'],
                        full_score=float(full[i]), score=float(values[i, j]), delta=float(changes[i]),
                        full_length=f['length'], length=g['length'], eos=g['eos']))
        print('Scored confirmation', unit, len(items), flush=True)
    write_csv(OUTPUT/'confirmation_summary.csv', summaries)
    for unit in COUNTS:
        write_csv(OUTPUT/f'confirmation_per_question_{unit}.csv', [r for r in details if r['dataset'] == unit])
    figure2(summaries)
    return summaries


def figure2(rows):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, unit in zip(axes.flat, COUNTS):
        metric = 'CIDEr' if unit in ('coco_caption', 'textcaps') else 'official_generation_score'
        selected = [r for r in rows if r['dataset'] == unit and r['metric'] == metric]
        first = selected[0]
        ax.axhline(100*first['Full'], color='#333333', ls='--', label='Full')
        ax.axhspan(100*first['full_ci95_low'], 100*first['full_ci95_high'], color='#333333', alpha=.08)
        for mode in MODES:
            values = sorted([r for r in selected if r['mode'] == mode], key=lambda r: r['window_start'])
            x, y = [r['window_start'] for r in values], [100*r['score'] for r in values]
            lo = [100*(r['score']-r['score_ci95_low']) for r in values]
            hi = [100*(r['score_ci95_high']-r['score']) for r in values]
            ax.errorbar(x, y, yerr=[lo, hi], fmt='o-', ms=3, color=COLORS[mode], label=mode)
            for r in values:
                if r['important_harm']:
                    ax.plot(r['window_start'], 100*r['score'], 'x', color='#b30000', ms=9)
        ax.set(title=f"{unit} (q={first['n_questions']}, images={first['n_images']})",
               xlabel='Frozen three-layer window start', ylabel='CIDEr x100' if metric == 'CIDEr' else 'Official score (%)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Figure 2. Independent confirmation; marginal95% CIs; red x = familywise important harm')
    save_figure(fig, 'figure2_confirmation_performance')


def figure3(discovery_rows, confirmation):
    fig, axes = plt.subplots(6, 2, figsize=(12, 18), layout='constrained')
    for pair, unit in zip(axes, COUNTS):
        for t in TIMES:
            rows = sorted([r for r in discovery_rows if r['dataset'] == unit and r['time'] == t], key=lambda r: r['window_start'])
            if rows:
                pair[0].plot([r['window_start'] for r in rows], [100*r['top1_flip_rate'] for r in rows], label='P-1' if t == 0 else f't={t}')
        pair[0].set(title=unit, ylabel='Discovery Top-1 flip (%)', xlabel='Window start')
        metric = 'CIDEr' if unit in ('coco_caption', 'textcaps') else 'official_generation_score'
        for mode in MODES:
            rows = sorted([r for r in confirmation if r['dataset'] == unit and r['mode'] == mode and r['metric'] == metric], key=lambda r: r['window_start'])
            binary = rows[0]['correct_to_wrong'] is not None
            pair[1].plot([r['window_start'] for r in rows],
                         [r['correct_to_wrong'] if binary else r['score_decreased'] for r in rows], 'o-', color=COLORS[mode], label=mode, ms=3)
        pair[1].set(ylabel='Full correct -> wrong' if binary else f'Lower {metric} count (not binary error)', xlabel='Frozen window start')
        for ax in pair:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, ncol=3)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle('Figure 3. Prediction changes are distinct from independently measured generation harm')
    save_figure(fig, 'figure3_prediction_change_and_harm')


def figure4(rows):
    fig, axes = plt.subplots(6, 4, figsize=(17, 16), layout='constrained')
    for row_axes, unit in zip(axes, COUNTS):
        for ax, t in zip(row_axes[:3], (0, 4, 16)):
            values = sorted([r for r in rows if r['dataset'] == unit and r['time'] == t], key=lambda r: r['window_start'])
            if not values:
                ax.text(.5, .5, 'No native reach', transform=ax.transAxes, ha='center')
                continue
            for feature, label in [('bos_mass', 'BOS'), ('image_mass', 'Image'), ('question_mass', 'Question'), ('history_mass', 'History')]:
                ax.plot([r['window_start'] for r in values], [r[feature] for r in values], label=label)
            ax.set(title=f'{unit}: P-1' if t == 0 else f'{unit}: t={t}', xlabel='Window start', ylabel='Native Full mass')
        selected = [r for r in rows if r['dataset'] == unit]
        row_axes[3].scatter([r['bos_mass'] for r in selected], [r['mean_js'] for r in selected],
                            c=[r['time'] for r in selected], cmap='viridis', s=12, alpha=.65)
        row_axes[3].set(xlabel='BOS mass (window mean)', ylabel='Mean single-step JS', title='Descriptive; see controlled regression CSV')
        for ax in row_axes:
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle('Figure 4. Native attention allocation; layer/task/time/confidence adjusted estimates reported separately')
    save_figure(fig, 'figure4_attention_mass_and_necessity')


def main():
    lock = (OUTPUT/'analysis.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT/'completion.json').exists():
        raise ValueError('Completed aggregate exists; do not overwrite original results')
    discovery_rows = discovery_summary(load_stage('discovery'))
    confirmation = confirmation_summary(load_stage('confirmation'))
    figure3(discovery_rows, confirmation)
    figure4(discovery_rows)
    freeze_json(OUTPUT/'completion.json', dict(status='complete', discovery_questions=2000,
        confirmation_questions=5700, official_primary_comparisons=144, caption_secondary_comparisons=48,
        four_figures_png_pdf=True, statistics='image-paired bootstrap50000; simultaneous bounds; all negative results retained'))
    lines = ['# Paper RQ1: when and where is an Attention update necessary?', '',
        'Complete:2000 image-disjoint Discovery questions and5700 independent Confirmation questions. '
        'All figures use actual experiment outputs. Model scope is only pinned LLaVA-1.5-7B FP16/eager.', '',
        '## Primary confirmation results', '',
        '| Dataset | Window | Mode | Full | Score | Delta | Paired95%CI | Low-harm supported (familywise) |',
        '|---|---:|---|---:|---:|---:|---|---|']
    for r in confirmation:
        if r['metric'] == 'SPICE':
            continue
        lines.append(f"| {r['dataset']} | L{r['window_start']}-L{r['window_start']+2} | {r['mode']} | "
                     f"{r['Full']:.4f} | {r['score']:.4f} | {r['delta']:+.4f} | "
                     f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['supports_low_harm']} |")
    lines += ['', '## RQ answers from complete Discovery', '',
              '| Task | Native time | Reached questions / images | Most JS-sensitive window | Mean JS | Least JS-sensitive window | Mean JS |',
              '|---|---|---|---|---:|---|---:|']
    for unit in COUNTS:
        for t in TIMES:
            rows = [r for r in discovery_rows if r['dataset'] == unit and r['time'] == t]
            if not rows:
                lines.append(f'| {unit} | {t} | 0 / 0 | Not reached | NA | Not reached | NA |')
                continue
            high, low = max(rows, key=lambda r: r['mean_js']), min(rows, key=lambda r: r['mean_js'])
            lines.append(f"| {unit} | {'P-1' if t == 0 else 't='+str(t)} | {high['n_questions']} / {high['n_images']} | "
                         f"L{high['window_start']}-L{high['window_start']+2} | {high['mean_js']:.6f} | "
                         f"L{low['window_start']}-L{low['window_start']+2} | {low['mean_js']:.6f} |")
    lines += ['', 'The table directly compares Prefill with reached Decode steps. Differences of maxima are '
              'descriptive, not a statistically established migration. Late-step composition can differ by native EOS.', '',
              '## Independent low-harm confirmation', '']
    for unit in COUNTS:
        rows = [r for r in confirmation if r['dataset'] == unit and r['metric'] != 'SPICE']
        supported = [f"L{r['window_start']}-L{r['window_start']+2}/{r['mode']}" for r in rows if r['supports_low_harm']]
        lines.append(f"- {unit}: {len(supported)}/24 conditions pass the frozen primary-score noninferiority bound. "
                     +(', '.join(supported) if supported else 'None has sufficient familywise evidence; this is not proof of harm.'))
    lines += ['', 'Controlled Attention-feature associations and image-cluster95% intervals are in '
              '`mechanism_adjusted_associations.csv`. Coefficients adjust task/time/window and native confidence; '
              'the raw across-layer panels alone are not evidence that BOS dominance causes lower necessity.', '']
    lines += ['', '## Interpretation boundaries', '',
        '- Discovery single-step JS, fixed-token LogP and Top-1 changes are NOT accuracy. Each intervention '
        'uses an isolated clone of exactly the native Full history; no free continuation after that single step.',
        '- Later-step results condition on native reach; early-EOS tasks contribute fewer or no late-step questions. '
        'Do not interpret this survivor-conditioned change as within-image migration for missing questions.',
        '- Confirmation is actual cached greedy generation. Prefill-only alters only original P-1; Decode-only '
        'alters each later current query. Scores and harm are independently measured, including negative windows.',
        '- Samples are image-isolated across datasets and splits by canonical source ID, byte/RGB SHA256. '
        'No guarantee about perceptually similar or resized/reencoded near-duplicates beyond available identities/RGB.',
        '- Bootstrap resamples whole images (POPE complete six-question clusters). Overlapping windows/timepoints '
        'are never treated as independent images. Marginal95%CI and conservative simultaneous bounds both saved. '
        'CI crossing0 is not evidence for equivalence; predeclared noninferiority margins apply.',
        '- If every paired delta is zero, the frozen exact zero-negative-event guard must also pass. '
        'A degenerate bootstrap[0,0] with too few independent images is reported as insufficient rare-harm '
        'resolution, not established population noninferiority.',
        '- Caption CIDEr is standard full-cohort-IDF scoring; its paired CI is conditional on that fixed reference '
        'IDF calibration. SPICE is separately provided with a secondary comparison family.',
        '- GQA primary accuracy follows the official equality expression after the pinned LLaVA submission '
        'converter; GQA validity/plausibility/consistency metrics are not claimed. TextVQA is image-only, no OCR hints.',
        '- Mechanism CSV conditions on task x time x window fixed effects and confidence, with image-cluster '
        'robust uncertainty. It is descriptive association, not a causal explanation for necessity.',
        '- Attention output is still computed before zeroing. No actual skipped computation or speedup is claimed.', '']
    (OUTPUT/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

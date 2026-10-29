"""Read completed checkpoints only; plot all questions and first-token-hit subset."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .masks import PATHS, window
from .run import OUTPUT, TASKS, write_csv

NAMES = {'gqa_chooserel': 'ChooseRel', 'gqa_chooseattr': 'ChooseAttr',
         'gqa_queryattr': 'QueryAttr', 'gqa_logicalobj': 'LogicalObj'}
LABELS = {'image_to_last': 'Image -> Last', 'question_to_last': 'Question -> Last',
          'image_to_question': 'Image -> Question'}
COLORS = {'image_to_last': '#3575b5', 'question_to_last': '#d45539', 'image_to_question': '#248458'}


def collect():
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    binding = json.loads((OUTPUT / 'execution_binding.json').read_text())
    baselines, rows = [], []
    for sample in manifest['samples']:
        data = json.loads((OUTPUT / 'checkpoints' / f'{sample["dataset"]}__{sample["question_id"]}.json').read_text())
        if data['binding'] != binding or not data['native_restored_bitwise']:
            raise ValueError('Mixed code or failed Full restoration')
        base = data['baseline']
        if (base['dataset'], base['question_id']) != (sample['dataset'], sample['question_id']):
            raise ValueError('Checkpoint sample mismatch')
        expected = {(path, center) for path in PATHS for center in range(32)}
        actual = [(r['path'], r['center']) for r in data['rows']]
        if set(actual) != expected or len(actual) != 96:
            raise ValueError('Missing or duplicate observations')
        for r in data['rows']:
            if r['target_token_id'] != base['target_token_id'] or r['full_logp'] != base['full_logp']:
                raise ValueError('Target token or Native reference changed')
            if not all(math.isfinite(r[key]) for key in ('delta_logp', 'probability_change_pct', 'masked_logp')):
                raise ValueError('Nonfinite score')
            if (r['window_start'], r['window_end']) != (window(r['center'])[0], window(r['center'])[-1]):
                raise ValueError('Wrong window')
        baselines.append(base)
        rows.extend(data['rows'])
    if Counter(b['dataset'] for b in baselines) != Counter({task: 24 for task in TASKS}):
        raise ValueError('Incomplete tasks')
    if len({(b['dataset'], b['question_id']) for b in baselines}) != 96:
        raise ValueError('Duplicate question')
    return baselines, rows, binding


def plot(curves, counts, subset, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, task in zip(axes.flat, TASKS):
        for route in PATHS:
            selected = [r for r in curves if r['subset'] == subset and r['dataset'] == task and r['path'] == route]
            if selected:
                ax.plot([r['center'] for r in selected], [r['mean_probability_change_pct'] for r in selected],
                        color=COLORS[route], lw=2, label=LABELS[route])
        ax.axhline(0, color='#777777', linewidth=.7)
        ax.set_title(f'{NAMES[task]} (n={counts[task]})', fontsize=13)
        ax.set_xlim(0, 31)
        ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
        ax.set_xlabel('Window center layer (9 layers, clipped at edges)')
        ax.set_ylabel('Fixed Full-token probability change (%)')
        ax.grid(alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('LLaVA-1.5-7B | ' + ('All frozen questions' if subset == 'all' else 'Native first-token-hit subset (not full-answer EM)'), fontsize=15)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    baselines, rows, binding = collect()
    if json.loads((OUTPUT / 'completion.json').read_text())['status'] != 'complete':
        raise ValueError('Partial run; no complete report allowed')
    curves, summary = [], []
    for subset in ('all', 'first_token_hit'):
        grouped = defaultdict(list)
        for row in rows:
            if subset == 'all' or row['full_first_token_hit']:
                grouped[(row['dataset'], row['path'], row['center'])].append(row)
        for task in TASKS:
            for route in PATHS:
                series = []
                for center in range(32):
                    group = grouped.get((task, route, center), [])
                    if not group:
                        continue
                    item = {'subset': subset, 'dataset': task, 'path': route, 'center': center, 'n': len(group),
                            'mean_probability_change_pct': statistics.mean(r['probability_change_pct'] for r in group),
                            'mean_delta_logp': statistics.mean(r['delta_logp'] for r in group)}
                    curves.append(item)
                    series.append(item)
                if series:
                    peak = min(series, key=lambda r: r['mean_probability_change_pct'])
                    tied = [r['center'] for r in series if r['mean_probability_change_pct'] == peak['mean_probability_change_pct']]
                    c = peak['center']
                    summary.append({'subset': subset, 'dataset': task, 'path': route, 'n': peak['n'],
                        'peak_center': c, 'tied_peak_centers': ','.join(map(str, tied)),
                        'sensitive_window': f'L{window(c)[0]}-L{window(c)[-1]}',
                        'mean_probability_change_pct_at_peak': peak['mean_probability_change_pct'],
                        'maximum_mean_probability_drop_pct': max(0, -peak['mean_probability_change_pct']),
                        'mean_delta_logp_at_peak': peak['mean_delta_logp'],
                        'early_mean_change_pct_centers0_10': statistics.mean(r['mean_probability_change_pct'] for r in series[:11]),
                        'middle_mean_change_pct_centers11_21': statistics.mean(r['mean_probability_change_pct'] for r in series[11:22]),
                        'late_mean_change_pct_centers22_31': statistics.mean(r['mean_probability_change_pct'] for r in series[22:])})
    write_csv(OUTPUT / 'curves.csv', curves)
    write_csv(OUTPUT / 'sensitivity_summary.csv', summary)
    counts = Counter(b['dataset'] for b in baselines)
    hit_counts = Counter(b['dataset'] for b in baselines if b['full_first_token_hit'])
    plot(curves, counts, 'all', OUTPUT / 'information_flow_all.png')
    plot(curves, hit_counts, 'first_token_hit', OUTPUT / 'information_flow_first_token_hit.png')
    table = ['| Task | Path | n | Peak center | Sensitive window | Maximum mean drop (%) | Mean delta LogP |',
             '|---|---|---:|---:|---|---:|---:|']
    for r in summary:
        if r['subset'] == 'all':
            table.append(f'| {NAMES[r["dataset"]]} | {LABELS[r["path"]]} | {r["n"]} | L{r["peak_center"]} | '
                         f'{r["sensitive_window"]} | {r["maximum_mean_probability_drop_pct"]:.2f} | {r["mean_delta_logp_at_peak"]:.3f} |')
    subset_lines = []
    for task in TASKS:
        bs = [b for b in baselines if b['dataset'] == task]
        multi = sum(any(len(ids) > 1 for ids in b['gold_token_ids'].values()) for b in bs)
        subset_lines.append(f'- {NAMES[task]}: all n=24; Native first-token hit n={hit_counts[task]}; multi-token Gold n={multi}.')
    peaks = {}
    for route in PATHS:
        peaks[route] = ', '.join(f'{NAMES[r["dataset"]]} L{r["peak_center"]} ({r["maximum_mean_probability_drop_pct"]:.1f}%)'
                               for r in summary if r['subset'] == 'all' and r['path'] == route)
    report = f'''# GPU1 nine-layer information-flow experiment

Status: **complete**, 96/96 frozen questions, 9,216/9,216 masked prefills. No Decode, training or extra window lengths.

## Main result

![All 96 questions](information_flow_all.png)

{chr(10).join(table)}

Maximum drop is the minimum over the 32 means of **per-question** relative probability changes, not the relative change of mean probabilities. A sensitive window is not evidence that a single layer alone is responsible. Edge windows have 5-8 rather than 9 layers; overlapping windows are not independent.

## Cohorts and supplementary result

{chr(10).join(subset_lines)}

![Native first-token hits only](information_flow_first_token_hit.png)

"Correct" here is explicitly **Gold-first-token text hit after strip/lower normalization**, not complete-answer exact match. No full Decode was run. Multi-token answers may share a prefix; punctuation and tokenization can affect this check. Strict token-ID agreement is additionally saved in baselines.csv. Thus this supplementary cohort must not be reported as the official paper's full-answer-correct cohort or as VQA accuracy. Main curves always include all 24 questions per task, including Native errors.

## Protocol

- Fixed seed 20261029. For each frozen 240-question pool, sort by question_id/image_id, independently sample 24 with Python Random(seed), then sort selected IDs. No prediction-based selection, replacement, or rebalancing. The official author-filtered source pools themselves are not an unfiltered GQA population.
- ChooseRel questions/images come from Phase 3.2; the other tasks from the Phase 4.0 frozen suite. All selected local images match their frozen SHA256. Uniform existing short-answer instruction and original HF chat template. For ChooseRel this appends the instruction relative to the historical bare-question pilot; historical Full scores are not used as baselines.
- LLaVA-1.5-7B-hf b234b804b114d9e37bb655e11cbbb5f5e971b7a9, FP16/eager, Transformers 4.49.0, only physical GPU1. Four identical image/prompt replicas per microbatch, each assigned a different center. Native Full uses the identical batch shape. No cross-question padding or mixed baselines.
- At center c, mask layers max(0,c-4)..min(31,c+4). Instance-local attention prehooks set only the specified query/key pairs to negative infinity in the existing causal additive mask, before softmax. All heads; original positions, MLP and tokens retained. Image-to-Question does not edit Last's direct links. Original true Question spans are dynamically located, excluding the added instruction and chat-template tokens.
- No KV cache; prefill only. Track the Native Full vocabulary argmax token even when the masked argmax changes. FP32 full-vocabulary log_softmax of FP16 model logits; relative change = 100*expm1(masked LogP - Full LogP). Also save both probabilities and delta LogP. Only the final position's LM head projection is needed (logits_to_keep=1); no hidden computation is skipped.

## Validation and provenance

CPU tests and four real GPU smoke questions pass: target mask positions only, blocked softmax weights exactly zero, original inputs intact, correct clipped layer coverage, hook cleanup on normal/exceptional exit. Native Full last-position vocabulary logits restore **bitwise exactly** after intervention for every formal question. No nonfinite scores, OOM, training, full Decode or other-GPU model jobs.

- Frozen execution commit: `{binding['code_commit']}`
- Base: `7ccb44941cb48e901e6e23e96bae6da173fe2ce5`
- Full source hashes, model/environment, config, selection, smoke details and completion are saved beside this report. Checkpoints are per question and bound to the same execution commit/config/manifest; exact coverage checked before aggregation.
- Official reference: [code](https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM/tree/c9a1111dac4baa175f67716fab473789320187c0), [paper](https://arxiv.org/abs/2411.18620). Verified official centered window and probability-change metric in InformationFlow.py / methods.py. The paper's main examples use LLaVA-1.5-13B and retain full-answer-correct generations; this is a small 7B HF adaptation, not an exact numerical reproduction.

## Descriptive peak locations

- Image -> Last: {peaks['image_to_last']}.
- Question -> Last: {peaks['question_to_last']}.
- Image -> Question: {peaks['image_to_question']}.

The paper describes image-to-question integration in lower/middle layers and propagation of the multimodal question representation to Last in higher layers, with a smaller direct Image-to-Last contribution. Interpret the present curves against that ordering, not as evidence of identical numeric peak layers across model sizes. The 24-question cohorts, overlapping windows, first-token metric and nonidentical correctness protocol limit generalization; no post-result tuning or additional conditions were run.

## Files

- `per_question.csv`: every masked path/center/question, fixed token, probabilities, LogP changes.
- `baselines.csv`: question, Gold, Native token, normalized first-token text hit, strict token-ID hit and true positions.
- `curves.csv`: all-sample and first-token-hit means, including mean delta LogP.
- `sensitivity_summary.csv`: peak windows plus descriptive early (centers 0-10), middle (11-21), late (22-31) means.
- `checkpoints/`, `manifest.json`, `config.json`, `execution_binding.json`, `environment.json`, `smoke.json`, `completion.json`, CPU and GPU text logs.
'''
    (OUTPUT / 'report.md').write_text(report)
    print('\n'.join(table), flush=True)
    print('First-token-hit counts:', dict(hit_counts), flush=True)


if __name__ == '__main__':
    main()

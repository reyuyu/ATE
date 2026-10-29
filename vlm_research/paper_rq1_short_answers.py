"""CPU-only partial delivery of completed RQ1 short-answer confirmation.

Reuse the frozen aggregation without changing its source, comparison family or
GPU workers. All output redirection is local to this separate Python process.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1 import analyze
from paper_rq1.inventory import OUTPUT
from paper_rq1.protocol import COUNTS
from paper_rq1.run import gate, samples, verify
from paper_rq1.scorers import short_score

SHORT = {k: v for k, v in COUNTS.items() if k not in ('coco_caption', 'textcaps')}
DESTINATION = OUTPUT/'short_answer_snapshot'
NAMES = {'gqa_balanced': 'GQA Balanced', 'scienceqa_img': 'ScienceQA-IMG',
         'pope_adversarial': 'POPE Adversarial', 'textvqa': 'TextVQA'}


def validate_generations(sample, result, starts):
    generations = result['generations']
    expected = {(None, 'Full')} | {(s, m) for s in starts for m in analyze.MODES}
    actual = [(g['window_start'], g['mode']) for g in generations]
    if len(actual) != 25 or set(actual) != expected:
        raise ValueError('Missing or duplicate frozen generation conditions')
    for g in generations:
        tokens = g['token_ids']
        if not 0 < len(tokens) <= 32 or len(tokens) != g['length']:
            raise ValueError('Invalid generated token length')
        if bool(g['eos']) != (tokens[-1] == 2) or 2 in tokens[:-1]:
            raise ValueError('EOS or post-EOS generation inconsistency')
        if not np.isfinite(g['score']) or g['score'] != short_score(sample, g['generated_text']):
            raise ValueError('Stored score differs from the frozen official short scorer')


def load_completed_short():
    config = verify()
    gate('smoke')
    gate('discovery')
    binding = digest(config)
    windows_path = OUTPUT/'frozen_windows.json'
    windows = json.loads(windows_path.read_text())
    if windows['execution_binding'] != binding:
        raise ValueError('Window selection is not bound to the frozen execution')
    collected, manifest, seen, images, environments = [], [], set(), {}, []
    fields = ('python', 'torch', 'transformers', 'cuda_runtime', 'cudnn', 'gpu',
              'model_revision', 'model_path', 'parameter_dtypes', 'code_commit', 'binding')
    for rank in range(4):
        folder = OUTPUT/f'confirmation/shard{rank}'
        environment = json.loads((folder/'environment.json').read_text())
        if environment['binding'] != binding or environment['code_commit'] != config['code_commit']:
            raise ValueError('Worker execution differs')
        environments.append({k: environment[k] for k in fields})
        for sample in samples(rank, 'confirmation'):
            if sample['dataset'] not in SHORT:
                continue
            key = sample['dataset'], sample['question_id']
            path = folder/'checkpoints'/(sample['dataset']+'__'+sample['question_id'].replace(':', '_')+'.json')
            checkpoint = json.loads(path.read_text())
            if key in seen or checkpoint['binding'] != binding or checkpoint['sample_sha256'] != digest(sample):
                raise ValueError('Duplicate question or changed checkpoint/sample binding')
            if (checkpoint['dataset'], checkpoint['question_id']) != key or checkpoint['image_identity'] != sample['image_identity']:
                raise ValueError('Question/image identity differs')
            image = sample['image_identity']
            if image in images and images[image] != (sample['dataset'], rank):
                raise ValueError('Image was assigned to multiple datasets/workers')
            images[image] = sample['dataset'], rank
            seen.add(key)
            validate_generations(sample, checkpoint['result'], windows['window_starts'])
            collected.append((sample, checkpoint['result'], rank))
            manifest.append(dict(dataset=key[0], question_id=key[1], rank=rank,
                                 image_identity=image, sample_sha256=digest(sample),
                                 checkpoint_path=str(path.relative_to(OUTPUT)), checkpoint_sha256=sha256(path)))
    if len({digest(e) for e in environments}) != 1:
        raise ValueError('Worker model/environment differs')
    counts = Counter(s['dataset'] for s, r, rank in collected)
    if dict(counts) != {k: v[1] for k, v in SHORT.items()}:
        raise ValueError('A short-answer dataset is partial; refuse a complete-short claim')
    return collected, dict(execution_binding=binding, frozen_worker_commit=config['code_commit'],
        frozen_windows_sha256=sha256(windows_path), window_starts=windows['window_starts'],
        dataset_question_counts=dict(counts),
        dataset_image_counts=dict(Counter(v[0] for v in images.values())),
        question_count=len(collected), image_count=len(images), generation_count=25*len(collected),
        checkpoints=manifest, worker_environments=environments,
        checks=dict(all_frozen_short_questions_present=True, all_25_conditions_present=True,
                    scores_recomputed_exact=True, sample_bindings_valid=True,
                    images_not_split_across_workers=True, frozen_sources_unchanged=True))


def plot_short(rows):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, unit in zip(axes.flat, SHORT):
        selected = [r for r in rows if r['dataset'] == unit]
        first = selected[0]
        ax.axhline(100*first['Full'], ls='--', color='#333333', label='Full')
        ax.axhspan(100*first['full_ci95_low'], 100*first['full_ci95_high'], color='#333333', alpha=.08)
        for mode in analyze.MODES:
            points = sorted([r for r in selected if r['mode'] == mode], key=lambda r: r['window_start'])
            ax.errorbar([r['window_start'] for r in points], [100*r['score'] for r in points],
                yerr=[[100*(r['score']-r['score_ci95_low']) for r in points],
                      [100*(r['score_ci95_high']-r['score']) for r in points]],
                fmt='o-', ms=4, color=analyze.COLORS[mode], label=mode)
            for r in points:
                if r['important_harm']:
                    ax.plot(r['window_start'], 100*r['score'], 'x', color='#b30000', ms=10)
        ax.set(title=f"{NAMES[unit]} (q={first['n_questions']}, images={first['n_images']})",
               xlabel='Frozen three-layer window start', ylabel='Official generation score (%)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Completed short-answer confirmation only; marginal 95% CI; red x: familywise important harm')
    analyze.save_figure(fig, 'short_answer_confirmation_performance')


@contextmanager
def short_aggregation_scope(destination):
    # This process has no GPU model. Restore globals even if CPU scoring fails.
    previous = analyze.OUTPUT, analyze.COUNTS, analyze.figure2
    analyze.OUTPUT, analyze.COUNTS, analyze.figure2 = destination, SHORT, plot_short
    try:
        yield
    finally:
        analyze.OUTPUT, analyze.COUNTS, analyze.figure2 = previous


def write_report(rows, audit):
    lines = ['# Paper RQ1: Completed Short-Answer Confirmation', '',
        '**Status: complete for all four short-answer datasets; the full six-task study remains partial.**', '',
        'This CPU-only snapshot does not wait for or summarize unfinished COCO Caption/TextCaps. '
        'The running GPU worker sources, model, data, windows and protocols are unchanged.', '',
        f"Frozen GPU code: `{audit['frozen_worker_commit']}`. "
        f"Questions: {audit['question_count']}; independent images: {audit['image_count']}; "
        f"actual greedy generations: {audit['generation_count']}.", '',
        '## Full and Frozen-Window Ranges', '',
        'Ranges show all eight preregistered windows, not a newly selected optimum. '
        'TextVQA is the official ten-answer soft score; other rows are accuracy. All entries are percentages.', '',
        '| Dataset | Questions / images | Full | Prefill-only range | Decode-only range | Combined range | Low-harm supported / 24 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for unit in SHORT:
        selected = [r for r in rows if r['dataset'] == unit]
        first = selected[0]
        ranges = []
        for mode in analyze.MODES:
            scores = [100*r['score'] for r in selected if r['mode'] == mode]
            ranges.append(f'{min(scores):.2f}-{max(scores):.2f}')
        lines.append(f"| {NAMES[unit]} | {first['n_questions']} / {first['n_images']} | "
                     f"{100*first['Full']:.2f} | "+' | '.join(ranges)+
                     f" | {sum(r['supports_low_harm'] for r in selected)} / 24 |")
    lines += ['', '![Completed short-answer confirmation](figures/short_answer_confirmation_performance.png)', '',
        '## All Frozen Conditions', '',
        'Changes and intervals below are percentage points. In binary tasks, harm = Full-correct to wrong, '
        'correction = Full-wrong to correct. TextVQA reports any soft-score decrease/increase instead.', '',
        '| Dataset | Window | Mode | Score % | Delta pp | Paired 95% CI pp | Harm / correction (soft down / up for TextVQA) | Familywise low-harm |',
        '|---|---|---|---:|---:|---|---:|---|']
    for r in rows:
        harm = r['correct_to_wrong'] if r['correct_to_wrong'] is not None else r['score_decreased']
        correction = r['wrong_to_correct'] if r['wrong_to_correct'] is not None else r['score_increased']
        lines.append(f"| {NAMES[r['dataset']]} | L{r['window_start']}-L{r['window_start']+2} | {r['mode']} | "
            f"{100*r['score']:.2f} | {100*r['delta']:+.2f} | "
            f"[{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}] | {harm} / {correction} | {r['supports_low_harm']} |")
    lines += ['', '## Interpretation and Validity', '',
        '- Full is paired on the same GPU and question as all 24 interventions; no historical baseline is mixed in.',
        '- Sources are verified unchanged; all 3,700 frozen questions have exactly 25 conditions. '
        'All 92,500 stored short scores were recomputed using the pinned scorer and matched exactly.',
        '- Aggregation is the original frozen `paper_rq1.analyze.confirmation_summary` function, '
        'redirected only in this separate CPU process to the four completed datasets and this new output folder.',
        '- Bootstrap: 50,000 paired image-cluster replicates, including complete six-question POPE clusters. '
        'Marginal 95% and Bonferroni familywise bounds remain in CSV. The original 144-primary-comparison '
        'family is retained, NOT reduced to 96 because Caption is pending.',
        '- Low-harm means the familywise lower bound exceeds the frozen -1 percentage point margin, '
        'and the frozen zero-effect rare-harm guard passes when applicable. '
        'Failure to establish low harm does not prove damage. A CI containing zero does not establish equivalence.',
        '- Candidate Margin/Gold LogP are not the scoring endpoint of this free-generation confirmation. '
        'Discovery fixed-Full-token DeltaLogP is a separate quantity and is not accuracy.',
        '- No incomplete Caption score is included. No aggregate completion for all 5,700 confirmation questions is claimed.',
        '- All eight windows and all three modes are shown, including negative outcomes; '
        'the range table is descriptive and does not select deployment windows.',
        '- Attention output is computed before zeroing: no actual compute skipping or speedup is claimed.', '',
        '## Artifacts', '',
        '- `confirmation_summary.csv`: all 96 conditions, paired intervals, flips, text/length/EOS and noninferiority decisions.',
        '- `confirmation_per_question_<dataset>.csv`: all 88,800 paired intervention rows with generated answers and scores.',
        '- `pope_secondary_metrics.csv`: accuracy, precision, recall, F1 and yes ratio.',
        '- `snapshot_manifest.json`: original checkpoint paths/SHA256, input/config bindings and worker environments.',
        '- `snapshot_completion.json`: complete-short / partial-study status and output hashes.',
        '- Original short checkpoints preserve Full and intervention token IDs and all 25 generated answers.', '']
    path = DESTINATION/'report.md'
    if path.exists() and path.read_text() != '\n'.join(lines):
        raise ValueError('Refuse to overwrite a different snapshot report')
    if not path.exists():
        with path.open('x') as f:
            f.write('\n'.join(lines))


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if (DESTINATION/'snapshot_completion.json').exists():
        raise ValueError('This snapshot is already complete')
    collected, audit = load_completed_short()
    freeze_json(DESTINATION/'snapshot_manifest.json', audit)
    freeze_json(DESTINATION/'frozen_windows.json', json.loads((OUTPUT/'frozen_windows.json').read_text()))
    with short_aggregation_scope(DESTINATION):
        rows = analyze.confirmation_summary(collected)
    if len(rows) != 96 or len({(r['dataset'], r['window_start'], r['mode']) for r in rows}) != 96:
        raise ValueError('Incomplete short aggregate')
    verify()
    write_report(rows, audit)
    outputs = {str(p.relative_to(DESTINATION)): sha256(p) for p in DESTINATION.rglob('*') if p.is_file()}
    freeze_json(DESTINATION/'snapshot_completion.json', dict(status='complete_short_answers',
        full_study_status='partial_caption_pending', completed_questions=3700, planned_confirmation_questions=5700,
        completed_short_generations=92500, comparison_family_primary=144, bootstrap_replicates=50000,
        frozen_worker_commit=audit['frozen_worker_commit'], execution_binding=audit['execution_binding'],
        snapshot_script_sha256=sha256(Path(__file__)),
        creation_checkout_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        created_utc=datetime.now(timezone.utc).isoformat(), output_sha256=outputs))
    print(json.dumps(dict(status='complete_short_answers', questions=len(collected), conditions=len(rows),
                          report=str(DESTINATION/'report.md'))), flush=True)


if __name__ == '__main__':
    main()

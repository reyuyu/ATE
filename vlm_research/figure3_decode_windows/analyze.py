"""Paired image-cluster statistics and Figure3 from all six generated answers."""
import csv
import json
import numpy as np
from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1.scorers import gqa_prediction
from paper_rq1_early_joint_attention.analyze import anomaly
from paper_rq1_early_joint_attention.run import checkpoint_valid
from paper_rq1_early_joint_attention.statistics import bootstrap
from .run import OUT, CONDITIONS, TASKS, SEED, read, verify, cohort, rank_for


def length_group(full):
    if not full['eos']:
        return 'Full missing EOS'
    n = sum(t != 2 for t in full['token_ids'])
    return 'Immediate EOS' if n == 0 else 'EOS-only tail' if n == 1 else 'Content tail'


def write_csv(name, records):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def summarize(values, images, gens, label):
    stats = bootstrap(values[:, 1:] - values[:, [0]], images, [5] * 5, seed=SEED)
    rows = []
    for k, key in enumerate(CONDITIONS):
        s = stats[k - 1] if k else None
        flags = [anomaly(g[k]['token_ids'], g[k]['text'], g[k]['eos']) for g in gens]
        rows.append(dict(group=label, condition=key, n=len(values), images=len(set(images)),
            correct=int(values[:, k].sum()), accuracy_pct=100 * float(values[:, k].mean()),
            delta_pp=100 * float((values[:, k] - values[:, 0]).mean()),
            correct_to_wrong=int(((values[:, 0] == 1) & (values[:, k] == 0)).sum()),
            wrong_to_correct=int(((values[:, 0] == 0) & (values[:, k] == 1)).sum()),
            ci95_low_pp=100 * s['ci95_low'] if s else '', ci95_high_pp=100 * s['ci95_high'] if s else '',
            corrected95_low_pp=100 * s['family95_low'] if s else '', corrected95_high_pp=100 * s['family95_high'] if s else '',
            noninferiority=bool(s['family95_low'] > -.01) if s else '',
            degenerate_bootstrap=s['degenerate_bootstrap'] if s else '',
            eos=sum(g[k]['eos'] for g in gens), truncated=sum(g[k]['truncated'] for g in gens),
            empty=sum(f['empty_answer'] for f in flags), repetition=sum(f['repetition_flag'] for f in flags),
            text_changed=sum(g[k]['text'] != g[0]['text'] for g in gens),
            mean_generated_tokens=float(np.mean([g[k]['length'] for g in gens]))))
    return rows


def figures(summary, taskwise):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10})
    labels = ['L0-L5', 'L6-L11', 'L12-L17', 'L18-L23', 'L26-L31']
    rows = summary[1:]
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    for k, row in enumerate(rows):
        color = '#12877e' if k == 0 else '#59636e'
        ax.errorbar(k, row['delta_pp'], yerr=[[row['delta_pp'] - row['corrected95_low_pp']],
            [row['corrected95_high_pp'] - row['delta_pp']]], fmt='o', color=color,
            markersize=8 if k == 0 else 6, capsize=5, linewidth=1.7)
    ax.axhline(0, color='#737373', linewidth=.8)
    ax.axhline(-1, color='#bb4c4c', linestyle='--', label='Noninferiority boundary: -1 pp')
    ax.set_xticks(range(5), labels)
    ax.set_xlabel('Six-layer joint Attention ablation window (Decode-only)')
    ax.set_ylabel('Accuracy change vs Full (percentage points)')
    ax.set_title('Figure 3. Depth-dependent Impact of Joint Attention\nAblation During Decoding', fontsize=11, pad=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.17); ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=False, fontsize=9)
    fig.text(.5, .015, '3,000 paired questions / 2,409 images; image-cluster bootstrap; five-comparison corrected 95% CI',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / 'figure3.png', dpi=260)
    fig.savefig(OUT / 'figure3.pdf'); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
    points = [r['accuracy_pct'] for r in taskwise]
    lower, upper = max(0, min(points) - 3), min(100, max(points) + 3)
    for ax, task in zip(axes.flat, TASKS):
        current = [r for r in taskwise if r['group'] == task]
        ax.axhline(current[0]['accuracy_pct'], color='#888888', linestyle='--', label='Full')
        for k, row in enumerate(current[1:]):
            ax.plot(k, row['accuracy_pct'], 'o', color='#12877e' if k == 0 else '#59636e')
        ax.set_xticks(range(5), labels, rotation=25, fontsize=8)
        ax.set_title(task + ' (n=500)', fontsize=10)
        ax.set_ylim(lower, upper); ax.grid(axis='y', alpha=.17)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('Exact-answer accuracy (%)')
    fig.suptitle('Supplement: six GQA task scores (shared accuracy scale)', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'task_scores.png', dpi=240)
    fig.savefig(OUT / 'task_scores.pdf'); plt.close(fig)


def main():
    c = verify()
    binding = digest(c)
    completions = [read(OUT / f'shard{k}/completion.json') for k in range(3)]
    assert all(r['status'] == 'complete' and r['binding'] == binding and r['hooks_clean'] and r['parameters_unchanged'] for r in completions)
    samples = cohort()
    per_question, all_gens, images, scores = [], [], [], []
    for sample in samples:
        directory = OUT / f"shard{rank_for(sample)}/checkpoints/{sample['task']}__{sample['question_id']}"
        current = []
        for key, condition in CONDITIONS.items():
            path = directory / f'{key}.json'
            assert checkpoint_valid(path, binding, sample, key)
            g = read(path)['generation']
            assert g['prediction'] == gqa_prediction(g['text']) and g['score'] == int(g['prediction'] == sample['answer'])
            ids = g['token_ids']
            assert 0 < len(ids) <= 32 and 2 not in ids[:-1]
            assert g['eos'] == (ids[-1] == 2) and g['truncated'] == (len(ids) == 32 and ids[-1] != 2)
            assert g['edits'] == (0 if key == 'Full' else 6 * (len(ids) - 1))
            assert g['own_prefill'] and g['own_prefix']
            current.append(g)
        assert len({g['token_ids'][0] for g in current}) == 1
        full = current[0]
        group = length_group(full)
        for key, g in zip(CONDITIONS, current):
            flags = anomaly(g['token_ids'], g['text'], g['eos'])
            per_question.append(dict(task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], gold=sample['answer'], condition=key,
                text=g['text'], prediction=g['prediction'], token_ids=json.dumps(g['token_ids']), score=g['score'],
                full_score=full['score'], delta_score=g['score'] - full['score'],
                full_content_tokens=sum(t != 2 for t in full['token_ids']), full_length_group=group,
                generated_tokens=g['length'], eos=g['eos'], truncated=g['truncated'],
                empty=flags['empty_answer'], repetition=flags['repetition_flag']))
        all_gens.append(current); images.append(sample['image_sha256']); scores.append([g['score'] for g in current])
    assert sum(len(list((OUT / f'shard{k}/checkpoints').glob('*/*.json'))) for k in range(3)) == 18000
    values = np.asarray(scores, dtype=float)
    images = np.asarray(images)
    summary = summarize(values, images, all_gens, 'Overall')
    print('Overall complete', flush=True)
    taskwise, lengthwise = [], []
    for task in TASKS:
        selected = np.array([i for i, s in enumerate(samples) if s['task'] == task])
        taskwise += summarize(values[selected], images[selected], [all_gens[i] for i in selected], task)
        print('Task complete', task, flush=True)
    for group in ['EOS-only tail', 'Content tail', 'Immediate EOS', 'Full missing EOS']:
        selected = np.array([i for i, g in enumerate(all_gens) if length_group(g[0]) == group], dtype=int)
        if len(selected):
            lengthwise += summarize(values[selected], images[selected], [all_gens[i] for i in selected], group)
            print('Full-length group complete', group, len(selected), flush=True)
    direct_stats = bootstrap(values[:, 2:] - values[:, [1]], images, [4] * 4, seed=SEED)
    direct = []
    for k, s in enumerate(direct_stats, start=2):
        harm = int(((values[:, 1] == 1) & (values[:, k] == 0)).sum())
        fix = int(((values[:, 1] == 0) & (values[:, k] == 1)).sum())
        direct.append(dict(comparison=f'W{k - 1}-W0', n=3000, delta_pp=100 * s['mean'],
            correct_to_wrong=harm, wrong_to_correct=fix, ci95_low_pp=100 * s['ci95_low'], ci95_high_pp=100 * s['ci95_high'],
            corrected95_low_pp=100 * s['family95_low'], corrected95_high_pp=100 * s['family95_high']))
    write_csv('accuracy_summary.csv', summary)
    write_csv('task_accuracy.csv', taskwise)
    write_csv('full_length_groups.csv', lengthwise)
    write_csv('depth_vs_W0_paired.csv', direct)
    write_csv('per_question.csv', per_question)
    figures(summary, taskwise)
    result = dict(status='complete', questions=3000, generations=18000, images=2409,
        binding=binding, summary=summary, direct_comparisons=direct, full_length_groups=lengthwise,
        gpu_generation_seconds=[s['generation_seconds'] for s in completions],
        figure_sources={name: sha256(OUT / name) for name in ['accuracy_summary.csv', 'task_accuracy.csv']})
    freeze_json(OUT / 'result.json', result)
    print(json.dumps(dict(summary=summary, direct=direct), indent=2), flush=True)


if __name__ == '__main__':
    main()

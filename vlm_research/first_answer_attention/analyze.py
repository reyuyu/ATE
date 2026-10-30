"""Only the five frozen contrasts; reuse all native Full free-generation scores."""
import csv
import json
import numpy as np
from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1.scorers import gqa_prediction
from paper_rq1_early_joint_attention.analyze import anomaly
from paper_rq1_early_joint_attention.run import checkpoint_valid
from paper_rq1_early_joint_attention.statistics import bootstrap
from .run import OUT, CONDITIONS, TASKS, SEED, DIRECT, cohort, rank_for, read, verify, baseline

KEYS = ['Full', *CONDITIONS]


def write_csv(name, records):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def summarize(values, images, gens, label):
    stats = bootstrap(values[:, 1:] - values[:, [0]], images, [5] * 5, seed=SEED)
    rows = []
    for k, key in enumerate(KEYS):
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
            first_top1_flip_pct=100 * float(np.mean([g[k]['first_top1_flip'] for g in gens])) if k else 0.,
            mean_first_logp_loss=float(np.mean([g[k]['first_logp_loss'] for g in gens])) if k else 0.,
            eos=sum(g[k]['eos'] for g in gens), truncated=sum(g[k]['truncated'] for g in gens),
            empty=sum(f['empty_answer'] for f in flags), repetition=sum(f['repetition_flag'] for f in flags),
            text_changed=sum(g[k]['text'] != g[0]['text'] for g in gens),
            mean_generated_tokens=float(np.mean([g[k]['length'] for g in gens]))))
    return rows


def figure(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 10})
    labels = ['A: L0-L4\nP-1 only', 'B: L0-L5\nP-1 only', 'C: L0-L4\nP-1 + Decode',
        'D: L0-L5\nP-1 + Decode', 'E: L2-L5\nP-1 + Decode']
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={'width_ratios': [1, 1.35]})
    colors = ['#59636e', '#16877c', '#ce7844', '#16877c', '#ce7844', '#6d5a9c']
    axes[0].bar(KEYS, [r['accuracy_pct'] for r in summary], color=colors, width=.65)
    for i, row in enumerate(summary):
        axes[0].annotate(f"{row['accuracy_pct']:.2f}", (i, row['accuracy_pct']), xytext=(0, 4),
            textcoords='offset points', ha='center', fontsize=8)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Complete-answer accuracy (%)')
    axes[0].set_title('3,000 frozen GQA questions')
    ax = axes[1]
    for i, row in enumerate(summary[1:]):
        ax.errorbar(i, row['delta_pp'], yerr=[[row['delta_pp'] - row['corrected95_low_pp']],
            [row['corrected95_high_pp'] - row['delta_pp']]], fmt='o', color=colors[i + 1], capsize=4, linewidth=1.6)
    ax.axhline(0, color='#777777', linewidth=.8)
    ax.axhline(-1, color='#ad4141', linestyle='--', label='Noninferiority boundary: -1 pp')
    ax.set_xticks(range(5), labels, fontsize=8)
    ax.set_ylabel('Accuracy change vs Full (pp)')
    ax.set_title('First-answer and content-generation intervention')
    ax.legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    fig.text(.5, .015, 'Full reused from Figure 3; 2,409 paired image clusters; five-comparison corrected 95% bootstrap CI',
        ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(OUT / 'first_answer_comparison.pdf')
    fig.savefig(OUT / 'first_answer_comparison.png', dpi=250)
    plt.close(fig)


def main():
    c = verify()
    binding = digest(c)
    done = [read(OUT / f'shard{k}/completion.json') for k in range(3)]
    assert all(d['status'] == 'complete' and d['binding'] == binding and d['hooks_clean'] and d['parameters_unchanged'] for d in done)
    samples = cohort()
    per_question, gens, images, scores = [], [], [], []
    for sample in samples:
        current = [baseline(sample, c['previous_binding'])]
        directory = OUT / f"shard{rank_for(sample)}/checkpoints/{sample['task']}__{sample['question_id']}"
        for key, condition in CONDITIONS.items():
            path = directory / f'{key}.json'
            assert checkpoint_valid(path, binding, sample, key)
            g = read(path)['generation']
            assert g['edits'] == len(condition['layers']) * (g['length'] if condition['decode'] else 1)
            assert g['full_first_token'] == current[0]['token_ids'][0]
            assert g['first_token'] == g['token_ids'][0]
            assert g['first_top1_flip'] == (g['first_token'] != g['full_first_token'])
            assert np.isfinite([g['first_logp_loss'], g['full_first_token_logp'], g['intervened_full_token_logp']]).all()
            current.append(g)
        # A/C and B/D must have precisely identical Prefill logits, despite their later different prefixes.
        assert current[1]['first_logits_sha256'] == current[3]['first_logits_sha256']
        assert current[2]['first_logits_sha256'] == current[4]['first_logits_sha256']
        assert len({g['full_first_logits_sha256'] for g in current[1:]}) == 1
        for key, g in zip(KEYS, current):
            ids = g['token_ids']
            assert g['prediction'] == gqa_prediction(g['text']) and g['score'] == int(g['prediction'] == sample['answer'])
            assert 0 < len(ids) <= 32 and 2 not in ids[:-1]
            assert g['eos'] == (ids[-1] == 2) and g['truncated'] == (len(ids) == 32 and ids[-1] != 2)
            assert g['own_prefill'] and g['own_prefix']
            flags = anomaly(ids, g['text'], g['eos'])
            per_question.append(dict(task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], gold=sample['answer'], condition=key,
                text=g['text'], prediction=g['prediction'], token_ids=json.dumps(ids), score=g['score'],
                full_score=current[0]['score'], delta_score=g['score'] - current[0]['score'],
                full_first_token=current[0]['token_ids'][0], first_token=ids[0],
                first_top1_flip=g.get('first_top1_flip', False), first_logp_loss=g.get('first_logp_loss', 0.),
                full_first_token_logp=g.get('full_first_token_logp', ''),
                intervened_full_token_logp=g.get('intervened_full_token_logp', ''),
                full_first_logits_sha256=g.get('full_first_logits_sha256', current[1]['full_first_logits_sha256']),
                first_logits_sha256=g.get('first_logits_sha256', current[1]['full_first_logits_sha256']),
                generated_tokens=g['length'], eos=g['eos'], truncated=g['truncated'],
                empty=flags['empty_answer'], repetition=flags['repetition_flag']))
        gens.append(current)
        images.append(sample['image_sha256'])
        scores.append([g['score'] for g in current])
    assert sum(len(list((OUT / f'shard{k}/checkpoints').glob('*/*.json'))) for k in range(3)) == 15000
    values, identities = np.asarray(scores, dtype=float), np.asarray(images)
    assert int(values[:, 0].sum()) == 2627
    summary = summarize(values, identities, gens, 'Overall')
    print('Overall', json.dumps(summary), flush=True)
    taskwise = []
    for task in TASKS:
        selected = np.asarray([i for i, s in enumerate(samples) if s['task'] == task])
        taskwise += summarize(values[selected], identities[selected], [gens[i] for i in selected], task)
        print('Task', task, flush=True)
    direct_values = np.column_stack([values[:, KEYS.index(a)] - values[:, KEYS.index(b)] for a, b in DIRECT])
    direct_stats = bootstrap(direct_values, identities, [6] * 6, seed=SEED)
    direct = []
    for (a, b), s in zip(DIRECT, direct_stats):
        before, after = values[:, KEYS.index(b)], values[:, KEYS.index(a)]
        direct.append(dict(comparison=f'{a}-{b}', n=3000, delta_pp=100 * s['mean'],
            correct_to_wrong=int(((before == 1) & (after == 0)).sum()),
            wrong_to_correct=int(((before == 0) & (after == 1)).sum()),
            ci95_low_pp=100 * s['ci95_low'], ci95_high_pp=100 * s['ci95_high'],
            corrected95_low_pp=100 * s['family95_low'], corrected95_high_pp=100 * s['family95_high']))
    write_csv('accuracy_summary.csv', summary)
    write_csv('task_accuracy.csv', taskwise)
    write_csv('paired_comparisons.csv', direct)
    write_csv('per_question.csv', per_question)
    figure(summary)
    result = dict(status='complete', binding=binding, questions=3000, images=2409, new_generations=15000,
        reused_full_generations=3000, summary=summary, direct_comparisons=direct,
        gpu_seconds=[d['seconds'] for d in done], validation=dict(cpu=True, three_gpu_smoke=True,
            baseline_compatibility=True, full_first_argmax_matches_all3000=True,
            AC_BD_first_logits_exact_all3000=True, complete_coverage=True),
        artifact_sha256={n: sha256(OUT / n) for n in ['accuracy_summary.csv', 'task_accuracy.csv',
            'paired_comparisons.csv', 'per_question.csv', 'first_answer_comparison.pdf', 'first_answer_comparison.png']})
    freeze_json(OUT / 'result.json', result)
    print('Complete', json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Image-cluster paired contrasts, compact targeted row summaries and paper figures."""
import csv
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import numpy as np
from PIL import Image

from benchmarks.dataset_registry import digest, sha256
from paper_rq1.scorers import gqa_prediction
from paper_rq1_early_joint_attention.statistics import bootstrap
from .capture import head_mass, vector_change
from .protocol import OUT, KEYS, SEED, old, read, identity, prior, verify, execution, atomic_json

CONTRASTS = ('L0L1_minus_Full', 'L2L5_minus_Full', 'L0L5_minus_Full', 'H01_given25', 'interaction_extra_harm')
LABELS = ('Full', 'L0-L1 OFF', 'L2-L5 OFF', 'L0-L5 OFF')


def contrasts(scores):
    full, f, e, d = np.asarray(scores, dtype=float).T
    return np.column_stack((f-full, e-full, d-full, e-d, f+e-d-full))


def write_csv(path, rows):
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize(samples, generations, group):
    scores = np.asarray([[g[k]['score'] for k in KEYS] for g in generations], dtype=float)
    images = np.asarray([s['image_sha256'] for s in samples])
    flips = np.asarray([[g[k]['first_top1_flip'] for k in KEYS[1:]] for g in generations], dtype=float)
    losses = np.asarray([[g[k]['first_logp_loss'] for k in KEYS[1:]] for g in generations], dtype=float)
    values = np.column_stack((scores, contrasts(scores), flips, losses))
    families = [1]*4 + [5 if group == 'Overall' else 1]*5 + [1]*6
    stats = bootstrap(values, images, families, seed=SEED)
    rows = []
    for k, name in enumerate(KEYS):
        delta = stats[4+k-1] if k else None
        flags = [g[name] for g in generations]
        rows.append(dict(group=group, condition=name, questions=len(samples), images=len(set(images)),
            correct=int(scores[:, k].sum()), accuracy_pct=100*scores[:, k].mean(),
            accuracy_ci95_low_pct=100*stats[k]['ci95_low'], accuracy_ci95_high_pct=100*stats[k]['ci95_high'],
            delta_full_pp=100*(scores[:, k]-scores[:, 0]).mean(),
            delta_ci95_low_pp=100*delta['ci95_low'] if delta else 0.,
            delta_ci95_high_pp=100*delta['ci95_high'] if delta else 0.,
            family95_low_pp=100*delta['family95_low'] if delta else 0.,
            family95_high_pp=100*delta['family95_high'] if delta else 0.,
            correct_to_wrong=int(((scores[:, 0] == 1) & (scores[:, k] == 0)).sum()),
            wrong_to_correct=int(((scores[:, 0] == 0) & (scores[:, k] == 1)).sum()),
            first_token_flips=sum(g.get('first_top1_flip', False) for g in flags),
            first_token_flip_pct=100*flips[:, k-1].mean() if k else 0.,
            flip_ci95_low_pct=100*stats[9+k-1]['ci95_low'] if k else 0.,
            flip_ci95_high_pct=100*stats[9+k-1]['ci95_high'] if k else 0.,
            mean_logp_loss=losses[:, k-1].mean() if k else 0.,
            logp_loss_ci95_low=stats[12+k-1]['ci95_low'] if k else 0.,
            logp_loss_ci95_high=stats[12+k-1]['ci95_high'] if k else 0.,
            truncated=sum(g['truncated'] for g in flags), empty=sum(not g['prediction'] for g in flags)))
    comparisons = []
    for name, stat in zip(CONTRASTS, stats[4:9]):
        comparisons.append(dict(group=group, contrast=name, n=len(samples), images=len(set(images)),
            estimate_pp=100*stat['mean'], ci95_low_pp=100*stat['ci95_low'], ci95_high_pp=100*stat['ci95_high'],
            family95_low_pp=100*stat['family95_low'], family95_high_pp=100*stat['family95_high'],
            repetitions=stat['repetitions']))
    first = comparisons[0]
    comparisons.append(dict(group=group, contrast='H01_alone', n=len(samples), images=len(set(images)),
        estimate_pp=-first['estimate_pp'], ci95_low_pp=-first['ci95_high_pp'], ci95_high_pp=-first['ci95_low_pp'],
        family95_low_pp=-first['family95_high_pp'], family95_high_pp=-first['family95_low_pp'],
        repetitions=first['repetitions']))
    return rows, comparisons


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def visual_map(arrays, condition, layer, image):
    raw = arrays[f'{condition}_attention_row_{layer}'].astype(float)[:, image].mean(0)
    return (raw / raw.sum()).reshape(24, 24), float(raw.sum())


def figure_a(meta, arrays, folder):
    maps = [visual_map(arrays, 'full', layer, meta['keys']['image']) for layer in (0, 1)]
    high = max(a.max() for a, mass in maps)
    norm = LogNorm(vmin=1e-5, vmax=max(high, 1.01e-5))
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.1), constrained_layout=True)
    axes[0].imshow(Image.open(meta['image_path']).convert('RGB'))
    axes[0].set_title('Original image')
    axes[1].imshow(Image.open(folder / 'clip_input.png'))
    axes[1].set_title('Actual CLIP crop (336 x 336)')
    for ax in axes[:2]:
        ax.axis('off')
    for ax, (grid, mass), layer in zip(axes[2:], maps, (0, 1)):
        im = ax.imshow(grid, cmap='magma', norm=norm, interpolation='nearest')
        ax.set_title(f'Full L{layer} P-1 -> Image\nRaw image mass: {100*mass:.2f}%')
        ax.set_xlabel('Patch column')
        ax.set_ylabel('Patch row')
        ax.set_xticks([0, 8, 16, 23])
        ax.set_yticks([0, 8, 16, 23])
    fig.colorbar(im, ax=axes[2:], fraction=.025, pad=.02, label='P(patch | image); logarithmic color scale')
    question = textwrap.fill(meta['question'], 112)
    fig.suptitle(f"Figure A | {meta['task']} {meta['question_id']} | outcome-blind case\n{question}", fontsize=11)
    save(fig, folder / 'figure_A_visual_attention')


def bos_values(meta, arrays, condition):
    return np.asarray([head_mass(arrays[f'{condition}_attention_row_{layer}'], meta['keys'])['bos'] for layer in range(2, 6)])


def figure_bc(cases):
    differences = [100*(bos_values(m, a, 'off') - bos_values(m, a, 'full')) for m, a, folder in cases]
    limit = max(float(np.max(np.abs(d))) for d in differences)
    norm = TwoSlopeNorm(vmin=-max(limit, 1e-8), vcenter=0, vmax=max(limit, 1e-8))
    fb, ab = plt.subplots(4, 3, figsize=(12, 11), sharex=True, sharey=True, constrained_layout=True)
    fc, ac = plt.subplots(4, 3, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    for index, (meta, arrays, folder) in enumerate(cases):
        full, off = bos_values(meta, arrays, 'full'), bos_values(meta, arrays, 'off')
        ax = ab.flat[index]
        ax.plot(range(2, 6), 100*full.mean(1), 'o-', color='#245c91', label='Full', lw=1.5, ms=3)
        ax.plot(range(2, 6), 100*off.mean(1), 's--', color='#bc493c', label='L0-L1 OFF', lw=1.5, ms=3)
        ax.set_title(f"{meta['task']} | {meta['question_id']}", fontsize=9)
        ax.set_xticks(range(2, 6))
        ax.set_ylim(0, 101)
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
        ax2 = ac.flat[index]
        heat = ax2.imshow(differences[index], cmap='RdBu_r', norm=norm, aspect='auto', interpolation='nearest')
        ax2.set_title(f"{meta['task']} | {meta['question_id']}", fontsize=9)
        ax2.set_yticks(range(4), ['L2', 'L3', 'L4', 'L5'])
        ax2.set_xticks([0, 8, 16, 24, 31])
        # Individual case B/C uses the same global color scale as the twelve-case panel.
        one, axs = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        axs[0].plot(range(2, 6), 100*full.mean(1), 'o-', color='#245c91', label='Full')
        axs[0].plot(range(2, 6), 100*off.mean(1), 's--', color='#bc493c', label='L0-L1 OFF')
        axs[0].set(xlabel='Layer', ylabel='P-1 -> BOS mass (%)', title='Figure B: head mean', xticks=range(2, 6), ylim=(0, 101))
        axs[0].legend(frameon=False)
        image = axs[1].imshow(differences[index], cmap='RdBu_r', norm=norm, aspect='auto', interpolation='nearest')
        axs[1].set(xlabel='Head', ylabel='Layer', title='Figure C: OFF - Full')
        axs[1].set_yticks(range(4), ['L2', 'L3', 'L4', 'L5'])
        one.colorbar(image, ax=axs[1], label='BOS mass change (pp)')
        one.suptitle(f"{meta['task']} {meta['question_id']} | original P-1, first answer prediction", fontsize=11)
        save(one, folder / 'figure_BC_bos_attention')
    ab.flat[0].legend(frameon=False, fontsize=8)
    fb.supxlabel('Layer')
    fb.supylabel('P-1 -> BOS mass (%), mean of 32 heads')
    fb.suptitle('Figure B | 12 preselected cases | first answer prediction, not EOS', fontsize=13)
    fc.supxlabel('Attention head')
    fc.suptitle('Figure C | BOS mass difference: L0-L1 OFF - Full', fontsize=13)
    fc.colorbar(heat, ax=list(ac.flat), fraction=.018, pad=.015, label='BOS mass change (percentage points)')
    save(fb, OUT / 'figures/figure_B_bos_mass_12cases')
    save(fc, OUT / 'figures/figure_C_layer_head_12cases')


def figure_d(meta, arrays, folder, number):
    full, off = bos_values(meta, arrays, 'full'), bos_values(meta, arrays, 'off')
    diff = 100*(off-full)
    limit = max(float(np.abs(diff).max()), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), constrained_layout=True)
    axes[0].imshow(Image.open(meta['image_path']).convert('RGB'))
    axes[0].axis('off')
    axes[0].set_title('Original image')
    axes[1].plot(range(2, 6), 100*full.mean(1), 'o-', color='#245c91', label='Full')
    axes[1].plot(range(2, 6), 100*off.mean(1), 's--', color='#bc493c', label='L0-L1 OFF')
    axes[1].set(xlabel='Layer', ylabel='P-1 -> BOS mass (%)', xticks=range(2, 6), ylim=(0, 101), title='First-answer prediction only')
    axes[1].legend(frameon=False)
    heat = axes[2].imshow(diff, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest')
    axes[2].set_yticks(range(4), ['L2', 'L3', 'L4', 'L5'])
    axes[2].set(xlabel='Head', title='BOS mass: OFF - Full')
    fig.colorbar(heat, ax=axes[2], label='Change (pp)', fraction=.05)
    q = textwrap.fill(meta['question'], 105)
    first_flip = meta['full_first_token'] != meta['off_first_token']
    title = (f"Figure D{number} | POST-HOC ILLUSTRATIVE DEGRADATION | {meta['task']} {meta['question_id']}\n{q}\n"
        f"Gold: {meta['answer']} | Full: {meta['full_answer']!r} (correct) | OFF: {meta['off_answer']!r} (wrong)\n"
        f"First token changed: {first_flip}. Selection is outcome-based; not confirmatory evidence.")
    fig.suptitle(title, fontsize=10)
    save(fig, OUT / f'figures/figure_D{number}_posthoc_degradation')


def mechanism(samples):
    protocol = verify()
    selected = {(r['task'], r['question_id']) for r in protocol['cases']}
    illustrated = {(r['task'], r['question_id']): i+1 for i, r in enumerate(read(OUT / 'posthoc_cases.json')['cases'])}
    masses, heads, vectors, summaries, panels = [], [], [], [], []
    for sample in samples:
        key = sample['task'], sample['question_id']
        if key not in selected | set(illustrated):
            continue
        folder = OUT / 'cases' / identity(sample)
        meta = read(folder / 'metadata.json')
        assert meta['npz_sha256'] == sha256(folder / 'p_minus_one.npz')
        assert meta['clip_png_sha256'] == sha256(folder / 'clip_input.png')
        assert meta['sample_sha256'] == digest(sample) and meta['bos_kv_l2_l5_exact']
        with np.load(folder / 'p_minus_one.npz') as stored:
            arrays = {k: stored[k] for k in stored.files}
        tag = dict(task=sample['task'], question_id=sample['question_id'], image_sha256=sample['image_sha256'],
            preselected=key in selected, posthoc_illustration=key in illustrated)
        for layer in range(6):
            for condition in ('full', 'off'):
                mass = head_mass(arrays[f'{condition}_attention_row_{layer}'], meta['keys'])
                total = sum(mass.values())
                assert np.max(np.abs(total-1)) <= .002
                masses.append({**tag, 'layer': layer, 'condition': condition,
                    **{name+'_mass': float(value.mean()) for name, value in mass.items()},
                    'effective_attention_output_norm': meta[f'{condition}_stats'][layer]['attention_output_norm']})
            if layer < 2:
                continue
            full, off = (head_mass(arrays[f'{c}_attention_row_{layer}'], meta['keys'])['bos'] for c in ('full', 'off'))
            for head in range(len(full)):
                heads.append({**tag, 'layer': layer, 'head': head, 'full_bos_mass': full[head],
                    'off_bos_mass': off[head], 'delta_bos_mass_pp': 100*(off[head]-full[head])})
            for name in ('hidden_in', 'attention_input', 'query_raw', 'query_rope', 'hidden_out', 'bos_key', 'bos_value'):
                change = vector_change(arrays[f'full_{name}_{layer}'], arrays[f'off_{name}_{layer}'])
                vectors.append({**tag, 'layer': layer, 'measurement': name, **change})
                if name in ('bos_key', 'bos_value'):
                    assert change['exact']
        bf, bo = bos_values(meta, arrays, 'full'), bos_values(meta, arrays, 'off')
        delta = bo-bf
        summaries.append({**tag, 'mean_delta_bos_pp': 100*delta.mean(),
            'mean_absolute_head_delta_pp': 100*np.abs(delta).mean(), 'max_absolute_head_delta_pp': 100*np.abs(delta).max(),
            'changed_head_layer_entries': int(np.count_nonzero(delta)), 'head_layer_entries': int(delta.size),
            'bos_kv_exact': True, 'first_token_flip': meta['full_first_token'] != meta['off_first_token']})
        if key in selected:
            figure_a(meta, arrays, folder)
            panels.append((meta, arrays, folder))
        if key in illustrated:
            figure_d(meta, arrays, folder, illustrated[key])
    assert len(panels) == 12
    figure_bc(panels)
    for name, rows in [('case_layer_mass.csv', masses), ('case_head_bos_changes.csv', heads),
                        ('case_vector_changes.csv', vectors), ('case_summary.csv', summaries)]:
        write_csv(OUT / name, rows)
    # Cases, not heads, are the observational units. No head-level p-values.
    frozen = [r for r in summaries if r['preselected']]
    result = dict(preselected=12, bos_kv_exact_all=True, cases_with_nonzero_bos_change=sum(r['changed_head_layer_entries'] > 0 for r in frozen),
        median_mean_absolute_head_delta_pp=float(np.median([r['mean_absolute_head_delta_pp'] for r in frozen])),
        max_absolute_head_delta_pp=max(r['max_absolute_head_delta_pp'] for r in frozen),
        layer_means=[dict(layer=layer, full_bos_mass=float(np.mean([r['bos_mass'] for r in masses if r['preselected'] and r['layer']==layer and r['condition']=='full'])),
            off_bos_mass=float(np.mean([r['bos_mass'] for r in masses if r['preselected'] and r['layer']==layer and r['condition']=='off']))) for layer in range(2, 6)])
    return result


def main():
    binding = execution()
    completion = read(OUT / 'completion.json')
    assert completion['status'] == 'complete' and completion['binding'] == digest(binding)
    assert completion['hooks_clean'] and completion['parameters_unchanged']
    samples = old.cohort()
    references = {(r['task'], r['question_id']): r for r in read(OUT / 'reuse_bindings.json')}
    generations, records = [], []
    for sample in samples:
        g = prior(sample, references[sample['task'], sample['question_id']])
        cp = read(OUT / 'checkpoints' / f'{identity(sample)}.json')
        assert cp['binding'] == digest(binding) and cp['sample_sha256'] == digest(sample)
        assert cp['result_sha256'] == digest(cp['generation'])
        g['L0L1_OFF'] = cp['generation']
        for name in KEYS:
            generation = g[name]
            assert generation['prediction'] == gqa_prediction(generation['text'])
            assert generation['score'] == int(generation['prediction'] == sample['answer'])
            assert generation['own_prefill'] and generation['own_prefix']
            assert 0 < generation['length'] <= 32 and 2 not in generation['token_ids'][:-1]
            if name != 'Full':
                assert generation['first_token'] == generation['token_ids'][0]
                assert np.isfinite(generation['first_logp_loss'])
            records.append(dict(task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], condition=name, gold=sample['answer'],
                answer=generation['text'], prediction=generation['prediction'], score=generation['score'],
                delta_score=generation['score']-g['Full']['score'], first_token=generation['token_ids'][0],
                first_token_flip=generation.get('first_top1_flip', False), first_logp_loss=generation.get('first_logp_loss', 0.),
                full_first_token_logp=generation.get('full_first_token_logp', g['L0L5_OFF']['full_first_token_logp']),
                tokens=json.dumps(generation['token_ids']), generated_tokens=generation['length'],
                eos=generation['eos'], truncated=generation['truncated'], reused=name != 'L0L1_OFF'))
        generations.append(g)
    assert len(list((OUT / 'checkpoints').glob('*.json'))) == 3000 and len(records) == 12000
    overall, paired = summarize(samples, generations, 'Overall')
    taskwise, task_paired = [], []
    for task in old.TASKS:
        indices = [i for i, s in enumerate(samples) if s['task'] == task]
        a, b = summarize([samples[i] for i in indices], [generations[i] for i in indices], task)
        taskwise += a
        task_paired += b
        print('Summarized', task, flush=True)
    for name, rows in [('accuracy_summary.csv', overall), ('paired_effects.csv', paired),
                        ('task_accuracy.csv', taskwise), ('task_paired_effects.csv', task_paired), ('per_question.csv', records)]:
        write_csv(OUT / name, rows)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    mechanism_result = mechanism(samples)
    result = dict(status='complete', code_commit=binding['code_commit'], source_sha256=binding['source_sha256'],
        questions=3000, images=2409, new_condition_generations=3000, reused_generations=9000,
        summary=overall, paired_effects=paired, mechanism=mechanism_result,
        gpu_seconds=read(OUT / 'accuracy_completion.json')['seconds'],
        cpu_validation=read(OUT / 'cpu_validation.json')['tests'], gpu_smoke_passed=True,
        artifact_sha256={str(p.relative_to(OUT)): sha256(p) for p in sorted(OUT.rglob('*')) if p.suffix in ('.csv', '.png', '.pdf')})
    atomic_json(OUT / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

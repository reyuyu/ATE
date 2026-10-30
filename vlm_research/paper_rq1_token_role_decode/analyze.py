"""CPU-only final analysis of all frozen images and native-token conditions."""
import csv
import io
import json
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1.analyze import write_csv
from .protocol import ROOT, OUT, LAYERS, TYPES, MODES, PROTOCOL
from .run import verify, gate, verify_targets
from .statistics import cluster_mean_ci, aggregate_positions, fixed_effects, FAMILY

COLORS = {'Function': '#237f92', 'Visual': '#bd642e', 'Relation': '#7855a6'}


def examples_csv(path, rows):
    if rows:
        write_csv(path, rows)
        return
    # Zero examples is a real result, represented by a header, never a fake row.
    buffer = io.StringIO(newline='')
    csv.writer(buffer, lineterminator='\n').writerow(['image_id', 'generation_step', 'word_type', 'word', 'loss', 'top1_flip', 'js_divergence'])
    content = buffer.getvalue().encode()
    if path.exists():
        if path.read_bytes() != content: raise ValueError('Changed empty example table')
    else:
        with path.open('xb') as stream: stream.write(content)


def save(fig, name):
    folder = OUT / 'figures'; folder.mkdir(exist_ok=True)
    fig.savefig(folder / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(folder / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def collect():
    verify(); verify_targets(); statuses = gate('ablation')
    if list(OUT.glob('**/failures/*.json')):
        raise ValueError('Recorded worker failures must be resolved and disclosed before completion')
    manifest = json.loads((OUT / 'sample_manifest.json').read_text()); rows = []; positions = 0
    chosen = json.loads((OUT / 'selected_token_manifest.json').read_text())
    for sample in manifest['samples']:
        name = sample['image_identity'].replace(':', '_')
        path = OUT / f"ablation/shard{sample['rank']}/checkpoints/{name}.json"
        value = json.loads(path.read_text())
        targets = chosen['images'][sample['image_identity']]['selected']
        if value['sample_sha256'] != digest(sample) or sha256(value['raw_path']) != value['raw_sha256']:
            raise ValueError('Changed formal sample/logits')
        if len(value['rows']) != len(targets)*20 or value['positions'] != len(targets):
            raise ValueError('Incomplete conditions')
        for target in targets:
            group = [r for r in value['rows'] if r['generation_step'] == target['generation_step']]
            expected = {('Full', -1), ('All32', -1)} | {(mode, layer) for mode in MODES for layer in LAYERS}
            if {(r['mode'], r['layer']) for r in group} != expected or len(group) != 20:
                raise ValueError('Duplicate or omitted token-layer conditions')
        with np.load(value['raw_path']) as raw:
            if not np.isfinite(raw['altered_logits']).all() or raw['altered_logits'].shape[0] != 19*len(targets):
                raise ValueError('Invalid raw altered logits')
        for audit in value['audit']:
            if not audit['replay_logits_exact'] or audit['parent_cache_sha256_before'] != audit['parent_cache_sha256_after']:
                raise ValueError('Cache or Full replay failed')
        rows.extend(value['rows']); positions += len(targets)
    if len({r['image_identity'] for r in manifest['samples']}) != 200:
        raise ValueError('Not200 original images')
    write_csv(OUT / 'per_token_layer_ablation.csv', rows)
    freeze_json(OUT / 'result_validation.json', dict(passed=True, images=200, target_positions=positions,
        intervention_forwards=positions*19, full_baseline_rows=positions, rows=len(rows),
        missing_or_duplicate_conditions=0, replay_all_steps_exact=True, cache_hash_checks=positions,
        statuses=statuses, source_commit=verify()['code_commit'], selected_sha256=sha256(OUT / 'selected_token_manifest.json')))
    return rows


def summarize(rows):
    averaged = aggregate_positions(rows); summaries = []
    for mode in MODES + ['All32']:
        for layer in LAYERS + ['mean9'] if mode != 'All32' else [-1]:
            source = averaged if layer == 'mean9' else rows
            for role in TYPES:
                sub = [r for r in source if r['mode'] == mode and r['layer'] == layer and r['word_type'] == role]
                if not sub: continue
                loss = np.array([r['loss'] for r in sub]); absolute = np.array([r['abs_delta_logp'] for r in sub])
                ci = cluster_mean_ci(sub, loss)
                summaries.append(dict(mode=mode, layer=layer, word_type=role, **ci,
                    median_loss=float(np.median(loss)), p90_loss=float(np.quantile(loss, .9)), p95_loss=float(np.quantile(loss, .95)),
                    mean_abs_delta_logp=float(absolute.mean()), p95_abs_delta_logp=float(np.quantile(absolute, .95)),
                    mean_top1_flip=float(np.mean([r['top1_flip'] for r in sub])),
                    mean_js=float(np.mean([r['js_divergence'] for r in sub])),
                    mean_full_logp=float(np.mean([r['full_token_logp'] for r in sub])),
                    mean_full_entropy=float(np.mean([r['full_token_entropy'] for r in sub])),
                    mean_generation_step=float(np.mean([r['generation_step'] for r in sub])),
                    mean_relative_position=float(np.mean([r['relative_generation_position'] for r in sub]))))
    write_csv(OUT / 'token_type_summary.csv', summaries)
    paired = []
    for mode in MODES:
        for layer in LAYERS + ['mean9']:
            source = averaged if layer == 'mean9' else rows
            sub = [r for r in source if r['mode'] == mode and r['layer'] == layer]
            lookup = {(r['image_sha256'], r['word_type']): r for r in sub}
            for role in ['Visual', 'Relation']:
                pairs = [(r, lookup[(r['image_sha256'], 'Function')]) for r in sub if r['word_type'] == role and (r['image_sha256'], 'Function') in lookup]
                if not pairs: continue
                left = [a for a, b in pairs]; delta = [a['loss']-b['loss'] for a, b in pairs]
                family = FAMILY if role == 'Visual' else 1
                paired.append(dict(mode=mode, layer=layer, comparison=role+'-Function',
                    **cluster_mean_ci(left, delta, family),
                    median_difference=float(np.median(delta)),
                    confidence_difference=float(np.mean([a['full_token_logp']-b['full_token_logp'] for a, b in pairs])),
                    relative_position_difference=float(np.mean([a['relative_generation_position']-b['relative_generation_position'] for a, b in pairs])),
                    inference='Planned22-contrast family' if role == 'Visual' else 'Secondary descriptive only, no familywise claim'))
    write_csv(OUT / 'paired_token_type_comparison.csv', paired)
    controlled = []; metadata = {}
    for mode in MODES:
        sub = [r for r in averaged if r['mode'] == mode]
        meta, effects = fixed_effects(sub); metadata[mode] = meta
        controlled.extend(dict(mode=mode, **r) for r in effects)
    if controlled: write_csv(OUT / 'controlled_effect_analysis.csv', controlled)
    else:
        with (OUT / 'controlled_effect_analysis.csv').open('x', newline='') as stream:
            csv.writer(stream).writerow(['mode','variable','effect','ci95_low','ci95_high'])
    freeze_json(OUT / 'controlled_model_audit.json', metadata)
    modes = []
    for role in TYPES:
        sub = [r for r in averaged if r['mode'] == 'Whole' and r['word_type'] == role]
        lookup = {(r['image_sha256'], r['generation_step']): r for r in averaged if r['mode'] == 'Image'}
        if sub:
            delta = [r['loss']-lookup[(r['image_sha256'], r['generation_step'])]['loss'] for r in sub]
            modes.append(dict(word_type=role, comparison='Whole-Image (same token, mean9)', **cluster_mean_ci(sub, delta)))
    write_csv(OUT / 'whole_image_paired_comparison.csv', modes)
    all32 = []; near = []; severe = []
    for role in TYPES:
        sub = [r for r in rows if r['mode'] == 'All32' and r['word_type'] == role]
        if not sub: continue
        loss = np.array([r['loss'] for r in sub]); flips = np.array([r['top1_flip'] for r in sub], float)
        ci = cluster_mean_ci(sub, loss); fci = cluster_mean_ci(sub, flips)
        small = [r for r in sub if r['abs_delta_logp'] <= .01 and not r['top1_flip'] and r['js_divergence'] <= 1e-4]
        bad = [r for r in sub if r['loss'] >= 1.]
        near.extend(small); severe.extend(sorted(bad, key=lambda r: -r['loss']))
        all32.append(dict(word_type=role, **ci, median_loss=float(np.median(loss)), p90_loss=float(np.quantile(loss, .9)), p95_loss=float(np.quantile(loss, .95)),
            mean_abs_delta_logp=float(np.mean([r['abs_delta_logp'] for r in sub])),
            top1_flips=int(flips.sum()), top1_flip_rate=float(flips.mean()), flip_ci95_low=fci['ci95_low'], flip_ci95_high=fci['ci95_high'],
            severe_count=len(bad), severe_fraction=len(bad)/len(sub), loss_above_point1=float((loss>.1).mean()),
            loss_above_point5=float((loss>.5).mean()), near_unchanged_count=len(small),
            severe_definition='loss>=1 nat; not final answer harm', near_unchanged_definition=PROTOCOL['near_unchanged']))
    write_csv(OUT / 'all32_ablation_summary.csv', all32)
    examples_csv(OUT / 'all32_near_unchanged_examples.csv', near)
    examples_csv(OUT / 'all32_severe_examples.csv', severe)
    return averaged, summaries, paired, controlled, modes, all32


def figures(rows, averaged, summary, paired, controlled, modes, all32):
    lookup = {(r['mode'], r['layer'], r['word_type']): r for r in summary}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    values = [lookup[(mode, layer, role)]['mean'] for mode in MODES for layer in LAYERS for role in TYPES if (mode, layer, role) in lookup]
    bound = max(max(np.abs(values)), 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
    for ax, mode in zip(axes, MODES):
        matrix = np.array([[lookup.get((mode, layer, role), {}).get('mean', np.nan) for layer in LAYERS] for role in TYPES])
        image = ax.imshow(matrix, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect='auto')
        ax.set_xticks(range(9), ['L'+str(l) for l in LAYERS]); ax.set_yticks(range(3), TYPES); ax.set_title(mode+' Attention off')
        for i, role in enumerate(TYPES):
            for j, layer in enumerate(LAYERS):
                r = lookup.get((mode, layer, role))
                if r: ax.text(j, i, f"{r['mean']:.3f}\nn={r['n_images']}", ha='center', va='center', fontsize=7, color='white' if abs(r['mean']) > .55*bound else 'black')
    fig.colorbar(image, ax=axes, label='Mean Full-token LogP loss (nats), signed shared scale', shrink=.8)
    fig.suptitle('Figure1: Layer x native token role; nine measured layers only')
    save(fig, 'figure1_layer_token_type_sensitivity')
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
    for j, mode in enumerate(MODES):
        x = np.arange(3)+(j-.5)*.15
        r = [lookup[(mode, 'mean9', role)] for role in TYPES]
        y = np.array([a['mean'] for a in r]); lo = np.array([a['ci95_low'] for a in r]); hi = np.array([a['ci95_high'] for a in r])
        axes[0].errorbar(x, y, yerr=[np.maximum(y-lo,0), np.maximum(hi-y,0)], fmt='o' if j==0 else 's', capsize=4, label=mode)
        raw = next(a for a in paired if a['mode']==mode and a['layer']=='mean9' and a['comparison']=='Visual-Function')
        axes[1].errorbar(j, raw['mean'], yerr=[[max(raw['mean']-raw['ci95_low'],0)], [max(raw['ci95_high']-raw['mean'],0)]], fmt='o', capsize=4)
        for k, role in enumerate(['Visual', 'Relation']):
            rr = next((a for a in controlled if a['mode']==mode and a['variable']==role+'_vs_Function'), None)
            if rr:
                axes[2].errorbar(j+(k-.5)*.15, rr['effect'], yerr=[[max(rr['effect']-rr['ci95_low'],0)], [max(rr['ci95_high']-rr['effect'],0)]], fmt='o' if k==0 else 's', color=COLORS[role], label=role+'-Function' if j==0 else None, capsize=4)
    axes[0].set_xticks(range(3), TYPES); axes[0].set_title('Unadjusted per-role mean9'); axes[0].legend()
    axes[1].set_xticks(range(2), MODES); axes[1].set_title('Same-image Visual-Function')
    axes[2].set_xticks(range(2), MODES); axes[2].set_title('Within-image adjusted role effects'); axes[2].legend(fontsize=8)
    for ax in axes: ax.axhline(0, color='black', ls=':', lw=1); ax.set_ylabel('LogP loss / difference (nats), ordinary95% CI'); ax.grid(alpha=.15)
    fig.suptitle('Figure2: Role associations before/after confidence, entropy and position controls; not a causal role effect')
    save(fig, 'figure2_raw_and_controlled_role_comparison')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    image_lookup = {(r['image_id'], r['generation_step']): r for r in averaged if r['mode']=='Image'}
    for role in TYPES:
        sub = [r for r in averaged if r['mode']=='Whole' and r['word_type']==role]
        axes[0].scatter([image_lookup[(r['image_id'],r['generation_step'])]['loss'] for r in sub], [r['loss'] for r in sub], s=14, alpha=.55, color=COLORS[role], label=role)
    axes[0].axhline(0,color='black',ls=':',lw=.7); axes[0].axvline(0,color='black',ls=':',lw=.7)
    axes[0].set_xlabel('Image contribution off, mean9 LogP loss'); axes[0].set_ylabel('Whole Attention off, mean9 LogP loss'); axes[0].legend()
    for i, r in enumerate(modes): axes[1].errorbar(i, r['mean'], yerr=[[max(r['mean']-r['ci95_low'],0)],[max(r['ci95_high']-r['mean'],0)]],fmt='o',color=COLORS[r['word_type']],capsize=4)
    axes[1].set_xticks(range(len(modes)),[r['word_type'] for r in modes]); axes[1].axhline(0,color='black',ls=':'); axes[1].set_ylabel('Paired Whole-Image loss, ordinary95% CI')
    fig.suptitle('Figure3: Same token, same original Full history; direct image contribution is not all visual information')
    save(fig, 'figure3_whole_vs_image_contribution')
    function = [r for r in rows if r['mode']=='All32' and r['word_type']=='Function']
    case = max(function, key=lambda r:r['loss']) if function else rows[0]
    samples = json.loads((OUT/'sample_manifest.json').read_text())['samples']; sample=next(s for s in samples if s['image_identity']==case['image_id'])
    chosen = json.loads((OUT/'selected_token_manifest.json').read_text())['images'][case['image_id']]
    fig=plt.figure(figsize=(15,8),layout='constrained'); grid=fig.add_gridspec(2,3)
    photo=fig.add_subplot(grid[0,0]); text=fig.add_subplot(grid[0,1:]); photo.imshow(Image.open(sample['image_path'])); photo.axis('off'); text.axis('off')
    selected_words='\n'.join(f"{w['word_type']}: {w['word']!r}, t={w['generation_step']}" for w in chosen['selected'])
    text.text(0,.95,'\n'.join(textwrap.wrap(chosen['caption'],76))+'\n\n'+selected_words+'\n\nPost-hoc illustration: largest Function All32 loss.\nNot representative evidence; all samples remain in statistics.',va='top',fontsize=12)
    for ax,mode in [(fig.add_subplot(grid[1,:2]),'Whole'),(fig.add_subplot(grid[1,2]),'Image')]:
        for role in TYPES:
            sub=sorted([r for r in rows if r['image_id']==case['image_id'] and r['mode']==mode and r['word_type']==role],key=lambda r:LAYERS.index(r['layer']))
            if sub: ax.plot([LAYERS.index(r['layer']) for r in sub],[r['loss'] for r in sub],'o-',color=COLORS[role],label=role+': '+sub[0]['word'])
        ax.set_xticks(range(9),LAYERS); ax.set_xlabel('Measured layer (categorical, no intermediate measurements)'); ax.set_ylabel('Full-token LogP loss'); ax.set_title(mode); ax.axhline(0,color='black',ls=':'); ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.suptitle('Figure4: '+case['image_id']+' native Caption and selected target tokens')
    save(fig,'figure4_real_caption_case')
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for i,r in enumerate(all32):
        color=COLORS[r['word_type']]
        axes[0].errorbar(i,r['mean'],yerr=[[max(r['mean']-r['ci95_low'],0)],[max(r['ci95_high']-r['mean'],0)]],fmt='o',capsize=4,color=color)
        axes[1].errorbar(i,r['top1_flip_rate'],yerr=[[max(r['top1_flip_rate']-r['flip_ci95_low'],0)],[max(r['flip_ci95_high']-r['top1_flip_rate'],0)]],fmt='o',capsize=4,color=color)
        axes[2].bar(i,r['severe_fraction'],color=color,width=.55); axes[2].text(i,r['severe_fraction']+.02,f"{r['severe_count']}/{r['n_observations']}",ha='center',fontsize=9)
    for ax in axes: ax.set_xticks(range(len(all32)),[r['word_type'] for r in all32]); ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Mean LogP loss, ordinary95% CI'); axes[1].set_ylabel('Top1 flip rate, ordinary95% CI'); axes[2].set_ylabel('Loss>=1 nat fraction'); axes[2].set_ylim(0,1.1)
    fig.suptitle('Figure5: All32 Attention off at this one Decode Query only; no accuracy or safe-skipping claim')
    save(fig,'figure5_all32_attention_off')


def report(summary, paired, controlled, all32):
    freeze = json.loads((OUT/'target_freeze.json').read_text()); config=verify()
    lines=['# Observation 2: Token-Dependent Decode Attention Necessity','',
        'Status: complete frozen200-image conditional mechanism study. No training, full-answer accuracy or speedup claim.',
        '',f"Collector commit: `{config['code_commit']}`. Exact selected positions: `{freeze['counts']}`. "
        f"Alignment failures: {freeze['alignment_failures']}. All200 images retained; missing roles are not fabricated.",
        '', '## Nine-Layer Mean Loss', '', '| Mode | Role | Positions/images | Mean loss | Ordinary95% CI | Top1 flip (mean per-layer rate) |', '|---|---|---:|---:|---|---:|']
    for r in summary:
        if r['layer']=='mean9': lines.append(f"| {r['mode']} | {r['word_type']} | {r['n_observations']}/{r['n_images']} | {r['mean']:.6f} | [{r['ci95_low']:.6f},{r['ci95_high']:.6f}] | {100*r['mean_top1_flip']:.2f}% |")
    lines += ['', '## Planned Visual-Function Contrasts', '', '| Mode | Analysis | Difference | Ordinary95% CI | Familywise95% CI (22) | Images |','|---|---|---:|---|---|---:|']
    for mode in MODES:
        for r in paired:
            if r['mode']==mode and r['layer']=='mean9' and r['comparison']=='Visual-Function':
                lines.append(f"| {mode} | Same-image unadjusted | {r['mean']:.6f} | [{r['ci95_low']:.6f},{r['ci95_high']:.6f}] | [{r['familywise95_low']:.6f},{r['familywise95_high']:.6f}] | {r['n_images']} |")
        for r in controlled:
            if r['mode']==mode and r['variable']=='Visual_vs_Function':
                lines.append(f"| {mode} | Within-image adjusted | {r['effect']:.6f} | [{r['ci95_low']:.6f},{r['ci95_high']:.6f}] | [{r['familywise95_low']:.6f},{r['familywise95_high']:.6f}] | {r['n_images']} |")
    lines += ['', '## All32 Diagnostic', '', '| Role | Mean loss | Flip | Loss>=1 nat | Near unchanged |','|---|---:|---:|---:|---:|']
    for r in all32: lines.append(f"| {r['word_type']} | {r['mean']:.6f} | {r['top1_flips']}/{r['n_observations']} | {r['severe_count']}/{r['n_observations']} | {r['near_unchanged_count']}/{r['n_observations']} |")
    lines += ['', '## Interpretation Boundaries', '',
        'Full-token LogP loss and Top1 flip are not complete-answer accuracy. Lexicon Visual labels are ungrounded candidates. '
        'Roles are conditioned on native generation. Current-query Image removal preserves original weights/denominator and '
        'does not remove image information already stored in text states. Whole-Image differences are not an additive attribution.', '',
        'Controls: native Full LogP, entropy, absolute/relative generation position; image fixed effects on matching Function/Visual-image cohort. '
        'Main token_length=1 is constant and excluded from regression, not falsely estimated. '
        'Cluster bootstrap uses whole images. The22-contrast family contains four global raw/adjusted mode contrasts and18 mode/layer contrasts; Relation comparisons are descriptive. '
        'No per-head/layer independence or universal safety conclusion. No unmeasured layer interpolation.', '',
        'Full replay logits match saved native logits bit-exactly at every generated step. All interventions own cloned caches; '
        'parent SHA256 before/after every target, no alias/metadata mutation, no free continuation. '
        'Smoke compares original model.generate and verifies no-op, image-observer equality and FP16/FP32 propagated contribution bounds. '
        'No tolerance was fitted to the data.', '',
        'See `research_summary.md` for the Chinese eight-question interpretation and `README.md` for exact resume commands.']
    for p in sorted((OUT/'figures').glob('*.png')): lines += ['', f'![{p.stem}](figures/{p.name})']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    rows=collect(); result=summarize(rows); figures(rows,*result); report(*[result[i] for i in [1,2,3,5]])
    freeze_json(OUT/'analysis_completion.json',dict(status='complete_collected_analysis',images=200,all_positions_complete=True,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),Path(__file__).with_name('statistics.py')]},
        code_commit=verify()['code_commit'], no_accuracy_claim=True,no_speedup_claim=True,
        files_sha256={str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.lock','.tmp'] and p.name not in ['analysis_completion.json']}))
    print('Analysis complete; all200 images and selected positions validated.',flush=True)


if __name__=='__main__': main()

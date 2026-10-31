"""Native P-1 mass statistics and processor-aligned publication figures."""
import csv
import json
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1_early_joint_attention.statistics import bootstrap
from paper_rq1_single_layer_gqa6.analyze import descriptive_correlations
from .engine import METRICS
from .run import OUT, ORIGINAL, SEED, TASKS, cohort, read, verify

COLORS = {'image_mass': '#CE7028', 'prompt_mass': '#167D8D', 'bos_mass': '#454255'}
LABELS = {'image_mass': 'Image', 'prompt_mass': 'Prompt', 'bos_mass': 'BOS'}
MAIN = list(COLORS)
LAYERS = [0, 3, 15, 31]


def write_csv(name, records):
    assert records
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def mapped_field(grid, geometry):
    """Invert the actual resize and crop, leaving unseen margins transparent."""
    w, h = geometry['original_size']
    rw, rh = geometry['resized_size']
    left, top = geometry['crop_left'], geometry['crop_top']
    assert np.asarray(grid).shape == (24, 24)
    field = Image.fromarray(np.asarray(grid, np.float32)).resize((336, 336), Image.Resampling.BILINEAR)
    canvas = Image.new('F', (rw, rh), 0)
    mask = Image.new('F', (rw, rh), 0)
    canvas.paste(field, (left, top))
    mask.paste(1., (left, top, left+336, top+336))
    return (np.asarray(canvas.resize((w, h), Image.Resampling.BILINEAR)),
        np.asarray(mask.resize((w, h), Image.Resampling.BILINEAR)))


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 240,
        'axes.labelsize': 10, 'legend.fontsize': 9})


def save(fig, stem):
    fig.savefig(OUT / (stem + '.png'), bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / (stem + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def curves(summary, group, ax, labels=False):
    for key in MAIN:
        rows = [r for r in summary if r['group'] == group and r['source'] == key]
        assert len(rows) == 32
        x, mean, low, high = ([r[k] for r in rows] for k in ['layer','mean','ci95_low','ci95_high'])
        ax.plot(x, mean, color=COLORS[key], lw=1.8, label=LABELS[key])
        ax.fill_between(x, low, high, color=COLORS[key], alpha=.18, linewidth=0)
    for lo, hi in [(0,1), (2,5), (14,16)]:
        ax.axvspan(lo-.45, hi+.45, color='#787878', alpha=.08, zorder=-2)
        if labels:
            ax.text((lo+hi)/2, 1.015, f'L{lo}\u2013L{hi}', ha='center', va='bottom', fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(-.45,31.45)
    ax.set_ylim(0,1)
    ax.set_xticks([0,3,5,10,15,20,25,31])
    ax.grid(axis='y', alpha=.16, linewidth=.5)


def mass_figures(summary):
    fig, ax = plt.subplots(figsize=(7.2,3.4))
    curves(summary,'Overall',ax,True)
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean attention mass')
    ax.legend(loc='upper center', bbox_to_anchor=(.58,1.02), ncol=3, frameon=False)
    fig.subplots_adjust(top=.88,bottom=.16,left=.1,right=.98)
    save(fig,'figure_x_attention_mass')
    fig, axes = plt.subplots(2,3,figsize=(10.2,5.8),sharex=True,sharey=True)
    for ax, task in zip(axes.flat,TASKS):
        curves(summary,task,ax)
        ax.set_title(f'{task} (n=500)', fontsize=10)
    for ax in axes[-1]: ax.set_xlabel('Layer index')
    for ax in axes[:,0]: ax.set_ylabel('Mean attention mass')
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.95))
    save(fig,'supplement_six_tasks')


def case_figure(records, conditional=False):
    selected = read(OUT / 'showcase_selection.json')['question_ids']
    lookup = {s['question_id']: s for s in cohort()}
    fig = plt.figure(figsize=(10.8,10.5))
    layout = fig.add_gridspec(8,6,height_ratios=[.23,1]*4,
        width_ratios=[1,1,1,1,1,.045],left=.015,right=.965,top=.985,bottom=.035,wspace=.08,hspace=.16)
    for i,qid in enumerate(selected):
        sample = lookup[qid]
        result = records[qid]['result']
        maps = [np.asarray(result['spatial_maps'][str(layer)], float) for layer in LAYERS]
        masses = [result['rows'][layer]['image_mass'] for layer in LAYERS]
        if conditional:
            maps = [m / mass for m,mass in zip(maps,masses)]
        vmax = max(float(m.max()) for m in maps)
        norm = Normalize(0,vmax)
        textax = fig.add_subplot(layout[2*i,:5])
        textax.axis('off')
        line = f"{sample['task']} | QID {qid} | Q: {sample['question']} | Full: {result['full_text'].strip()}"
        textax.text(0,.42,'\n'.join(textwrap.wrap(line,125)),va='center',ha='left',fontsize=9)
        image = Image.open(OUT / 'case_images' / f'{qid}.png').convert('RGB')
        for j in range(5):
            ax = fig.add_subplot(layout[2*i+1,j])
            ax.imshow(image)
            if j:
                field, mask = mapped_field(maps[j-1],result['geometry'])
                rgba = plt.get_cmap('magma')(norm(field))
                rgba[...,3] = mask*.62
                ax.imshow(rgba)
                g = result['geometry']
                left, top = g['crop_left']/g['scale_x'], g['crop_top']/g['scale_y']
                ax.add_patch(Rectangle((left,top),336/g['scale_x'],336/g['scale_y'],
                    fill=False,ec='white',lw=.6,linestyle=':'))
                ax.set_title(f'L{LAYERS[j-1]} | Image mass {masses[j-1]:.3f}',fontsize=8,pad=3)
            else: ax.set_title('Original image',fontsize=8,pad=3)
            ax.set_axis_off()
        cax = fig.add_subplot(layout[2*i+1,5])
        cb = fig.colorbar(ScalarMappable(norm=norm,cmap='magma'),cax=cax)
        cb.ax.tick_params(labelsize=6)
        cb.set_label('Conditional patch probability' if conditional else 'Raw patch probability',fontsize=7)
    fig.text(.015,.008,'Shared scale across depth within each case; dotted boundary = CLIP crop; outside-crop pixels are unobserved.',fontsize=8)
    save(fig,'supplement_cases_conditional' if conditional else 'figure_y_visual_cases')


def sensitivity(all_rows, binding):
    for name, expected in binding['historical_csv'].items():
        assert sha256(ORIGINAL / name) == expected
    index = {(r['task'],r['question_id'],r['layer']):r for r in all_rows}
    joined = []
    seen = set()
    with (ORIGINAL / 'first_token_per_question.csv').open() as f:
        for h in csv.DictReader(f):
            key = (h['task'],h['question_id'],int(h['layer']))
            assert key in index and key not in seen
            seen.add(key)
            current = index[key]
            assert current['image_id'] == h['image_id']
            joined.append(dict(**current,abs_delta_logp=abs(float(h['delta_logp'])),
                delta_logp=float(h['delta_logp']),js=float(h['js'])))
    assert len(seen) == 96000
    bos_error = 0.
    seen_bos = set()
    with (ORIGINAL / 'bos_mass_per_question.csv').open() as f:
        for h in csv.DictReader(f):
            key = (h['task'],h['question_id'],int(h['layer']))
            assert key in index and key not in seen_bos
            seen_bos.add(key)
            bos_error = max(bos_error,abs(float(h['bos_mass'])-index[key]['bos_mass']))
    assert len(seen_bos) == 96000
    aligned, correlations = [], []
    for group in ['Overall',*TASKS]:
        subset = [r for r in joined if group=='Overall' or r['task']==group]
        layer_rows=[]
        for layer in range(32):
            rows=[r for r in subset if r['layer']==layer]
            layer_rows.append(dict(group=group,layer=layer,n=len(rows),
                **{m:float(np.mean([r[m] for r in rows])) for m in MAIN},
                mean_abs_delta_logp=float(np.mean([r['abs_delta_logp'] for r in rows])),
                mean_delta_logp=float(np.mean([r['delta_logp'] for r in rows])),
                mean_js=float(np.mean([r['js'] for r in rows]))))
        aligned.extend(layer_rows)
        for m in MAIN:
            for target in ['mean_abs_delta_logp','mean_js']:
                p,s = descriptive_correlations([r[m] for r in layer_rows],[r[target] for r in layer_rows])
                correlations.append(dict(group=group,source=m,target=target,n_layers=32,pearson=p,spearman=s,
                    interpretation='Descriptive depth-level association, not independent or causal evidence'))
    write_csv('aligned_layer_sensitivity.csv',aligned)
    write_csv('mass_sensitivity_correlations.csv',correlations)
    return bos_error


def main():
    c = verify()
    binding = digest(c)
    complete = read(OUT / 'completion.json')
    assert complete['status']=='complete' and complete['binding']==binding
    records, rows, values, image_ids, tasks = {}, [], [], [], []
    assert len(list((OUT / 'checkpoints').glob('*.json')))==3000
    for sample in cohort():
        record=read(OUT/'checkpoints'/f"{sample['task']}__{sample['question_id']}.json")
        assert record['binding']==binding and record['sample_sha256']==digest(sample)
        assert record['result_sha256']==digest(record['result'])
        assert len(record['result']['rows'])==32
        records[sample['question_id']]=record
        matrix=[]
        for layer, r in enumerate(record['result']['rows']):
            assert r['layer']==layer
            rows.append(dict(task=sample['task'],question_id=sample['question_id'],image_id=sample['image_id'],
                image_sha256=sample['image_sha256'],query=sample['input_mapping']['p_minus_one'],**r))
            matrix.extend(r[m] for m in METRICS)
        values.append(matrix)
        image_ids.append(sample['image_sha256'])
        tasks.append(sample['task'])
    assert len(rows)==96000 and len(set(image_ids))==2409
    write_csv('per_question_layer.csv',rows)
    values=np.asarray(values,float)
    assert np.isfinite(values).all()
    summary=[]
    for group in ['Overall',*TASKS]:
        mask=np.ones(3000,bool) if group=='Overall' else np.asarray(tasks)==group
        subset=values[mask]
        identities=[v for v,keep in zip(image_ids,mask) if keep]
        boot=bootstrap(subset,identities,[1]*subset.shape[1],repetitions=10000,seed=SEED)
        std=subset.std(axis=0,ddof=1)
        for layer in range(32):
            for j,m in enumerate(METRICS):
                k=layer*len(METRICS)+j
                b=boot[k]
                summary.append(dict(group=group,layer=layer,source=m,n_questions=len(subset),
                    n_images=len(set(identities)),mean=b['mean'],std=float(std[k]),ci95_low=b['ci95_low'],ci95_high=b['ci95_high']))
        print('Summarized',group,len(subset),flush=True)
    write_csv('layer_summary.csv',summary)
    bos_error=sensitivity(rows,c)
    spatial=[]
    for s in read(OUT/'case_pool.json')['samples']:
        record=records[s['question_id']]
        for layer in LAYERS:
            grid=np.asarray(record['result']['spatial_maps'][str(layer)])
            mass=record['result']['rows'][layer]['image_mass']
            assert np.isclose(grid.sum(),mass,atol=1e-12,rtol=0)
            for y,x in np.ndindex(24,24):
                spatial.append(dict(task=s['task'],question_id=s['question_id'],layer=layer,patch_row=y,patch_col=x,
                    raw_probability=float(grid[y,x]),conditional_visual_probability=float(grid[y,x]/mass)))
    write_csv('case_patch_attention.csv',spatial)
    style()
    mass_figures(summary)
    case_figure(records)
    case_figure(records,True)
    window_rows=[]
    for lo,hi in [(0,1),(2,5),(14,16),(25,31)]:
        means={m:float(np.mean([r['mean'] for r in summary if r['group']=='Overall' and lo<=r['layer']<=hi and r['source']==m])) for m in MAIN}
        window_rows.append(dict(layers=f'L{lo}-L{hi}',**means))
    write_csv('depth_band_means.csv',window_rows)
    freeze_json(OUT/'analysis_validation.json',dict(status='complete',binding=binding,questions=3000,images=2409,
        rows=96000,case_pool=12,showcases=4,full_generation='Reused matched native Figure3 Full answers; no new generation',
        max_head_row_sum_error=max(r['max_head_row_sum_error'] for r in rows),
        max_source_sum_error=max(r['source_sum_error'] for r in rows),
        historical_bos_max_abs_difference=bos_error,statistics=c['statistics'],
        artifacts={p.name:sha256(p) for p in sorted(OUT.iterdir()) if p.suffix in ['.csv','.png','.pdf']}))
    print(json.dumps(window_rows,indent=2),flush=True)


if __name__=='__main__':
    main()

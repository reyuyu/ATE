"""Post-hoc case layout refinement only; all numbers come from completed CSV."""
import csv
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from benchmarks.dataset_registry import sha256
from paper_rq1.analyze import write_csv
from .protocol import OUT, LAYERS, TYPES
from .analyze import COLORS, save


def main():
    rows = list(csv.DictReader((OUT/'per_token_layer_ablation.csv').open()))
    for row in rows:
        row['generation_step'] = int(row['generation_step'])
        row['layer'] = int(row['layer'])
        row['loss'] = float(row['loss'])
    # Same disclosed case-selection rule as the original automated figure.
    case = max([r for r in rows if r['mode']=='All32' and r['word_type']=='Function'], key=lambda r:r['loss'])
    image_id = case['image_id']
    selected = json.loads((OUT/'selected_token_manifest.json').read_text())['images'][image_id]
    sample = next(r for r in json.loads((OUT/'sample_manifest.json').read_text())['samples'] if r['image_identity']==image_id)
    original = json.loads((OUT/'analysis_completion.json').read_text())
    for suffix in ['png','pdf']:
        name = 'figures/figure4_real_caption_case.'+suffix
        archive = OUT/'audit/pre_delivery'/name
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            if sha256(OUT/name) != original['files_sha256'][name]:
                raise ValueError('Original Figure4 changed before archival')
            shutil.copyfile(OUT/name, archive)
        if sha256(archive) != original['files_sha256'][name]:
            raise ValueError('Original Figure4 archive differs')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig = plt.figure(figsize=(16,13),layout='constrained')
    grid = fig.add_gridspec(3,2,height_ratios=[1.1,.7,1])
    image_ax = fig.add_subplot(grid[0,0])
    image_ax.imshow(Image.open(sample['image_path'])); image_ax.axis('off')
    text_ax = fig.add_subplot(grid[0,1]); text_ax.axis('off')
    text_ax.text(0,.94,selected['caption'],wrap=True,va='top',fontsize=12)
    detail = []
    summary = []
    for target in sorted(selected['selected'],key=lambda r:r['generation_step']):
        role = target['word_type']; t = target['generation_step']
        sub = [r for r in rows if r['image_id']==image_id and r['generation_step']==t]
        means = {m:float(np.mean([r['loss'] for r in sub if r['mode']==m])) for m in ['Whole','Image']}
        all32 = next(r['loss'] for r in sub if r['mode']=='All32')
        detail.append(f"{role}: '{target['word']}', t={t}; All32 loss={all32:.3f}")
        summary.append(dict(image_id=image_id,word=target['word'],word_type=role,generation_step=t,
            whole_mean9=means['Whole'],image_mean9=means['Image'],all32_loss=all32))
    text_ax.text(0,.72,'\n\n'.join(detail),va='top',fontsize=11)
    text_ax.text(0,.28,'Post-hoc: largest Function All32 loss.\nNot representative; no sample removed.\n\nMiddle strip: native token positions; only the\nthree frozen target positions were intervened.',va='top',fontsize=10)
    ax = fig.add_subplot(grid[1,:])
    tokens = selected['annotation']['tokens']
    ax.set_xticks([r['generation_step'] for r in tokens],
        [r['decoded_piece'] or f"id{r['token_id']}" for r in tokens],rotation=25,ha='right')
    for r in summary:
        for offset,mode,marker in [(-.06,'whole_mean9','o'),(.06,'image_mean9','s')]:
            ax.scatter(r['generation_step']+offset,r[mode],marker=marker,color=COLORS[r['word_type']],s=55)
        ax.annotate(r['word_type'],(r['generation_step'],max(r['whole_mean9'],r['image_mean9'])),
            xytext=(0,12),textcoords='offset points',ha='center',color=COLORS[r['word_type']],fontsize=9)
    ax.plot([],[],'ko',label='Whole mean9');ax.plot([],[],'ks',label='Image mean9')
    ax.axhline(0,color='black',ls=':');ax.grid(alpha=.15);ax.margins(y=.5)
    ax.set_xlabel('Actual native token position (including untested words; no values imputed)')
    ax.set_ylabel('Single-layer mean9 loss');ax.legend(loc='upper left',fontsize=9)
    ax.set_xlim(-.5,max(r['generation_step'] for r in tokens)+.5)
    axes = []
    for i,mode in enumerate(['Whole','Image']):
        ax = fig.add_subplot(grid[2,i],sharey=axes[0] if axes else None);axes.append(ax)
        for role in TYPES:
            sub = sorted([r for r in rows if r['image_id']==image_id and r['mode']==mode and r['word_type']==role],key=lambda r:LAYERS.index(r['layer']))
            if sub:
                ax.plot([LAYERS.index(r['layer']) for r in sub],[r['loss'] for r in sub],
                    'o-',color=COLORS[role],label=role+': '+sub[0]['word'])
        ax.set_xticks(range(9),LAYERS);ax.set_xlabel('Measured layer (categorical; no unmeasured values)')
        ax.set_ylabel('Full-token LogP loss (nats)');ax.set_title(mode)
        ax.axhline(0,color='black',ls=':');ax.legend(fontsize=9);ax.grid(alpha=.15)
    fig.suptitle('Figure4: '+image_id+' native caption and token-aligned post-hoc case')
    save(fig,'figure4_real_caption_case')
    write_csv(OUT/'case_study_token_positions.csv',summary)


if __name__=='__main__':main()

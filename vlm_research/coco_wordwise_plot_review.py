"""Delivery-only figure labels: expose conditional denominators and head means."""
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coco_wordwise_sources.common import OUT, MAIN, SELECTED, read, write, sha256, verify
from coco_wordwise_sources.figures import COLORS, NAMES, save


def main():
    verify()
    grouped = defaultdict(list)
    with (OUT/'per_token_contributions.csv').open() as stream:
        for r in csv.DictReader(stream):
            if r['source'] in MAIN:
                grouped[int(r['layer']),r['source'],r['image_sha256']].append(
                    [float(r['mass']),float(r['relative_residual_ratio'])])
    with (OUT/'mass_contribution_consistency.csv').open() as stream:
        rates = list(csv.DictReader(stream))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,3,figsize=(11,6.5))
    for j,layer in enumerate(SELECTED):
        ax=axes[0,j]
        for k,source in enumerate(MAIN):
            points=np.array([np.mean(v,axis=0) for (l,s,_),v in grouped.items() if l==layer and s==source])
            ax.scatter(points[:,0],points[:,1],s=9,alpha=.5,color=COLORS[k],label=NAMES[k])
        ax.set_title(f'L{layer}')
        ax.set_xlabel('Source mass (caption mean)')
        if j==0:ax.set_ylabel('Projected norm / residual norm')
        if j==2:ax.legend(fontsize=8,frameon=False)
        ax=axes[1,j]
        for k,source in enumerate(['IMAGE','GENERATED_HISTORY']):
            r=next(r for r in rates if int(r['layer'])==layer and r['source']==source and r['include_eos']=='False')
            mean,low,high=[100*float(r[n]) for n in ['mean','ci95_low','ci95_high']]
            ax.errorbar(k,mean,yerr=[[mean-low],[high-mean]],fmt='o',capsize=4,color=COLORS[1 if k==0 else 3])
        ax.set_xticks([0,1],['Image','History'])
        ax.set_xlim(-.4,1.4)
        ax.set_ylim(-3,103)
        ax.set_title(f"{r['n_images']} images / {r['eligible_pairs']} pairs",fontsize=9)
        ax.set_xlabel('Head-mean mass-matched subset',fontsize=8)
        if j==0:ax.set_ylabel('Contribution change within subset (%)')
        ax.grid(axis='y',alpha=.15)
    fig.tight_layout()
    save(fig,'figure4_mass_vs_contribution')
    write(OUT/'figure_review.json',dict(source_sha256={str(__file__):sha256(__file__)},
        data_unchanged=True,caption_points='Same per-image means, all native incremental positions',
        rates_unchanged=True,denominators_explicit=True,head_average_scope_explicit=True,
        original_frozen_collector_unchanged=True))


if __name__=='__main__':main()

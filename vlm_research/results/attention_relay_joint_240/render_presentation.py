"""Reposition labels and integer count ticks only; reuse unchanged frozen CSV values."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT=Path(__file__).resolve().parent


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def save(fig,name):
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'figures/{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    acc=read('accuracy_summary.csv')
    colors=['#666666','#1b9e77','#377eb8','#c44e52']
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for i,r in enumerate(acc):
        value=float(r['accuracy_pct'])
        lo=float(r['accuracy_ci95_low_pct']);hi=float(r['accuracy_ci95_high_pct'])
        axes[0].bar(i,value,color=colors[i],width=.65)
        axes[0].errorbar(i,value,yerr=[[value-lo],[hi-value]],fmt='none',color='black',capsize=4)
        axes[0].text(i,value-12,f"{r['correct']}/240",ha='center',color='white',fontweight='bold',fontsize=9)
    axes[0].set(xticks=range(4),xticklabels=[r['condition'] for r in acc],
        ylabel='Complete-answer accuracy (%)',ylim=(0,100),title='Four independently generated conditions')
    for i,r in enumerate(acc[1:]):
        value=float(r['delta_full_pp'])
        axes[1].errorbar(value,i,xerr=[[value-float(r['delta_ci95_low_pp'])],[float(r['delta_ci95_high_pp'])-value]],
            fmt='o',color=colors[i+1],capsize=4)
    axes[1].axvline(0,color='.55',lw=1)
    axes[1].axvline(-1,color='.6',lw=1,ls=':')
    axes[1].set(yticks=range(3),yticklabels=[r['condition'] for r in acc[1:]],
        xlabel='Accuracy difference vs Full (pp)',title='Paired image-cluster 95% intervals')
    save(fig,'figure1_accuracy')
    comp=read('paired_comparisons.csv')[:4];tasks=read('joint_error_tasks.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for i,r in enumerate(comp):
        value=float(r['estimate_pp'])
        axes[0].errorbar(value,i,xerr=[[value-float(r['ci95_low_pp'])],[float(r['ci95_high_pp'])-value]],
            fmt='o',color='#c44e52',capsize=4)
    axes[0].axvline(0,color='.5',lw=1)
    axes[0].set(yticks=range(4),yticklabels=[r['comparison'] for r in comp],
        xlabel='Accuracy contrast (pp)',title='Joint losses and 2x2 interaction')
    counts=[int(r['joint_unique_harm']) for r in tasks]
    axes[1].bar(range(6),counts,color='#c44e52')
    axes[1].set(xticks=range(6),xticklabels=[r['task'] for r in tasks],ylabel='Questions',
        ylim=(0,max(counts+[1])+1),title=f'Joint-only Full-correct losses: {sum(counts)}')
    axes[1].tick_params(axis='x',rotation=35)
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for i,v in enumerate(counts):axes[1].text(i,v+.08,str(v),ha='center')
    save(fig,'figure2_joint_interaction')


if __name__=='__main__':main()

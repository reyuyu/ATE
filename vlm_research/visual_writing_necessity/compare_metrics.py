"""Compare frozen Gold NLL and accuracy; no inference or statistics rerun."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/gqa3000_visual_writing_necessity'
OUT=RESULTS/'metric_comparison'


def main():
    source=RESULTS/'condition_summary.csv'
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    completed=json.loads((RESULTS/'analysis_complete.json').read_text())
    assert completed['passed'] and source_hash==completed['hashes']['condition_summary.csv']
    with source.open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==34 and all(r['subset']=='overall' and int(r['questions'])==3000 for r in rows)
    baseline=next(r for r in rows if r['condition']=='All-Writer')
    drops=sorted((r for r in rows if r['condition'].startswith('Drop-L')),key=lambda r:int(r['condition'][6:]))
    x=np.array([int(r['condition'][6:]) for r in drops]);assert np.array_equal(x,np.arange(32))
    OUT.mkdir(exist_ok=True)
    fields=['layer','condition','gold_nll','delta_gold_nll','delta_gold_nll_ci_low','delta_gold_nll_ci_high',
        'delta_gold_nll_corrected_low','delta_gold_nll_corrected_high','accuracy_pct','delta_accuracy_pp',
        'delta_accuracy_ci_low','delta_accuracy_ci_high','delta_accuracy_corrected_low','delta_accuracy_corrected_high',
        'correct_to_wrong','wrong_to_correct']
    with (OUT/'layer_metric_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        writer.writerows(dict(layer=int(r['condition'][6:]),**{k:r[k] for k in fields if k!='layer'}) for r in drops)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
    fig,axes=plt.subplots(2,1,sharex=True,figsize=(10,6.4),layout='constrained')
    specs=[('delta_gold_nll','delta_gold_nll','#087e8b',r'$\Delta$ Gold NLL (nat/token)',
            'Gold-answer log loss: positive = worse'),
           ('delta_accuracy_pp','delta_accuracy','#a23b72',r'$\Delta$ Accuracy (pp)',
            'Generated-answer accuracy: negative = worse')]
    for ax,(field,prefix,color,ylabel,title) in zip(axes,specs):
        y=np.array([float(r[field]) for r in drops])
        lo=np.array([float(r[prefix+'_ci_low']) for r in drops]);hi=np.array([float(r[prefix+'_ci_high']) for r in drops])
        clo=np.array([float(r[prefix+'_corrected_low']) for r in drops]);chi=np.array([float(r[prefix+'_corrected_high']) for r in drops])
        ax.axvspan(7.5,11.5,color='#e9c46a',alpha=.14,zorder=0,label='Historical L8-L11 window')
        ax.vlines(x,clo,chi,color=color,alpha=.22,lw=4,label='32-comparison corrected 95% CI')
        ax.vlines(x,lo,hi,color=color,lw=1.1,label='Pointwise 95% CI')
        ax.plot(x,y,'o-',color=color,ms=3.5,lw=1.35,label='Single-layer drop vs All-Writer')
        ax.axhline(0,color='#454545',lw=.8,ls='--');ax.grid(axis='y',alpha=.15)
        ax.set(ylabel=ylabel,title=title,xlim=(-.65,31.65))
    nll=np.array([float(r['delta_gold_nll']) for r in drops]);acc=np.array([float(r['delta_accuracy_pp']) for r in drops])
    axes[0].annotate(f'L0: {nll[0]:+.3f}',xy=(0,nll[0]),xytext=(2,nll[0]+.18),
        arrowprops=dict(arrowstyle='-',color='#555555'),fontsize=9)
    axes[0].annotate(f'L13: {nll[13]:+.4f}',xy=(13,nll[13]),xytext=(15,.18),
        arrowprops=dict(arrowstyle='-',color='#555555'),fontsize=9)
    axes[0].set_ylim(min(float(r['delta_gold_nll_corrected_low']) for r in drops)-.08,.28)
    axes[1].annotate(f'L11: {acc[11]:+.2f} pp',xy=(11,acc[11]),xytext=(14,acc[11]-.6),
        arrowprops=dict(arrowstyle='-',color='#555555'),fontsize=9)
    axes[0].legend(frameon=False,fontsize=8,loc='lower right')
    axes[1].set(xlabel='Dropped visual writing layer',xticks=range(32))
    axes[1].tick_params(axis='x',labelsize=8)
    fig.suptitle('Visual Writing: Log Loss vs Accuracy',fontsize=15)
    for ext in ['png','pdf']:fig.savefig(OUT/f'logloss_accuracy_aligned.{ext}')
    plt.close(fig)
    # An explicit two-axis view is supplementary; both zero lines are aligned.
    fig,left=plt.subplots(figsize=(10,4.2),layout='constrained');right=left.twinx()
    a=left.plot(x,nll,'o-',color='#087e8b',lw=1.3,ms=3,label='Delta Gold NLL (left)')
    b=right.plot(x,acc,'s-',color='#a23b72',lw=1.3,ms=3,label='Delta Accuracy (right)')
    left.set_ylim(-1.,1.);right.set_ylim(-4.6,4.6)
    left.axhline(0,color='#454545',ls='--',lw=.8)
    left.axvspan(7.5,11.5,color='#e9c46a',alpha=.12)
    left.set(xlabel='Dropped visual writing layer',ylabel='Delta Gold NLL (nat/token)',
        title='Different Units; Zero-Aligned Axes; Original Signs Retained',xticks=range(32),xlim=(-.65,31.65))
    right.set_ylabel('Delta Accuracy (pp)');right.spines['right'].set_visible(True)
    left.tick_params(axis='y',colors='#087e8b');right.tick_params(axis='y',colors='#a23b72')
    left.tick_params(axis='x',labelsize=8);left.legend(a+b,[line.get_label() for line in a+b],frameon=False,fontsize=9,loc='upper right')
    for ext in ['png','pdf']:fig.savefig(OUT/f'logloss_accuracy_overlay.{ext}')
    plt.close(fig)
    metadata=dict(source=str(source.relative_to(ROOT)),source_sha256=source_hash,questions=3000,image_clusters=2409,
        baseline='All-Writer, not native Full',baseline_accuracy_pct=float(baseline['accuracy_pct']),
        baseline_gold_nll=float(baseline['gold_nll']),fresh_inference=False,bootstrap_rerun=False,
        intervals='Original image-cluster paired bootstrap, pointwise and Bonferroni-32 95% intervals',
        primary='Two vertically aligned panels, original signs and separate units; all 32 layers retained',
        supplementary='Dual axes, symmetric limits and aligned zeros; numeric heights do not share units',
        warning='Gold NLL uses exact official answer tokens; it is not normalized generated-answer accuracy or first-Full-token LogP loss.')
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()

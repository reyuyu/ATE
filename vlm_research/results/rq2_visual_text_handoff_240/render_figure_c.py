"""Publication-only redraw from final CSVs; leave frozen runtime and statistics alone."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT=Path(__file__).resolve().parent
COLORS=['#414141','#147d64','#2876a4','#bc543a']
KEYS=list('ABCD')


def main():
    swaps=list(csv.DictReader((OUT/'image_swap_results.csv').open()))
    hidden=list(csv.DictReader((OUT/'hidden_state_handoff.csv').open()))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10.5,7.1))
    for i,key in enumerate(KEYS):
        sr=[r for r in swaps if r['condition']==key]
        axes[0,0].scatter(np.full(len(sr),i),[float(r['first_js']) for r in sr],s=13,alpha=.65,color=COLORS[i])
        axes[0,1].bar(i-.16,100*np.mean([r['tokens_changed']=='True' for r in sr]),width=.3,color=COLORS[i])
        axes[0,1].bar(i+.16,100*np.mean([int(r['both_correct']) for r in sr]),width=.3,color=COLORS[i],alpha=.4,hatch='//')
        for group,style in [('Question','-'),('Instruction','--'),('Template',':'),('P-1','-.')]:
            selected=[r for r in hidden if r['condition']==key and r['group']==group]
            layers=sorted({int(r['layer']) for r in selected})
            values=[np.mean([float(r['relative_l2']) for r in selected if int(r['layer'])==l]) for l in layers]
            axes[1,0].plot(layers,values,style,color=COLORS[i],lw=1.4)
    axes[0,0].set(xticks=range(4),xticklabels=KEYS,ylabel='First-token distribution JS (nat)',title='Same text, different real images')
    axes[0,1].set(xticks=range(4),xticklabels=KEYS,ylim=(0,105),ylabel='Image pairs (%)',title='Solid: output changes | hatched: both correct')
    axes[1,0].set(xlabel='Decoder layer output',ylabel='Image-swap relative L2',title='Fixed 6-pair read-only state subset')
    d=[float(r['first_logits_max_difference']) for r in swaps if r['condition']=='D']
    axes[1,1].plot(range(len(d)),d,'o',color=COLORS[3],ms=4)
    axes[1,1].axhline(0,color='#999999',lw=.6)
    axes[1,1].set(ylim=(-.01,.01),xlabel='Frozen image pair',ylabel='D maximum absolute logits difference',title='D: exact Prefill + all-Decode invariance')
    handles=[Line2D([0],[0],color=c,label=k,lw=2) for k,c in zip(KEYS,COLORS)]
    handles += [Line2D([0],[0],color='#333333',ls=style,label=group,lw=1.5) for group,style in [('Question','-'),('Instruction','--'),('Template',':'),('P-1','-.')]]
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=9,frameon=False,bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(left=.08,right=.985,bottom=.19,top=.95,wspace=.31,hspace=.43)
    for ext in ['png','pdf']:fig.savefig(OUT/f'figure_c_image_swap_causality.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()

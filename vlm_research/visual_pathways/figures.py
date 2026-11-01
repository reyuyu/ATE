"""Paper-sized unsmoothed curves and paired intervention summaries."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import OUT,TASKS,read

COLORS={'qwen':'#216b9b','llava_history':'#c35139','absolute':'#216b9b','signed':'#2e8064','flip':'#94599f'}


def csv_read(path):
    with Path(path).open() as handle:return list(csv.DictReader(handle))


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'xtick.labelsize':8,'ytick.labelsize':8,'savefig.facecolor':'white'})


def save(fig,name):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=280,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def curve(ax,rows,metric,color,label=None,percent=False):
    rows=sorted(rows,key=lambda r:int(r['layer']))
    x=np.array([int(r['layer']) for r in rows])
    scale=100 if percent else 1
    y=np.array([float(r[metric])*scale for r in rows])
    low=np.array([float(r[metric+'_ci_low'])*scale for r in rows])
    high=np.array([float(r[metric+'_ci_high'])*scale for r in rows])
    ax.plot(x,y,color=color,lw=1.5,marker='o',markersize=2.4,label=label)
    ax.fill_between(x,low,high,color=color,alpha=.15,lw=0)
    ax.axhline(0,color='#777777',lw=.6)
    ax.set_xlabel('Layer index');ax.grid(axis='y',alpha=.15)
    ax.set_xticks(list(range(0,int(x.max())+1,5)))
    return x,y


def scan_figures():
    rows=csv_read(OUT/'qwen/layer_summary.csv')
    coverage=read(OUT/'qwen/coverage.json')
    values=[r for r in rows if r['cohort']=='all' and r['task']=='Overall']
    fig,axes=plt.subplots(1,3,figsize=(11.5,3),layout='constrained')
    for ax,metric,label,color,percentage in zip(axes,
        ['mean_absolute_logp_change','mean_signed_loss','top1_flip_rate'],
        ['Mean absolute LogP change (nat)','Mean signed LogP loss (nat)','Top-1 flip rate (%)'],
        [COLORS['absolute'],COLORS['signed'],COLORS['flip']],[False,False,True]):
        x,y=curve(ax,values,metric,color,percent=percentage)
        ax.set_ylabel(label)
        peak=int(y.argmax());ax.annotate(f'L{x[peak]}',xy=(x[peak],y[peak]),xytext=(0,6),
            textcoords='offset points',ha='center',fontsize=8,color=color)
    fig.suptitle(f'Qwen3-VL: P-1 single-layer Image OFF | N={coverage["questions"]:,} ({coverage["status"]})',fontsize=11)
    save(fig,'qwen_overall_sensitivity')
    for metric,label,suffix,percentage in [
        ('mean_absolute_logp_change','Mean absolute LogP change (nat)','absolute',False),
        ('mean_signed_loss','Mean signed LogP loss (nat)','signed',False),
        ('top1_flip_rate','Top-1 flip rate (%)','flip',True)]:
        fig,axes=plt.subplots(2,3,figsize=(10.5,5.8),layout='constrained')
        for ax,task in zip(axes.flat,TASKS):
            selected=[r for r in rows if r['cohort']=='all' and r['task']==task]
            if not selected:
                ax.text(.5,.5,'Not yet collected',ha='center',transform=ax.transAxes);continue
            curve(ax,selected,metric,COLORS[suffix],percent=percentage)
            ax.set_title(f'{task} (N={selected[0]["questions"]})');ax.set_ylabel(label)
        fig.suptitle('Qwen3-VL: source-specific sensitivity by task (pointwise image-cluster 95% CI)',fontsize=11)
        save(fig,'qwen_six_tasks_'+suffix)


def cross_model():
    q=[r for r in csv_read(OUT/'qwen/layer_summary.csv') if r['cohort']=='pilot240' and r['task']=='Overall']
    l=[r for r in csv_read(OUT/'llava_historical_imageoff_layer_summary.csv') if r['task']=='Overall']
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    for rows,name,label in [(q,'qwen','Qwen3-VL-8B (BF16)'),(l,'llava_history','LLaVA-1.5-7B (FP16, historical)')]:
        x,y=curve(axes[0],rows,'mean_absolute_logp_change',COLORS[name],label)
        axes[1].plot(x/x.max(),y/y.sum(),color=COLORS[name],marker='o',markersize=2.4,lw=1.4,label=label)
    axes[0].set_ylabel('Mean absolute LogP change (nat)')
    axes[0].set_xticks(range(0,36,5));axes[0].set_xlim(-.5,35.5)
    axes[1].set_xlabel('Relative decoder depth');axes[1].set_ylabel('Share of summed layer-wise mean changes')
    axes[1].grid(axis='y',alpha=.15);axes[0].legend(frameon=False,fontsize=7)
    axes[0].set_title('(a) Measured sensitivity');axes[1].set_title('(b) Descriptive depth profile')
    fig.suptitle('Exact same 240 questions / 235 images; native prompts and token targets differ',fontsize=10)
    fig.text(.5,-.055,'InternVL: STOPPED at native-logit smoke (0/240); no curve is imputed.',ha='center',fontsize=8,color='#9c3434')
    save(fig,'cross_model_paired240')


def relay():
    folder=OUT/'llava_relay'
    rows=csv_read(folder/'accuracy_summary.csv');interaction=read(folder/'extension_decision.json')
    fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
    x=np.arange(4);y=np.array([float(r['accuracy_percent']) for r in rows])
    low=np.array([float(r['accuracy_ci_low']) for r in rows]);high=np.array([float(r['accuracy_ci_high']) for r in rows])
    colors=['#597781','#d2a34a','#597781','#d2a34a']
    axes[0].bar(x,y,color=colors,width=.6)
    axes[0].errorbar(x,y,yerr=[y-low,high-y],fmt='none',color='#222222',capsize=3,lw=.9)
    axes[0].set_xticks(x,['A\nRelay ON','B\nON + Skip','C\nRelay OFF','D\nOFF + Skip'])
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Final-answer accuracy (%)')
    for i,v in enumerate(y):axes[0].text(i,v+5.5,f'{v:.2f}',ha='center',fontsize=8)
    points=csv_read(folder/'interaction.csv')
    for ax,names,label in [(axes[1],['relay_on_skip_logp_harm','relay_off_skip_logp_harm'],'Skip harm: fixed-Full-token LogP loss (nat)'),
                           (axes[2],['relay_on_skip_accuracy_harm_pp','relay_off_skip_accuracy_harm_pp'],'Skip harm: accuracy decrease (pp)')]:
        selected=[next(r for r in points if r['metric']==n) for n in names]
        vals=np.array([float(r['mean']) for r in selected]);lo=np.array([float(r['ci_low']) for r in selected]);hi=np.array([float(r['ci_high']) for r in selected])
        ax.errorbar([0,1],vals,yerr=[vals-lo,hi-vals],fmt='o',color='#216b9b',capsize=4)
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks([0,1],['Relay ON','Relay OFF']);ax.set_ylabel(label)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('LLaVA 240: L2-L5 P-1 Attention skip does not become more harmful with relay OFF',fontsize=10)
    save(fig,'llava_relay_interaction')


def rescue():
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.4),layout='constrained')
    for ax,model,label,keep in zip(axes,['llava','qwen'],['LLaVA-1.5-7B','Qwen3-VL-8B'],['L14/L19','L26']):
        folder=OUT/(model+'_rescue')
        if not (folder/'accuracy_summary.csv').exists():
            ax.text(.5,.5,'Pending / not completed',transform=ax.transAxes,ha='center');continue
        rows=csv_read(folder/'accuracy_summary.csv');baseline=read(folder/'native_full_reference.json')
        y=np.array([float(r['accuracy_percent']) for r in rows]);low=np.array([float(r['accuracy_ci_low']) for r in rows]);high=np.array([float(r['accuracy_ci_high']) for r in rows])
        ax.bar(np.arange(3),y,width=.6,color=['#597781','#bababa','#d2a34a'])
        ax.errorbar(np.arange(3),y,yerr=[y-low,high-y],fmt='none',color='#333333',capsize=4,lw=1)
        ax.axhline(baseline['accuracy_percent'],color='#b43d40',ls='--',lw=1,label='Native Full (relay ON)')
        ax.set_xticks(np.arange(3),['All layers\nImage access','All layers\nImage OFF',f'Keep only\n{keep}'])
        ax.set_ylim(0,100);ax.set_ylabel('Final-answer accuracy (%)');ax.set_title(label+' | Relay OFF')
        for i,value in enumerate(y):ax.text(i,value+6,f'{value:.2f}',ha='center',fontsize=8)
        ax.legend(frameon=False,fontsize=7,loc='lower left')
    fig.suptitle('Fixed sparse visual access under damaged Prefill relay (240 paired questions)',fontsize=10)
    save(fig,'sparse_visual_rescue')


if __name__=='__main__':
    style()
    if (OUT/'qwen/layer_summary.csv').exists():scan_figures();cross_model()
    relay();rescue()

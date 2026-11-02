"""Publication figure from frozen analysis tables; no fitted or smoothed curves."""
import csv,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages
from .common import *

COLORS={'Semantic-first':'#007C78','Question-Last':'#B95B25','Question-Random mean2':'#5459A8','R-Random mean2':'#8B6578'}
LABELS={'Semantic-first':'Semantic-first','Question-Last':'Question-last','Question-Random mean2':'Question random (2 seeds)','R-Random mean2':'R random (2 seeds)'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})

def table(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def num(r,k):return float(r[k])
def select(rows,pop,c):return next(r for r in rows if r['population']==pop and r['condition']==c)
def mark(ax,letter,title):ax.set_title(letter+'  '+title,loc='left',fontweight='bold',pad=9)
SUPPLEMENT=None
def save(fig,name,main=False,overview=False):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for ext in (['png','pdf'] if main else ['png']):
        path=folder/(name+'.'+ext);assert not path.exists();fig.savefig(path,bbox_inches='tight',facecolor='white')
    if not main and not overview:
        assert SUPPLEMENT is not None;SUPPLEMENT.savefig(fig,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def curve_axes(axn,axa,rows,pop,legend=True):
    for policy,color in COLORS.items():
        rr=[select(rows,pop,f'{policy}-Q{b}') for b in BUDGETS];x=np.array([100*num(r,'actual_R_budget_mean') for r in rr])
        for ax,metric,low,high in [(axn,'delta_gold_nll','delta_nll_ci_low','delta_nll_ci_high'),(axa,'accuracy_percent','accuracy_ci_low','accuracy_ci_high')]:
            y=np.array([num(r,metric) for r in rr]);lo=np.array([num(r,low) for r in rr]);hi=np.array([num(r,high) for r in rr])
            ax.errorbar(x,y,yerr=np.array([y-lo,hi-y]),marker='o',ms=3,lw=1,capsize=2,color=color,label=LABELS[policy],zorder=3)
    for c,marker,col in [('No-Writer','X','#333333'),('All-Writer','*','#111111')]:
        rr=select(rows,pop,c);x=100*num(rr,'actual_R_budget_mean')
        for ax,metric,low,high in [(axn,'delta_gold_nll','delta_nll_ci_low','delta_nll_ci_high'),(axa,'accuracy_percent','accuracy_ci_low','accuracy_ci_high')]:
            y=num(rr,metric);ax.errorbar([x],[y],yerr=[[y-num(rr,low)],[num(rr,high)-y]],fmt=marker,color=col,ms=6,capsize=2,label=c)
    axn.axhline(0,color='#888888',lw=.6,ls='--');axn.set_ylabel('$\Delta$ Gold NLL\n(nat/token)');axa.set_ylabel('VQA accuracy (%)')
    axa.set_xlabel('Actual open R queries (%)');axn.tick_params(labelbottom=False)
    for ax in [axn,axa]:ax.set_xlim(-3,104);ax.grid(axis='y',alpha=.18)
    if legend:axa.legend(loc='lower right',fontsize=6,frameon=False,ncol=1)

def token_panel(ax):
    ax.axis('off');mark(ax,'A','Which tokens receive vision?')
    qid=read(OUT/'input_design_audit.json')['figure_A_question_id'];samples=read(OUT/'sample_manifest.json')['samples'];s=next(s for s in samples if s['question_id']==qid);spec=read(OUT/read(OUT/'selection_index.json')[key(s)]['path'])
    groups=[g for g in spec['groups'] if g['question']];k=spec['natural']['k'];R=len(spec['relay_positions'])
    ax.text(0,.97,s['question'],transform=ax.transAxes,va='top',fontsize=8.5)
    ax.text(1,.97,f'Fixed example: GQA {qid}',transform=ax.transAxes,ha='right',va='top',fontsize=6.5,color='#555555')
    policies=[('Semantic-k','Semantic','#007C78'),('Question-Last-k','Question-last','#B95B25'),(f'Question-Random-k-{SEEDS[0]}','Question random','#5459A8')]
    for row,(c,label,color) in enumerate(policies):
        y=.64-row*.25;ax.text(.18,y,label,transform=ax.transAxes,ha='right',va='center',weight='bold',color=color,fontsize=7.5)
        selected=set(spec['writers'][c]);x=.20;yy=y
        for g in groups:
            word=g['decoded'].strip() or '\u00b7';width=min(.65,.009*len(word)+.018)
            if x+width>.99:x=.20;yy-=.20
            active=bool(set(g['positions'])&selected)
            ax.add_patch(FancyBboxPatch((x,yy-.09),width,.18,boxstyle='round,pad=0.003',linewidth=.5,
                edgecolor=color if active else '#dddddd',facecolor=color if active else '#f1f1f1',transform=ax.transAxes))
            ax.text(x+width/2,yy-.002,word,color='white' if active else '#444444',ha='center',va='center',fontsize=7,transform=ax.transAxes)
            x+=width+.006
    ax.text(0,-.12,f'Each row: {k} tokenizer tokens / {R} R queries. Whole words stay intact. R also includes instruction/template tokens.',fontsize=6.5,transform=ax.transAxes,color='#555555')

def natural_panel(ax,summary,paired,nllax=None):
    mark(ax,'B','Receiver selection matters')
    pop='all3000_descriptive';names=['All-Writer','Semantic-k','Question-Random-k mean3','Question-Stratified-k mean2','Question-Last-k','R-Last-k','R-Random-k mean3']
    labels=['All R','Semantic-k','Q-random (3 seeds)','Q-position random (2)','Q-last-k','R-last-k','R-random (3 seeds)']
    for i,(c,label) in enumerate(zip(names,labels)):
        r=select(summary,pop,c);y=len(names)-1-i;v=num(r,'accuracy_percent');color='#007C78' if c=='Semantic-k' else '#555555'
        ax.errorbar(v,y,xerr=[[v-num(r,'accuracy_ci_low')],[num(r,'accuracy_ci_high')-v]],fmt='o',color=color,ms=4,capsize=2)
        # All rows share the natural k except All-writer, shown explicitly.
        if nllax is not None:
            v=num(r,'delta_gold_nll');nllax.errorbar(v,y,xerr=[[v-num(r,'delta_nll_ci_low')],[num(r,'delta_nll_ci_high')-v]],fmt='o',color=color,ms=4,capsize=2)
        else:ax.text(1.02,y,f"{num(r,'delta_gold_nll'):+.3f}\n[{num(r,'delta_nll_ci_low'):+.3f}, {num(r,'delta_nll_ci_high'):+.3f}]",transform=ax.get_yaxis_transform(),va='center',fontsize=6.5,color=color)
    ax.set_yticks(range(len(names)),labels[::-1]);ax.set_xlabel('VQA accuracy (%) · pointwise 95% CI');ax.grid(axis='x',alpha=.18);ax.set_ylim(-.7,len(names)-.15)
    if nllax is not None:
        nllax.set_ylim(ax.get_ylim());nllax.set_yticks([]);nllax.axvline(0,color='#aaaaaa',ls='--',lw=.6);nllax.grid(axis='x',alpha=.18)
        nllax.set_xlabel('$\Delta$ Gold NLL vs All (nat/token)\nPaired pointwise 95% CI')
        primary=next((r for r in paired if r['population']==pop and r['left']=='Semantic-k' and r['right']=='Question-Random-k mean3'),None)
        if primary:
            text=f"Semantic - Q-random: $\Delta$Acc {num(primary,'accuracy_delta_pp'):+.2f}pp [{num(primary,'accuracy_ci_low'):+.2f}, {num(primary,'accuracy_ci_high'):+.2f}]\n"
            text+=f"$\Delta$NLL {num(primary,'delta_gold_nll'):+.3f} [{num(primary,'nll_ci_low'):+.3f}, {num(primary,'nll_ci_high'):+.3f}] (paired 95% CI)"
            nllax.text(0,1.03,text,transform=nllax.transAxes,fontsize=6.5,va='bottom')
    else:ax.text(1.02,1.02,'$\Delta$ Gold NLL vs All\n(nat/token; paired 95% CI)',transform=ax.transAxes,fontsize=7,va='bottom')

def swap_panel(ax,swap):
    mark(ax,'D','Does sparse writing preserve grounding?')
    names=['All-Writer','No-Writer','Semantic-k',f'Question-Random-k-{SEEDS[0]}','Question-Last-k'];labels=['All','No','Semantic','Q-random','Q-last'];x=np.arange(5);width=.35
    for offset,m,col,label in [(-width/2,'both_correct','#007C78','Both endpoints correct'),(width/2,'answer_changed','#D5A34A','Answer changes with image')]:
        rr=[next(r for r in swap if r['condition']==c) for c in names];v=np.array([100*num(r,m) for r in rr]);lo=np.array([100*num(r,m+'_ci_low') for r in rr]);hi=np.array([100*num(r,m+'_ci_high') for r in rr])
        ax.bar(x+offset,v,width,color=col,label=label,yerr=np.array([v-lo,hi-v]),capsize=2,error_kw={'linewidth':.6})
    count=int(swap[0]['pairs']);ax.set_xticks(x,labels);ax.set_ylabel('Image pairs (%)');ax.set_ylim(0,125);ax.set_yticks([0,50,100]);ax.legend(loc='upper center',fontsize=7,frameon=False,ncol=2)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.text(0,-.33,f'{count} frozen different-Gold image pairs; linked-pair 95% CI.\nNo-writer: all effective logits strictly identical across images.',transform=ax.transAxes,fontsize=7)

def main():
    global SUPPLEMENT
    assert read(OUT/'statistics_validation.json')['passed'] and read(OUT/'image_swap_validation.json')['passed']
    summary=table('condition_summary.csv');paired=table('paired_comparisons.csv');swap=table('image_swap_summary.csv')
    fig=plt.figure(figsize=(7.2,10));outer=fig.add_gridspec(4,1,height_ratios=[1.15,1.9,2.6,1.65],hspace=.55)
    a=fig.add_subplot(outer[0]);token_panel(a)
    bg=outer[1].subgridspec(1,2,wspace=.23);b=fig.add_subplot(bg[0]);bn=fig.add_subplot(bg[1]);natural_panel(b,summary,paired,bn)
    cg=outer[2].subgridspec(2,1,hspace=.11);cn=fig.add_subplot(cg[0]);ca=fig.add_subplot(cg[1],sharex=cn)
    mark(cn,'C','Token budget-quality frontier');curve_axes(cn,ca,summary,'all3000_descriptive')
    d=fig.add_subplot(outer[3]);swap_panel(d,swap)
    fig.suptitle('Receiver-selective visual writing | LLaVA-1.5-7B / GQA3000',fontsize=10,weight='bold',y=.987)
    fig.text(.12,.006,'All answer queries block direct image access at all 32 layers. Curve budgets are matched by whole-word token counts.\nCurve intervals are pointwise. Confirmatory tests use a separate prespecified Bonferroni family.',fontsize=6.5)
    fig.subplots_adjust(bottom=.09,top=.945,left=.17,right=.98);save(fig,'receiver_selective_visual_writing_main',main=True)
    folder=OUT/'figures';supplement_path=folder/'receiver_selective_visual_writing_supplement.pdf';assert not supplement_path.exists();SUPPLEMENT=PdfPages(supplement_path)
    fig,axs=plt.subplots(2,1,figsize=(7.2,6.6),sharex=True,gridspec_kw={'hspace':.08})
    curve_axes(*axs,summary,'image_disjoint_confirmatory');axs[0].set_title('Image-disjoint confirmation subset (n = 2640)',loc='left',weight='bold');fig.tight_layout();save(fig,'confirmation_budget_curve')
    fig=plt.figure(figsize=(11.2,10));grid=fig.add_gridspec(3,2,hspace=.5,wspace=.3)
    for idx,t in enumerate(TASKS):
        sg=grid[idx//2,idx%2].subgridspec(2,1,hspace=.08);an=fig.add_subplot(sg[0]);ac=fig.add_subplot(sg[1],sharex=an)
        curve_axes(an,ac,summary,'task_'+t,legend=idx==0);an.set_title(t+' (n = 500)',loc='left',weight='bold')
    fig.subplots_adjust(bottom=.07,top=.94,left=.07,right=.98);fig.suptitle('Six frozen GQA tasks | descriptive token budget-quality curves',fontsize=12);save(fig,'six_task_budget_curves',overview=True)
    for idx,t in enumerate(TASKS):
        fig,axs=plt.subplots(2,1,figsize=(7.2,6.6),sharex=True,gridspec_kw={'hspace':.08})
        curve_axes(*axs,summary,'task_'+t);axs[0].set_title(t+' (n = 500)',loc='left',weight='bold');fig.tight_layout();save(fig,f'task_{idx+1}_{t}_budget_curve')
    # Separate large natural-budget plot for readable manuscript/supplement reuse.
    fig,axs=plt.subplots(1,2,figsize=(7.2,4.7),gridspec_kw={'wspace':.23});natural_panel(axs[0],summary,paired,axs[1]);fig.subplots_adjust(left=.2,right=.98,bottom=.3,top=.85);save(fig,'natural_k_receiver_selection')
    SUPPLEMENT.close();SUPPLEMENT=None
    atomic_json(OUT/'figure_validation.json',dict(passed=True,main_panels=['A fixed predeclared real question','B natural matched k','C both NLL and Accuracy versus actual R fraction','D frozen image swaps'],
        formats=['300dpi PNG','7.2-inch journal-width vector main PDF','Eight-page vector supplement: confirmation, six separate task figures, natural matched-k'],smoothing=False,all_negative_points_included=True,plotting_inputs=['condition_summary.csv','paired_comparisons.csv','budget_curve.csv','image_swap_per_pair.csv','image_swap_summary.csv','token_manifests'],
        ordinary_pointwise_CI_distinct_from_primary_family=True))

if __name__=='__main__':main()

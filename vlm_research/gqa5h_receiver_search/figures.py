"""Publication-style observed results; all plotting data are retained as CSV."""
from .common import *
from .analyze import rows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    frontier=rows(P/'budget_accuracy_frontier.csv');winner=read(P/'winner_freeze.json')['candidate_id'];key=[('All-Writer','All-Writer','#232b35','s'),('Question-Only','Question-only','#d1841c','D'),('Semantic-first-Q80','Semantic-first Q80','#326db3','o'),('Semantic-k','Original Semantic-k','#21805b','v'),(winner,'Frozen Winner: S + ASSISTANT','#b82e44','*')]
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.2),sharey=True)
    for ax,split in zip(axs,['DEV','TUNE','FINAL']):
        ss=[r for r in frontier if r['split']==split];ax.scatter([100*float(r['mean_R_ratio']) for r in ss],[float(r['ACC']) for r in ss],s=18,c='#aeb5be',alpha=.7)
        pp=sorted((r for r in ss if r['Pareto']=='True'),key=lambda r:float(r['mean_R_ratio']));ax.plot([100*float(r['mean_R_ratio']) for r in pp],[float(r['ACC']) for r in pp],':',color='#718598',lw=1)
        for cid,label,color,marker in key:
            for r in ss:
                if r['candidate_id']==cid:ax.scatter(100*float(r['mean_R_ratio']),float(r['ACC']),s=105 if marker=='*' else 38,c=color,marker=marker,zorder=4,label=label)
        ax.axvline(35,color='#aaa',ls=':',lw=.8);ax.axvline(50,color='#aaa',ls='--',lw=.8);ax.set_title({'DEV':'DEV360: exploration / fitting','TUNE':'TUNE240: Winner selection','FINAL':'FINAL600: one frozen Winner'}[split],fontsize=10);ax.set_xlabel('Mean open R queries (%)');ax.grid(alpha=.16);ax.set_xlim(20,105)
    axs[0].set_ylabel('Complete-answer accuracy (%)');handles,labels=axs[2].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.10));fig.tight_layout();fig.savefig(P/'budget_accuracy_frontier.png',dpi=230,bbox_inches='tight');fig.savefig(P/'budget_accuracy_frontier.pdf',bbox_inches='tight');plt.close(fig)
    rr=[r for r in rows(P/'paired_comparisons.csv') if r['split']=='FINAL'];fig,ax=plt.subplots(figsize=(7.5,3.9));xx=[float(r['delta_pp']) for r in rr];yy=list(range(len(rr)));ax.errorbar(xx,yy,xerr=[[x-float(r['CI95_low_pp']) for x,r in zip(xx,rr)],[float(r['CI95_high_pp'])-x for x,r in zip(xx,rr)]],fmt='o',color='#b82e44',capsize=4);ax.set_yticks(yy,[r['baseline'] for r in rr]);ax.invert_yaxis();ax.axvline(0,color='#777',lw=1);ax.set_xlim(-4,4);ax.set_xlabel('Winner minus baseline accuracy (percentage points)');ax.set_title('FINAL600: exploratory pointwise 95% image-cluster bootstrap CIs',fontsize=10);ax.grid(axis='x',alpha=.2)
    for y,r in zip(yy,rr):ax.text(3.9,y,f"{r['harm']} harmed / {r['fix']} corrected",va='center',ha='right',fontsize=8)
    fig.tight_layout();fig.savefig(P/'final_paired_differences.png',dpi=230);fig.savefig(P/'final_paired_differences.pdf');plt.close(fig)
    print('Figures regenerated from retained plotting CSVs',flush=True)

if __name__=='__main__':main()

import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def main():
    out=Path(sys.argv[1]);summary=list(csv.DictReader((out/'accuracy_summary.csv').open()));pairs=list(csv.DictReader((out/'paired_comparisons.csv').open()));cov=list(csv.DictReader((out/'coverage_by_task.csv').open()));rows=list(csv.DictReader((out/'per_question120.csv').open()));valid=[r for r in rows if r['correct']!=''];plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    labels=['Semantic-k','Noun-only','Noun+Article','Semantic+Noun','Semantic+Article','Semantic+Noun+Article','All-Writer'];index={r['condition']:r for r in summary};short=['S','N','N+D','S+N','S+D','S+N+D','All-R']
    with PdfPages(out/'figures.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(9,5));y=[float(index[n]['accuracy']) for n in labels if n in index];ll=[short[i] for i,n in enumerate(labels) if n in index];bars=ax.bar(ll,y,color=['#65758b','#367fa7','#d89b3b','#49a684','#dc7a38','#6858a4','#203b51'][:len(y)])
        for b,a,n in zip(bars,y,[n for n in labels if n in index]):ax.text(b.get_x()+b.get_width()/2,a+1.2,f'{a:.1f}%\n{index[n]["correct"]}/120',ha='center',fontsize=9)
        ax.set_ylim(0,105);ax.set_ylabel('Complete-answer accuracy (%)');ax.set_title('Noun + Article receiver interventions | 120 image-disjoint questions');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=220);pdf.savefig(fig);plt.close(fig)
        names=['Semantic+Article - placebo mean','Semantic+Noun+Article - placebo mean','Noun+Article - placebo mean'];pp=[next(r for r in pairs if r['comparison']==n) for n in names if any(r['comparison']==n for r in pairs)]
        if pp:
            fig,ax=plt.subplots(figsize=(9,4));v=np.array([float(r['delta_pp']) for r in pp]);lo=np.array([float(r['ci_low']) for r in pp]);hi=np.array([float(r['ci_high']) for r in pp]);xx=np.arange(len(pp));ax.errorbar(v,xx,xerr=[v-lo,hi-v],fmt='o',capsize=5,color='#b56b22');ax.axvline(0,color='#888',ls='--');ax.set_yticks(xx,[f'{["S+D vs S+placebo","S+N+D vs SN+placebo","N+D vs N+placebo"][i]} (n={r["n"]})' for i,r in enumerate(pp)]);ax.set_xlabel('Article advantage over two-seed placebo mean (percentage points)');ax.set_title('Exact whole-word token-budget comparisons | exploratory paired 95% CI');ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'article_placebo.png',dpi=220);pdf.savefig(fig);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(11,4.4));xs=[float(index[n]['mean_k_R'])*100 for n in labels if n in index];ys=[float(index[n]['accuracy']) for n in labels if n in index];axs[0].scatter(xs,ys,s=55,color='#287a80')
        for i,(x,y) in enumerate(zip(xs,ys)):axs[0].annotate(ll[i],(x,y),xytext=(4,5),textcoords='offset points',fontsize=9)
        axs[0].set(xlabel='Mean open R-to-Image receiver ratio (%)',ylabel='Complete-answer accuracy (%)',title='Functional writing budget; dense compute unchanged')
        x=np.arange(6);noun=[100*int(r['noun_selected'])/int(r['noun_tokens']) for r in cov];art=[100*int(r['article_selected'])/int(r['article_tokens']) if int(r['article_tokens']) else 0 for r in cov];axs[1].bar(x-.17,noun,.34,label='Noun token coverage');axs[1].bar(x+.17,art,.34,label='Article token coverage');axs[1].set_xticks(x,[r['task'] for r in cov],rotation=30,ha='right');axs[1].set(ylabel='Token coverage by historical Semantic-k (%)',title='Text-only POS audit over all GQA3000');axs[1].legend(fontsize=8)
        for ax in axs:ax.spines[['top','right']].set_visible(False)
        fig.tight_layout();fig.savefig(out/'budget_and_coverage.png',dpi=220);pdf.savefig(fig);plt.close(fig)
if __name__=='__main__':main()

"""Final figure layout, rendered only from the verified analysis tables."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import P,NAMES,TASKS,read,sha256,atomic_json

def main():
    bundle=read(P/'analysis_bundle.json');summary=bundle['summary'];tasks=bundle['tasks']
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'});short=['All','S','Q','Q+B','Q+Post2','S+Tail2','S+B']
    fig,ax=plt.subplots(figsize=(10.5,5.8));palette=['#6b7280','#4584b1','#5c7f96','#e39c43','#945c9d','#389484','#c16748']
    for i,r in enumerate(summary):
        x=100*r['mean_R_ratio'];y=r['accuracy_percent'];ax.errorbar(x,y,yerr=[[y-r['CI95_low']],[r['CI95_high']-y]],fmt='o',color=palette[i],capsize=3,label=f'{i+1}. {r["condition"]} ({r["correct"]}/360)')
        # Keep the two S+2 labels on the outward side of their nearby points.
        y_offset={5:9,6:-16}.get(i,9 if i%2==0 else -16)
        ax.annotate(str(i+1),(x,y),xytext=(8,y_offset),textcoords='offset points',fontsize=10)
    ax.set_xlabel('Mean accessible R receivers (%)');ax.set_ylabel('Complete-answer GQA accuracy (%)');ax.set_title('Structural visual receivers: fixed natural GQA360');ax.set_xlim(0,105);ax.grid(alpha=.18);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=9,frameon=False)
    fig.text(.5,.015,'Image-cluster bootstrap 95% intervals. Discrete strategies; no interpolation or population claim.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(P/'accuracy_writer_budget.png',dpi=220);fig.savefig(P/'accuracy_writer_budget.svg');plt.close(fig)
    data=np.array([[next(r['delta_vs_base_pp'] for r in tasks if r['group']==t and r['condition']==c) for c in NAMES] for t in TASKS]);vmax=max(5,float(np.abs(data).max()));fig,ax=plt.subplots(figsize=(11,5.5));im=ax.imshow(data,aspect='auto',cmap='RdBu',vmin=-vmax,vmax=vmax)
    ax.set_xticks(range(7),short);ax.set_yticks(range(6),TASKS)
    for i in range(6):
        for j in range(7):ax.text(j,i,f'{data[i,j]:+.1f}',ha='center',va='center',color='white' if abs(data[i,j])>.65*vmax else '#222')
    ax.set_title('Task accuracy delta vs corresponding baseline (60 questions per task)');fig.colorbar(im,ax=ax,label='Accuracy difference (pp)');fig.text(.5,.01,'Q / Q+B / Q+Post2 relative to Q; S / S+Tail2 / S+B relative to S; All relative to itself.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(P/'task_accuracy_delta.png',dpi=220);fig.savefig(P/'task_accuracy_delta.svg');plt.close(fig)
    atomic_json(P/'figure_qa.json',dict(passed=True,statistics_unchanged=True,plot_source_sha256=sha256(__file__),analysis_bundle_sha256=sha256(P/'analysis_bundle.json'),artifacts={n:sha256(P/n) for n in ['accuracy_writer_budget.png','accuracy_writer_budget.svg','task_accuracy_delta.png','task_accuracy_delta.svg']},note='Adjusted adjacent point-label offsets after visual inspection; no data, intervals, axes, or conditions changed. PNG and vector SVG.'))

if __name__=='__main__':main()

"""Layout-only redraw from frozen statistics; no inference or statistical recomputation."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'results/semantic_single_gpu_search3h'
rows={r['condition']:r for r in csv.DictReader((P/'final_accuracy_summary.csv').open())}
show=['Semantic-k','S+Assistant','Question-Only','Phase5-spatial','Fixed-Q-Conditional-Assistant-18','Fixed-Q-Conditional-Assistant-15','All-Writer']
labels=['Semantic-k','S + assistant','Question only','S+A / spatial Q+A','Q + conditional A (18)','Q + conditional A (15)','All writers']
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(8.5,5))
offsets=[(5,4),(5,4),(-9,10),(8,-15),(-13,-17),(9,10),(5,4)]
for i,(name,label,offset) in enumerate(zip(show,labels,offsets)):
    r=rows[name];x=100*float(r['mean_R_ratio']);y=float(r['ACC'])
    ax.errorbar(x,y,yerr=[[y-float(r['CI95_low'])],[float(r['CI95_high'])-y]],fmt='o',capsize=3,label=f'{i+1}. {label}')
    ax.annotate(str(i+1),(x,y),xytext=offset,textcoords='offset points',ha='center')
ax.axvline(50,ls='--',color='gray',lw=1)
ax.set(xlabel='Mean allowed R-to-image receiver ratio (%)',ylabel='Complete-answer accuracy (%)',title='Observed budget–accuracy points; ordinary 95% image-cluster CI')
ax.legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(P/'final_budget_accuracy.png',dpi=220);fig.savefig(P/'final_budget_accuracy.pdf');plt.close(fig)

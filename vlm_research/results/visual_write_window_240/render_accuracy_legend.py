from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path('/data/lk/ATE-worktrees/visual-write-window-240/vlm_research/results/visual_write_window_240')
with (p/'accuracy_summary.csv').open(newline='') as f:rows=list(csv.DictReader(f))
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(13,5));x=np.arange(18);ys=[float(v['accuracy_pct']) for v in rows]
upper=[float(v['ci95_accuracy_high_pct']) for v in rows];lower=[float(v['ci95_accuracy_low_pct']) for v in rows]
ax.bar(x,ys,color=['#486b89','#999999']+['#ce7258']*8+['#75a5ad']*8)
ax.errorbar(x,ys,yerr=[np.array(ys)-lower,np.array(upper)-ys],fmt='none',color='black',capsize=2)
for k,v in enumerate(rows):ax.text(k,upper[k]+1.2,v['correct'],ha='center',va='bottom',fontsize=8)
ax.axhline(ys[0],ls='--',color='#486b89',label='All-Writer');ax.axhline(ys[1],ls=':',color='gray',label='No-Writer')
ax.set(xticks=x,xticklabels=[v['condition'] for v in rows],ylabel='Accuracy (%)',ylim=(0,103));ax.tick_params(axis='x',rotation=50)
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.09),ncol=2)
ax.set_title('240 questions per condition; exploratory 95% image-cluster bootstrap intervals',fontsize=10,pad=10)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(p/f'figure_accuracy.{ext}',dpi=180,bbox_inches='tight',pad_inches=.2)
plt.close(fig)
audit=json.loads((p/'presentation_audit.json').read_text());audit['accuracy_legend_placed_outside_data']=True
(p/'presentation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
c=json.loads((p/'completion.json').read_text());c['artifacts']={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file() and f.name not in ['completion.json','worker.lock']}
(p/'completion.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
print('Accuracy legend moved outside data; final hashes refreshed')

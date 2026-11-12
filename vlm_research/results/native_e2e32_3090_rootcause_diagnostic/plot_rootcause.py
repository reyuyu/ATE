"""CPU-only figures from saved evidence. Does not import torch or run requests."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path(__file__).parent
if (D/'ate3090-rootcause-diagnostic').is_dir():D=D/'ate3090-rootcause-diagnostic'
rows=list(csv.DictReader((D/'request_timeline.csv').open(encoding='utf-8')))
steps=list(csv.DictReader((D/'idle60_all_step_events.csv').open(encoding='utf-8')))
extra=json.loads((D/'additional_rootcause_evidence.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
fig,axs=plt.subplots(2,1,figsize=(11,7.6),gridspec_kw={'height_ratios':[1.4,1]})
colors={'fixed_warmup':'#a6adb4','continuous_measured':'#2566aa','continuous_before':'#2566aa','continuous_middle':'#2566aa','continuous_after':'#2566aa','idle15_forward':'#d29018','idle15_reverse':'#d29018','idle60_forward':'#bf3a3a','idle60_reverse':'#bf3a3a'}
labels={'fixed_warmup':'A: 12 warmups','continuous_measured':'A: 28 continuous','continuous_before':'B: continuous','idle15_forward':'B: 15 s idle','idle60_forward':'B: 60 s idle'}
seen=set()
for r in rows:
    if r['phase']=='C':continue
    c=r['condition'];label=labels.get(c)
    if c.startswith('continuous_') and c!='continuous_measured':label='B: continuous'
    if c.startswith('idle15'):label='B: 15 s idle'
    if c.startswith('idle60'):label='B: 60 s idle'
    axs[0].scatter(int(r['request']),float(r['E2E_ms']),s=26,color=colors[c],label=label if label not in seen else None)
    seen.add(label)
axs[0].axvline(40.5,color='#888',ls=':',lw=1)
axs[0].annotate('Request 49: 2558.9 ms\nOnly 1/8 idle60 requests in historical slow band',xy=(49,2558.889),xytext=(27,2660),fontsize=9,arrowprops={'arrowstyle':'->','color':'#555'})
axs[0].set(title='Unprofiled A/B requests (all 68 shown, including 12 warmups)',xlabel='Diagnostic request number',ylabel='E2E@32 (ms)',ylim=(2180,2810))
axs[0].legend(loc='upper right',fontsize=8,frameon=False,ncol=2)
axs[0].grid(axis='y',alpha=.2)
for prof,label,color,marker in [('False','C: unprofiled scout','#7c4c9c','o'),('True','C: profiler capture','#e37642','s')]:
    sub=[r for r in rows if r['phase']=='C' and r['profiled']==prof]
    axs[1].plot([int(r['request']) for r in sub],[float(r['TPOT_ms']) for r in sub],marker=marker,color=color,label=label)
axs[1].set(title='Profiler phase kept separate: later scouts may retain instrumentation effects',xlabel='Diagnostic request number',ylabel='TPOT (ms/token)',ylim=(60,105))
axs[1].legend(loc='lower right',frameon=False,fontsize=9)
axs[1].grid(axis='y',alpha=.2)
fig.tight_layout(pad=1.8)
for ext in ['png','svg']:fig.savefig(D/f'diagnostic_timeline.{ext}')
plt.close(fig)

fig,axs=plt.subplots(2,1,figsize=(11,7.3),gridspec_kw={'height_ratios':[1.5,1]})
ids=sorted({int(r['request']) for r in steps})
for i,q in enumerate(ids):
    sub=sorted([r for r in steps if int(r['request'])==q and int(r['step'])>0],key=lambda r:int(r['step']))
    axs[0].plot([int(r['step']) for r in sub],[float(r['CUDA_Event_ms']) for r in sub],color='#c03c3c' if q==49 else '#73899b',lw=2.2 if q==49 else 1,alpha=1 if q==49 else .65,label='Request 49 (slow)' if q==49 else 'All other 7 idle60 requests' if q==50 else None)
axs[0].axvline(5.5,color='#bbb',ls=':',lw=1)
axs[0].annotate('Slowdown begins at decode 6\nFirst 5 mean 65.45 ms; last 25 mean 75.27 ms',xy=(6,82.205),xytext=(10,80),fontsize=9,arrowprops={'arrowstyle':'->','color':'#555'})
axs[0].set(title='All eight idle60 requests: 31 decode Event durations (prefill excluded)',xlabel='Decode step',ylabel='CUDA Event time (ms)',xlim=(1,31),ylim=(57,86))
axs[0].legend(loc='lower right',frameon=False,fontsize=9)
axs[0].grid(axis='y',alpha=.2)
a=extra['profiler_decode_aggregates'];total=a['Event_sum_ms']['mean'];busy=a['traced_GPU_activity_union_sum_ms']['mean'];gap=a['internal_GPU_gaps_sum_ms']['mean'];edge=total-busy-gap
left=0
for val,color,name in [(busy,'#397c82','Traced GPU activity'),(gap,'#d69b42','Internal gaps'),(edge,'#acb2b9','Event edges / residual')]:
    axs[1].barh([0],val,left=left,color=color,height=.36,label=f'{name}: {val:.1f} ms')
    if val>100:axs[1].text(left+val/2,0,f'{val:.1f} ms',va='center',ha='center',color='white',weight='bold')
    left+=val
axs[1].set(title='PROFILED ONLY: average 31-step Event total across 8 captures',xlabel='Milliseconds',yticks=[],xlim=(0,3300),ylim=(-.45,.7))
axs[1].text(0,.43,'This is not a clean fast/slow comparison and cannot allocate the historical 65→74 ms/token difference.',fontsize=9)
axs[1].legend(loc='lower center',bbox_to_anchor=(.5,-.48),ncol=3,frameon=False,fontsize=8)
fig.tight_layout(pad=1.8)
for ext in ['png','svg']:fig.savefig(D/f'decode_and_profiled_gaps.{ext}',bbox_inches='tight')
plt.close(fig)
print('Saved 2 figures as PNG and SVG from existing CPU evidence only.')

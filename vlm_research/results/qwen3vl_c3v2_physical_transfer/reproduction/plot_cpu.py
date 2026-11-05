import pathlib,csv,json
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
W=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer');O=W/'vlm_research/results/qwen3vl_c3v2_physical_transfer';F=O/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
rows=list(csv.DictReader((O/'smoke_physical_execution.csv').open()));lookup={r['condition']:r for r in rows if r['rank']=='0'}
fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
values=[.4453125/.5,.08348099887371063/.02]
ax[0].bar(['Max absolute\nlogit error','Mean absolute\nlogit error'],values,color=['#4e79a7','#e15759']);ax[0].axhline(1,color='black',ls='--',label='Frozen limit');ax[0].set_ylabel('Observed error / frozen limit');ax[0].set_title('A  Numerical acceptance: FAILED');ax[0].legend(loc='upper left')
for i,y in enumerate(values):ax[0].text(i,y+.12,f'{y:.2f}x',ha='center')
cs=['F','P','B','C'];v=[float(lookup[c]['language_matrix_gflops']) for c in cs]
ax[1].bar(cs,v,color=['#999999','#59a14f','#4e79a7','#e15759']);ax[1].set_ylabel('Language matrix GFLOPs');ax[1].set_title('B  Observed executed GEMM shapes')
for i,y in enumerate(v):ax[1].text(i,y+25,f'{y:.1f}',ha='center',fontsize=9)
ax[1].set_ylim(0,max(v)*1.2)
x=np.arange(len(cs));live=[float(lookup[c]['kv_live_bytes'])/2**20 for c in cs];cap=[float(lookup[c]['kv_capacity_bytes'])/2**20 for c in cs]
ax[2].bar(x-.18,live,.36,label='Live');ax[2].bar(x+.18,cap,.36,label='Allocated capacity');ax[2].set_xticks(x,cs);ax[2].set_ylabel('KV MiB');ax[2].set_title('C  Actual persistent KV');ax[2].legend()
fig.suptitle('Failed physical-equivalence smoke: one GQA image, replicated on two RTX 3090 GPUs\nNot a quality or latency benchmark',fontsize=12)
for suffix in ['png','pdf']:fig.savefig(F/f'smoke_failure_diagnostic.{suffix}',dpi=220)
plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
for c,color in [('F','#777777'),('B','#4e79a7'),('P','#59a14f'),('C','#e15759')]:
 file=O/f'shard0/smoke/GQA__06905471__{c}.json';r=json.loads(file.read_text())['result']['audit'];ls=[v for v in r['layers'] if v['step']==0]
 ax.plot([v['layer'] for v in ls],[v['hidden_rows'] for v in ls],label=c,color=color,lw=2,ls='--' if c in ['B','C'] else '-')
ax.axvline(21.5,ls=':',color='black');ax.set(xlabel='Decoder layer (0-based)',ylabel='Actual prefill hidden rows',title='Physical exit observed at L22: 96 image rows removed, 26 text rows retained',xticks=list(range(0,36,4))+[35],yticks=[26,122]);ax.legend(title='Smoke only')
for suffix in ['png','pdf']:fig.savefig(F/f'physical_hidden_rows.{suffix}',dpi=220)
plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.axis('off');ax.text(.5,.62,'QUALITY / SPEED EVALUATION NOT RUN',ha='center',fontsize=16,color='#b2182b');ax.text(.5,.38,'Frozen F-versus-C numerical gate failed on both GPUs.\nNo formal accuracy, bootstrap interval, noninferiority,\nE2E speedup or full-model FLOPs claim is available.',ha='center',fontsize=12);ax.text(.5,.12,'Planned new-image cohorts: GQA 1012 questions; VQAv2 503 questions',ha='center',fontsize=10)
for suffix in ['png','pdf']:fig.savefig(F/f'quality_speed_status.{suffix}',dpi=220)
print('6 figure files written')

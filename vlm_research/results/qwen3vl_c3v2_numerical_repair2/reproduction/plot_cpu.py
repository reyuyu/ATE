import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-numerical-repair2/vlm_research/results/qwen3vl_c3v2_numerical_repair2');D=O/'figures';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def rows(name):return list(csv.DictReader((O/name).open()))
def save(fig,name):
 fig.savefig(D/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(D/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
r=rows('logits_comparison.csv');c=rows('repair_candidates.csv');variants=['previous-prefill-zero-key','decode-zero-key','FP32-L22+decode','FP32-late+decode'];labels=['Previous repair','Decode zero keys','Local FP32 at L22','Local FP32 at L22-L35']
fig,ax=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
for panel,metric in zip(ax,['mean_abs','max_abs']):
 for step,color,delta in [(0,'#4C78A8',-.18),(1,'#F58518',.18)]:
  vals=[float(next(x for x in r if x['run']==('phase1_rank0' if v==variants[0] else 'phase2_rank0') and x['variant']==v and x['reference']=='F' and x['candidate']=='C' and int(x['step'])==step)[metric]) for v in variants]
  bars=panel.bar(np.arange(4)+delta,vals,width=.36,color=color,label='First token' if step==0 else 'EOS prediction');panel.bar_label(bars,fmt='%.4f',fontsize=8,padding=3)
 panel.axhline(.02 if metric=='mean_abs' else .5,color='black',ls='--',lw=1.2,label='Unchanged acceptance limit');panel.set_xticks(range(4),labels,rotation=15,ha='right');panel.set_ylabel(('Mean' if metric=='mean_abs' else 'Maximum')+' absolute logits difference');panel.set_ylim(0,.13 if metric=='mean_abs' else .78);panel.legend(fontsize=8)
fig.suptitle('Full physical C remains outside the original numerical acceptance gate');save(fig,'logits_repair_comparison')
before=rows('phase1_component_differences.csv');after=rows('phase2_rank0_component_differences.csv');fig,ax=plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
for axis,stage in zip(ax,['attention_AV_output','layer_output']):
 for dat,label,color in [(before,'Previous repair','#E45756'),(after,'Decode zero-key repair','#4C78A8')]:
  rr=[x for x in dat if x['comparison']=='F__B' and x['step']=='1' and x['stage']==stage and ('variant' not in x or x['variant']=='decode-zero-key')];rr.sort(key=lambda x:int(x['layer']));axis.plot([int(x['layer']) for x in rr],[float(x['mean_abs']) for x in rr],label=label,color=color,marker='.',ms=3)
 axis.set_yscale('symlog',linthresh=1e-11);axis.set_ylim(bottom=0);axis.axvline(12,color='gray',ls=':',lw=1);axis.set_xlabel('Decoder layer (0-based)');axis.set_ylabel('Mean absolute difference');axis.set_title('Decode '+stage.replace('_',' '));axis.legend(fontsize=8)
fig.suptitle('Original case only: F/B Decode first diverges at L12 AV; repair restores exact agreement');save(fig,'decode_first_divergence')
ops=rows('operator_differences.csv');fig,ax=plt.subplots(1,2,figsize=(11,3.7),layout='constrained');names=['k_proj','v_proj','gate_proj','up_proj'];types=['BF16_native_26','FP32_local_round_BF16','BF16_batched_GEMM','BF16_no_reduced_accumulation'];colors=['#4C78A8','#F58518','#54A24B','#B279A2']
for axis,layer in zip(ax,[22,23]):
 for j,(kind,color) in enumerate(zip(types,colors)):
  vals=[float(next(r for r in ops if int(r['layer'])==layer and r['operator']==name and r['variant']==kind)['mean_abs']) for name in names];axis.bar(np.arange(4)+(j-1.5)*.19,vals,width=.19,color=color,label=kind.replace('_',' '))
 axis.set_xticks(range(4),['K','V','Gate','Up']);axis.set_ylabel('Mean difference vs frozen dense BF16');axis.set_title(f'Layer {layer}: same retained input rows')
ax[1].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.0));ax[1].set_ylim(0,.0021);ax[0].set_ylim(0,.0021);fig.suptitle('Local FP32 does not restore the frozen dense K/V projections');save(fig,'operator_precision_comparison')
audit=json.loads((O/'physical_execution_audit.json').read_text());p=next(x for x in audit['executions'] if x['run']=='phase2_rank0' and x['name']=='decode-zero-key__C');fig,ax=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
ax[0].plot(range(36),[122]*36,label='Dense F',color='gray');ax[0].plot(range(36),[r['hidden_rows'] for r in p['hidden_rows_by_layer'] if r['step']==0],label='Physical C',color='#4C78A8');ax[0].set_xlabel('Decoder layer');ax[0].set_ylabel('Actual Prefill hidden rows');ax[0].set_yticks([26,122]);ax[0].legend()
ax[1].bar(['Persistent\ntext KV','Temporary\nattention keys'],[27,123],color=['#4C78A8','#F58518']);ax[1].set_ylabel('Key positions per active Decode layer');ax[1].set_ylim(0,140);ax[1].text(1,125,'96 masked zero columns',ha='center',fontsize=8)
extra=sum(x['extra_QK_AV_flops'] for x in p['decode_padding']);compact=4*32*27*128*30;ax[2].bar(['Compact\nQK/AV','Zero-key repair\nQK/AV'],[compact/1e6,(compact+extra)/1e6],color=['#54A24B','#E45756']);ax[2].set_ylabel('Decode QK + AV MFLOPs');ax[2].set_ylim(0,76);ax[2].text(.5,68,'Six layers still fully skipped',ha='center',fontsize=8)
fig.suptitle('Real skip, hidden exit and compact KV persist; temporary padding adds computation');save(fig,'physical_shapes_and_overhead')
print('Four PNG/PDF pairs generated')

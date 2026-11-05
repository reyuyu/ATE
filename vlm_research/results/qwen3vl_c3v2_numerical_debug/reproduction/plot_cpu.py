import pathlib,csv,json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-numerical-debug/vlm_research/results/qwen3vl_c3v2_numerical_debug');F=O/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def csvread(n):return list(csv.DictReader((O/n).open()))
def save(fig,n):
 for suffix in ['png','pdf']:fig.savefig(F/f'{n}.{suffix}',dpi=220)
 plt.close(fig)
original=csvread('attention_component_differences.csv');repaired=csvread('repair_component_differences.csv')
def curve(rows,pair,scope,stage,metric):
 return [float(next(r for r in rows if r['comparison']==pair and r['layer']==str(layer) and r['scope']==scope and r['stage']==stage)[metric]) for layer in range(36)]
fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for col,metric,title in [(0,'mean_abs','Mean absolute hidden error'),(1,'max_abs','Maximum absolute hidden error')]:
 ax[col].plot(range(36),curve(original,'F__C','p_minus_1','layer_output',metric),label='Original C',color='#e15759')
 ax[col].plot(range(36),curve(repaired,'F__C-repaired','p_minus_1','layer_output',metric),label='Minimal repair C',color='#4e79a7')
 ax[col].axvline(6.5,color='gray',ls=':',label='First P-1 difference: L7' if col==0 else None);ax[col].axvline(21.5,color='black',ls='--',label='Physical exit at L22' if col==0 else None)
 ax[col].set(xlabel='Decoder layer (0-based)',ylabel=title,title=title);ax[col].set_yscale('symlog',linthresh=1e-5);ax[col].legend(fontsize=8)
fig.suptitle('P-1 hidden-state error relative to F: original fixed GQA smoke only')
save(fig,'p_minus_1_layerwise_error')
stages=['layer_input','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm','scores_before_mask','attention_probabilities','attention_AV_output','o_proj_raw','layer_output']
def heat(pair,layers,title,name):
 values=np.array([[float(next(r for r in original if r['comparison']==pair and r['layer']==str(l) and r['scope']=='text' and r['stage']==s)['mean_abs']) for l in layers] for s in stages]);masked=np.ma.masked_equal(values,0)
 fig,ax=plt.subplots(figsize=(10,6),layout='constrained');cmap=plt.get_cmap('magma').copy();cmap.set_bad('#dddddd');im=ax.imshow(np.ma.log10(masked),aspect='auto',vmin=-10,vmax=-1,cmap=cmap)
 ax.set_xticks(range(len(layers)),[f'L{x}' for x in layers]);ax.set_yticks(range(len(stages)),[s.replace('_',' ') for s in stages]);ax.set_title(title+'\nGray = exactly equal; aligned retained text positions');fig.colorbar(im,ax=ax,label='log10(mean absolute difference)');save(fig,name)
heat('F__B',list(range(8)),'Early physical bypass: F versus B','early_bypass_components')
heat('W22-functional__P',list(range(20,24)),'L22 physical exit: matched W22-functional versus P','exit_boundary_components')
rows=csvread('repair_comparison.csv');labels=[r['comparison'].replace('__',' / ')+'\nstep '+r['step'] for r in rows];x=np.arange(len(rows));fig,ax=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for j,metric,label,limit in [(0,'mean_abs','Mean absolute logit error',.02),(1,'max_abs','Maximum absolute logit error',.5)]:
 a=[float(r['before_'+metric]) for r in rows];b=[float(r['after_'+metric]) for r in rows]
 ax[j].bar(x-.18,a,.36,label='Before',color='#e15759');ax[j].bar(x+.18,b,.36,label='Minimal repair',color='#4e79a7');ax[j].axhline(limit,ls='--',color='black',label=f'Unchanged limit = {limit:g}');ax[j].set_xticks(x,labels);ax[j].set_ylabel(label);ax[j].set_ylim(0,max(max(a),max(b),limit)*1.25);ax[j].legend(fontsize=8,loc='upper right')
 for i,z in enumerate(b):ax[j].text(i+.18,z+max(max(a),limit)*.03,f'{z:.4f}',ha='center',fontsize=8)
fig.suptitle('Partial early repair does not pass complete C acceptance; no further samples run')
save(fig,'repair_logits_comparison')
d=json.loads((O/'physical_execution_audit.json').read_text());cases={('repair' if e['version']=='minimal_repair' else 'original')+'_'+e['condition']:e for e in d['executions']};fig,ax=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
for key,color,style in [('original_F','#777777','-'),('original_B','#59a14f','--'),('original_C','#e15759','-'),('repair_C','#4e79a7','--')]:
 e=cases[key];ls=e['hidden_rows_by_layer'];ax[0].plot([r['layer'] for r in ls],[r['hidden_rows'] for r in ls],label=key.replace('_',' '),color=color,ls=style)
ax[0].set(xlabel='Decoder layer',ylabel='Actual prefill hidden rows',title='Physical hidden exit remains present',yticks=[26,122]);ax[0].legend(fontsize=8)
keys=['original_F','original_B','original_C','repair_C'];x=np.arange(4);live=[cases[k]['kv']['live_bytes']/2**20 for k in keys];capacity=[cases[k]['kv']['capacity_bytes']/2**20 for k in keys];ax[1].bar(x-.18,live,.36,label='Live');ax[1].bar(x+.18,capacity,.36,label='Capacity');ax[1].set_xticks(x,[k.replace('_','\n') for k in keys]);ax[1].set(ylabel='Persistent KV MiB',title='No dummy key is persisted');ax[1].legend(fontsize=8)
fig.suptitle('Shape / storage audit only: not an E2E benchmark; no speedup claim')
save(fig,'physical_shapes_kv_audit')
print('Five PNG/PDF figure pairs written')

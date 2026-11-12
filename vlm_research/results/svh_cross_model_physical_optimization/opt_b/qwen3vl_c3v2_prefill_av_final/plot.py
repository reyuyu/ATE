import pathlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=pathlib.Path(__file__).resolve().parents[2]/'vlm_research/results/qwen3vl_c3v2_prefill_av_final'
def rows(n):return list(csv.DictReader((O/n).open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
f,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained');layers=rows('layerwise_errors.csv');x=np.arange(22,36)
for variant,label,color in [('before_audited_reuse_with_new_strict_logits_replay','Before (audited parent)','#70889b'),('after','After Prefill zero keys','#178e77')]:
 a=[r for r in layers if r['sample']=='GQA__06905471' and r['variant']==variant and r['rank'] in ['historical0','0'] and r['step']=='0' and r['stage']=='layer_output' and r['scope']=='retained_text'];ys=[float(next(r['mean_abs'] for r in a if r['layer']==str(i))) for i in x];ax[0,0].plot(x,ys,'o-',label=label,color=color,ms=4)
ax[0,0].set_title('A  Original input: layer-output error');ax[0,0].set_xlabel('Layer (0-based)');ax[0,0].set_ylabel('Mean absolute hidden error');ax[0,0].legend(fontsize=8)
lg=rows('logits_comparison.csv');idx=np.arange(4)
for tag,offset,label,color in [('F__C_before',-.18,'Before','#70889b'),('F__C_after',.18,'After','#178e77')]:
 ys=[float(next(r['mean_abs'] for r in lg if r['rank']=='0' and r['sample']==s and r['comparison']==tag and r['step']==str(t))) for s,t in [('GQA__06905471',0),('GQA__06905471',1),('GQA__02497673',0),('GQA__02497673',1)]];ax[0,1].bar(idx+offset,ys,.34,label=label,color=color)
ax[0,1].axhline(.02,ls='--',color='#bd3a38',label='Mean gate = 0.02');ax[0,1].set_xticks(idx,['Original\nfirst','Original\nEOS','Median\nfirst','Median\nEOS']);ax[0,1].set_ylabel('Mean absolute F/C logit error');ax[0,1].set_title('B  Original passes; median fails');ax[0,1].legend(fontsize=8)
pc=[r for r in rows('physical_cost_summary.csv') if r['rank']=='0'];z=np.arange(2)
for field,offset,label,color in [('late_QK_AV_flops_before',-.18,'Before','#70889b'),('late_QK_AV_flops_after',.18,'After','#178e77')]:ax[1,0].bar(z+offset,[int(r[field])/1e9 for r in pc],.34,label=label,color=color)
ax[1,0].set_xticks(z,['Original: 26 queries','Median: 29 queries']);ax[1,0].set_ylabel('QK + AV matrix FLOPs (billions)');ax[1,0].set_title('C  L22-L35 Prefill arithmetic increases');ax[1,0].legend(fontsize=8)
for field,offset,label,color in [('late_attention_single_call_ms_sum_before',-.18,'Before','#70889b'),('late_attention_single_call_ms_sum_after',.18,'After','#178e77')]:ax[1,1].bar(z+offset,[float(r[field]) for r in pc],.34,label=label,color=color)
ax[1,1].set_xticks(z,['Original','Median']);ax[1,1].set_ylabel('Sum of 14 local GPU-event calls (ms)');ax[1,1].set_title('D  Local attention timing (single pass)');ax[1,1].legend(fontsize=8);ax[1,1].set_xlabel('Not E2E; no repeats; host gaps may contribute',fontsize=8)
(O/'figures').mkdir(exist_ok=True)
for ext in ['png','pdf']:f.savefig(O/'figures'/('prefill_av_repair_and_costs.'+ext),dpi=180)

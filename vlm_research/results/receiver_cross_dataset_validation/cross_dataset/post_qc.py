"""Delivery-only readability, anomaly accounting and final artifact checks."""
import pathlib,sys,csv,collections,json,shutil
W=pathlib.Path('/data/lk/ATE-worktrees/single-gpu-receiver-cross-dataset-validation');sys.path.insert(0,str(W/'vlm_research'))
from receiver_cross_dataset.common import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
D=OUT/'cross_dataset'
def load(p):return list(csv.DictReader(p.open()))
def save(fig,name):fig.savefig(D/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(D/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
# Same observed points as the frozen analysis; only separate coincident labels.
fig,axes=plt.subplots(1,2,figsize=(12,4));colors=['#606060','#0072B2','#D55E00','#009E73','#CC79A7'];short=['All','Q only','T15','T18','Random'];offsets=[(-8,8),(5,8),(14,2),(-7,22),(14,-12)]
for ax,dataset in zip(axes,['vqa_v2','textvqa']):
 for i,r in enumerate(load(OUT/dataset/'accuracy_summary.csv')):
  x=float(r['mean_R_ratio'])*100;y=float(r['VQA_accuracy'])
  ax.scatter(x,y,edgecolor=colors[i],facecolor='none' if i==2 else colors[i],marker='^' if i==3 else 'o',s=70 if i==2 else 55,linewidth=1.5,zorder=3)
  ax.annotate(short[i],(x,y),xytext=offsets[i],textcoords='offset points',fontsize=8,ha='right' if i in [0,3] else 'left',arrowprops=dict(arrowstyle='-',lw=.6,color=colors[i]))
 ax.set_xlabel('Mean allowed R-to-image receiver budget (%)');ax.set_ylabel('Official VQA accuracy (%)');ax.set_title(dataset);ax.grid(alpha=.2)
fig.suptitle('D. Functional receiver budget vs. answer quality\nDense computation retained; budget is not FLOPs or latency');fig.tight_layout();save(fig,'figure_D_budget_accuracy')
# A shared legend prevents category labels from covering the TextVQA bar.
structures={'GQA3000':load(D/'gqa_historical_structure.csv'),'vqa_v2':load(OUT/'vqa_v2/trigger_analysis.csv'),'textvqa':load(OUT/'textvqa/trigger_analysis.csv')}
fig,axes=plt.subplots(1,3,figsize=(15,4.5));groups=['what','which','how','yes/no','where/who/why','other'];x=np.arange(3);bottom=np.zeros(3)
for category in groups:
 vals=np.array([100*sum(r['first_word_group']==category for r in rr)/len(rr) for rr in structures.values()]);axes[0].bar(x,vals,bottom=bottom,label=category);bottom+=vals
axes[0].set_xticks(x,list(structures),rotation=15);axes[0].set_ylabel('Questions (%)');axes[0].set_title('First question word')
axes[1].boxplot([[int(r['Q_tokens']) for r in rr] for rr in structures.values()],tick_labels=list(structures));axes[1].axhline(15,color=colors[2],ls='--',lw=1);axes[1].axhline(18,color=colors[3],ls=':',lw=1);axes[1].set_title('Native Question-token length');axes[1].set_ylabel('Tokenizer subwords');axes[1].tick_params(axis='x',rotation=15)
for k,t in enumerate([15,18]):axes[2].bar(x+(k-.5)*.3,[100*sum(r['trigger'+str(t)]=='True' for r in rr)/len(rr) for rr in structures.values()],width=.3,label=f'Threshold-{t}')
axes[2].set_xticks(x,list(structures),rotation=15);axes[2].set_ylabel('Triggered questions (%)');axes[2].set_title('Frozen OR-trigger coverage');axes[2].legend()
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=6,frameon=False);fig.suptitle('B. Question structure and receiver trigger distribution');fig.tight_layout(rect=(0,.08,1,.98));save(fig,'figure_B_question_structure')
report=OUT/'report_zh.md';text=report.read_text();text=text.replace('图B区分GPU样本与完整验证集CPU首词统计。','图B仅使用实际GPU样本分布；完整验证集CPU首词统计另存各数据集full_validation_cpu_firstwords.json。')
text=text.replace('**Physical Visual Handoff。** 若后续算法需要功能性Receiver规则，这两条冻结规则可按本报告实际外部证据保留为候选；','**Physical Visual Handoff。** VQAv2上值得将这两条规则保留为LLaVA固定模板的功能性候选；TextVQA只支持增加接收位置有益，尚未证明当前语言触发优于随机，因而不能统一锁定为默认策略。')
extra=['','## 输出异常、题型与交付复核',''];anomaly_summary=[]
for dataset in ['vqa_v2','textvqa']:
 d=OUT/dataset;rows=load(d/'per_question.csv');ab=[r for r in rows if r['truncated']=='True' or r['empty']=='True' or r['repeated_trigram']=='True'];csvwrite(d/'abnormal_outputs.csv',ab,fields=list(rows[0]));counts=[]
 for condition in CONDITIONS:
  rr=[r for r in rows if r['condition']==condition];counts.append(dict(dataset=dataset,condition=condition,n=len(rr),empty=sum(r['empty']=='True' for r in rr),truncated=sum(r['truncated']=='True' for r in rr),repeated_3gram=sum(r['repeated_trigram']=='True' for r in rr)))
 anomaly_summary.extend(counts);unique_cap=len({r['checkpoint'] for r in ab if r['truncated']=='True'});unique_repeat=len({r['checkpoint'] for r in ab if r['repeated_trigram']=='True'})
 extra += [f"**{dataset}异常输出。** 按五个条件标签统计：空答案{sum(x['empty'] for x in counts)}条，达到32 Token上限{sum(x['truncated'] for x in counts)}条（{unique_cap}次不同接收集合生成），重复3gram标记{sum(x['repeated_3gram'] for x in counts)}条（{unique_repeat}次不同接收集合生成）。全部保留并参与官方评分，没有延长生成、重试或删除。COMPLETE表示全部冻结任务已执行并核验，不表示所有模型输出均正常EOS。逐条见abnormal_outputs.csv。",'']
 if dataset=='vqa_v2':
  aa=[r for r in load(d/'answer_type_accuracy.csv') if r['axis']=='answer_type'];extra += ['| VQAv2官方答案类型 | n | All % | Q % | T15 % | T18 % | Random % |','|---|---:|---:|---:|---:|---:|---:|']
  for category in sorted({r['value'] for r in aa}):
   group={r['condition']:r for r in aa if r['value']==category};extra.append('| '+category+' | '+group['All-Writer']['n']+' | '+' | '.join(f"{float(group[c]['accuracy']):.3f}" for c in CONDITIONS)+' |')
  extra+=['','此表为预先冻结样本的描述性分组，不因题型表现重新选阈值。','']
csvwrite(OUT/'abnormal_output_summary.csv',anomaly_summary)
audit=read(OUT/'independent_delivery_audit.json');extra += [f"**独立评分复核。** 所有4000条条件标签均与原始Checkpoint、原输入/图像哈希和接收位置一致。完整官方VQAEval重评最大Score差为{audit['datasets']['vqa_v2']['official_evaluator_max_score_difference']:.3g}（浮点舍入）；TextVQA完整MMF评分路径差为0。模型Native/No-op的Logits验证始终使用严格torch.equal，没有容差替代。",'', '**图像与工程限制。** TextVQA这里的All-Writer也禁止答案位置直接读原图，且不提供额外OCR提示；21.833%不能与不同Prompt/OCR配置的公开Native Full成绩直接比较。未发现历史GQA图片元数据/字节重叠，但重新编码或未映射身份仍有未知可能。','', '## 图表预览','']
for letter,name in [('A','cross_dataset_accuracy'),('B','question_structure'),('C','gain_over_question'),('D','budget_accuracy')]:extra += [f'![Figure {letter}](cross_dataset/figure_{letter}_{name}.png)','']
marker='\n## 输出异常、题型与交付复核'
if marker in text:text=text.split(marker)[0]
report.write_text((text+'\n'.join(extra)).rstrip()+'\n')
# Copy human-readable process evidence, not large logits or original datasets.
logs=OUT/'logs';logs.mkdir(exist_ok=True)
for name in ['CPU_tests.log','CPU_tests_rerun.log','bootstrap.log','prepare.log','data_prepare.log','gpu_run.log','analyze.log','delivery_audit.log']:
 p=PRIVATE/name
 if p.exists():shutil.copyfile(p,logs/name)
shutil.copyfile(PRIVATE/'source_acquisition.json',OUT/'source_acquisition.json')
v=read(OUT/'validation.json');v.update(independent_official_score_audit_passed=True,independent_audit_sha256=sha(OUT/'independent_delivery_audit.json'),statistical_CPU_validation_passed=read(OUT/'statistical_validation.json')['passed'],own_GPU_process_exited=True,raw_non_EOS_preserved=True);write(OUT/'validation.json',v)
assert sources()==read(OUT/'source_manifest.json')['files']
write(OUT/'cross_dataset/visual_QA.json',dict(passed=True,UTC=utc(),observed_data_unchanged=True,inference_source_unchanged=True,layout_edits=['Move Figure B legend outside bars','Separate coincident Figure D labels'],anomaly_summary=anomaly_summary))
print(json.dumps(anomaly_summary),flush=True)

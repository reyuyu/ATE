"""Independent CPU token rescoring and paired whole-image-cluster statistics."""
import csv,importlib.util,json,os
from collections import Counter
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from .common import *
def csvout(name,rows):
 rows=list(rows);assert rows
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def bootstrap(values,images,seed):
 names,inv=np.unique(images,return_inverse=True);k=len(names)
 sizes=np.bincount(inv).astype(float);sums=np.zeros((k,values.shape[1]));np.add.at(sums,inv,values)
 rng=np.random.default_rng(seed);result=np.empty((20000,values.shape[1]))
 for start in range(0,20000,128):
  w=rng.multinomial(k,np.full(k,1/k),size=min(128,20000-start))
  result[start:start+len(w)]=(w@sums)/(w@sizes)[:,None]
 assert np.isfinite(result).all()
 return result
def ci(x):return [float(x) for x in np.quantile(x,[.025,.975])*100]
def main():
 p=read(OUT,'protocol.json');ph=sha(OUT/'protocol.json');done=read(OUT,'generation_complete.json')
 assert done['passed'] and done['total_analysis_rows']==12000 and read(OUT,'smoke_summary.json')['passed']
 for path,name in [(P,'A'),(B,'B')]:
  assert sha(path/'protocol.json')==p['baseline_'+name+'_protocol_sha256']
  assert sha(path/'per_question.csv')==p['baseline_'+name+'_predictions_sha256']
 for src,h in read(OUT,'source_manifest.json').items():assert sha(src)==h,src
 samples=read(OUT,'sample_manifest.json')['samples'];hashes=read(OUT,'persistence_complete.json')['SHA256']
 scope={};exec(read(P/'references/qwen/vlm_research/results/qwen3vl_visual_writing_necessity_3000','official_gqa_prediction.json')['function_source'],scope)
 scorer=scope['gqa_prediction'];tok=AutoTokenizer.from_pretrained(MODEL,use_fast=False,local_files_only=True)
 spec=importlib.util.spec_from_file_location('frozen_joint_conv',MODEL/'conversation.py');conv=importlib.util.module_from_spec(spec);spec.loader.exec_module(conv)
 sep=conv.get_conv_template(read(MODEL,'config.json')['template']).sep.strip();eos=tok.convert_tokens_to_ids(sep)
 values=np.empty((3000,4));records=[];rows=[];flags={c:Counter() for c in CONDITIONS}
 for idx,s in enumerate(samples):
  k=key(s);assert sha(OUT/'checkpoints'/k)==hashes[k]
  r=read(OUT/'checkpoints',k)
  assert r['complete'] and r['protocol_sha256']==ph and r['GPU_UUID']==p['GPU_UUID'] and r['index']==idx
  assert r['sample_sha256']==digest(s) and set(r['conditions'])==set(CONDITIONS)
  assert r['question_id']==s['question_id'] and r['image_sha256']==s['image_sha256'] and r['gold']==s['answer']
  assert digest(r['input'])==r['input_binding']
  for j,c in enumerate(CONDITIONS):
   g=r['conditions'][c];ids=g['token_ids'];assert 1<=len(ids)<=32
   text=tok.decode(ids,skip_special_tokens=True).split(sep)[0].strip();pred=scorer(text);correct=int(pred==s['answer'].lower())
   assert g['text']==text and g['prediction']==pred and int(g['correct'])==correct
   assert g['input_binding']==r['input_binding'] and g['independent_fresh_cache']
   assert g['writer_layers']==list(range(20 if c in ('B','D') else 28)) and g['P1_Decode_image_off']
   assert g['mask_audit']['blocked_probability_max']==0
   if c in ('C','D'):
    assert g['answer_attention_output_zero_layers']==list(range(2,6))
    assert g['mask_audit']['zero_output_max']==0 and g['mask_audit']['zero_output_events']==4*len(ids)
   n=r['input']['roles']['n'];assert g['final_cache_lengths']==[n+len(ids)-1]*28
   assert all(shape==[1,4,n+len(ids)-1,128] for shape in g['cache_shapes'])
   f=dict(eos=ids[-1]==eos,truncated=len(ids)==32 and ids[-1]!=eos,empty=not text,
    repeated_trigram=len(ids)>=6 and len({tuple(ids[q:q+3]) for q in range(len(ids)-2)})<len(ids)-2)
   for flag,v in f.items():assert bool(g[flag])==v;flags[c][flag]+=int(v)
   values[idx,j]=correct
   rows.append(dict(index=idx,task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],
    condition=c,gold=s['answer'],text=text,prediction=pred,correct=correct,token_ids=json.dumps(ids),generated_tokens=len(ids),**f,
    input_binding=r['input_binding'],reused_baseline=g['reused'],source_checkpoint_sha256=g.get('source_checkpoint_sha256',''),
    checkpoint_sha256=hashes[k],GPU_UUID=p['GPU_UUID'],protocol_sha256=ph))
  records.append(r)
 assert len(rows)==12000
 for c in CONDITIONS:assert Counter(x['task'] for x in rows if x['condition']==c)==Counter({t:500 for t in TASKS})
 csvout('per_question.csv',rows)
 images=np.array([s['image_sha256'] for s in samples]);draws=bootstrap(values,images,p['bootstrap_seed'])
 summary=[];taskrows=[];labels={'A':'All-Writer','B':'L20-L27 writing OFF','C':'L2-L5 answer o_proj output zero','D':'Joint candidate'}
 for j,c in enumerate(CONDITIONS):
  lo,hi=ci(draws[:,j])
  summary.append(dict(condition=c,description=labels[c],questions=3000,correct=int(values[:,j].sum()),accuracy_pct=float(values[:,j].mean()*100),
   CI95_low=lo,CI95_high=hi,reused_baseline=bool(records[0]['conditions'][c]['reused']),**dict(flags[c])))
  for task in TASKS:
   ix=[i for i,s in enumerate(samples) if s['task']==task]
   taskrows.append(dict(task=task,condition=c,questions=500,correct=int(values[ix,j].sum()),accuracy_pct=float(values[ix,j].mean()*100),delta_vs_A_pp=float((values[ix,j]-values[ix,0]).mean()*100)))
 csvout('accuracy_summary.csv',summary);csvout('task_accuracy.csv',taskrows)
 pairs=[('A','B'),('A','C'),('B','D'),('A','D')];comparisons=[];harms=[]
 for base,test in pairs:
  bi=CONDITIONS.index(base);ti=CONDITIONS.index(test);lo,hi=ci(draws[:,ti]-draws[:,bi]);count=Counter()
  for idx,s in enumerate(samples):
   x=records[idx]['conditions'][base];y=records[idx]['conditions'][test];ctw=int(x['correct'] and not y['correct']);wtc=int(not x['correct'] and y['correct'])
   count.update(correct_to_wrong=ctw,wrong_to_correct=wtc,first_token_flip=int(x['token_ids'][0]!=y['token_ids'][0]),full_text_change=int(x['text']!=y['text']))
   if ctw:harms.append(dict(comparison=test+'-'+base,task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],
    baseline_text=x['text'],intervention_text=y['text'],baseline_token_ids=json.dumps(x['token_ids']),intervention_token_ids=json.dumps(y['token_ids']),input_binding=records[idx]['input_binding']))
  comparisons.append(dict(comparison=test+'-'+base,baseline=base,intervention=test,questions=3000,delta_pp=float((values[:,ti]-values[:,bi]).mean()*100),CI95_low=lo,CI95_high=hi,**dict(count),ordinary_unadjusted_CI=True))
 csvout('paired_comparisons.csv',comparisons);csvout('harm_cases.csv',harms or [dict(comparison='none',note='No correct-to-wrong cases')])
 taskcomparisons=[]
 for task in TASKS:
  ix=[i for i,s in enumerate(samples) if s['task']==task];v=values[ix];d=bootstrap(v,images[ix],p['bootstrap_seed']+TASKS.index(task)+1)
  for base,test in pairs:
   bi=CONDITIONS.index(base);ti=CONDITIONS.index(test);lo,hi=ci(d[:,ti]-d[:,bi])
   taskcomparisons.append(dict(task=task,comparison=test+'-'+base,questions=500,delta_pp=float((v[:,ti]-v[:,bi]).mean()*100),CI95_low=lo,CI95_high=hi,
    correct_to_wrong=int(((v[:,bi]==1)&(v[:,ti]==0)).sum()),wrong_to_correct=int(((v[:,bi]==0)&(v[:,ti]==1)).sum()),ordinary_unadjusted_CI=True))
 csvout('task_paired_comparisons.csv',taskcomparisons)
 iv=values[:,3]-values[:,1]-values[:,2]+values[:,0];idraw=draws[:,3]-draws[:,1]-draws[:,2]+draws[:,0];lo,hi=ci(idraw)
 interaction=dict(comparison='(D-B)-(C-A)',interaction_pp=float(iv.mean()*100),CI95_low=lo,CI95_high=hi,negative_means_additional_joint_harm=True,ordinary_unadjusted_CI=True)
 csvout('interaction_analysis.csv',[interaction]);save(OUT,'score_pattern_counts.json',dict(Counter(''.join(str(int(x)) for x in r) for r in values)))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import MaxNLocator
 fig,(ax,bx)=plt.subplots(1,2,figsize=(11.8,4.4),gridspec_kw={'width_ratios':[1.1,1]})
 colors=['#465A74','#EBA752','#7F8FB3','#4A9F8D'];acc=[r['accuracy_pct'] for r in summary]
 ax.bar(CONDITIONS,acc,color=colors,width=.62)
 for i,v in enumerate(acc):ax.text(i,v+1.6,f'{v:.2f}%',ha='center',fontsize=11)
 ax.set_ylim(0,100);ax.set_ylabel('GQA Accuracy (%)');ax.set_title('(a) Frozen 2x2 functional validation')
 ax.set_xticks(range(4),['A'+chr(10)+'All-Writer','B'+chr(10)+'Writing OFF','C'+chr(10)+'Attn output 0','D'+chr(10)+'Joint SVH'])
 ax.spines[['right','top']].set_visible(False)
 for i,r in enumerate(comparisons):
  color=colors[3] if r['comparison']=='D-B' else colors[0]
  bx.errorbar(r['delta_pp'],3-i,xerr=[[r['delta_pp']-r['CI95_low']],[r['CI95_high']-r['delta_pp']]],fmt='o',color=color,capsize=4)
  bx.annotate(f"{r['delta_pp']:+.2f}",(r['delta_pp'],3-i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
 bx.axvline(0,color='gray',lw=1);bx.axvline(-1,color='#B67940',ls=':',lw=1)
 bx.set_yticks(range(4),[r['comparison']+(' (core)' if r['comparison']=='D-B' else '') for r in comparisons[::-1]])
 bx.set_ylim(-.55,3.65);bx.set_xlabel('Paired accuracy difference (pp), ordinary 95% CI')
 bx.set_title('(b) Effects relative to A and conditional D-B');bx.spines[['right','top']].set_visible(False);bx.xaxis.set_major_locator(MaxNLocator(6))
 fig.text(.5,.015,'Same 3,000 GQA questions / 2,409 image clusters. Functional output zeroing; no physical speed claim. L20 is post hoc.',ha='center',fontsize=8.5)
 fig.tight_layout(rect=(0,.055,1,1));fig.savefig(OUT/'figure_joint_validation.png',dpi=320,bbox_inches='tight');fig.savefig(OUT/'figure_joint_validation.pdf',bbox_inches='tight');plt.close(fig)
 pair={r['comparison']:r for r in comparisons};d=summary[3]
 def fmt(r):return f"{r['delta_pp']:+.4f} pp，95%CI [{r['CI95_low']:+.4f}, {r['CI95_high']:+.4f}]"
 head=f"联合候选D的完整答案Accuracy为 **{d['accuracy_pct']:.4f}%（{d['correct']}/3000）**。D−A：{fmt(pair['D-A'])}；D−B：{fmt(pair['D-B'])}。交互(D−B)−(C−A)：{interaction['interaction_pp']:+.4f} pp，95%CI [{lo:+.4f}, {hi:+.4f}]。"
 if hi<0:head+='普通95%CI低于0，存在额外负向联合效应的探索性证据。'
 elif lo>0:head+='交互方向为正，未观察到负向非加性效应。'
 else:head+='交互CI跨0，尚不能确认负向联合效应，也不能证明等价或无交互。'
 report=[head,'','| 条件 | 写入层 | 答案Attention | Correct / 3000 | Accuracy |','|---|---|---|---:|---:|']
 for r in summary:
  c=r['condition'];report.append(f"| {c} | L0–L{'19' if c in ('B','D') else '27'} | {'L2–L5 o_proj输出置零' if c in ('C','D') else '正常'} | {r['correct']} | {r['accuracy_pct']:.4f}% |")
 report+=['','所有条件P−1与真实Decode全部28层均禁止直接读原始Image Key。Image Query原生。只改变R→Image权限和目标答案行o_proj输出；Norm、MLP、Residual、视觉Hidden和Dense KV保留。输出置零不等于物理跳算。','',
  '| 对比（负值为损失） | ΔACC pp | 普通95%CI | 原对→错 | 原错→对 | 首Token翻转 | 完整文本改变 |','|---|---:|---|---:|---:|---:|---:|']
 for r in comparisons:report.append(f"| {r['comparison']} | {r['delta_pp']:+.4f} | [{r['CI95_low']:+.4f}, {r['CI95_high']:+.4f}] | {r['correct_to_wrong']} | {r['wrong_to_correct']} | {r['first_token_flip']} | {r['full_text_change']} |")
 report+=['','六任务完整结果（各500题）：','','| Task | A | B | C | D | D−A pp | D−B pp |','|---|---:|---:|---:|---:|---:|---:|']
 for task in TASKS:
  v={r['condition']:r['accuracy_pct'] for r in taskrows if r['task']==task}
  report.append(f"| {task} | {v['A']:.2f}% | {v['B']:.2f}% | {v['C']:.2f}% | {v['D']:.2f}% | {v['D']-v['A']:+.2f} | {v['D']-v['B']:+.2f} |")
 report+=['','CompareAttr、QueryAttr及其他任务的全部配对CI和损害计数见task_paired_comparisons.csv。所有原对变错案例含问题、Gold、完整答案、Token IDs和图片SHA，完整列于harm_cases.csv，没有筛选或隐藏。','',
  '| 条件 | EOS | 截断 | 空输出 | 重复三元组 |','|---|---:|---:|---:|---:|']
 for r in summary:report.append(f"| {r['condition']} | {r.get('eos',0)} | {r.get('truncated',0)} | {r.get('empty',0)} | {r.get('repeated_trigram',0)} |")
 decision=read(OUT,'baseline_reuse_decision.json')
 report+=['',f"A/B复用：{decision['reuse_A_B']}；全部历史逐题数据已与原CSV、Checkpoint SHA、Prompt、Gold、输入绑定和独立评分核验。最短/典型/最长GPU回放见smoke_summary.json。新增正式生成{done['new_requests']}次，最终四条件共12000条记录。",
  f"模型OpenGVLab/InternVL3-8B @{REV}；BF16/Eager，Torch2.6.0+cu124、Transformers4.49.0、B1/Greedy/正常EOS/max_new_tokens32/logits_to_keep1、256视觉Token；GPU {p['GPU_UUID']}。",
  '','使用20,000次Image SHA整簇配对Bootstrap，问题加权；四项差值与一项交互为未作多重比较校正的普通95%CI，任务子组也为探索性。−1pp只作参考。CI跨0不证明等价；L20为看过原结果后提出的候选，当前样本不构成独立确认的非劣效证据，也不支持物理跳算、KV压缩或速度结论。','',
  f"持久化结果：{OUT}；独立工作树：{ROOT}；分支codex/internvl3-svh-joint-l20-skip2to5-3000。完整预测per_question.csv，全部Checkpoint在checkpoints/，恢复方式见recovery.json。历史实验未覆盖，本轮未推送GitHub、未创建PR、未运行其他Benchmark。"]
 (OUT/'report_zh.md').write_text(chr(10).join(report)+chr(10))
 save(OUT,'validation.json',dict(passed=True,questions=3000,image_clusters=2409,conditions=4,rows=12000,new_generations=done['new_requests'],
  baseline_A_B_reused=decision['reuse_A_B'],independent_token_decode_and_official_scores_exact=True,all_checkpoint_SHA256_verified=True,
  all_input_bindings_verified=True,all_target_o_proj_outputs_zero=True,all_P1_decode_image_probability_zero=True,
  all_28_dense_cache_shapes_correct=True,source_fingerprints_unchanged=True,same_GPU_UUID=p['GPU_UUID'],smoke_passed=True,
  no_training=True,no_E2E_or_physical_experiment=True,no_GitHub_push=True,image_cluster_bootstrap=20000,ordinary_unadjusted_CIs=True,post_hoc_candidate=True))
 save(OUT,'analysis_source_manifest.json',{str(Path(__file__).resolve()):sha(__file__)})
 print('JOINT_ANALYSIS_COMPLETE',json.dumps(dict(correct={r['condition']:r['correct'] for r in summary},comparisons=comparisons,interaction=interaction)),flush=True)
if __name__=='__main__':main()

"""Accuracy, paired cluster uncertainty and fixed restoration gate only."""
import csv,collections
import numpy as np
from .common import *
ALL=['Native','F']+CONDITIONS
def csvout(name,rows):
 rows=list(rows)
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['no_cases'],lineterminator='\n');w.writeheader();w.writerows(rows)
def bootstrap(samples,values):
 groups=sorted({s['image_sha256'] for s in samples});idx={h:i for i,h in enumerate(groups)};g=len(groups);counts=np.zeros(g);sums=np.zeros((g,values.shape[1]))
 for s,v in zip(samples,values):counts[idx[s['image_sha256']]]+=1;sums[idx[s['image_sha256']]]+=v
 rng=np.random.default_rng(SEED);out=[]
 for _ in range(80):
  w=rng.multinomial(g,np.full(g,1/g),size=250);out.append((w@sums)/(w@counts)[:,None])
 return np.concatenate(out)
def main():
 check_sources();m=read(OUT/'sample_manifest.json');samples=m['samples'];n=len(samples);assert n==1000
 for rank in range(2):
  assert read(OUT/f'smoke/rank{rank}/completion.json')['status']=='PASSED'
  assert read(OUT/f'accuracy/rank{rank}/completion.json')['status']=='COMPLETE'
 records={};points=np.zeros((n,7));wide=[];long=[];shaindex=[];anomalies=[]
 for j,s in enumerate(samples):
  q=s['question_id'];row={k:s[k] for k in ['question_id','question','image_id','image_sha256','gold','task','rank']}
  for i,c in enumerate(ALL):
   p=HIST/f'checkpoints/rank{s["rank"]}/{q}__{("Dense-F" if c=="F" else c)}.json' if c in ['Native','F'] else OUT/f'checkpoints/rank{s["rank"]}/{q}__{c}.json'
   d=read(p);r=d['result'];assert d['binding']['input_sha256']==digest(m['input_bindings'][q]);assert d['binding']['image_sha256']==s['image_sha256']
   if c in CONDITIONS:
    assert d['binding']['source_manifest_sha256']==digest(read(OUT/'source_manifest.json'))
    assert d['binding']['protocol_sha256']==sha(OUT/'protocol.json') and r['all_logits_finite'] and r['audit']['dense_cache_all_layers']
    assert not any(k in r for k in ['timing','e2e_ms','flops','memory','peak_allocated'])
   pred=r['text'].rstrip('.').lower();score=pred==s['gold'];assert score==r['correct'];records[q,c]=r;points[j,i]=score
   for key in ['text','correct','eos','truncated','empty','first_token','generated_tokens']:row[c+'__'+key]=r[key]
   row[c+'__token_ids']=json.dumps(r['token_ids'])
   long.append(dict(question_id=q,image_sha256=s['image_sha256'],condition=c,reused=c in ['Native','F'],gold=s['gold'],answer=r['text'],prediction=pred,correct=int(score),token_ids=json.dumps(r['token_ids']),eos=r['eos'],empty=r['empty'],truncated=r['truncated'],first_token=r['first_token']))
   shaindex.append(dict(question_id=q,condition=c,path=str(p.relative_to(ROOT)),sha256=sha(p),reused=c in ['Native','F']))
   if r['empty'] or r['truncated'] or not r['eos']:anomalies.append(dict(question_id=q,condition=c,text=r['text'],eos=r['eos'],truncated=r['truncated'],empty=r['empty']))
  assert records[q,'B']['first_token']==records[q,'F']['first_token'];assert records[q,'A']['first_token']==records[q,'C']['first_token']
  wide.append(row)
 assert points[:,0].sum()==694 and points[:,1].sum()==588
 boot=bootstrap(samples,points);aci=np.quantile(boot,[.025,.975],axis=0)*100
 summary=[]
 for i,c in enumerate(ALL):summary.append(dict(condition=c,n=n,correct=int(points[:,i].sum()),accuracy_percent=points[:,i].mean()*100,accuracy_ci_low=aci[0,i],accuracy_ci_high=aci[1,i],delta_F_pp=(points[:,i]-points[:,1]).mean()*100,delta_Native_pp=(points[:,i]-points[:,0]).mean()*100,reused=c in ['Native','F']))
 comparisons=read(OUT/'protocol.json')['bootstrap']['comparisons']+read(OUT/'protocol.json')['bootstrap']['additional_frozen_comparisons'];pairs=[];harms=[];recovered=[]
 for ca,cb in comparisons:
  a,b=ALL.index(ca),ALL.index(cb);v=points[:,a]-points[:,b];draw=(boot[:,a]-boot[:,b])*100;ci=np.quantile(draw,[.025,.975]);adj=np.quantile(draw,[.005,.995]) if cb=='F' else [None,None]
  pairs.append(dict(candidate=ca,reference=cb,delta_pp=v.mean()*100,ci_low_pp=ci[0],ci_high_pp=ci[1],F_family_bonferroni95_low_pp=adj[0],F_family_bonferroni95_high_pp=adj[1],reference_correct_to_wrong=int((v<0).sum()),reference_wrong_to_correct=int((v>0).sum()),unchanged_correctness=int((v==0).sum()),answer_changes=sum(records[s['question_id'],ca]['text']!=records[s['question_id'],cb]['text'] for s in samples),first_token_flips=sum(records[s['question_id'],ca]['first_token']!=records[s['question_id'],cb]['first_token'] for s in samples)))
  for i,s in enumerate(samples):
   if v[i]:
    q=s['question_id'];row=dict(candidate=ca,reference=cb,question_id=q,image_sha256=s['image_sha256'],task=s['task'],question=s['question'],gold=s['gold'],candidate_answer=records[q,ca]['text'],reference_answer=records[q,cb]['text'],delta=int(v[i]))
    (recovered if v[i]>0 else harms).append(row)
 taskrows=[]
 for task in sorted({s['task'] for s in samples}):
  ix=[i for i,s in enumerate(samples) if s['task']==task]
  for j,c in enumerate(ALL):taskrows.append(dict(task=task,condition=c,n=len(ix),correct=int(points[ix,j].sum()),accuracy_percent=points[ix,j].mean()*100,delta_F_pp=(points[ix,j]-points[ix,1]).mean()*100,delta_Native_pp=(points[ix,j]-points[ix,0]).mean()*100))
 csvout('accuracy_summary.csv',summary);csvout('paired_comparisons.csv',pairs);csvout('per_question.csv',wide);csvout('generation_outputs.csv',long);csvout('harm_cases.csv',harms);csvout('recovered_cases.csv',recovered);csvout('task_accuracy.csv',taskrows);csvout('output_anomalies.csv',anomalies);csvout('checkpoint_sha256.csv',shaindex)
 best=max(CONDITIONS,key=lambda c:points[:,ALL.index(c)].sum());bp=next(p for p in pairs if p['candidate']==best and p['reference']=='F');passed=bp['delta_pp']>=3 and bp['ci_low_pp']>0
 gate=dict(best_condition=best,delta_F_pp=bp['delta_pp'],pointwise_95ci_low=bp['ci_low_pp'],pointwise_95ci_high=bp['ci_high_pp'],user_accuracy_gate_passed=bool(passed),F_family_bonferroni95_low=bp['F_family_bonferroni95_low_pp'],simultaneous_evidence_positive=bool(bp['F_family_bonferroni95_low_pp']>0),performance_test_started=False,physical_implementation_started=False,stop_after_accuracy=True,selection_is_exploratory=True)
 write(OUT/'restoration_gate.json',gate)
 write(OUT/'validation.json',dict(status='PASSED',at=now(),new_generations=5000,reused_baselines=2000,n=1000,images=918,smoke_cases=20,all_logits_finite=True,dense_no_physical_image_exit=True,cache_mrope_deepstack_mask_audited=True,first_token_B_equals_F_all1000=True,first_token_C_equals_A_all1000=True,bootstrap_replicates=20000,score_recomputed=True,source_and_input_hashes_bound=True,anomalies=len(anomalies),timing_FLOPs_and_performance_tests_run=False,conditions_added_after_freeze=False))
 plots(summary,pairs);report(summary,pairs,gate)
 print('ACCURACY',summary);print('GATE',gate)
def plots(summary,pairs):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42});folder=OUT/'figures';folder.mkdir(exist_ok=True)
 rr=[r for r in summary if r['condition'] in CONDITIONS];labels=['A: P-1 vision','B: Decode vision','C: Both vision','D: Early text attn','E: Late R writing'];x=np.arange(5);y=np.array([r['accuracy_percent'] for r in rr]);lo=np.array([r['accuracy_ci_low'] for r in rr]);hi=np.array([r['accuracy_ci_high'] for r in rr])
 fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');ax.bar(x,y,color=['#4477AA','#66AACC','#228833','#CCBB44','#AA3377'],yerr=np.vstack([y-lo,hi-y]),capsize=3);ax.axhline(58.8,color='grey',linestyle='--',label='Dense F: 58.8%');ax.axhline(69.4,color='black',linestyle=':',label='Native: 69.4%');ax.set(xticks=x,xticklabels=labels,ylabel='GQA complete-answer accuracy (%)',ylim=(0,100),title='Fixed dense restoration pathways: 1,000 questions / 918 images');ax.legend()
 for i,v in enumerate(y):ax.text(i,hi[i]+1,f'{v:.1f}%',ha='center')
 for fmt in ['png','pdf']:fig.savefig(folder/f'accuracy_restoration.{fmt}',dpi=220)
 plt.close(fig)
 rr=[p for p in pairs if p['reference']=='F'];y=np.array([p['delta_pp'] for p in rr]);lo=np.array([p['ci_low_pp'] for p in rr]);hi=np.array([p['ci_high_pp'] for p in rr]);fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');ax.errorbar(y,x,xerr=np.vstack([y-lo,hi-y]),fmt='o',capsize=4);ax.axvline(0,color='grey');ax.axvline(3,color='orange',ls='--',label='+3 pp criterion');ax.set(yticks=x,yticklabels=labels,xlabel='Accuracy difference vs Dense F (pp), exploratory 95% cluster CI');ax.legend();ax.invert_yaxis()
 for fmt in ['png','pdf']:fig.savefig(folder/f'paired_restoration.{fmt}',dpi=220)
 plt.close(fig)
def report(summary,pairs,gate):
 names=dict(A='P−1视觉回看恢复',B='Decode视觉回看恢复',C='P−1与Decode视觉回看共同恢复',D='L1–L6早层完整文本Attention恢复',E='L22–L35深层R→Image写入恢复');best=gate['best_condition'];s={r['condition']:r for r in summary};p={(r['candidate'],r['reference']):r for r in pairs}
 lines=['# Qwen3-VL C3质量损失归因与恢复：固定1000题Accuracy实验','',f'本轮最大点估计恢复来自**{best}：{names[best]}**。正确{s[best]["correct"]}/1000，Accuracy {s[best]["accuracy_percent"]:.3f}%，相对Dense F恢复{gate["delta_F_pp"]:+.3f}pp，按Image SHA整簇配对95%CI [{gate["pointwise_95ci_low"]:+.3f}, {gate["pointwise_95ci_high"]:+.3f}]。这衡量的是F背景下恢复该通路的效果，不能直接分摊原10.6pp损失。','','## 设置与约束','','严格复用上一轮同一1000道GQA val_balanced问题、918张图、原Prompt、Tokenizer、图像Processor及短答案评分转换。Native 694/1000和Dense F 588/1000经2000份Checkpoint、输入、源码与模型绑定审计后复用。本轮新增A–E共5000份完整Greedy生成，BF16/Eager、max_new_tokens=32、logits_to_keep=1、每请求独立真实Dense KV。模型revision为`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，Transformers 4.57.6。','','|条件|相对F唯一恢复项|其余约束|','|---|---|---|','|A|非跳算层P−1→Image|Decode禁图；早层输出仍置零；深层R禁图|','|B|非跳算层真实Decode→Image|P−1禁图；早层输出仍置零；深层R禁图|','|C|同时恢复A和B|早层输出仍置零；深层R禁图|','|D|L1–L6 P−1/Decode Attention输出|直接读图仍全层禁止；深层R禁图|','|E|L22–L35 Prefill R→Image|P−1/Decode仍禁图；早层输出仍置零|','','索引0-based。非跳算层为L0及L7–L35；本轮“跳算”仅指Dense功能参考的Attention输出置零，全部实际Attention、Image Hidden与KV均计算、保留，不进行物理删除。D恢复的是直接Image Key仍屏蔽时的Attention计算结果，可读取已经携带视觉信息的文本Key。MLP、Residual、Norm、mRoPE、DeepStack及非目标Mask保持正常。','','## 全部结果','','|条件|正确/1000|ACC|相对F Δpp|相对Native Δpp|','|---|---:|---:|---:|---:|']
 for r in summary:lines.append(f'|{r["condition"]}|{r["correct"]}|{r["accuracy_percent"]:.3f}%|{r["delta_F_pp"]:+.3f}|{r["delta_Native_pp"]:+.3f}|')
 lines+=['','|候选−参考|Δpp|普通配对95%CI|参考对→错|参考错→对|答案变化|首Token翻转|','|---|---:|---|---:|---:|---:|---:|']
 for r in pairs:lines.append(f'|{r["candidate"]}−{r["reference"]}|{r["delta_pp"]:+.3f}|[{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}]|{r["reference_correct_to_wrong"]}|{r["reference_wrong_to_correct"]}|{r["answer_changes"]}|{r["first_token_flips"]}|')
 lines+=['','## 统计与解释边界','','20,000次整图配对Bootstrap，seed=20261106，以918个Image SHA簇均匀有放回抽取，保留同图全部题，按题目加权。普通95%CI为探索性区间；五项相对F比较另提供Bonferroni同时95%区间（各项分位数0.005/0.995），见CSV，防止从五项中选最大值后把普通区间当作确认性证据。最佳条件按全1000题正确数、A/B/C/D/E顺序打破平局，冻结后未增删条件。','','各通路是在F其余限制仍存在时分别恢复。作用可能非加性，恢复量不能直接相加；最大点估计也不能证明其他通路完全冗余。此1000题已用于上一轮探索，不是独立新验证集。六类历史GQA采样体系未用于本轮重新分组，任务明细沿用此验证集官方types.semantic五类。','','Decode-only的B在Prefill与F完全一致，因此首Token应一致；20题Smoke核验完整首Token Logits逐位一致，1000题生成首Token ID也全部一致。A和C的Prefill同样一致。B若有答案质量变化，发生在后续Token或EOS阶段，不能解释为修改了第一个答案Token。','','## 预设停止门槛','',f'最佳新增条件相对F提升≥3pp且普通配对95%CI下界>0：**{"通过" if gate["user_accuracy_gate_passed"] else "未通过"}**。五项同时区间下界为{gate["F_family_bonferroni95_low"]:+.3f}pp。']
 if gate['user_accuracy_gate_passed']:
  plans={
   'A':'物理方案需要让P−1在L0/L7–L35访问真实视觉KV，同时保留L1–L6的真实跳算；Decode仍可使用紧凑文本KV。但要获得原生深层视觉KV，需先验证Prefill视觉Hidden退出L22与新增P−1访问的兼容性，不能直接假定原物理退出方案仍成立。',
   'B':'物理方案需要在L0/L7–L35为真实Decode保留视觉KV，早层仍可跳算。深层视觉KV来源与L22视觉Hidden退出存在兼容性要求，必须独立验证，不能沿用纯文本紧凑Cache而声称权限已恢复。',
   'C':'物理方案需要在L0/L7–L35保留P−1及Decode的真实视觉访问，L1–L6仍真实跳算。需保留所需层的视觉KV并审查L22视觉Hidden退出，不能继续使用全层无图的Decode Cache。',
   'D':'物理方案应保留L1–L6 P−1及Decode的文本Attention，取消这六层的整段Attention跳算；其Query可仅使用真实文本KV，直接Image Key仍屏蔽。其余层仍可能使用紧凑文本KV和既有视觉退出，但须先逐层验证与该Dense D参考的功能一致及最终质量。',
   'E':'物理方案必须保留L22–L35的R视觉写入计算，重新审查L22即删除Image Hidden的策略；P−1/Decode仍禁图，因此写入完成后仍可探索释放直接视觉KV。新的退出时机需先完成物理正确性与质量验证，不能在本轮宣布物理收益。'}
  lines+=['','最有效恢复通路的**待验证物理方案**：'+plans[best],'','本轮未实现上述方案，未运行E2E、FLOPs或性能测试。后续先独立验证物理正确性与质量，再另行授权测速。']
 else:lines+=['','触发停止规则。证据不足以支持进入物理恢复与性能评测，本轮保存完整负面结果并停止。未追加策略或层组合。']
 lines+=['','## 质量审计及交付','','CPU测试覆盖全部36层、Prefill/Decode、五条件的精确目标Mask以及D/F Mask相等、E仅改深层R、非目标Mask不变。GPU Smoke核验Native/No-op全部有效Logits逐位一致，新F与继承F及历史Smoke Logits严格一致、禁用实际Attention概率为0、每层完整Dense Hidden/KV、36层MLP/Norm/Residual、原mRoPE位置及DeepStack视觉特征一致、请求独立Cache与Hook清理。全部新增生成逐步检查Logits有限。','','逐题完整答案和Token、所有改善与损害案例、任务统计、SHA索引、冻结协议与验证记录保存在同目录；模型和原始图片仍位于服务器资源目录，其哈希有记录。Accuracy曲线和配对图只展示质量，不包含任何速度或FLOPs推断。']
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()

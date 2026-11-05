"""Quality-only analysis of the frozen historical Keep/All-Writer comparison."""
import csv, collections
import numpy as np
from .common import *

CONDITIONS = ['Native', 'F', 'Keep-L0-L21', 'E', 'All-Writer']

def csvout(name, rows):
 rows=list(rows)
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['no_cases'],lineterminator='\n');w.writeheader();w.writerows(rows)

def bootstrap(samples, values):
 groups=sorted({s['image_sha256'] for s in samples});index={h:i for i,h in enumerate(groups)};g=len(groups)
 counts=np.zeros(g);sums=np.zeros((g,values.shape[1]))
 for s,v in zip(samples,values):counts[index[s['image_sha256']]]+=1;sums[index[s['image_sha256']]]+=v
 rng=np.random.default_rng(SEED);draws=[]
 for _ in range(80):
  weights=rng.multinomial(g,np.full(g,1/g),size=250);draws.append((weights@sums)/(weights@counts)[:,None])
 return np.concatenate(draws)

def main():
 check_sources();manifest=read(OUT/'sample_manifest.json');samples=manifest['samples'];gate=read(OUT/'equivalence_validation.json');assert gate['reuse_D_for_Keep']
 assert len(samples)==1000 and len({s['image_sha256'] for s in samples})==918
 for rank in range(2):assert read(OUT/f'accuracy/rank{rank}/completion.json')['status']=='COMPLETE'
 original_index=list(csv.DictReader((OUT/'baseline_index.csv').open()))
 for r in original_index:assert sha(ROOT/r['path'])==r['sha256']
 records={};matrix=np.zeros((1000,len(CONDITIONS)));wide=[];long=[];index=[];anomalies=[];new_count=0
 for i,s in enumerate(samples):
  q=s['question_id'];row={k:s[k] for k in ['question_id','question','image_id','image_sha256','gold','task','rank']}
  for j,c in enumerate(CONDITIONS):
   reused=c!='All-Writer';p=baseline_path(s,'D' if c=='Keep-L0-L21' else c) if reused else OUT/f'checkpoints/rank{s["rank"]}/{q}__All-Writer.json'
   d=read(p);r=d['result'];b=d['binding'];assert b['input_sha256']==digest(manifest['input_bindings'][q]) and b['image_sha256']==s['image_sha256']
   if not reused:
    new_count+=1;assert b['source_manifest_sha256']==digest(read(OUT/'source_manifest.json'));assert b['protocol_sha256']==sha(OUT/'protocol.json')
    a=r['source_audit'];assert a['hook_cleanup'] and a['non_target_mask_exact'] and a['blocked_probability_max']==0
    assert r['writer_layers']==list(range(36)) and r['P1_Decode_image_off'] and r['hook_cleanup'] and r['all_logits_finite']
   pred=r['text'].rstrip('.').lower();score=pred==s['gold'];assert score==r['correct'];matrix[i,j]=score;records[q,c]=r
   for key in ['text','correct','eos','truncated','empty','first_token','generated_tokens']:row[c+'__'+key]=r[key]
   row[c+'__token_ids']=json.dumps(r['token_ids'])
   long.append(dict(question_id=q,image_sha256=s['image_sha256'],task=s['task'],condition=c,reused=reused,reused_condition='D' if c=='Keep-L0-L21' else c if reused else '',gold=s['gold'],answer=r['text'],prediction=pred,correct=int(score),token_ids=json.dumps(r['token_ids']),eos=r['eos'],empty=r['empty'],truncated=r['truncated'],first_token=r['first_token']))
   index.append(dict(question_id=q,condition=c,path=str(p.relative_to(ROOT)),sha256=sha(p),reused=reused))
   if r['empty'] or r['truncated'] or not r['eos']:anomalies.append(dict(question_id=q,condition=c,text=r['text'],empty=r['empty'],truncated=r['truncated'],eos=r['eos']))
  wide.append(row)
 assert new_count==1000 and list(matrix.sum(axis=0)[:4])==[694,588,585,680]
 draws=bootstrap(samples,matrix);ci=np.quantile(draws,[.025,.975],axis=0)*100;summary=[]
 for j,c in enumerate(CONDITIONS):summary.append(dict(condition=c,n=1000,correct=int(matrix[:,j].sum()),accuracy_percent=matrix[:,j].mean()*100,ci_low_percent=ci[0,j],ci_high_percent=ci[1,j],delta_Native_pp=(matrix[:,j]-matrix[:,0]).mean()*100,reused=c!='All-Writer'))
 protocol=read(OUT/'protocol.json');comparisons=[protocol['bootstrap']['primary']]+protocol['bootstrap']['secondary'];pairs=[];harms=[];recoveries=[]
 for ca,cb in comparisons:
  a,b=CONDITIONS.index(ca),CONDITIONS.index(cb);delta=matrix[:,a]-matrix[:,b];lo,hi=np.quantile((draws[:,a]-draws[:,b])*100,[.025,.975])
  pairs.append(dict(candidate=ca,reference=cb,delta_pp=delta.mean()*100,ci_low_pp=lo,ci_high_pp=hi,reference_correct_to_wrong=int((delta<0).sum()),reference_wrong_to_correct=int((delta>0).sum()),unchanged_correctness=int((delta==0).sum()),answer_changes=sum(records[s['question_id'],ca]['text']!=records[s['question_id'],cb]['text'] for s in samples),first_token_flips=sum(records[s['question_id'],ca]['first_token']!=records[s['question_id'],cb]['first_token'] for s in samples),ci_type='exploratory pointwise 95% image-cluster paired bootstrap'))
  for i,s in enumerate(samples):
   if not delta[i]:continue
   q=s['question_id'];r=dict(candidate=ca,reference=cb,question_id=q,image_sha256=s['image_sha256'],task=s['task'],question=s['question'],gold=s['gold'],candidate_answer=records[q,ca]['text'],reference_answer=records[q,cb]['text'],delta=int(delta[i]));(recoveries if delta[i]>0 else harms).append(r)
 tasks=[]
 for task in sorted({s['task'] for s in samples}):
  ix=[i for i,s in enumerate(samples) if s['task']==task]
  for j,c in enumerate(CONDITIONS):tasks.append(dict(task=task,condition=c,n=len(ix),correct=int(matrix[ix,j].sum()),accuracy_percent=matrix[ix,j].mean()*100,delta_Keep_pp=(matrix[ix,j]-matrix[ix,2]).mean()*100))
 smoke_steps=[]
 for s in samples:
  if s['question_id'] not in manifest['smoke_question_ids']:continue
  d=read(OUT/f'smoke/rank{s["rank"]}/{s["question_id"]}.json');assert d['comparison']['strict_equal'] and all(d['checks'].values())
  for t in d['comparison']['steps']:smoke_steps.append(dict(question_id=s['question_id'],rank=s['rank'],**t))
 for name,rs in [('accuracy_summary.csv',summary),('paired_comparisons.csv',pairs),('per_question.csv',wide),('generation_outputs.csv',long),('checkpoint_sha256.csv',index),('harm_cases.csv',harms),('recovered_cases.csv',recoveries),('task_accuracy.csv',tasks),('output_anomalies.csv',anomalies),('smoke_logit_comparison.csv',smoke_steps)]:csvout(name,rs)
 write(OUT/'validation.json',dict(status='PASSED',at=now(),source_manifest_verified=True,original_baseline_files_unchanged=4000,new_accuracy_generations=1000,reused_Keep_D=1000,reused_reference_generations=3000,questions=1000,images=918,smoke_cases=20,strict_Keep_D_logit_vectors=len(smoke_steps),strict_Keep_D_all_valid_logits_equal=True,original_D_smoke_logits_reproduced=True,Native_noop_strict=True,mask_probability_zero=True,non_target_mask_exact=True,all_new_writer_layers=list(range(36)),all_P1_Decode_Image_access_off=True,early_attention_normal=True,independent_cache_storage_smoke=True,hooks_clean=True,new_all_logits_finite=True,anomalies=len(anomalies),bootstrap_repetitions=20000,performance_or_FLOPs_tests_run=False))
 plots(summary,pairs);report(summary,pairs,smoke_steps,anomalies)
 print('ACCURACY',json.dumps(summary));print('PAIRS',json.dumps(pairs))

def plots(summary,pairs):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42});folder=OUT/'figures';folder.mkdir(exist_ok=True)
 x=np.arange(len(summary));y=np.array([r['accuracy_percent'] for r in summary]);lo=np.array([r['ci_low_percent'] for r in summary]);hi=np.array([r['ci_high_percent'] for r in summary])
 fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');ax.bar(x,y,color=['#777777','#AA4499','#4477AA','#CCBB44','#228833'],yerr=np.vstack([y-lo,hi-y]),capsize=4);ax.set(xticks=x,xticklabels=['Native','Dense F','Keep L0-21 (= D)','E','All-Writer'],ylabel='GQA complete-answer accuracy (%)',ylim=(0,85),title='Same frozen 1,000 GQA questions / 918 image clusters')
 for i,r in enumerate(summary):ax.text(i,hi[i]+1,f'{r["correct"]}/1000\n{y[i]:.1f}%',ha='center',fontsize=9)
 for fmt in ['png','pdf']:fig.savefig(folder/f'accuracy_comparison.{fmt}',dpi=220)
 plt.close(fig)
 y=np.arange(len(pairs));v=np.array([p['delta_pp'] for p in pairs]);lo=np.array([p['ci_low_pp'] for p in pairs]);hi=np.array([p['ci_high_pp'] for p in pairs])
 fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');ax.errorbar(v,y,xerr=np.vstack([v-lo,hi-v]),fmt='o',capsize=4,color='#4477AA');ax.axvline(0,color='gray',ls='--');ax.set(yticks=y,yticklabels=[f'{p["candidate"]} minus {p["reference"]}' for p in pairs],xlabel='Accuracy difference (pp); exploratory pointwise 95% cluster CI');ax.invert_yaxis()
 for fmt in ['png','pdf']:fig.savefig(folder/f'paired_differences.{fmt}',dpi=220)
 plt.close(fig)

def report(summary,pairs,steps,anomalies):
 s={r['condition']:r for r in summary};p=pairs[0];old=read(OUT/'historical_cohort_audit.json')
 lines=['# Qwen3-VL历史Keep-L0–L21：相同1000题复现与All-Writer质量对照','',f'**历史Keep与最新D通过严格复核；All-Writer为{s["All-Writer"]["correct"]}/1000（{s["All-Writer"]["accuracy_percent"]:.3f}%），Keep为585/1000（58.500%）。All-Writer−Keep为{p["delta_pp"]:+.3f}pp，图片整簇配对探索性95%CI [{p["ci_low_pp"]:+.3f}, {p["ci_high_pp"]:+.3f}]。**','','## 历史Keep是否等价于最新D？','',f'两者均保留L1–L6 Attention正常执行；P−1及所有真实Decode在全部36层禁止读取原始Image Key；Prefill R仅L0–L21可读图。直接调用未修改的历史`qwen3vl_keep_l0_21.engine.generate`，20道冻结Smoke的完整Token序列相同，全部{len(steps)}份有效生成Logits均严格`torch.equal`，逐向量Mean/Max Abs差异为0。最新D同时逐步复现原历史Smoke Logits SHA与输出Token。Native/历史No-op也全部逐位一致。','', '所有1000题的原始输入、图片、mRoPE与角色位置经CPU重建核对；36层Mask规则CPU逐项相等。依照预设复用门槛，Keep准确率与完整答案复用已完成D，不重新推理1000题Keep。严格逐位运行复核的直接范围是20题，其余980题依据输入、协议及控制规则等价审计复用，不宣称已逐位重测全1000题。','','## 冻结协议与样本','','基于`c2ab2892994a39149f7c3203241fa6f6e9ceb341`，执行源码冻结于`d27497db9a5091fb87830645bab6f7a2b908bfd6`。模型Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`，BF16/Eager，Transformers 4.57.6，Greedy、max_new_tokens=32、logits_to_keep=1，每请求独立完整真实KV。','', '原样复用最新1000题GQA val_balanced，918个图像SHA簇；Manifest SHA256为`bb09f894a04f111f430841f4d074ff0d770585956a04c5401436351e235d4a88`。R为最后一个Image Token之后至P−1之前的位置，包含原生vision_end标记。只做Softmax前Hard Mask；Image Query、非目标连接、MLP、Residual、Norm、mRoPE、DeepStack、QKV与完整图像KV不变。','', '沿用此1000题历史评分器：原始`tokenizer.decode(..., skip_special_tokens=True)`后执行`rstrip(".").lower()`，与Gold严格比对；不额外strip。历史生成包装器的strip后文本另字段保存，不用它改写本次评分。模型/原始图片保留在服务器资源目录，Git中保存其绑定和审计。','','## 全部质量结果','','|条件|R读图层|早层P−1/Decode Attention|正确/1000|ACC|相对Native Δpp|','|---|---|---|---:|---:|---:|']
 roles={'Native':('全部','正常；答案直接读图开放'),'F':('L0–L21','L1–L6输出置零'),'Keep-L0-L21':('L0–L21','正常'),'E':('L0–L35','L1–L6输出置零'),'All-Writer':('L0–L35','正常')}
 for r in summary:
  c=r['condition'];lines.append(f'|{c}|{roles[c][0]}|{roles[c][1]}|{r["correct"]}|{r["accuracy_percent"]:.3f}%|{r["delta_Native_pp"]:+.3f}|')
 lines+=['','Native之外四条件的P−1/Decode均在全部36层直接禁图。F/E中的输出置零是继承的Dense功能对照，不属于本轮新增Keep/All-Writer；本轮没有物理删除、跳算或新消融条件。Native、F、E及D共4000份原Checkpoint经SHA、输入、评分、源码绑定审计后复用，新增All-Writer共1000份生成。','','|候选−参考|Δpp|普通配对95%CI|参考对→错|参考错→对|答案变化|首Token翻转|','|---|---:|---|---:|---:|---:|---:|']
 for r in pairs:lines.append(f'|{r["candidate"]}−{r["reference"]}|{r["delta_pp"]:+.3f}|[{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}]|{r["reference_correct_to_wrong"]}|{r["reference_wrong_to_correct"]}|{r["answer_changes"]}|{r["first_token_flips"]}|')
 lines+=['','## 关闭深层R视觉写入损失多少？','',f'在早层Attention正常、答案Query全层直接禁图的固定背景下，将R读图从L0–L35缩为L0–L21，Accuracy变化为{-p["delta_pp"]:+.3f}pp（Keep−All-Writer），对应95%CI [{-p["ci_high_pp"]:+.3f}, {-p["ci_low_pp"]:+.3f}]。恢复深层R写入使{p["reference_wrong_to_correct"]}题由错变对，同时使{p["reference_correct_to_wrong"]}题由对变错。该比较定位的是此背景下深层视觉写入权限的总干预效果，不说明每个深层均必要，也不能等同于实际加速收益。','', 'F→Keep仅取消早层输出置零；E→All-Writer也仅取消早层输出置零。两个对照有助于区分早层Attention恢复与深层R写入恢复，不应把不同背景下的效应直接相加解释为独立因果份额。', '', f'本轮All-Writer比E低{abs(s["All-Writer"]["accuracy_percent"]-s["E"]["accuracy_percent"]):.3f}pp，Keep比F低{abs(s["Keep-L0-L21"]["accuracy_percent"]-s["F"]["accuracy_percent"]):.3f}pp；这两项普通配对区间均跨0，不能声称早层Attention恢复带来质量收益。相反，关闭深层R视觉写入在本轮同题控制下造成明显损害。All-Writer仍比Native低{abs(s["All-Writer"]["delta_Native_pp"]):.3f}pp，不能把恢复全部R写入等同于恢复Native。', '', '两种早层Attention状态下的晚层写入收益点估计在本批均为9.2pp，这是本批探索观察，不据此断言效应在其他数据上完全可加或没有交互。','','## 为什么不能套用历史3000题分数？','',f'历史Keep/All-Writer的87.5%/87.9%来自另一份六任务均衡3000题（2409张图），本轮是官方val_balanced冻结1000题，官方types.semantic分为5类。两组仅重合{old["shared_question_ids"]}个Question ID，本轮{old["current_questions_with_historical_images"]}题的图片出现在旧3000中。此前87.5%不能作为本轮同题Keep分数，也不能据此认定代码复现失败。本轮样本已用于前轮探索，不是独立确认集。','','## 统计与完整性','','20,000次Image SHA整簇配对Bootstrap，seed=20261106；每次有放回抽取918个图片簇，保留同图所有问题，以题数加权。全部区间为预设比较的普通探索性95%CI，未作确认性多重检验结论，CI跨0不代表无损或非劣效。任务表保留官方五类，不混入历史六任务标签。','',f'20题GPU Smoke验证禁用实际Attention概率严格为0、非目标Mask不变、P−1/Decode全层禁图、原生位置与DeepStack/MLP保持、Cache独立、Hook清理。全部1000题新增生成逐步检查Logits有限，完整保存Token、EOS、截断、空答案和异常；异常记录共{len(anomalies)}条（包含复用参考，见output_anomalies.csv）。','', '本轮只做质量，没有运行测速、FLOPs统计、Kernel搜索或物理优化。图表仅展示准确率和配对质量差异。生成、检查和统计结束后释放两卡，提交独立分支并停止。']
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()

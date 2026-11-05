"""Fixed-cutoff quality statistics; no performance measurement."""
import csv,collections
import numpy as np
from .common import *
from ..qwen_keep1000.analyze import bootstrap
ALL=['Native','F','Keep-L0-L21','Keep-L0-L24','Keep-L0-L26','E','All-Writer']

def csvout(name,rows):
 rows=list(rows)
 with (OUT/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['no_cases'],lineterminator='\n');writer.writeheader();writer.writerows(rows)

def main():
 check_sources();m=read(OUT/'sample_manifest.json');samples=m['samples'];p=read(OUT/'protocol.json');records={};matrix=np.zeros((1000,len(ALL)));wide=[];long=[];index=[];anomalies=[]
 for rank in range(2):assert read(OUT/f'accuracy/rank{rank}/completion.json')['status']=='COMPLETE'
 for i,s in enumerate(samples):
  q=s['question_id'];row={k:s[k] for k in ['question_id','question','image_id','image_sha256','gold','task','rank']}
  for j,c in enumerate(ALL):
   reused=c not in CONDITIONS;path=reference_path(s,c) if reused else OUT/f'checkpoints/rank{s["rank"]}/{q}__{c}.json';d=read(path);r=d['result'];bind=d['binding']
   assert bind['input_sha256']==digest(m['input_bindings'][q]) and bind['image_sha256']==s['image_sha256']
   if not reused:
    assert bind['source_manifest_sha256']==digest(read(OUT/'source_manifest.json')) and bind['protocol_sha256']==sha(OUT/'protocol.json')
    assert r['writer_layers']==list(range(CONDITIONS[c]+1)) and r['P1_Decode_image_off'] and r['source_audit']['blocked_probability_max']==0 and r['source_audit']['hook_cleanup'] and r['all_logits_finite']
   score=r['text'].rstrip('.').lower()==s['gold'];assert score==r['correct'];matrix[i,j]=score;records[q,c]=r
   for key in ['text','correct','eos','truncated','empty','first_token','generated_tokens']:row[c+'__'+key]=r[key]
   row[c+'__token_ids']=json.dumps(r['token_ids'])
   long.append(dict(question_id=q,image_sha256=s['image_sha256'],condition=c,reused=reused,gold=s['gold'],answer=r['text'],correct=int(score),token_ids=json.dumps(r['token_ids']),eos=r['eos'],truncated=r['truncated'],empty=r['empty'],first_token=r['first_token']))
   index.append(dict(question_id=q,condition=c,path=str(path.relative_to(ROOT)),sha256=sha(path),reused=reused))
   if not r['eos'] or r['truncated'] or r['empty']:anomalies.append(dict(question_id=q,condition=c,text=r['text'],eos=r['eos'],truncated=r['truncated'],empty=r['empty']))
  wide.append(row)
 assert len(samples)==1000 and list(matrix.sum(0)[[0,1,2,5,6]])==[694,588,585,680,677]
 draws=bootstrap(samples,matrix);bounds=np.quantile(draws,[.025,.975],axis=0)*100;summary=[]
 for j,c in enumerate(ALL):summary.append(dict(condition=c,n=1000,correct=int(matrix[:,j].sum()),accuracy_percent=matrix[:,j].mean()*100,accuracy_ci_low=bounds[0,j],accuracy_ci_high=bounds[1,j],delta_Keep21_pp=(matrix[:,j]-matrix[:,2]).mean()*100,delta_All_Writer_pp=(matrix[:,j]-matrix[:,-1]).mean()*100,reused=c not in CONDITIONS))
 pairs=[];harm=[];recover=[]
 for ca,cb in p['comparisons']:
  a,b=ALL.index(ca),ALL.index(cb);delta=matrix[:,a]-matrix[:,b];ci=np.quantile((draws[:,a]-draws[:,b])*100,[.025,.975])
  pairs.append(dict(candidate=ca,reference=cb,delta_pp=delta.mean()*100,ci_low_pp=ci[0],ci_high_pp=ci[1],reference_correct_to_wrong=int((delta<0).sum()),reference_wrong_to_correct=int((delta>0).sum()),unchanged_correctness=int((delta==0).sum()),answer_changes=sum(records[s['question_id'],ca]['text']!=records[s['question_id'],cb]['text'] for s in samples),first_token_flips=sum(records[s['question_id'],ca]['first_token']!=records[s['question_id'],cb]['first_token'] for s in samples)))
  for i,s in enumerate(samples):
   if not delta[i]:continue
   q=s['question_id'];r=dict(candidate=ca,reference=cb,question_id=q,image_sha256=s['image_sha256'],task=s['task'],question=s['question'],gold=s['gold'],candidate_answer=records[q,ca]['text'],reference_answer=records[q,cb]['text'],delta=int(delta[i]));(recover if delta[i]>0 else harm).append(r)
 taskrows=[]
 for task in sorted({s['task'] for s in samples}):
  ix=[i for i,s in enumerate(samples) if s['task']==task]
  for j,c in enumerate(ALL):taskrows.append(dict(task=task,condition=c,n=len(ix),correct=int(matrix[ix,j].sum()),accuracy_percent=matrix[ix,j].mean()*100,delta_Keep21_pp=(matrix[ix,j]-matrix[ix,2]).mean()*100))
 for name,rows in [('accuracy_summary.csv',summary),('paired_comparisons.csv',pairs),('per_question.csv',wide),('generation_outputs.csv',long),('checkpoint_sha256.csv',index),('harm_cases.csv',harm),('recovered_cases.csv',recover),('task_accuracy.csv',taskrows),('output_anomalies.csv',anomalies)]:csvout(name,rows)
 write(OUT/'validation.json',dict(status='PASSED',at=now(),new_generations=2000,reused_generations=5000,questions=1000,image_clusters=918,smoke_questions=4,source_input_protocol_hashes_verified=True,all_new_logits_finite=True,actual_prohibited_probability_zero=True,non_target_mask_native=True,anomalies=len(anomalies),bootstrap_repetitions=20000,performance_tests_run=False))
 plot(summary);report(summary,pairs,anomalies);print('SCORES',json.dumps(summary));print('PAIRS',json.dumps(pairs))

def plot(summary):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'pdf.fonttype':42,'font.size':10});folder=OUT/'figures';folder.mkdir(exist_ok=True)
 names=['Keep-L0-L21','Keep-L0-L24','Keep-L0-L26','All-Writer'];rr=[next(r for r in summary if r['condition']==c) for c in names];y=np.array([r['accuracy_percent'] for r in rr]);lo=np.array([r['accuracy_ci_low'] for r in rr]);hi=np.array([r['accuracy_ci_high'] for r in rr])
 fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');ax.bar(range(4),y,color=['#4477AA','#66AACC','#228833','#AA3377'],yerr=np.vstack([y-lo,hi-y]),capsize=4);ax.axhline(69.4,color='black',ls=':',label='Native: 69.4%');ax.set(xticks=range(4),xticklabels=['Keep L0-21\n22 layers','Keep L0-24\n25 layers','Keep L0-26\n27 layers','All-Writer\n36 layers'],ylabel='GQA complete-answer accuracy (%)',ylim=(0,82),title='Fixed 1,000 questions: inclusive writer-layer cutoffs');ax.legend(loc='upper left')
 for i,r in enumerate(rr):ax.text(i,hi[i]+1,f'{r["correct"]}/1000\n{y[i]:.1f}%',ha='center')
 for fmt in ['png','pdf']:fig.savefig(folder/f'accuracy_cutoffs.{fmt}',dpi=220)
 plt.close(fig)

def report(summary,pairs,anomalies):
 lines=['# Qwen3-VL Keep-L0–L24 / L26：固定1000题快速质量验证','','在上一轮完全相同的1000道GQA val_balanced问题（918个Image SHA簇）上新增两个固定条件。所有层编号0-based、包含端点：Keep-L0–L24保留25层R读图；Keep-L0–L26保留27层。P−1与全部真实Decode始终在36层禁读原始Image Key；L1–L6 Attention正常。','','模型为Qwen3-VL-8B-Instruct，revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`；BF16/Eager、Transformers 4.57.6、原Prompt/Processor、Greedy、max_new_tokens=32、logits_to_keep=1，每请求独立完整真实KV。只做Softmax前Hard Mask，Image Query及其余Attention、QKV、MLP、Residual、mRoPE、DeepStack与KV均保持原生。不进行物理删除、测速或新增搜索。','','## Accuracy','','|条件|正确/1000|Accuracy|相对Keep-L0–L21 Δpp|相对All-Writer Δpp|','|---|---:|---:|---:|---:|']
 for r in summary:lines.append(f'|{r["condition"]}|{r["correct"]}|{r["accuracy_percent"]:.3f}%|{r["delta_Keep21_pp"]:+.3f}|{r["delta_All_Writer_pp"]:+.3f}|')
 lines+=['','Native、F、Keep-L0–L21、E、All-Writer均复用上一轮经审计记录，本轮新增2×1000份完整生成。F/E的L1–L6 P−1/Decode输出置零仅为历史对照；两个新Keep条件不包含这一干预。','','## 预设配对比较','','|候选−参考|Δpp|普通95%CI|参考对→错|参考错→对|答案变化|首Token翻转|','|---|---:|---|---:|---:|---:|---:|']
 for r in pairs:lines.append(f'|{r["candidate"]}−{r["reference"]}|{r["delta_pp"]:+.3f}|[{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}]|{r["reference_correct_to_wrong"]}|{r["reference_wrong_to_correct"]}|{r["answer_changes"]}|{r["first_token_flips"]}|')
 lines+=['','20,000次Image SHA整簇配对Bootstrap，seed=20261106，整簇重抽后按题数加权。区间是普通探索性95%CI，未作确认性多重检验；CI跨0不能解释为非劣效或无损。本1000题已用于研究开发，不是新独立确认集。保留官方types.semantic五类任务分组。','','## 正确性与范围','','5项CPU测试覆盖精确层端点、36层Prefill目标Mask、Decode全层禁图、非目标连接不变、样本及图像分片，以及新增层差集。两卡各选此前冻结Smoke中的最短/最长输入，共4题；通用Keep21完整Token及全部有效Logits严格复现历史Keep，Native/No-op有效Logits严格torch.equal，两个新条件禁止实际概率严格0、独立Cache存储与Hook清理通过。2000份正式新增生成均检查Logits有限和目标Mask数量。','',f'所有新旧答案、完整Token、评分、EOS、截断、空输出、首Token变化及全部损害/恢复案例均保留。共{len(anomalies)}条输出异常，详见output_anomalies.csv。评分严格沿用原始解码文本的rstrip(".").lower()规则，不额外strip。','', '数据与源码基于`dd464a691c30a0e047ddfb59177ec6a8ad2a40be`；新执行源码冻结于`a5057c4fd6bca2b4cade5258749efeaa6060f11b`。历史记录、main不修改。所有结果完成后释放两卡并停止，不扩展其他层集合。']
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()

"""Official soft answer accuracy and three frozen paired quality contrasts."""
import csv,collections
import numpy as np
from .common import *
LABELS=['N','A','B','Historical-C3-V2']

def csvout(name,rows):
 rows=list(rows)
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['no_cases'],lineterminator='\n');w.writeheader();w.writerows(rows)

def cluster_bootstrap(samples,values):
 groups=sorted({s['image_sha256'] for s in samples});g=len(groups);ix={h:i for i,h in enumerate(groups)};sums=np.zeros((g,values.shape[1]));counts=np.zeros(g)
 for s,v in zip(samples,values):sums[ix[s['image_sha256']]]+=v;counts[ix[s['image_sha256']]]+=1
 rng=np.random.default_rng(SEED);draws=[]
 for _ in range(80):
  weight=rng.multinomial(g,np.full(g,1/g),size=250);draws.append((weight@sums)/(weight@counts)[:,None])
 return np.concatenate(draws)

def main():
 source=verify_sources();manifest=read(OUT/'sample_manifest.json');keys=sorted(manifest);samples=[manifest[k] for k in keys];part=read(OUT/'partition.json');bindings=read(OUT/'input_bindings.json');records={};matrix=np.zeros((500,4));wide=[];long=[];index=[];anomalies=[]
 for rank in range(2):assert read(OUT/f'accuracy/rank{rank}/completion.json')['status']=='COMPLETE'
 for i,key in enumerate(keys):
  s=manifest[key];row=dict(key=key,question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],question=s['question'],prompt=s['prompt_text'],official_answers=json.dumps(s['answer'],ensure_ascii=False))
  for j,c in enumerate(LABELS):
   reused=c in ['N','Historical-C3-V2'];path=OUT/(f'baselines/{key}/{c}.json' if reused else f'checkpoints/{key}/{c}.json');d=read(path);r=d['generation'];score=float(textvqa_score(r['text'],s['answer']));assert abs(score-d['score'])<1e-12
   if not reused:
    b=d['binding'];assert b['sample_sha256']==digest(s) and b['input_sha256']==digest(bindings[key]) and b['image_sha256']==s['image_sha256']
    assert b['source_manifest_sha256']==source and b['protocol_sha256']==sha(OUT/'protocol.json') and b['rank']==part['image_shard'][s['image_sha256']]
    assert r['audit']['blocked_probability_max']==0 and r['all_logits_finite'] and r['audit']['all32_R_image_open']
   matrix[i,j]=score;records[key,c]=r;first=r['token_ids'][0]
   for field in ['text','token_ids','eos','truncated','empty','repeated']:row[c+'__'+field]=json.dumps(r[field],ensure_ascii=False) if field=='token_ids' else r[field]
   row[c+'__score']=score;row[c+'__first_token']=first
   long.append(dict(key=key,question_id=s['question_id'],image_sha256=s['image_sha256'],condition=c,reused=reused,answer=r['text'],soft_score=score,token_ids=json.dumps(r['token_ids']),first_token=first,first_logits_sha256=r['first_logits_sha256'],eos=r['eos'],truncated=r['truncated'],empty=r['empty'],repeated=r['repeated']))
   index.append(dict(key=key,condition=c,path=str(path.relative_to(ROOT)),sha256=sha(path),reused=reused))
   if not r['eos'] or r['truncated'] or r['empty']:anomalies.append(dict(key=key,condition=c,text=r['text'],eos=r['eos'],truncated=r['truncated'],empty=r['empty']))
  wide.append(row)
 assert abs(matrix[:,0].mean()*100-49.68)<1e-9 and abs(matrix[:,-1].mean()*100-22.34)<1e-9
 draws=cluster_bootstrap(samples,matrix);bounds=np.quantile(draws,[.025,.975],axis=0)*100;summary=[]
 for j,c in enumerate(LABELS):
  delta=matrix[:,j]-matrix[:,0]
  summary.append(dict(condition=c,n=500,sum_soft_score=matrix[:,j].sum(),accuracy_percent=matrix[:,j].mean()*100,accuracy_ci_low_percent=bounds[0,j],accuracy_ci_high_percent=bounds[1,j],delta_N_pp=delta.mean()*100,score_up_vs_N=int((delta>1e-12).sum()),score_down_vs_N=int((delta< -1e-12).sum()),score_same_vs_N=int((np.abs(delta)<=1e-12).sum()),token_sequence_changes_vs_N=sum(records[k,c]['token_ids']!=records[k,'N']['token_ids'] for k in keys),answer_text_changes_vs_N=sum(records[k,c]['text']!=records[k,'N']['text'] for k in keys),first_token_changes_vs_N=sum(records[k,c]['token_ids'][0]!=records[k,'N']['token_ids'][0] for k in keys),eos=sum(records[k,c]['eos'] for k in keys),truncated=sum(records[k,c]['truncated'] for k in keys),empty=sum(records[k,c]['empty'] for k in keys),repeated=sum(records[k,c]['repeated'] for k in keys),reused=c in ['N','Historical-C3-V2']))
 pairs=[];changes=[]
 for ca,cb in read(OUT/'protocol.json')['primary_comparisons']:
  a,b=LABELS.index(ca),LABELS.index(cb);v=matrix[:,a]-matrix[:,b];ci=np.quantile((draws[:,a]-draws[:,b])*100,[.025,.975])
  pairs.append(dict(candidate=ca,reference=cb,delta_pp=v.mean()*100,ci_low_pp=ci[0],ci_high_pp=ci[1],candidate_score_higher=int((v>1e-12).sum()),candidate_score_lower=int((v< -1e-12).sum()),score_unchanged=int((np.abs(v)<=1e-12).sum()),complete_token_sequence_changes=sum(records[k,ca]['token_ids']!=records[k,cb]['token_ids'] for k in keys),answer_text_changes=sum(records[k,ca]['text']!=records[k,cb]['text'] for k in keys),first_token_changes=sum(records[k,ca]['token_ids'][0]!=records[k,cb]['token_ids'][0] for k in keys)))
  for i,key in enumerate(keys):
   if abs(v[i])<=1e-12:continue
   s=manifest[key];changes.append(dict(candidate=ca,reference=cb,key=key,question_id=s['question_id'],image_sha256=s['image_sha256'],question=s['question'],official_answers=json.dumps(s['answer'],ensure_ascii=False),candidate_answer=records[key,ca]['text'],reference_answer=records[key,cb]['text'],candidate_score=matrix[i,a],reference_score=matrix[i,b],score_delta=v[i]))
 negatives=[]
 for i,key in enumerate(keys):
  for c in ['A','B']:
   j=LABELS.index(c)
   if matrix[i,j]<matrix[i,0]-1e-12:
    s=manifest[key];negatives.append(dict(key=key,question_id=s['question_id'],image_sha256=s['image_sha256'],question=s['question'],condition=c,official_answers=json.dumps(s['answer'],ensure_ascii=False),Native_answer=records[key,'N']['text'],candidate_answer=records[key,c]['text'],Native_score=matrix[i,0],candidate_score=matrix[i,j],delta=matrix[i,j]-matrix[i,0]))
 for name,rows in [('accuracy_summary.csv',summary),('paired_comparisons.csv',pairs),('per_question.csv',wide),('generation_outputs.csv',long),('checkpoint_sha256.csv',index),('paired_score_changes.csv',changes),('negative_cases.csv',negatives),('output_anomalies.csv',anomalies)]:csvout(name,rows)
 atomic(OUT/'validation.json',dict(status='PASSED',at=now(),questions=500,image_clusters=309,new_dense_generations=1000,reused_N=500,historical_C3_reference=500,all_soft_scores_recomputed=True,all_input_source_and_protocol_bindings_verified=True,all_logits_finite=True,prohibited_probability_zero=True,all32_R_visual_writing_retained=True,all_MLP_norm_and_dense_KV_preserved=True,smoke_N_A_B_prefill_prefix_hidden_strict=True,native_noop_valid_logits_strict=True,anomalies=len(anomalies),bootstrap_repetitions=20000,performance_tests_run=False))
 plot(summary,pairs);report(summary,pairs,anomalies);print('ACCURACY',json.dumps(summary));print('PAIRS',json.dumps(pairs))

def plot(summary,pairs):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42});folder=OUT/'figures';folder.mkdir(exist_ok=True);y=np.array([r['accuracy_percent'] for r in summary]);lo=np.array([r['accuracy_ci_low_percent'] for r in summary]);hi=np.array([r['accuracy_ci_high_percent'] for r in summary])
 fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained');ax.bar(range(4),y,color=['#777777','#4477AA','#228833','#CC9977'],yerr=np.vstack([y-lo,hi-y]),capsize=4);ax.set(xticks=range(4),xticklabels=['N: Native','A: All-Writer\nAnswer-NoImage','B: All-Writer\nC3-Answer','Historical physical\nC3-V2 (different writing)'],ylabel='Official TextVQA soft accuracy (%)',ylim=(0,65),title='Same frozen TextVQA500 / 309 image clusters')
 for i,v in enumerate(y):ax.text(i,hi[i]+1,f'{v:.2f}%',ha='center')
 for fmt in ['png','pdf']:fig.savefig(folder/f'accuracy_comparison.{fmt}',dpi=220)
 plt.close(fig)
 y=np.arange(3);v=np.array([p['delta_pp'] for p in pairs]);lo=np.array([p['ci_low_pp'] for p in pairs]);hi=np.array([p['ci_high_pp'] for p in pairs]);fig,ax=plt.subplots(figsize=(8,3.4),layout='constrained');ax.errorbar(v,y,xerr=np.vstack([v-lo,hi-v]),fmt='o',capsize=4);ax.axvline(0,color='gray',ls='--');ax.set(yticks=y,yticklabels=['N minus A: answer image access','A minus B: early answer attention','N minus B: combined difference'],xlabel='Soft accuracy difference (pp); exploratory pointwise 95% cluster CI');ax.invert_yaxis()
 for fmt in ['png','pdf']:fig.savefig(folder/f'paired_differences.{fmt}',dpi=220)
 plt.close(fig)

def report(summary,pairs,anomalies):
 s={r['condition']:r for r in summary};p={(r['candidate'],r['reference']):r for r in pairs};na=p['N','A'];ab=p['A','B'];nb=p['N','B']
 mechanism='答案阶段直接禁图' if na['delta_pp']>ab['delta_pp'] else '在答案禁图背景下再关闭L2–L5答案Attention'
 lines=['# LLaVA TextVQA500：全层视觉写入与Dense C3式答案读取','',f'**N={s["N"]["accuracy_percent"]:.2f}%，A={s["A"]["accuracy_percent"]:.2f}%，B={s["B"]["accuracy_percent"]:.2f}%。** 在本批固定的顺序干预中，较大的净质量差来自**{mechanism}**：N−A={na["delta_pp"]:+.2f}pp，而A−B={ab["delta_pp"]:+.2f}pp。下面同时报告图片簇配对区间及所有负面结果，不将点估计当作普遍因果份额。','','## 冻结样本、协议与复用','','基于Commit `5627bc822a7827ae11edfaec84c31f6b73398e25`，原样复用c3v2_middle_visual_reaccess的500题、309个图片SHA簇，无新增OCR输入。所有Question ID、十个官方答案、原Prompt、完整Token、FP16图片Tensor及原图片SHA已在CPU逐题重建并与历史Native和原C3检查点输入绑定严格一致。','', '模型LLaVA-1.5-7B revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`，FP16/Eager、Transformers 4.49.0、Greedy、max_new_tokens=32、logits_to_keep=1，原generation_config EOS；每请求独立完整Native DynamicCache，所有32层保留Image Hidden和KV。评分调用已固定SHA的官方TextVQA十答案留一共识软评分器。没有重新抽题、筛除异常或修改Prompt。','','|条件|全部32层R→Image|P−1/真实Decode→Image|L2–L5答案Attention输出|执行图|','|---|---|---|---|---|','|N|开放|全层开放|正常|Native，审计复用|','|A|开放|全层禁止|正常|Dense完整计算|','|B|开放|全层禁止|仅答案位置置零|Dense完整计算|','|历史C3-V2|仅L0–L19开放|全层禁止|物理跳算|L20退出Image Hidden，压缩KV|','','层索引0-based。R为最后一个Image Token之后至原始P−1之前，包含问题及原模板。B不跳过L2–L5之前的任何Prefill Query，不直接修改R/Image Query输出；仅将o_proj的原始P−1行及每个真实Decode行置零。全部MLP、Norm、Residual、RoPE、QKV及完整KV更新仍正常。','', '未找到严格匹配的历史完整500题A/B条件。旧全层All-Writer的300题及500题物理C3/Middle结果无法替代本轮A/B。Native 49.68%和原物理C3-V2 22.34%的1000份历史记录经归档逐文件SHA、原Payload摘要、模型权重、源码和输入绑定复核后保留；Native另经六题真实Smoke完整Token及首Token Logits哈希严格复现。新增A/B各500题，Smoke不计入正式矩阵重复评分。原始历史归档仍位于继承目录，派生baselines仅保留质量字段及完整溯源，不把旧计时当作本轮测试。','','## 完整答案软Accuracy','','|条件|软分总和/500|Accuracy|相对N Δpp|相对N软分升/降/不变题数|','|---|---:|---:|---:|---|']
 for r in summary:lines.append(f'|{r["condition"]}|{r["sum_soft_score"]:.3f}/500|{r["accuracy_percent"]:.3f}%|{r["delta_N_pp"]:+.3f}|{r["score_up_vs_N"]}/{r["score_down_vs_N"]}/{r["score_same_vs_N"]}|')
 lines+=['','软分是十答案共识分，不是二值正确题数；不能把分数上升/下降称为对错翻转。原版C3-V2单列为历史参考，其不同的写入限制和物理路径不能与B混同。','','|比较|差值pp|普通配对95%CI|候选软分更高/更低/相同|完整Token序列变化|文本答案变化|首Token变化|','|---|---:|---|---|---:|---:|---:|']
 for r in pairs:lines.append(f'|{r["candidate"]}−{r["reference"]}|{r["delta_pp"]:+.3f}|[{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}]|{r["candidate_score_higher"]}/{r["candidate_score_lower"]}/{r["score_unchanged"]}|{r["complete_token_sequence_changes"]}|{r["answer_text_changes"]}|{r["first_token_changes"]}|')
 lines+=['','## 质量损失归因','',f'保留全部32层R视觉写入后，直接禁止答案位置读图的N−A为{na["delta_pp"]:+.3f}pp，95%CI [{na["ci_low_pp"]:+.3f}, {na["ci_high_pp"]:+.3f}]。在此禁图背景下，再把L2–L5答案Attention输出置零的A−B为{ab["delta_pp"]:+.3f}pp，95%CI [{ab["ci_low_pp"]:+.3f}, {ab["ci_high_pp"]:+.3f}]。合计N−B为{nb["delta_pp"]:+.3f}pp。','', f'**直接回答：在本批TextVQA500中，即使全部32层R写入保持原生，也不能保留Native质量；主要观察损失来自P−1/Decode直接禁图。** B比A的点估计为{s["B"]["accuracy_percent"]-s["A"]["accuracy_percent"]:+.3f}pp，A−B区间跨0，本批未显示再置零L2–L5答案Attention的明确额外净损失。这不构成早层跳算严格无损、有效提升或非劣效证明。','', '这是固定顺序下的两个条件效应：A−B仅衡量已经禁图时的早层答案Attention输出作用。本轮没有新增“答案可读图但早层置零”的第四个条件，因此不能估计完整二因素交互，也不能推广为所有背景下独立相加的通路份额。N−A同时改变P−1和真实Decode，不能进一步区分两阶段各自贡献。B与历史物理C3的差别还混合了深层R写入及物理数值路径，不用它单独断言深层写入的纯效应。','','## 统计、正确性与边界','','20,000次按309个Image SHA簇配对Bootstrap，固定seed=20261106。每次有放回抽取图片簇，保留同图全部问题，按题目等权重新计算软Accuracy。主三项比较在GPU前冻结；图中及CSV中95%CI均为普通探索性点位区间，未作为校正后的确认性检验，CI跨0不代表无损或等效。500题为历史复用探索样本，不是独立确认集。','', '10项CPU测试通过。两卡各3题Smoke验证Native/No-op及HF生成全部有效Logits严格torch.equal；Native完整Token与首Logits哈希复现历史。32层所有P−1之前的Prefill Hidden在N/A/B间逐层严格相等，原生视觉特征一致，RoPE相同；禁止的实际Attention概率严格0，非目标Mask及o_proj行不变，全部32层MLP/Norm/Block完整执行，独立完整KV存储、Cache位置及Hook清理均通过。1000份正式生成均核验输入、逐步Logits有限、实际Mask、块计数、Cache长度和模型参数版本。','', '首次Smoke启动因独立加载器遗漏历史模块级CUBLAS_WORKSPACE_CONFIG而在图像编码运算前中止；失败现场保留于diagnostics/smoke_attempt1。补齐历史相同的:4096:8配置后通过全部Smoke，未改精度、输入、数值门槛或实验条件，正式矩阵只在通过后启动。执行环境补充见execution_environment.json。','', f'所有EOS、截断、空答案、重复标记和负面案例完整保留。输出异常共{len(anomalies)}条，见output_anomalies.csv；重复为三Token子串启发式，不自动排除。逐题全部答案、完整Token、软分及配对变化见per_question.csv/generation_outputs.csv/paired_score_changes.csv；negative_cases.csv保留A/B相对Native全部软分下降题。','', '本轮没有运行任何测速、物理优化或额外层组合。原始模型/图片仍保留于服务器资源目录，其SHA和输入绑定在交付中可审计；GitHub包含完整答案和源码审计，不宣称包含模型权重或原始图像。提交独立分支、释放两张GPU后停止。']
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()

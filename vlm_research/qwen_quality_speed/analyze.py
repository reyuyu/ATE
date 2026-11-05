"""CPU recomputation from immutable per-request outputs and timing records."""
import csv,collections,math
import numpy as np
from .common import *

def csvout(name,rows):
 rows=list(rows);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['no_cases'],lineterminator='\n');w.writeheader();w.writerows(rows)
def intervals(x):return np.quantile(x,[.025,.975],axis=0)
def cluster_bootstrap(samples,values):
 groups=sorted({r['image_sha256'] for r in samples});ix={g:i for i,g in enumerate(groups)};g=len(groups)
 sums=np.zeros((g,values.shape[1]));counts=np.zeros(g)
 for r,v in zip(samples,values):sums[ix[r['image_sha256']]]+=v;counts[ix[r['image_sha256']]]+=1
 rng=np.random.default_rng(SEED);out=[]
 for _ in range(80):
  w=rng.multinomial(g,np.full(g,1/g),size=250)
  out.append((w@sums)/(w@counts)[:,None])
 return np.concatenate(out)
def main():
 m=read(OUT/'sample_manifest.json');samples=m['samples'];protocol=read(OUT/'protocol.json');n=len(samples)
 assert n==1000;check_sources();records={};rows=[];long=[];index=[]
 smokes=[]
 for rank in range(2):
  assert read(OUT/f'smoke/rank{rank}/completion.json')['status']=='PASSED'
  assert read(OUT/f'quality/rank{rank}/completion.json')['status']=='COMPLETE'
  assert read(OUT/f'benchmark/rank{rank}/completion.json')['status']=='COMPLETE'
 for p in sorted((OUT/'smoke').glob('rank*/*.json')):
  if p.stem in m['smoke_question_ids']:smokes.append(read(p))
 assert len(smokes)==20
 acc=np.zeros((n,4));anomalies=[]
 for j,s in enumerate(samples):
  q=s['question_id'];row={k:s[k] for k in ['question_id','image_id','image_sha256','question','gold','task','rank']}
  for i,c in enumerate(CONDITIONS):
   p=OUT/f'checkpoints/rank{s["rank"]}/{q}__{c}.json';d=read(p);r=d['result'];b=d['binding']
   assert b['input_sha256']==digest(m['input_bindings'][q]) and b['image_sha256']==s['image_sha256']
   assert b['source_manifest_sha256']==digest(read(OUT/'source_manifest.json'))
   prediction=r['text'].rstrip('.').lower();correct=prediction==s['gold']
   assert prediction==r['prediction'] and correct==r['correct'] and r['finite_checked']
   assert r['generated_tokens']==len(r['token_ids']) and 1<=len(r['token_ids'])<=32
   acc[j,i]=correct;records[q,c]=r
   for key in ['text','prediction','correct','eos','truncated','empty','generated_tokens','first_token']:row[f'{c}__{key}']=r[key]
   row[f'{c}__token_ids']=json.dumps(r['token_ids'])
   long.append(dict(question_id=q,image_sha256=s['image_sha256'],condition=c,gold=s['gold'],prediction=prediction,text=r['text'],correct=int(correct),token_ids=json.dumps(r['token_ids']),eos=r['eos'],truncated=r['truncated'],empty=r['empty'],first_token=r['first_token'],kv_live_bytes=r['kv']['live_bytes'],kv_capacity_bytes=r['kv']['capacity_bytes']))
   index.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
   if r['truncated'] or r['empty'] or not r['eos']:anomalies.append(dict(question_id=q,condition=c,text=r['text'],token_ids=json.dumps(r['token_ids']),eos=r['eos'],truncated=r['truncated'],empty=r['empty']))
  rows.append(row)
 boot=cluster_bootstrap(samples,acc);accci=intervals(boot)*100
 summary=[]
 for i,c in enumerate(CONDITIONS):
  rs=[records[s['question_id'],c] for s in samples]
  summary.append(dict(condition=c,n=n,correct=int(acc[:,i].sum()),accuracy_percent=acc[:,i].mean()*100,accuracy_ci_low=accci[0,i],accuracy_ci_high=accci[1,i],delta_native_pp=(acc[:,i]-acc[:,0]).mean()*100,native_correct_to_wrong=int(((acc[:,0]==1)&(acc[:,i]==0)).sum()),native_wrong_to_correct=int(((acc[:,0]==0)&(acc[:,i]==1)).sum()),eos=sum(r['eos'] for r in rs),empty=sum(r['empty'] for r in rs),truncated=sum(r['truncated'] for r in rs),mean_generated_tokens=np.mean([r['generated_tokens'] for r in rs])))
 csvout('accuracy_summary.csv',summary);csvout('per_question.csv',rows);csvout('generation_outputs.csv',long);csvout('output_anomalies.csv',anomalies);csvout('checkpoint_sha256.csv',index)
 paired=[];negative=[];recovered=[]
 disagreements=np.array([[records[s['question_id'],ca]['prediction']!=records[s['question_id'],cb]['prediction'] for ca,cb in protocol['bootstrap']['comparisons']] for s in samples],dtype=float)
 disagreement_ci=intervals(cluster_bootstrap(samples,disagreements))*100
 for pair_index,(ca,cb) in enumerate(protocol['bootstrap']['comparisons']):
  a,b=CONDITIONS.index(ca),CONDITIONS.index(cb);delta=acc[:,a]-acc[:,b];ci=intervals(boot[:,a]-boot[:,b])*100
  disagree=sum(records[s['question_id'],ca]['prediction']!=records[s['question_id'],cb]['prediction'] for s in samples)
  paired.append(dict(comparison=f'{ca} - {cb}',candidate=ca,reference=cb,delta_pp=delta.mean()*100,ci_low_pp=ci[0],ci_high_pp=ci[1],reference_correct_to_wrong=int((delta<0).sum()),reference_wrong_to_correct=int((delta>0).sum()),unchanged_score=int((delta==0).sum()),normalized_answer_disagreements=disagree,normalized_answer_disagreement_percent=disagree/n*100,disagreement_ci_low_percent=disagreement_ci[0,pair_index],disagreement_ci_high_percent=disagreement_ci[1,pair_index],raw_answer_disagreements=sum(records[s['question_id'],ca]['text']!=records[s['question_id'],cb]['text'] for s in samples),token_sequence_disagreements=sum(records[s['question_id'],ca]['token_ids']!=records[s['question_id'],cb]['token_ids'] for s in samples),first_token_flips=sum(records[s['question_id'],ca]['first_token']!=records[s['question_id'],cb]['first_token'] for s in samples)))
  for j,s in enumerate(samples):
   if delta[j]:
    rr=dict(comparison=f'{ca} - {cb}',question_id=s['question_id'],image_sha256=s['image_sha256'],question=s['question'],gold=s['gold'],candidate_answer=records[s['question_id'],ca]['text'],reference_answer=records[s['question_id'],cb]['text'],delta=int(delta[j]))
    (negative if delta[j]<0 else recovered).append(rr)
 csvout('paired_comparisons.csv',paired);csvout('negative_cases.csv',negative);csvout('recovered_cases.csv',recovered)
 taskrows=[]
 for task in sorted({s['task'] for s in samples}):
  ix=[i for i,s in enumerate(samples) if s['task']==task]
  for j,c in enumerate(CONDITIONS):taskrows.append(dict(task=task,condition=c,n=len(ix),correct=int(acc[ix,j].sum()),accuracy_percent=acc[ix,j].mean()*100,delta_native_pp=(acc[ix,j]-acc[ix,0]).mean()*100))
 csvout('task_accuracy.csv',taskrows)
 # Repeated runs are collapsed within question; repeated trials are not independent samples.
 raw=[];perfs=[];benchmark_indices=[]
 for q in m['benchmark_question_ids']:
  s=next(s for s in samples if s['question_id']==q)
  for regime in ['natural','fixed16']:
   for c in CONDITIONS:
    rr=[]
    for rep in range(5):
     p=OUT/f'benchmark/rank{s["rank"]}/raw/{q}__{regime}__{c}__{rep}.json';d=read(p);r=d['result']
     assert d['binding']['input_sha256']==digest(m['input_bindings'][q]) and d['reference_finite_checked'] and not r['finite_checked']
     assert not r.get('audit'), 'Audit probes must not be in timed runs'
     if regime=='natural':assert r['token_ids']==records[q,c]['token_ids']
     else:assert len(r['token_ids'])==16
     t=dict(question_id=q,image_sha256=s['image_sha256'],rank=s['rank'],condition=c,regime=regime,repeat=rep,generated_tokens=r['generated_tokens'],**r['timing'],**r['memory'],kv_live_bytes=r['kv']['live_bytes'],kv_capacity_bytes=r['kv']['capacity_bytes']);raw.append(t);rr.append(t)
     benchmark_indices.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
    row={k:rr[0][k] for k in ['question_id','image_sha256','rank','condition','regime']}
    for k in rr[0]:
     if k not in row and k!='repeat':row[k]=float(np.median([r[k] for r in rr]))
    perfs.append(row)
 assert len(raw)==1920 and len(perfs)==384
 csvout('benchmark_raw.csv',raw);csvout('benchmark_per_question.csv',perfs);csvout('benchmark_sha256.csv',benchmark_indices)
 perf_summary=[];speed_pairs=[]
 for regime in ['natural','fixed16']:
  by={(r['question_id'],r['condition']):r for r in perfs if r['regime']==regime};qs=m['benchmark_question_ids'];rng=np.random.default_rng(SEED+1);draw=rng.integers(0,48,(20000,48))
  native=np.array([by[q,'Native']['e2e_wall_ms'] for q in qs])
  for c in CONDITIONS:
   rr=[by[q,c] for q in qs];tt=np.array([r['e2e_wall_ms'] for r in rr]);speed=native.sum()/tt.sum();bs=native[draw].sum(1)/tt[draw].sum(1);ci=intervals(bs)
   row=dict(regime=regime,condition=c,questions=48,repeats_per_question=5,e2e_speedup_native=speed,speedup_ci_low=ci[0],speedup_ci_high=ci[1],median_per_question_speedup=float(np.median(native/tt)),e2e_mean_ms=tt.mean())
   for k in ['generated_tokens','vision_ms','prefill_language_ms','ttft_gpu_ms','ttft_wall_ms','decode_gpu_ms','decode_gpu_ms_per_token','peak_allocated','incremental_peak_allocated','peak_reserved','baseline_allocated','kv_live_bytes','kv_capacity_bytes']:row[k]=np.mean([r[k] for r in rr])
   perf_summary.append(row)
  for cand,ref in [('Physical-Original','Native'),('Physical-Repaired','Native'),('Physical-Repaired','Physical-Original')]:
   ct=np.array([by[q,cand]['e2e_wall_ms'] for q in qs]);rt=np.array([by[q,ref]['e2e_wall_ms'] for q in qs]);ci=intervals(rt[draw].sum(1)/ct[draw].sum(1))
   speed_pairs.append(dict(regime=regime,candidate=cand,reference=ref,speedup_reference_over_candidate=rt.sum()/ct.sum(),ci_low=ci[0],ci_high=ci[1]))
 csvout('benchmark_summary.csv',perf_summary);csvout('speed_paired_comparisons.csv',speed_pairs)
 card_rows=[]
 for rank in range(2):
  for regime in ['natural','fixed16']:
   native_time=sum(r['e2e_wall_ms'] for r in perfs if r['rank']==rank and r['regime']==regime and r['condition']=='Native')
   for c in CONDITIONS:
    rr=[r for r in perfs if r['rank']==rank and r['regime']==regime and r['condition']==c]
    card_rows.append(dict(rank=rank,uuid=UUIDS[rank],regime=regime,condition=c,n=len(rr),e2e_mean_ms=np.mean([r['e2e_wall_ms'] for r in rr]),e2e_speedup_native=native_time/sum(r['e2e_wall_ms'] for r in rr)))
 csvout('benchmark_per_gpu_summary.csv',card_rows)
 write(OUT/'validation.json',dict(status='PASSED',at=now(),quality_records=4000,quality_questions=1000,quality_images=len({s['image_sha256'] for s in samples}),smoke_questions=20,strict_native_noop_questions=20,strict_repaired_audit_on_off_questions=20,frozen_debug_repaired_strict_questions=sum(s['checks']['frozen_repaired_debug_strict'] for s in smokes),all_checkpoint_sha_bound=True,inputs_recomputed_equal=True,official_scoring_recomputed=True,finite_all_quality_steps=True,benchmark_records=1920,benchmark_questions=48,benchmark_repeats=5,benchmark_tokens_match_untimed_reference=True,benchmark_probes_disabled=True,bootstrap_replicates=20000,bootstrap_clusters='Image SHA256; question weighted; speed repeats collapsed first',abnormal_outputs=len(anomalies),old_strict_failures_not_modified=True,model_manifest_sha256=sha(OUT/'model_manifest.json'),source_manifest_sha256=digest(read(OUT/'source_manifest.json'))))
 plot(summary,paired,perf_summary,taskrows)
 report(summary,paired,perf_summary,m)
 print('ACCURACY',summary);print('SPEED',perf_summary)

def plot(summary,paired,perf,taskrows):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
 folder=OUT/'figures';folder.mkdir(exist_ok=True)
 labels=['Native','Dense F','Original','Repaired'];colors=['#555555','#5287B8','#22836D','#D27A36']
 fig,axs=plt.subplots(1,3,figsize=(12.5,3.7),layout='constrained')
 ay=np.array([r['accuracy_percent'] for r in summary]);alo=np.array([r['accuracy_ci_low'] for r in summary]);ahi=np.array([r['accuracy_ci_high'] for r in summary])
 axs[0].bar(labels,ay,color=colors,yerr=np.vstack([ay-alo,ahi-ay]),capsize=3)
 axs[0].set(ylabel='GQA exact-answer accuracy (%)',title='Exploratory quality: 1,000 questions',ylim=(0,100))
 for i,r in enumerate(summary):axs[0].text(i,r['accuracy_ci_high']+1,f'{r["accuracy_percent"]:.1f}',ha='center')
 x=np.arange(4)
 for j,regime in enumerate(['natural','fixed16']):
  rr=[r for r in perf if r['regime']==regime];y=np.array([r['e2e_speedup_native'] for r in rr]);lo=np.array([r['speedup_ci_low'] for r in rr]);hi=np.array([r['speedup_ci_high'] for r in rr]);axs[1].bar(x+(j-.5)*.35,y,.35,label='Natural EOS' if j==0 else 'Fixed 16 tokens',yerr=np.vstack([y-lo,hi-y]),capsize=2)
 axs[1].set(xticks=x,xticklabels=labels,ylabel='E2E speedup (Native / method)',title='Speed: 48 images, 5 repeats');axs[1].axhline(1,color='grey',lw=1);axs[1].legend(fontsize=8)
 rr=[r for r in perf if r['regime']=='fixed16'];bottom=np.zeros(4)
 for key,name,color in [('vision_ms','Vision','#999999'),('prefill_language_ms','Prefill language','#8CC5D1'),('decode_gpu_ms','Incremental Decode','#426EAA')]:
  y=np.array([r[key] for r in rr]);axs[2].bar(labels,y,bottom=bottom,label=name,color=color);bottom+=y
 axs[2].set(ylabel='Mean GPU event intervals (ms)',title='Fixed 16: forward phase costs');axs[2].legend(fontsize=8)
 for fmt in ['png','pdf']:fig.savefig(folder/f'quality_speed_overview.{fmt}',dpi=220)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.array([r['delta_pp'] for r in paired]);lo=np.array([r['ci_low_pp'] for r in paired]);hi=np.array([r['ci_high_pp'] for r in paired]);ax.errorbar(x,np.arange(len(x)),xerr=np.vstack([x-lo,hi-x]),fmt='o',capsize=3);ax.axvline(0,color='grey');ax.set(yticks=np.arange(len(x)),yticklabels=[r['comparison'] for r in paired],xlabel='Paired accuracy difference (pp), exploratory 95% cluster CI');ax.invert_yaxis()
 for fmt in ['png','pdf']:fig.savefig(folder/f'paired_accuracy.{fmt}',dpi=220)
 plt.close(fig)

def report(summary,paired,perf,m):
 s={r['condition']:r for r in summary};p={(r['candidate'],r['reference']):r for r in paired};b={(r['regime'],r['condition']):r for r in perf}
 lines=['# Qwen3-VL C3-V2：探索性质量与真实速度评测','',f'结果基于1000道冻结GQA val_balanced问题（{m["selection"]["selected_images"]}张图），四条件4000份独立完整生成；48张图同时测试自然EOS和固定16 Token，各条件预热2次、测量5次。本轮为探索性评测，旧严格BF16 Logits门槛失败记录保持原样，不作为本轮质量结论的通过证明。','', '## 直接回答','']
 for num,(ca,cb,text) in enumerate([('Physical-Original','Native','原始物理C3相对Native'),('Physical-Original','Dense-F','原始物理C3相对Dense F'),('Physical-Repaired','Physical-Original','数值修复相对原始物理版')],1):
  r=p[ca,cb];lines.append(f'{num}. **{text}**：{s[ca]["correct"]}/1000（{s[ca]["accuracy_percent"]:.3f}%）对{ s[cb]["correct"]}/1000（{s[cb]["accuracy_percent"]:.3f}%），Δ={r["delta_pp"]:+.3f}pp，95%CI [{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}]；原对变错{r["reference_correct_to_wrong"]}题、原错变对{r["reference_wrong_to_correct"]}题。')
 lines.extend(['','4. **真实E2E速度**（>1表示加速，<1表示变慢）：'])
 for c in ['Physical-Original','Physical-Repaired']:
  nn=b['natural',c];ff=b['fixed16',c];lines.append(f'   - {c}：自然EOS {nn["e2e_speedup_native"]:.3f}×（95%CI [{nn["speedup_ci_low"]:.3f}, {nn["speedup_ci_high"]:.3f}]）；固定16 Token {ff["e2e_speedup_native"]:.3f}×（95%CI [{ff["speedup_ci_low"]:.3f}, {ff["speedup_ci_high"]:.3f}]）。')
 orig=s['Physical-Original'];rep=s['Physical-Repaired'];os=b['fixed16','Physical-Original']['e2e_speedup_native'];rs=b['fixed16','Physical-Repaired']['e2e_speedup_native']
 if rep['correct']<=orig['correct'] and rs<os:rec='本批数据中修复版未增加正确数且更慢，原始物理版更值得作为后续优化起点；严格数值一致性失败仍保留，不把本结论当作数值验收通过。'
 elif rep['correct']>orig['correct'] and rs<os:rec='修复版提高了点估计正确数但付出速度代价；应按实际质量预算权衡，本批不足以证明修复具有普遍价值。'
 else:rec='需同时考虑下表中的准确率与固定长度速度；本批探索结果不足以把某版本宣称为普遍更优。'
 lines+=['',f'5. **后续版本选择**：{rec}','',f'原始物理版相对Native仍损失{s["Native"]["accuracy_percent"]-orig["accuracy_percent"]:.3f}pp，不能作为近无损替代。版本选择仅指本批质量—速度权衡，未授权或执行进一步修复、调参或搜索。若某速度比低于1，明确表示该场景没有实现加速。','','## Accuracy与配对变化','','|条件|正确/1000|ACC|相对Native pp|Native对→错 / 错→对|','|---|---:|---:|---:|---:|']
 for r in summary:lines.append(f'|{r["condition"]}|{r["correct"]}|{r["accuracy_percent"]:.3f}%|{r["delta_native_pp"]:+.3f}|{r["native_correct_to_wrong"]} / {r["native_wrong_to_correct"]}|')
 lines+=['','|配对比较|Δpp|探索性95%CI|参考对→错|参考错→对|标准化答案不一致|首Token翻转|','|---|---:|---|---:|---:|---:|---:|']
 for r in paired:lines.append(f'|{r["comparison"]}|{r["delta_pp"]:+.3f}|[{r["ci_low_pp"]:+.3f}, {r["ci_high_pp"]:+.3f}]|{r["reference_correct_to_wrong"]}|{r["reference_wrong_to_correct"]}|{r["normalized_answer_disagreements"]}/1000|{r["first_token_flips"]}|')
 lines+=['','所有CI均为预声明20,000次Image SHA256整簇配对Bootstrap，固定seed=20261106，按题目加权；同图问题一起重采样。比较均为探索性，未据此作无损或非劣效声明；区间跨0不代表等效。完整损害和纠正案例分别见`negative_cases.csv`与`recovered_cases.csv`。']
 for c in ['Physical-Original','Physical-Repaired']:
  r=p[c,'Dense-F'];lines.append(f'{c}与F标准化答案不一致率为{r["normalized_answer_disagreement_percent"]:.2f}%（{r["normalized_answer_disagreements"]}/1000），95%CI [{r["disagreement_ci_low_percent"]:.2f}%, {r["disagreement_ci_high_percent"]:.2f}%]。')
 lines+=['','Dense F与Native的差异反映既定C3功能干预的总体质量损失；Physical与F的差异反映物理实现及其数值偏差在本批输出上的额外影响。不能因为两种Physical与F的净准确率接近，就忽略其逐题答案不一致及新增损害。数值上更接近F也不保证更高的任务准确率。','','## 四条实现路径与正确性','','- Native为原生完整Qwen3-VL；Dense F保留稠密计算，仅实施既有C3功能干预。','- Original严格使用基线原始`Execution C`，L1–L6真实Attention跳算、L22视觉Hidden退出、紧凑文本KV。','- Repaired保留既有早层dead-key、Decode zero-image、L22–L35 Prefill zero-image，以及四个投影的固定cuBLASLt方案。未增加Kernel搜索、精度变更或新修复。','- 为公平测速，仅移除修复实现中的只读检查、日志及逐层计时探针；20题对比审计开/关的全部有效Logits严格相同，另对每卡最短/最长Smoke共4题与冻结debug修复路径逐位相同。','- 两卡各10题Smoke，Native/No-op全部有效Logits严格`torch.equal`；Mask禁用概率为0、36层MLP/Norm与残差检查、mRoPE绝对位置、DeepStack和Hook释放通过。物理路径保留真实Skip/视觉Hidden退出/紧凑持久KV。','- 自由生成均为BF16/Eager、greedy、max_new_tokens=32、logits_to_keep=1，每请求独立Cache；所有4000条逐步检查有限Logits。未重复追求旧F/C数值误差门槛。','','## 测速边界与内存','','计时为同步的模型请求E2E墙钟，包含CPU输入Tensor传入GPU、Vision、执行/Cache/Hook上下文建立、Token选择、EOS判断与上下文清理；不包含磁盘图片读取、Processor预处理、模型加载及计时后的文本解码。因此不是用户从文件到字符串的全服务延迟。所有条件使用同一计时入口、相同设备、相同样本，按冻结循环顺序交替测量。固定16含Prefill预测的第一个Token及15个真实增量Decode，忽略EOS停止仅用于等长度速度控制；不用于Accuracy。自然EOS的输出长度不同，速度同时受答案长度影响。','','每题先将5次重复折叠为中位数，再以Native总时间/方法总时间计算速度比；48张不同图片整簇Bootstrap形成速度CI，不把240次重复当作独立样本。计时无审计探针，另在计时外检查有效Logits有限及Token与未计时参考一致。GPU事件分别测Vision、首Forward（TTFT GPU）、增量Decode；Prefill language=TTFT GPU−Vision。48题按输入长度跨度冻结选择，测速结论限于此子集、两张RTX3090及batch=1，不是生产工作负载总体估计。','','|模式|条件|E2E ms|TTFT墙钟 ms|Vision ms|Prefill语言 ms|Decode GPU ms|生成Token均值|KV live MiB|KV allocated MiB|峰值allocated GiB|增量峰值 GiB|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in perf:lines.append(f'|{r["regime"]}|{r["condition"]}|{r["e2e_mean_ms"]:.2f}|{r["ttft_wall_ms"]:.2f}|{r["vision_ms"]:.2f}|{r["prefill_language_ms"]:.2f}|{r["decode_gpu_ms"]:.2f}|{r["generated_tokens"]:.2f}|{r["kv_live_bytes"]/2**20:.2f}|{r["kv_capacity_bytes"]/2**20:.2f}|{r["peak_allocated"]/2**30:.3f}|{r["incremental_peak_allocated"]/2**30:.3f}|')
 lines+=['','KV live与allocated分别记录，物理版预分配容量不会伪装成live节省。Repaired的临时补零内存计入PyTorch峰值，但不计入持久KV。`peak_reserved`记录缓存分配器保留量，受此前请求历史影响，仅作诊断；不是整进程nvidia-smi显存。所有显存结果为每请求实测，未从Mask比例推断速度。','','## 数据冻结与限制','',f'官方注释共有{m["selection"]["official_questions"]}题、{m["selection"]["official_images"]}张图，本地图片全部可用。使用与输出无关的SHA规则抽1000题，未改变Prompt、Processor、图像分辨率或选层；不是新独立确认实验，历史研究中可能见过部分图片。Image Token数随原始Processor变化，输入长度覆盖{min(x["input_length"] for x in m["input_bindings"].values())}–{max(x["input_length"] for x in m["input_bindings"].values())}，未强制为历史96 Image Token。', '', '完整答案使用既有固定GQA转换`text.rstrip(".").lower()`后与官方短答案做完整字符串匹配；没有首词截断或新答案解析器。所有条件使用同一评分。任务分组沿用官方types.semantic，见`task_accuracy.csv`。','','原始逐请求输出与Token在`checkpoints/`，全部计时重复在`benchmark/`，SHA索引及`validation.json`用于复核。冻结模型资产、输入、协议、执行源码和cuBLASLt桥接二进制哈希均保存。本轮只完成所要求的探索性质量与速度测试，停止后续修复及搜索。']
 lines+=['','## 复现资产与实现开销','','测速关闭可选的逐层审计Hook和修复路径只读探针；继承的原始物理执行器仍含绝对位置、Cache容量等断言，均作为当前实现开销计入E2E。本结果不是理想Kernel或实现性能上界。未修改继承的Original源码来追求更高速度。','','本次全部逐请求答案、Token、计时与CSV随独立分支交付。模型权重、原始图片及既有cuBLASLt桥接共享库保留在服务器资源目录，不宣称GitHub包含这些资产；其路径及SHA绑定见模型、输入与源码冻结清单。Smoke保存有效Logits的SHA及严格比对记录，避免新增大规模冗余Logits归档。']
 (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()

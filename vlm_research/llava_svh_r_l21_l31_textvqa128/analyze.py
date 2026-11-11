from common import *
import sys,collections,subprocess
import numpy as np
sys.path.insert(0,str(RUN/'references'))
from textvqa_eval import TextVQAAccuracyEvaluator
S=read(RUN/'state.json');PARENT=Path(S['parent_run']);METHODS=('Native','SVH-R_L11_L24','Late11_L21_31')
def main():
 verify_sources();sm=read(OUT/'sample_manifest.json')['Main'];official=TextVQAAccuracyEvaluator();assert read(OUT/'run_completion.json')['formal_records']==128 and read(OUT/'Native_reuse_calibration.json')['status']=='PASS'
 data={};origins={};auditrows=[]
 for base,reused in [(PARENT,True),(RUN,False)]:
  db=sqlite3.connect('file:'+str(base/'predictions.sqlite')+'?mode=ro',uri=True);assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
  po=base/'results';protocol=sha(po/'protocol.json');source=sha(po/'source_manifest.json');expected=256 if reused else 128;count=0
  for qid,label,h,raw in db.execute('select qid,method,checksum,record from predictions'):
   r=json.loads(raw);assert digest(r)==h and r['question_id']==qid and r['method']==label and r['GPU_UUID']==GPU and r['protocol_sha256']==protocol and r['source_manifest_sha256']==source
   method='SVH-R_L11_L24' if reused and label=='SVH-R' else label
   assert method in METHODS
   data[qid,method]=r;origins[qid,method]='REUSED' if reused else 'NEW';count+=1
   auditrows.append(dict(qid=qid,method=method,origin=origins[qid,method],source_commit=S['parent'] if reused else 'current branch source_manifest',checkpoint_checksum=h,original_method=label,original_protocol_sha256=protocol,original_source_manifest_sha256=source))
  assert count==expected;db.close()
 assert len(data)==384
 rows=[];kvrows=[];scores={m:[] for m in METHODS}
 for s in sm:
  qid=str(s['question_id'])
  for method in METHODS:
   r=data[qid,method];assert r['input_binding']==s['input_binding'] and r['image_sha256']==s['image_sha256'] and r['prompt_sha256']==s['prompt_sha256'] and r['all_logits_finite']
   assert r['generated_tokens']==len(r['token_ids'])<=32 and r['generated_tokens']>0 and r['real_decode_calls']==r['generated_tokens']-1
   assert r['eos']==(r['token_ids'][-1]==2) and r['truncated']==(r['generated_tokens']==32 and not r['eos'])
   gold=[official.answer_processor(a) for a in s['gold']];pred=official.answer_processor(r['text']);score=official._compute_answer_scores(s['gold']).get(pred,0.)
   ind=sum(min(1.,sum(gold[j]==pred for j in range(10) if i!=j)/3.) for i in range(10))/10.;assert abs(score-ind)<1e-12
   scores[method].append(score)
   rows.append(dict(question_id=qid,method=method,origin=origins[qid,method],image_sha256=s['image_sha256'],question=s['question'],prompt=s['prompt'],gold=s['gold'],prediction=r['text'],normalized_prediction=pred,score=score,independent_score=ind,token_ids=r['token_ids'],first_token=r['first_token'],generated_tokens=r['generated_tokens'],eos=r['eos'],empty=r['empty'],truncated=r['truncated'],repeated=r['repeated'],input_binding=s['input_binding'],checkpoint_checksum=digest(r),source_protocol_sha256=r['protocol_sha256'],source_manifest_sha256=r['source_manifest_sha256']))
   keep=range(32) if method=='Native' else range(11,25) if method=='SVH-R_L11_L24' else range(21,32)
   assert len(r['KV']['layers'])==32
   live=storage=0
   for i,k in enumerate(r['KV']['layers']):
    absent=method!='Native' and 2<=i<=5;image=0 if absent or i not in keep else 576
    used=0 if absent else s['input_tokens']-576+image+r['generated_tokens']-1
    assert k['used_tokens']==used and k['image_tokens']==image and k['live_bytes']==used*2*32*128*2
    if not absent and 'original_positions' in k:
     pos=(list(range(s['input_tokens'])) if image else list(s['text']))+list(range(s['input_tokens'],s['input_tokens']+r['generated_tokens']-1))
     assert k['original_positions']==pos
    live+=k['live_bytes'];storage+=k['storage_bytes']
    kvrows.append(dict(question_id=qid,method=method,origin=origins[qid,method],layer=i,generated_tokens=r['generated_tokens'],measurement='actual natural EOS endpoint; NOT KV@32 or speed',image_tokens=image,used_tokens=k['used_tokens'],capacity_tokens=k['capacity_tokens'],live_bytes=k['live_bytes'],storage_bytes=k['storage_bytes'],K_shape=k.get('K_shape'),V_shape=k.get('V_shape'),dtype=k.get('dtype'),positions_sha256=digest(k.get('original_positions'))))
   assert live==r['KV']['live_bytes'] and storage==r['KV']['storage_bytes']
 summary=[]
 for method in METHODS:
  rr=[x for x in rows if x['method']==method];lens=[x['generated_tokens'] for x in rr];agg=official.eval_pred_list([dict(pred_answer=x['prediction'],gt_answers=x['gold']) for x in rr],disable_tqdm=True);assert abs(agg-np.mean(scores[method]))<1e-12
  summary.append(dict(method=method,N=128,origin='NEW' if method=='Late11_L21_31' else 'REUSED',score_sum=sum(scores[method]),accuracy_percent=agg*100,empty=sum(x['empty'] for x in rr),truncated=sum(x['truncated'] for x in rr),EOS=sum(x['eos'] for x in rr),repeated=sum(x['repeated'] for x in rr),length_P50=float(np.percentile(lens,50)),length_P95=float(np.percentile(lens,95)),length_max=max(lens),natural_EOS_KV_live_MiB=float(np.mean([data[str(s['question_id']),method]['KV']['live_bytes']/2**20 for s in sm])),natural_EOS_KV_capacity_MiB=float(np.mean([data[str(s['question_id']),method]['KV']['storage_bytes']/2**20 for s in sm]))))
 groups=collections.defaultdict(list)
 for i,s in enumerate(sm):groups[s['image_sha256']].append(i)
 clusters=list(groups.values());counts=np.array([len(ix) for ix in clusters]);rng=np.random.default_rng(20261112);indices=rng.integers(0,len(clusters),(20000,len(clusters)));pairs=[];changes=[]
 for control in ('Native','SVH-R_L11_L24'):
  a=np.array(scores[control]);b=np.array(scores['Late11_L21_31']);delta=b-a;sums=np.array([delta[ix].sum() for ix in clusters]);bootstrap=sums[indices].sum(1)/counts[indices].sum(1)*100;lo,hi=np.percentile(bootstrap,[2.5,97.5])
  pp=dict(comparison='Late11_L21_31 minus '+control,delta_pp=float(delta.mean()*100),CI95_low_pp=float(lo),CI95_high_pp=float(hi),image_SHA_clusters=len(clusters),bootstrap=20000,CI='ordinary unadjusted95%,2 exploratory contrasts',improved=int((delta>1e-12).sum()),declined=int((delta< -1e-12).sum()),unchanged=int((abs(delta)<=1e-12).sum()),full_correct_to_not_full=int(((a==1)&(b<1)).sum()),not_full_to_full=int(((a<1)&(b==1)).sum()),positive_to_zero=int(((a>0)&(b==0)).sum()),zero_to_positive=int(((a==0)&(b>0)).sum()))
  cc=[]
  for i,s in enumerate(sm):
   qid=str(s['question_id']);x=data[qid,control];y=data[qid,'Late11_L21_31']
   cc.append(dict(comparison=pp['comparison'],question_id=qid,image_sha256=s['image_sha256'],question=s['question'],gold=s['gold'],control_prediction=x['text'],late_prediction=y['text'],control_score=float(a[i]),late_score=float(b[i]),delta=float(delta[i]),first_token_changed=x['token_ids'][0]!=y['token_ids'][0],complete_tokens_changed=x['token_ids']!=y['token_ids'],text_changed=x['text']!=y['text']))
  pp.update(first_token_changed=sum(x['first_token_changed'] for x in cc),complete_tokens_changed=sum(x['complete_tokens_changed'] for x in cc),text_changed=sum(x['text_changed'] for x in cc));pairs.append(pp);changes+=cc
 csvwrite(OUT/'accuracy_summary.csv',summary);csvwrite(OUT/'per_question.csv',rows);csvwrite(OUT/'paired_comparison.csv',pairs);csvwrite(OUT/'per_question_changes.csv',changes);csvwrite(OUT/'harm_cases.csv',[x for x in changes if x['delta']< -1e-12]);csvwrite(OUT/'kv_layout_per_layer.csv',kvrows);csvwrite(OUT/'reuse_record_audit.csv',auditrows)
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip();assert not apps
 gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip();assert GPU in gpu
 write(OUT/'gpu_release_external.json',dict(status='PASS_WORKER_EXITED_GPU_RELEASED',GPU=gpu,compute_apps=apps,internal=read(OUT/'gpu_release.json')))
 smoke=read(OUT/'smoke_validation.json');assert smoke['status']=='PASS_NATURAL_EOS_QUALITY_AND_9STEP_STRUCTURE'
 validation=dict(status='PASS_REQUESTED_NATURAL_EOS_QUALITY',records_verified=384,new_formal=128,reused=256,all_record_checksums_and_inputs_source_protocol=True,independent_official_scores_agree=True,actual32layer_KV_and_positions=True,Native_reuse_calibration='3/3 complete Tokens and text equal same GPU',Smoke=smoke['status'],numeric_scope='unchanged historical thresholds on natural EOS valid steps; synthetic postEOS only structural/finite with numeric failures retained',same_GPU=GPU,model_loads=read(OUT/'run_completion.json')['model_loads'],all_answers_including_wrong_empty_truncated_counted=True,GPU_released=True,speed='NOT_RUN',GitHub_branch=S['branch'])
 write(OUT/'validation.json',validation);write(OUT/'analysis_digest.json',dict(summary=summary,paired=pairs))
 n,old,late=summary;v,w=pairs
 failure_steps=[e for x in smoke['samples'] for e in x['synthetic_post_EOS_original_threshold_failures']]
 report=f'''# LLaVA-SVH-R L21–L31：相同128题TextVQA Image-only
晚层L21–L31回看得分 {late['accuracy_percent']:.6f}%，相比同机Native {n['accuracy_percent']:.6f}%变化 {v['delta_pp']:+.6f}pp；相比之前L11–L24的 {old['accuracy_percent']:.6f}%变化 {w['delta_pp']:+.6f}pp。

| 方法 | 题数 | 十答案软分Accuracy | 记录来源 |
|---|---:|---:|---|
| Native |128|{n['accuracy_percent']:.6f}%|同一GPU与输入，复用{S['parent']}|
| SVH-R L11–L24 |128|{old['accuracy_percent']:.6f}%|同上|
| SVH-R L21–L31 |128|{late['accuracy_percent']:.6f}%|本轮128次真实新增生成|

## 配对比较
晚层−Native：{v['delta_pp']:+.6f}pp，图片SHA簇配对Bootstrap普通95%CI [{v['CI95_low_pp']:+.6f}, {v['CI95_high_pp']:+.6f}]pp；软分上升{v['improved']}题，下降{v['declined']}题，不变{v['unchanged']}题。
晚层−原中层：{w['delta_pp']:+.6f}pp，CI [{w['CI95_low_pp']:+.6f}, {w['CI95_high_pp']:+.6f}]pp；上升{w['improved']}题，下降{w['declined']}题，不变{w['unchanged']}题。
按{len(clusters)}个图片SHA整簇配对重采样20000次，seed20261112，两个探索性比较均为普通未调整95%CI。本数据集已用于探索，并非独立确认集；CI跨0不能证明等价/非劣效，不能由128题推断完整TextVQA结果。

## 配置与实现
模型llava-hf/llava-1.5-7b-hf，Revision {REVISION}，FP16/Eager，Torch2.6.0+cu124，Transformers4.49.0，Batch1，Greedy，logits_to_keep=1，use_cache=True，正常EOS，max_new_tokens32。
严格保留之前LLaVA的128题ID、顺序、RGB图片SHA、问题、十答案Gold、原生HF chat及“Answer the question using a single word or phrase.”后缀、CLIP336 shortest-edge/center-crop、Tokenizer。Image-only，不注入OCR或Gold，prepared文件仅模型Tensor。未修改Prompt、精度或评分。

P−1和真实Decode仅L21–L31共11层可访问原始576 Image Key；L2–L5答案Attention继续物理跳算，R→Image从L20开始禁止。全部32语言Block仍处理原视觉Hidden；退出放在L31之后、最终逐Token RMSNorm之前，因此不宣称Prefill Block视觉计算减少。仅L21–L31持久保留原视觉KV，L2–L5无持久KV，其余缓存只留文本与真实增量位置。保持原RoPE逻辑位置。冻结的skip实现Prefill K/V仍计算全部行，仅最后P−1 Q/Attention/o_proj不计算；这些层Decode完整QKV/Attention/o_proj不执行。

未编辑固定{S['source_commit']}的26个核心依赖，新增独立TerminalMiddleWriter，只补上cutoff32的终端退出。CPU32层Mock验证原cutoff25/KEEP11..24与旧实现Tokens/Logits/Hidden严格torch.equal，晚层Dense/Physical CPU差异及实际退出由cpu_tests.json保存。

## GPU验收与复用
GPU始终{GPU}。三道外部Smoke覆盖短中长输入：Native/No-op有效Logits严格torch.equal、共享Native循环和原生HF generate完整Token一致、晚层Dense/Physical完整自然EOS Token一致，数值有效生成步使用原门槛atol0.10/rtol0.003，Prefill Hidden relativeRMSE<=0.01、cosine error<=1e-4。另9步强制生成检查真实Decode、投影调用、全部Norm/MLP、原RoPE、Mask禁图零概率与每层实际KV。
沿用上一轮已明确的v2检查范围：超出自然EOS的合成步不计入正常质量Logit门槛，仍保留全部数值差异；本轮此类门槛失败计{len(failure_steps)}步，见smoke_numerical.csv。未抬高阈值，不宣称固定长度数值测试通过。
正式前确定性选最短/中位/最长三道主集题重跑Native，与原完整Token/文本3/3一致，才复用原Native和L11–L24各128条原记录。旧记录原protocol/source/Commit/checksum完整保留，未伪装成本轮新生成。

## 完整性与异常
共384条逐题记录：128新增+256复用，全部独立重算官方十答案软评分并用leave-one-out公式再次核验；源、协议、输入及checkpoint SHA校验通过。晚层空输出{late['empty']}题、截断{late['truncated']}题、EOS {late['EOS']}题、重复启发式标记{late['repeated']}题，全部纳入评分。长度P50/P95/最大为{late['length_P50']}/{late['length_P95']}/{late['length_max']}。全部损害案例保留harm_cases.csv；不因答案错误或输出格式重跑/剔除。
未测速，KV统计为自然EOS实际终点，不是KV@32或性能改进。本轮模型加载一次，GPUWorker已退出，外部核验无计算进程。未运行GQA、OCRBench、训练、其他层搜索或任何额外质量Benchmark。

## 产物
本轮SSD完整检查点：{RUN}/predictions.sqlite
原始复用检查点：{PARENT}/predictions.sqlite
持久盘结果：{S['root']}/{S['result']}/
实现代码：{S['root']}/{S['pkg']}/
独立Git分支：{S['branch']}。GitHub仅上传轻量代码、CSV、协议、评分与验证证据；完整SQLite、详细日志在server_only，模型/图像/prepared Tensor/密钥不上传。SHA256SUMS与artifact_manifest.json区分上传及仅服务器文件。Git提交与远端逐文件回读结果存server_only/git_publish_receipt.json。
'''
 (OUT/'report_zh.md').write_text(report);print(json.dumps(dict(status='ANALYSIS_PASS',summary=summary,paired=pairs,synthetic_postEOS_numeric_failed_steps=len(failure_steps)),ensure_ascii=False))
if __name__=='__main__':main()

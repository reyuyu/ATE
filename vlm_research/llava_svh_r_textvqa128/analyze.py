from common import *
import sys,math,collections,subprocess,statistics
import numpy as np
sys.path.insert(0,str(RUN/'references'))
from textvqa_eval import TextVQAAccuracyEvaluator
def main():
 verify_sources();manifest=read(OUT/'sample_manifest.json');sm=manifest['Main'];official=TextVQAAccuracyEvaluator()
 assert read(OUT/'run_completion.json')['status']=='COMPLETE'
 db=sqlite3.connect('file:'+str(RUN/'predictions.sqlite')+'?mode=ro',uri=True);assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
 records={}
 for qid,label,h,raw in db.execute('select qid,method,checksum,record from predictions'):
  r=json.loads(raw);assert digest(r)==h and qid==r['question_id'] and label==r['method']
  records[qid,label]=r
 assert len(records)==256 and len(sm)==128
 protocol=read(OUT/'protocol.json');source=sha(OUT/'source_manifest.json');phash=sha(OUT/'protocol.json')
 rows=[];kvrows=[];summary=[];scores={x:[] for x in LABELS}
 for s in sm:
  qid=str(s['question_id']);b=read(OUT/'input_bindings.json')[qid];assert digest(b)==s['input_binding']
  for label in LABELS:
   r=records[qid,label];assert r['input_binding']==s['input_binding'] and r['image_sha256']==s['image_sha256'] and r['protocol_sha256']==phash and r['source_manifest_sha256']==source and r['GPU_UUID']==GPU
   assert r['all_logits_finite'] and len(r['token_ids'])==r['generated_tokens'] and 1<=r['generated_tokens']<=32 and r['real_decode_calls']==r['generated_tokens']-1
   assert r['eos']==(r['token_ids'][-1]==2) and r['truncated']==(r['generated_tokens']==32 and not r['eos'])
   gold=[official.answer_processor(a) for a in s['gold']];pred=official.answer_processor(r['text'])
   score=official._compute_answer_scores(s['gold']).get(pred,0.)
   independent=sum(min(1.,sum(gold[j]==pred for j in range(10) if i!=j)/3.) for i in range(10))/10.
   assert abs(score-independent)<1e-12,(qid,label,score,independent)
   scores[label].append(score)
   rows.append(dict(question_id=qid,method=label,image_sha256=s['image_sha256'],question=s['question'],prompt=s['prompt'],prompt_sha256=s['prompt_sha256'],gold=s['gold'],prediction=r['text'],normalized_prediction=pred,score=score,independent_score=independent,token_ids=r['token_ids'],first_token=r['first_token'],generated_tokens=r['generated_tokens'],eos=r['eos'],truncated=r['truncated'],empty=r['empty'],repeated=r['repeated'],input_binding=s['input_binding'],checkpoint_checksum=digest(r)))
   assert len(r['KV']['layers'])==32
   live=storage=0
   for i,k in enumerate(r['KV']['layers']):
    skipped=label=='SVH-R' and 2<=i<=5;image=0 if skipped else 576 if label=='Native' or i in KEEP else 0
    expected=0 if skipped else s['input_tokens']-576+image+r['generated_tokens']-1
    assert k['layer']==i and k['used_tokens']==expected and k['image_tokens']==image and k['live_bytes']==expected*2*32*128*2
    if not skipped:
     pos=(list(range(s['input_tokens'])) if image else list(s['text']))+list(range(s['input_tokens'],s['input_tokens']+r['generated_tokens']-1))
     if 'original_positions' in k:assert k['original_positions']==pos
    live+=k['live_bytes'];storage+=k['storage_bytes']
    kvrows.append(dict(question_id=qid,method=label,layer=i,generated_tokens=r['generated_tokens'],measurement='normal EOS endpoint, NOT KV@32',image_tokens=image,used_tokens=k['used_tokens'],capacity_tokens=k['capacity_tokens'],live_bytes=k['live_bytes'],storage_bytes=k['storage_bytes'],K_shape=k.get('K_shape'),V_shape=k.get('V_shape'),dtype=k.get('dtype'),original_positions_sha256=digest(k.get('original_positions')) if 'original_positions' in k else None))
   assert live==r['KV']['live_bytes'] and storage==r['KV']['storage_bytes']
 for label in LABELS:
  rr=[x for x in rows if x['method']==label];gens=[x['generated_tokens'] for x in rr]
  official_aggregate=official.eval_pred_list([dict(pred_answer=x['prediction'],gt_answers=x['gold']) for x in rr],disable_tqdm=True)
  assert abs(official_aggregate-np.mean(scores[label]))<1e-12
  summary.append(dict(method=label,N=128,score_sum=sum(scores[label]),accuracy_percent=100*official_aggregate,empty=sum(x['empty'] for x in rr),truncated=sum(x['truncated'] for x in rr),EOS=sum(x['eos'] for x in rr),repeated=sum(x['repeated'] for x in rr),length_P50=float(np.percentile(gens,50)),length_P95=float(np.percentile(gens,95)),length_max=max(gens),normal_EOS_KV_live_MiB=float(np.mean([records[str(s['question_id']),label]['KV']['live_bytes']/2**20 for s in sm])),normal_EOS_KV_capacity_MiB=float(np.mean([records[str(s['question_id']),label]['KV']['storage_bytes']/2**20 for s in sm]))))
 a=np.array(scores['Native']);b=np.array(scores['SVH-R']);delta=b-a
 groups=collections.defaultdict(list)
 for i,s in enumerate(sm):groups[s['image_sha256']].append(i)
 ix=list(groups.values());sums=np.array([delta[v].sum() for v in ix]);counts=np.array([len(v) for v in ix])
 rng=np.random.default_rng(20261112);sample=rng.integers(0,len(ix),size=(20000,len(ix)));boot=sums[sample].sum(1)/counts[sample].sum(1)*100;low,high=np.percentile(boot,[2.5,97.5])
 paired=dict(comparison='SVH-R minus Native',delta_pp=float(delta.mean()*100),CI95_low_pp=float(low),CI95_high_pp=float(high),image_SHA_clusters=len(ix),bootstrap=20000,seed=20261112,improved=int((delta>1e-12).sum()),declined=int((delta< -1e-12).sum()),unchanged=int((abs(delta)<=1e-12).sum()),full_correct_to_not_full=int(((a==1)&(b<1)).sum()),not_full_to_full=int(((a<1)&(b==1)).sum()),positive_to_zero=int(((a>0)&(b==0)).sum()),zero_to_positive=int(((a==0)&(b>0)).sum()))
 changes=[]
 for i,s in enumerate(sm):
  qid=str(s['question_id']);x=records[qid,'Native'];y=records[qid,'SVH-R']
  changes.append(dict(question_id=qid,image_sha256=s['image_sha256'],question=s['question'],gold=s['gold'],Native=x['text'],SVH_R=y['text'],Native_score=float(a[i]),SVH_R_score=float(b[i]),delta=float(delta[i]),first_token_changed=x['token_ids'][0]!=y['token_ids'][0],complete_tokens_changed=x['token_ids']!=y['token_ids'],text_changed=x['text']!=y['text']))
 paired.update(first_token_changed=sum(x['first_token_changed'] for x in changes),complete_tokens_changed=sum(x['complete_tokens_changed'] for x in changes),text_changed=sum(x['text_changed'] for x in changes))
 csvwrite(OUT/'accuracy_summary.csv',summary);csvwrite(OUT/'per_question.csv',rows);csvwrite(OUT/'paired_comparison.csv',[paired]);csvwrite(OUT/'per_question_changes.csv',changes);csvwrite(OUT/'harm_cases.csv',[x for x in changes if x['delta']< -1e-12]);csvwrite(OUT/'kv_layout_per_layer.csv',kvrows)
 release=read(OUT/'gpu_release.json')
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 final_gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 assert not apps and GPU in final_gpu
 write(OUT/'gpu_release_external.json',dict(status='PASS_WORKER_EXITED_GPU_RELEASED',internal_before_process_exit=release,final_GPU=final_gpu,compute_apps=apps,explanation='32MiB remained in terminating worker allocator at its internal snapshot; all allocations released by process exit, independently verified here without initializing CUDA'))
 validation=dict(status='PASS',formal_records=256,questions=128,conditions=2,IDs_order_images_questions_Gold_equal=True,input_tensors_hash_equal_between_methods=True,all_record_checksums_and_source_protocol_bound=True,independent_official_scores_agree=True,KV32_layers_and_logical_positions_verified=True,all_outputs_including_errors_and_truncation_retained=True,Native_NoOp_strict_equal=True,Dense_Physical_smoke_gate='PASS natural-EOS valid steps at unchanged historical FP16 thresholds +9step physical structure;original all9step numerical failure retained',initial_smoke_all9step_numerical='FAILED_SYNTHETIC_POST_EOS',model_loads_all_attempts=3,GPU=GPU,GPU_model_objects_released=True,model_loads=read(OUT/'run_completion.json')['model_loads'],speed_test='NOT_RUN',GitHub_branch='codex/llava-svh-r-textvqa128-3090')
 write(OUT/'validation.json',validation);write(OUT/'analysis_digest.json',dict(summary=summary,paired=paired))
 n,v=summary;report=f'''# LLaVA-1.5-7B-SVH-R：相同128题 TextVQA Image-only
LLaVA Native 为 {n['accuracy_percent']:.6f}%，SVH-R 为 {v['accuracy_percent']:.6f}%，SVH-R相对Native变化 {paired['delta_pp']:+.6f} 个百分点。图片SHA簇配对Bootstrap普通95%CI：[{low:+.6f}, {high:+.6f}] pp。

| 方法 | 题数 | 官方十答案软分 Accuracy | 分数和 |
|---|---:|---:|---:|
| Native | 128 | {n['accuracy_percent']:.6f}% | {n['score_sum']:.6f}/128 |
| llava-SVH-R | 128 | {v['accuracy_percent']:.6f}% | {v['score_sum']:.6f}/128 |

## 实验含义
严格复用此前InternVL小测试集的128道题、图片字节/SHA、题目顺序与十答案Gold，但LLaVA有其自身冻结Prompt、Tokenizer与图像预处理。该128题已用于探索，不能视为独立确认集。核心结论是本轮同机LLaVA内部的Native配对差值，不把跨模型分数差异全部归因于算法。

模型llava-hf/llava-1.5-7b-hf，Revision {REVISION}；FP16/Eager，Torch2.6.0+cu124，Transformers4.49.0，Greedy，Batch1，logits_to_keep=1，use_cache=True，max_new_tokens=32，原生EOS。原HF CLIP shortest-edge resize336、center-crop336（未新增square-pad），原生HF USER/image/ASSISTANT chat，问题后追加“Answer the question using a single word or phrase.”。Image-only，无OCR文字。Gold仅评分读取，不进模型输入。

## 冻结SVH-R
逐字节复用 5ee2a0dbe78df336229b55fed14a4a6d0d880bed 的MiddleWriter及依赖，不修改冻结算法。P−1和真实Decode仅L11–L24可读原始Image Key；L20起R→Image禁止；L25入口真正删除576行视觉Hidden。L2–L5 P−1的Q/Attention/o_proj输出计算被跳过，其他Prefill Query正常计算；固定历史实现的这些层Prefill K/V仍对全Prefill行计算，真实Decode全部QKV/Attention/o_proj不执行，且这些层无持久KV。所有层Norm、MLP与Residual保留。其余层只在L11–L24持久保存原始视觉KV，逻辑RoPE位置不重编号。

## 完整性与正确性
GPU {GPU}，Native和SVH-R同卡同输入，各128次正式生成，共256次；最终正式Worker加载模型一次，含初始Smoke失败和独立诊断共3次GPU模型加载。正式生成前3题Smoke与主集不重叠，覆盖短中长输入。Native与No-op有效末位Logits严格torch.equal，原生HF generate与共享Native循环完整Token一致。Dense/Physical通过预先冻结的历史FP16门槛（Logits allclose atol0.10 rtol0.003，Hidden relativeRMSE<=0.01、cosine error<=1e-4），自然EOS有效生成步通过上述数值阈值，完整自然EOS生成Tokens一致；另9步强制续跑只用于真实缓存/结构检查。逐层隐藏行、真实投影调用、Mask禁图概率、RoPE和实际K/V布局均核验。smoke_numerical.csv保存实测差异，不宣称所有输入数值逐位等同。

初始统一9步数值门槛曾失败：长Smoke题正常2个Token已EOS，强制EOS后第3步Logit MaxAbs0.104492、MeanAbs0.015672，原allclose条件未通过。该失败、旧协议、旧源码和诊断均保留于initial_attempt和smoke_diagnostic_v1；并未改写为通过。本轮用户请求的是正常EOS质量评测，GPU正式运行前建立范围明确的v2协议：有效生成步保持原数值门槛（3题最大有效差0.020508），EOS后合成步骤保留非有限、位置、缓存、物理结构检查，并记录数值失败。未改变算法、精度、Backend或抬高数值阈值，也不宣称固定长度生成的数值门槛已通过。

CPU32层float64 Mock内部沿用官方FP32 Softmax，初始1e-10断言观察到3.31e-8 Decode Logit舍入差；保留初始诊断，正式GPU前将CPU浮点容差明确冻结为1e-6（Hidden差0），未修改算法或正式GPU既定门槛。

## 配对变化与异常
软分上升{paired['improved']}题、下降{paired['declined']}题、不变{paired['unchanged']}题；首Token变化{paired['first_token_changed']}题、完整Tokens变化{paired['complete_tokens_changed']}题、文本变化{paired['text_changed']}题。完整正确→非完整正确{paired['full_correct_to_not_full']}题，反向{paired['not_full_to_full']}题。全部损害案例见harm_cases.csv，未剔除错误答案、重复或截断。
Native空输出{n['empty']}、截断{n['truncated']}、重复标记{n['repeated']}；SVH-R空输出{v['empty']}、截断{v['truncated']}、重复标记{v['repeated']}。重复标记仅为Token三元组启发式，不自动认定异常或停止。
长度P50/P95/最大：Native {n['length_P50']}/{n['length_P95']}/{n['length_max']}；SVH-R {v['length_P50']}/{v['length_P95']}/{v['length_max']}。

Bootstrap：按{len(ix)}个Image SHA整簇配对重采样20000次，seed20261112，普通未调整95%CI。CI跨0时不能解释为等价或独立非劣效证明。小样本仅供探索，不调整层或Prompt。

## 交付与范围
本地SSD完整检查点：{RUN}/predictions.sqlite
持久盘轻量结果与完整原始检查点：/home/featurize/work/ATE-llava-svh-r-textvqa128-3090/vlm_research/results/llava_svh_r_textvqa128_3090/
独立复现代码：/home/featurize/work/ATE-llava-svh-r-textvqa128-3090/vlm_research/llava_svh_r_textvqa128/
独立Git分支：codex/llava-svh-r-textvqa128-3090。artifact_manifest.json区分GitHub上传文件与仅服务器原件；SHA256SUMS覆盖轻量证据与实现来源。模型、图片、prepared Tensor与密钥不上传。
未测速度，缓存为自然EOS实际终点，不能当作KV@32或性能节省结果。内部退出快照仍有32MiB CUDA分配；Worker进程退出后，外部nvidia-smi核验GPU仅1MiB、无计算进程，相关记录均保留。最终进程状态以server_only/git_publish_receipt.json为准。未执行其他Benchmark、训练或新候选。
'''
 (OUT/'report_zh.md').write_text(report);db.close();print(json.dumps(dict(status='ANALYSIS_PASS',summary=summary,paired=paired),ensure_ascii=False))
if __name__=='__main__':main()

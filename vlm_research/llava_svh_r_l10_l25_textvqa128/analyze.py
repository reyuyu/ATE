from common import *
import sys,collections,subprocess
import numpy as np
sys.path.insert(0,str(RUN/'references'))
from textvqa_eval import TextVQAAccuracyEvaluator
S=read(RUN/'state.json');PARENT=Path(S['parent_run']);METHODS=('Native','SVH-R_L11_L24','L0_L25','L10_L25')
def main():
 verify_sources();sm=read(OUT/'sample_manifest.json')['Main'];official=TextVQAAccuracyEvaluator();assert read(OUT/'run_completion.json')['formal_records']==128 and read(OUT/'Native_reuse_calibration.json')['status']=='PASS'
 data={};origins={};auditrows=[]
 for base,reused in [(PARENT,True),(Path(S['zero25_run']),True),(RUN,False)]:
  db=sqlite3.connect('file:'+str(base/'predictions.sqlite')+'?mode=ro',uri=True);assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
  po=base/'results';protocol=sha(po/'protocol.json');source=sha(po/'source_manifest.json');expected=256 if base==PARENT else 128;count=0
  for qid,label,h,raw in db.execute('select qid,method,checksum,record from predictions'):
   r=json.loads(raw);assert digest(r)==h and r['question_id']==qid and r['method']==label and r['GPU_UUID']==GPU and r['protocol_sha256']==protocol and r['source_manifest_sha256']==source
   method='SVH-R_L11_L24' if reused and label=='SVH-R' else label
   assert method in METHODS
   data[qid,method]=r;origins[qid,method]='REUSED' if reused else 'NEW';count+=1
   auditrows.append(dict(qid=qid,method=method,origin=origins[qid,method],source_commit=(S['native_commit'] if base==PARENT else S['zero25_commit']) if reused else 'current branch source_manifest',checkpoint_checksum=h,original_method=label,original_protocol_sha256=protocol,original_source_manifest_sha256=source))
  assert count==expected;db.close()
 assert len(data)==512
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
   keep=range(32) if method=='Native' else range(11,25) if method=='SVH-R_L11_L24' else tuple(i for i in range(26) if i not in (2,3,4,5)) if method=='L0_L25' else KEEP
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
  summary.append(dict(method=method,N=128,origin='NEW' if method=='L10_L25' else 'REUSED',score_sum=sum(scores[method]),accuracy_percent=agg*100,empty=sum(x['empty'] for x in rr),truncated=sum(x['truncated'] for x in rr),EOS=sum(x['eos'] for x in rr),repeated=sum(x['repeated'] for x in rr),length_P50=float(np.percentile(lens,50)),length_P95=float(np.percentile(lens,95)),length_max=max(lens),natural_EOS_KV_live_MiB=float(np.mean([data[str(s['question_id']),method]['KV']['live_bytes']/2**20 for s in sm])),natural_EOS_KV_capacity_MiB=float(np.mean([data[str(s['question_id']),method]['KV']['storage_bytes']/2**20 for s in sm]))))
 groups=collections.defaultdict(list)
 for i,s in enumerate(sm):groups[s['image_sha256']].append(i)
 clusters=list(groups.values());counts=np.array([len(ix) for ix in clusters]);rng=np.random.default_rng(20261112);indices=rng.integers(0,len(clusters),(20000,len(clusters)));pairs=[];changes=[]
 for control in ('Native','SVH-R_L11_L24','L0_L25'):
  a=np.array(scores[control]);b=np.array(scores['L10_L25']);delta=b-a;sums=np.array([delta[ix].sum() for ix in clusters]);bootstrap=sums[indices].sum(1)/counts[indices].sum(1)*100;lo,hi=np.percentile(bootstrap,[2.5,97.5])
  pp=dict(comparison='L10_L25 minus '+control,delta_pp=float(delta.mean()*100),CI95_low_pp=float(lo),CI95_high_pp=float(hi),image_SHA_clusters=len(clusters),bootstrap=20000,CI='ordinary unadjusted95%,3 exploratory contrasts',improved=int((delta>1e-12).sum()),declined=int((delta< -1e-12).sum()),unchanged=int((abs(delta)<=1e-12).sum()),full_correct_to_not_full=int(((a==1)&(b<1)).sum()),not_full_to_full=int(((a<1)&(b==1)).sum()),positive_to_zero=int(((a>0)&(b==0)).sum()),zero_to_positive=int(((a==0)&(b>0)).sum()))
  cc=[]
  for i,s in enumerate(sm):
   qid=str(s['question_id']);x=data[qid,control];y=data[qid,'L10_L25']
   cc.append(dict(comparison=pp['comparison'],question_id=qid,image_sha256=s['image_sha256'],question=s['question'],gold=s['gold'],control_prediction=x['text'],new_prediction=y['text'],control_score=float(a[i]),new_score=float(b[i]),delta=float(delta[i]),first_token_changed=x['token_ids'][0]!=y['token_ids'][0],complete_tokens_changed=x['token_ids']!=y['token_ids'],text_changed=x['text']!=y['text']))
  pp.update(first_token_changed=sum(x['first_token_changed'] for x in cc),complete_tokens_changed=sum(x['complete_tokens_changed'] for x in cc),text_changed=sum(x['text_changed'] for x in cc));pairs.append(pp);changes+=cc
 csvwrite(OUT/'accuracy_summary.csv',summary);csvwrite(OUT/'per_question.csv',rows);csvwrite(OUT/'paired_comparison.csv',pairs);csvwrite(OUT/'per_question_changes.csv',changes);csvwrite(OUT/'harm_cases.csv',[x for x in changes if x['delta']< -1e-12]);csvwrite(OUT/'kv_layout_per_layer.csv',kvrows);csvwrite(OUT/'reuse_record_audit.csv',auditrows)
 apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip();assert not apps
 gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip();assert GPU in gpu
 write(OUT/'gpu_release_external.json',dict(status='PASS_WORKER_EXITED_GPU_RELEASED',GPU=gpu,compute_apps=apps,internal=read(OUT/'gpu_release.json')))
 smoke=read(OUT/'smoke_validation.json');assert smoke['status']=='PASS_NATURAL_EOS_QUALITY_AND_9STEP_STRUCTURE'
 validation=dict(status='PASS_REQUESTED_NATURAL_EOS_QUALITY',records_verified=512,new_formal=128,reused=384,all_record_checksums_and_inputs_source_protocol=True,independent_official_scores_agree=True,actual32layer_KV_and_positions=True,Native_reuse_calibration='3/3 complete Tokens and text equal same GPU',Smoke=smoke['status'],numeric_scope='unchanged historical thresholds on natural EOS valid steps; synthetic postEOS only structural/finite with numeric failures retained',same_GPU=GPU,model_loads=read(OUT/'run_completion.json')['model_loads'],all_answers_including_wrong_empty_truncated_counted=True,GPU_released=True,speed='NOT_RUN',GitHub_branch=S['branch'])
 write(OUT/'validation.json',validation);write(OUT/'analysis_digest.json',dict(summary=summary,paired=pairs))
 n,middle,previous,current=summary;v,w,z=pairs
 failures=[e for x in smoke['samples'] for e in x['synthetic_post_EOS_original_threshold_failures']]
 report=f'''# LLaVA-SVH-R L10–L25：TextVQA Image-only 128题
本轮L10–L25共16层回看，官方十答案软评分为{current['accuracy_percent']:.6f}%，软分合计{current['score_sum']:.4f}/128。同GPU Native {n['accuracy_percent']:.6f}%，差值{v['delta_pp']:+.6f}pp；相比上轮L0–L25 {previous['accuracy_percent']:.6f}%，差值{z['delta_pp']:+.6f}pp。

|配置|题数|官方软评分Accuracy|来源|
|---|---:|---:|---|
|Native Full|128|{n['accuracy_percent']:.6f}%|复用同GPU {S['native_commit']}|
|SVH-R L11–L24|128|{middle['accuracy_percent']:.6f}%|同上|
|SVH-R L0–L25，跳过L2–L5|128|{previous['accuracy_percent']:.6f}%|复用同GPU {S['zero25_commit']}|
|SVH-R L10–L25|128|{current['accuracy_percent']:.6f}%|本轮新增128次|

## 逐题配对
本轮−Native：{v['delta_pp']:+.6f}pp，图片SHA整簇Bootstrap普通95%CI [{v['CI95_low_pp']:+.6f}, {v['CI95_high_pp']:+.6f}]pp；上升{v['improved']}题、下降{v['declined']}题、不变{v['unchanged']}题。
本轮−L11–L24：{w['delta_pp']:+.6f}pp，95%CI [{w['CI95_low_pp']:+.6f}, {w['CI95_high_pp']:+.6f}]pp。
本轮−L0–L25：{z['delta_pp']:+.6f}pp，95%CI [{z['CI95_low_pp']:+.6f}, {z['CI95_high_pp']:+.6f}]pp；上升{z['improved']}题、下降{z['declined']}题、不变{z['unchanged']}题。
按128个图片SHA整簇同题配对重采样20000次，seed20261112，三个比较均为普通未调整95%CI。本128题已用于多轮探索，不是独立确认集；不据此宣称最佳层、严格等价/非劣效或完整5000题成绩。

## 冻结配置
模型llava-hf/llava-1.5-7b-hf，Revision {REVISION}；FP16/Eager、Torch2.6.0+cu124、Transformers4.49.0、Batch1、Greedy、正常EOS、max_new_tokens32、use_cache=True、logits_to_keep=1。
原128题Question ID、顺序、图片SHA、问题、十答案Gold、原生HF Chat及single-word-or-phrase后缀、CLIP336 shortest-edge/center-crop不变。Image-only，无OCR和Gold输入。仅复用本地已有模型、图片、prepared Tensor，未重新下载或全量安装。
P−1与真实Decode只在L10–L25共16层读取原始Image Key；L2–L5答案Attention持续真实物理跳算，R→Image仍仅L0–L19允许；进入L26物理删除576视觉Hidden，后六层仅处理非视觉Hidden。真实Compact KV只在L10–L25保留原视觉KV，L2–L5无持久KV；原逻辑RoPE、非目标文本Attention、32层Norm/MLP/Residual不变。前一轮L0–L25与本轮具有相同退出位置L26，区别是原视觉回看集合22层变16层；L11–L24历史配置退出位置L25，不能将该比较全部归因于层集合。
固定{S['source_commit']}的26个核心文件及独立TerminalMiddleWriter字节不变，独立目录只配置KEEP与cutoff并核验，不修改冻结实现。

## 验收与完整性
CPU验证原cutoff25严格torch.equal回归、本轮cutoff26实际退出、32层真实KV和逻辑位置；原131份输入（128主集+3外部Smoke）图片/模型Tensor SHA与历史一致，Gold保持分离。既有三个条件384条原记录逐题核验checksum、原协议/源码/输入SHA、GPU UUID和Manifest，保留原Commit。
GPU短/中/长三题Smoke：Native/No-op有效Logits严格torch.equal；Native循环与原生HF generate完整Token一致；本轮Dense/Physical完整自然EOS Token一致。沿用先前v2数值门槛：正常EOS有效步Logits atol0.10/rtol0.003，Hidden relativeRMSE<=0.01、cosineerror<=1e-4；强制9步EOS后仅结构与有限性，仍保留数值差异，本轮EOS后原阈值失败{len(failures)}步，不声明固定长度严格数值等价。审计确认L26真实删行、目标投影跳算、禁止连接概率0、原RoPE与每层KV正确。
正式前最短/中位/最长3题Native完整Token和文本校准通过才复用既有基线。仅新增128次正式生成，模型加载一次，不重跑旧128题条件。
全部512条记录由官方TextVQA十答案共识软评分和独立leave-one-out公式重算一致；完整预测、Token IDs、Gold、原始SHA、32层KV和逐题损害均保存。本轮空输出{current['empty']}、截断{current['truncated']}、EOS{current['EOS']}、重复启发式{current['repeated']}，全部计分；长度P50/P95/最大{current['length_P50']}/{current['length_P95']}/{current['length_max']}。
GPU始终{GPU}，结束已释放，无计算进程。不继续暂停的单层扫描，不运行InternVL、GQA、OCRBench、训练、其他窗口或任何性能测量。KV为正常EOS终点，不是KV@32，不作速度声明。

## 文件与同步
SSD完整本轮检查点：{RUN}/predictions.sqlite
持久化结果：{S['root']}/{S['result']}/
源码：{S['root']}/{S['pkg']}/
独立GitHub分支：{S['branch']}。代码、轻量CSV、协议、报告、验证和SHA清单同步GitHub；SQLite与详细日志仅保存在服务器，artifact_manifest逐项区分。模型、图像、prepared Tensor、缓存和密钥不上传。未修改main或历史实验，不创建PR。
'''
 (OUT/'report_zh.md').write_text(report)
 print(json.dumps(dict(status='ANALYSIS_PASS',summary=summary,paired=pairs,post_EOS_numeric_failed_steps=len(failures)),ensure_ascii=False))
if __name__=='__main__':main()

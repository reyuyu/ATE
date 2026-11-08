"""Independent CPU audit of every generation, official scoring and paired CIs."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import sys,json,gzip,hashlib,csv,sqlite3,subprocess,re,io
import numpy as np
from transformers import AutoTokenizer
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/featurize_4090_gqa_official_aligned_threeway';A=O.parent/'featurize_4090_gqa_native_protocol_audit'
D=Path('/home/featurize/data/ate-recovery')
sys.path[:0]=[str(R),str(A)]
from protocol_helpers import official_functions,official_prompt
from vlm_research.gqa_testdev_c3.common import score as frozen_score,prediction as frozen_prediction

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(n,rows):
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def manual_parse(text):
 while text.endswith('.'):text=text[:-1]
 return text.lower()
def save_gzip_text(name,text):
 path=O/name;raw=text.encode('utf-8')
 if path.exists():assert gzip.decompress(path.read_bytes())==raw,'Refuse replacing differing raw scientific evidence'
 else:path.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
progress=read(O/'progress.json');p=read(O/'protocol.json');config=read(O/'execution_config.json');binding=sha(O/'protocol.json')
assert progress['status'] in {'GENERATIONS_COMPLETE','COMPLETE'} and progress['completed']==37734
assert progress['counts']==dict.fromkeys(p['conditions'],12578) and config['protocol_sha256']==binding
assert sha(O/'run.py')==config['runner_sha256'] and sha(O/'prepare.py')==config['prepare_sha256']
assert sha(O/'question_order.csv')==p['question_order_sha256']
assert sha(A/'official_sources/source_manifest.json')==p['audit_source_manifest_sha256']
for name,info in read(A/'official_sources/source_manifest.json')['sources'].items():assert sha(A/'official_sources'/name)==info['sha256'],name
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Wait for the inference process to exit before final audit'
assert sha(O/'input_manifest.jsonl.gz')==p['manifest_sha256']
with gzip.open(O/'input_manifest.jsonl.gz','rt') as f:samples=[json.loads(l) for l in f]
questions={s['question_id']:s for s in samples};assert len(questions)==len(samples)==12578
goldpath=D/'restored_bundles/ate3090-gqa-testdev-balanced/source_annotations/gqa/testdev_balanced_questions.json'
assert sha(goldpath)==p['official_Gold_sha256'];gold=read(goldpath)
preparedpath=D/'audit_support/gqa_native_protocol/official_llava_gqa_testdev_balanced.jsonl';assert sha(preparedpath)==p['official_prepared_sha256']
prepared=[json.loads(l) for l in preparedpath.read_text().splitlines()]
assert [str(s['question_id']) for s in prepared]==[s['question_id'] for s in samples]
conv,_=official_functions();tokenizer=AutoTokenizer.from_pretrained(p['model_path'],local_files_only=True)
for s,official in zip(samples,prepared):
 q=s['question_id'];g=gold[q];assert s['question']==g['question'] and s['answer']==g['answer'] and s['image_id']==g['imageId']
 assert official['text']==s['prompt_text'] and Path(official['image']).stem==s['image_id'] and g['isBalanced']
 assert official_prompt(s,conv)==s['prompt'] and hashlib.sha256(s['prompt'].encode()).hexdigest()==s['prompt_sha256']
 temp=dict(s);saved=temp.pop('sample_binding');assert digest(temp)==saved
db=sqlite3.connect('file:'+str(O/'results.sqlite')+'?mode=ro',uri=True);rows={}
for q,label,payload,h in db.execute('SELECT qid,condition,payload,sha256 FROM results'):
 r=json.loads(payload);assert digest(r)==h and r['protocol_sha256']==binding and (q,label)==(r['question_id'],r['condition'])
 assert (q,label) not in rows and q in questions and label in p['conditions'];s=questions[q];g=r['generation']
 assert r['index']==s['index'] and r['sample_binding']==s['sample_binding'] and r['input_hashes']==s['input_hashes'] and r['prompt_sha256']==s['prompt_sha256']
 assert r['gold']==s['answer'] and r['image_id']==s['image_id'] and r['image_sha256']==s['image_sha256'] and r['question']==s['question']
 text=tokenizer.decode(g['token_ids'],skip_special_tokens=True).strip();assert text==g['text']
 normalized=manual_parse(text);score=int(normalized==gold[q]['answer'])
 assert score==r['score']==frozen_score(s,text) and normalized==r['parsed']==frozen_prediction(text)
 assert 1<=len(g['token_ids'])<=32 and g['eos']==(g['token_ids'][-1]==2) and g['truncated']==(len(g['token_ids'])==32 and not g['eos'])
 assert g['empty']==(not bool(text)) and g['all_effective_logits_finite'] and r['vision_forward_calls']==1
 rows[q,label]=r
db.close();assert len(rows)==37734 and set(rows)=={(q,label) for q in questions for label in p['conditions']}
support=D/'audit_support/gqa_official_aligned_threeway';support.mkdir(parents=True,exist_ok=True)
summaries=[];vectors={};bytype=[];converted_hashes={};official_metrics={}
official_wrapper="""import sys,runpy,json
entry,metric_path=sys.argv[1:3]
sys.argv=[entry]+sys.argv[3:]
ns=runpy.run_path(entry,run_name='__main__')
n=sum(bool(q['isBalanced']) for q in ns['questions'].values())
accuracy=ns['scores']['accuracy']
correct=round(accuracy*n/100)
metrics={'accuracy_percent_unrounded':accuracy,'balanced_questions':n,'predictions':len(ns['predictions']),'correct':correct,'original_script_executed_unmodified':True}
with open(metric_path,'w') as f:json.dump(metrics,f,indent=2);f.write('\\n')
"""
for label in p['conditions']:
 rr=[rows[s['question_id'],label] for s in samples];vectors[label]=np.array([r['score'] for r in rr],dtype=np.int64)
 name=label.replace('+','_');rawpath=support/(name+'_raw_for_official_conversion.jsonl');dst=support/(name+'_official_predictions.json')
 with rawpath.open('w') as f:
  for r in rr:f.write(json.dumps({'question_id':r['question_id'],'text':r['generation']['text']},ensure_ascii=False)+'\n')
 subprocess.run([sys.executable,str(A/'official_sources/release_convert_gqa.py'),'--src',str(rawpath),'--dst',str(dst)],check=True)
 converted=read(dst);assert len(converted)==12578 and all(x['prediction']==rows[x['questionId'],label]['parsed'] for x in converted)
 metric_path=O/('official_score_'+name+'.json')
 result=subprocess.run([sys.executable,'-c',official_wrapper,str(A/'official_sources/GQA_eval_llava.py'),str(metric_path),'--tier','testdev_balanced','--questions',str(goldpath),'--predictions',str(dst)],text=True,capture_output=True)
 (O/('official_score_'+name+'.log')).write_text(result.stdout+'\n'+result.stderr);assert result.returncode==0,result.stderr
 accuracy_match=re.search(r'Accuracy:\s*([0-9.]+)%',result.stdout)
 assert accuracy_match and abs(float(accuracy_match.group(1))-100*int(vectors[label].sum())/12578)<0.00501,result.stdout
 official_metrics[label]=read(metric_path)
 assert official_metrics[label]['correct']==int(vectors[label].sum()) and official_metrics[label]['balanced_questions']==official_metrics[label]['predictions']==12578
 assert abs(official_metrics[label]['accuracy_percent_unrounded']-100*int(vectors[label].sum())/12578)<1e-10
 converted_hashes[label]=sha(dst)
 save_gzip_text('predictions_'+name+'.jsonl.gz',''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rr))
 correct=int(vectors[label].sum());summaries.append({'condition':label,'questions':12578,'correct':correct,'accuracy_percent':100*correct/12578,
  'delta_native_pp':100*(correct-int(vectors['Native'].sum()))/12578,'empty':sum(r['generation']['empty'] for r in rr),
  'truncated':sum(r['generation']['truncated'] for r in rr),'max_generated_tokens':max(len(r['generation']['token_ids']) for r in rr),'official_evaluator_status':'PASS'})
 for kind in sorted(p['structural_counts']):
  subset=[r for r in rr if questions[r['question_id']]['types']['structural']==kind]
  bytype.append({'condition':label,'structural':kind,'questions':len(subset),'correct':sum(r['score'] for r in subset),'accuracy_percent':100*sum(r['score'] for r in subset)/len(subset)})
csvwrite('scores.csv',summaries);csvwrite('scores_by_type.csv',bytype)
early_native=read(O/'completed_native_early_audit.json')
assert early_native['status']=='PASS' and early_native['questions']==summaries[0]['questions'] and early_native['correct']==summaries[0]['correct']
early_c3=read(O/'completed_c3_early_audit.json')
assert early_c3['status']=='PASS' and early_c3['questions']==summaries[1]['questions'] and early_c3['correct']==summaries[1]['correct']
with gzip.open(O.parent/'featurize_new3090_c3v2_reaccess_sevenbench/gqa_historical_controls.jsonl.gz','rt') as f:old={r['question_id']:r['controls']['Native'] for r in map(json.loads,f)}
assert set(old)==set(questions) and sum(r['score'] for r in old.values())==7626
vectors['Historical Native']=np.array([old[s['question_id']]['score'] for s in samples],dtype=np.int64)
images=sorted({s['image_sha256'] for s in samples});cluster_index={s:i for i,s in enumerate(images)};codes=np.array([cluster_index[s['image_sha256']] for s in samples])
cluster_n=np.bincount(codes,minlength=len(images));assert len(images)==398
rng=np.random.default_rng(config['cluster_bootstrap']['seed']);indices=rng.integers(0,len(images),size=(config['cluster_bootstrap']['resamples'],len(images)))
denominator=cluster_n[indices].sum(axis=1);pairs=[]
for candidate,reference in [('C3-V2','Native'),('C3-V2+L11-L24','Native'),('C3-V2+L11-L24','C3-V2'),('Native','Historical Native')]:
 delta=vectors[candidate]-vectors[reference];cluster_delta=np.bincount(codes,weights=delta,minlength=len(images));distribution=100*cluster_delta[indices].sum(axis=1)/denominator
 ci=np.percentile(distribution,[2.5,97.5]);pairs.append({'candidate':candidate,'reference':reference,'paired_questions':12578,'image_SHA_clusters':len(images),
  'accuracy_delta_pp':100*int(delta.sum())/12578,'improved':int((delta==1).sum()),'worsened':int((delta==-1).sum()),'both_correct':int(((vectors[candidate]==1)&(vectors[reference]==1)).sum()),
  'both_wrong':int(((vectors[candidate]==0)&(vectors[reference]==0)).sum()),'bootstrap95_low_pp':float(ci[0]),'bootstrap95_high_pp':float(ci[1]),'resamples':len(indices),'bootstrap_seed':config['cluster_bootstrap']['seed']})
csvwrite('paired_differences.csv',pairs)
# Reproducibility check against the already-completed same-4090 official-aligned C arm.
prior=[json.loads(l) for l in (A/'gpu_predictions.jsonl').read_text().splitlines() if json.loads(l)['condition']=='C']
assert len(prior)==200
for r in prior:
 new=rows[r['question_id'],'Native'];assert new['input_hashes']==r['input_hashes'] and new['generation']['token_ids']==r['generation']['token_ids'] and new['score']==r['score']
with io.StringIO(newline='') as f:
 fields=['question_id','image_sha256','gold','Native_text','Native_score','C3_text','C3_score','Reaccess_text','Reaccess_score'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for s in samples:
  q=s['question_id'];r={'question_id':q,'image_sha256':s['image_sha256'],'gold':s['answer']}
  for label,prefix in [('Native','Native'),('C3-V2','C3'),('C3-V2+L11-L24','Reaccess')]:r[prefix+'_text']=rows[q,label]['generation']['text'];r[prefix+'_score']=rows[q,label]['score']
  w.writerow(r)
 save_gzip_text('per_question_score_audit.csv.gz',f.getvalue())
physical=read(O/'physical_validation_summary.json')
audit={'status':'PASS','questions_per_condition':12578,'records':37734,'missing_records':0,'duplicate_records':0,'Gold_question_image_mismatches':0,
 'independent_score_mismatches':0,'token_decode_mismatches':0,'shared_input_hash_mismatches':0,'official_conversion_and_evaluators':'PASS',
 'same_4090_prior_200_official_aligned_outputs_exact':True,'manifest_sha256':p['manifest_sha256'],'protocol_sha256':binding,
 'converted_predictions_sha256':converted_hashes,'official_unrounded_metrics':official_metrics,'all_errors_empty_and_truncations_retained':True,'CPU_only':True,'historical_native_correct':7626,
 'new_native_minus_historical_pp':100*(int(vectors['Native'].sum())-7626)/12578,'new_native_minus_official_rounded_62_pp':100*int(vectors['Native'].sum())/12578-62.0,
 'official_62_reference_limit':'Rounded public 7B ModelZoo value; original native/HF backend and weight numerical parity remain unproven.'}
save('independent_scoring_audit.json',audit)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip()
metadata={'status':'COMPLETE','finished_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':binding,'execution_config_sha256':sha(O/'execution_config.json'),
 'code_commit':p['code_commit'],'model_revision':p['revision'],'audit_reference_commit':p['audit_commit'],'scores':summaries,'paired_differences':pairs,
 'physical_validation':physical['status'],'strict_numerical_gate':physical['strict_numerical_gate'],'warning_waiver_used':False,
 'generation_progress':progress,'independent_scoring_audit':audit,'GPU_after_exit':gpu,'unrelated_experiments_started':False}
save('run_metadata.json',metadata)
examples=[questions[q] for q in p['physical_smoke_question_ids']]
save('prompt_examples.json',[{k:s[k] for k in ['question_id','question','prompt','prompt_sha256','input_ids','question_positions','input_hashes','sample_binding']} for s in examples])
lines=['# GQA 官方对齐协议三组正式评测','', '**状态：COMPLETE。** 三组各12,578题，共37,734条正式生成；全部逐题预测保留，独立CPU复核及官方评分脚本通过。','',
 '| 方法 | 正确数 / 12,578 | GQA Accuracy | 相对 Native |','|---|---:|---:|---:|']
for s in summaries:lines.append(f"| {s['condition']} | {s['correct']} | {s['accuracy_percent']:.7f}% | {'—' if s['condition']=='Native' else format(s['delta_native_pp'],'+.7f')+' pp'} |")
lines+=['','| 配对差值 | 改善 / 恶化 | Accuracy差值 | 按Image SHA簇Bootstrap 95% CI |','|---|---:|---:|---:|']
for x in pairs:lines.append(f"| {x['candidate']} − {x['reference']} | {x['improved']} / {x['worsened']} | {x['accuracy_delta_pp']:+.7f} pp | [{x['bootstrap95_low_pp']:+.4f}, {x['bootstrap95_high_pp']:+.4f}] pp |")
lines+=['',f"图像SHA簇398个，有放回抽取398簇并按实际题数重算Accuracy差值；10,000次，seed20261108，percentile区间。此为固定testdev集合上的配对不确定性估计，不用于证明模型与官方原生后端等价。",
 f"新协议Native比历史60.6296709%变化 **{audit['new_native_minus_historical_pp']:+.7f} pp**；比官方四舍五入公开62.0%变化 **{audit['new_native_minus_official_rounded_62_pp']:+.7f} pp**。官方值只作参考，未据此调参。历史与新协议的Prompt、预处理及硬件运行记录不同；三种新方法内部输入完全一致。新旧Native差值不能单独分解System、pad和跨硬件浮点残余贡献，本轮没有重跑旧协议全量基线。",'',
 '## 协议与可追溯性','',
 f"模型 `{p['model']}@{p['revision']}`；代码基线 `{p['code_commit']}`，冻结Physical源码哈希与历史GQA执行冻结逐项相同。复用4090本地盘模型、数据和既有Torch2.6.0+cu124 / Transformers4.49.0环境；FP16/Eager/batch1/greedy/EOS2/max_new_tokens32。",
 '完整Vicuna-v1 System + USER + 图像 + 官方原题及single-word-or-phrase指令；RGB→CLIP mean色补方形→CLIP336。按官方prepared jsonl原始顺序完成Native、C3、Reaccess；旧清单使用另一顺序，历史对照全部以qid对齐。每条新预测绑定同一个样本输入张量哈希、Prompt哈希及protocol哈希。',
 f"Protocol SHA256 `{binding}`；输入Manifest SHA256 `{p['manifest_sha256']}`；官方Gold SHA256 `{p['official_Gold_sha256']}`。详见protocol.json、execution_config.json、prompt_examples.json和本地完整input_manifest.jsonl.gz。",'',
 '## Physical验收','',
 '3道固定GQA题×2种Physical条件，共6次严格Dense数值比较全部PASS；没有使用PASS_WITH_NUMERICAL_WARNING豁免。L20/L25前视觉Hidden退出、R只在L0–L19读取Image Key、回看只在L11–L24、L2–L5物理跳算、Norm/MLP/Residual、RoPE和紧凑KV位置均通过。Dense仅作短序列检查，正式37,734条均使用各自冻结实现。',
 f"六次比较最大Hidden Relative RMSE为{max(x['numerical_equivalence']['hidden_relative_rmse'] for x in physical['checks']):.9f}，原阈值仍为0.01。2道旧协议题×2种C3的跨硬件输入及完整Token输出均匹配历史。上一轮POPE约0.0101545的严格FAIL保持原样，本轮没有重分类或修改阈值。",
 f"正式进程三组共享一次模型加载。此前新入口误将回看结果从仅含Native/C3的历史表读取，导致0条正式生成时退出；已修正为读取独立历史回看文件，并在CPU阶段验证来源。失败代码、日志和配置保存在attempt1_entry_lookup_failure；总加载次数记录为{progress['model_loads']}，未隐瞒。",'',
 '## 独立评分审计与资源','',
 '全部37,734条记录检查qid唯一性、数量、官方Question/Gold/Image映射、输入绑定、token解码、EOS/截断和原始输出。另以逐字符移除尾句号再lower的独立逻辑复算，并实际调用未修改的官方转换和GQA evaluator；0差异。此前同4090的200条官方对齐C组与本次Native的Token和分数全部一致。','',
 '| 条件 | 空输出 | 截断 | 最长生成Token数（含EOS） |','|---|---:|---:|---:|']
for s in summaries:lines.append(f"| {s['condition']} | {s['empty']} | {s['truncated']} | {s['max_generated_tokens']} |")
lines+=['',f"GPU已释放：`{gpu}`。无遗留推理进程；未启动其他Benchmark、层搜索或V²Drop。逐题SQLite同步事务持久化，断点只跳过绑定相同的已完成样本。",
 '完整预测压缩文件、Manifest、逐题评分表及必要配置留在独立本地目录；Windows备份和GitHub远端逐文件SHA校验结果见本地windows_backup_receipt.json及github_sync_receipt.json。GitHub仅同步轻量源码、正式分数、统计、哈希和报告，完整预测不上传。','',
 '[官方7B Model Zoo参考](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/MODEL_ZOO.md)；[此前协议审计](https://github.com/reyuyu/ATE/commit/25989d8c0e00cf53a980c726be94d84feaabbc14)。']
(O/'report_zh.md').write_text('\n'.join(lines)+'\n')
save('progress.json',dict(progress,status='COMPLETE',phase='GENERATION_AND_SCORING_COMPLETE',scoring_finished_at=datetime.now(timezone.utc).isoformat(),delivery_receipts='windows_backup_receipt.json and github_sync_receipt.json are created and verified in the subsequent archival/sync stage'))
print('INDEPENDENT_OFFICIAL_SCORING_COMPLETE',json.dumps({'scores':summaries,'pairs':pairs,'GPU':gpu},ensure_ascii=False),flush=True)

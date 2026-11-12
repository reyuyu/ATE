"""CPU-only full TextVQA coverage, independent official scoring and paired image-cluster statistics."""
import sys,json,gzip,csv,shutil,subprocess,hashlib
from pathlib import Path
O=Path('/home/featurize/ate-experiments/internvl-svh-r-new-textvqa5000-22875-20261113');L=Path('/home/featurize/ate-experiments/ocrbench-ate-frozen-completion-22875-20261113')
sys.path.insert(0,str(L));from run_common import atomic,sha
sys.path.insert(0,str(O));from scoring import Scorer
import numpy as np
from transformers import AutoTokenizer
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert json.loads((O/'progress.json').read_text())['status']=='COMPLETE_GENERATION_PENDING_CPU_AUDIT'
P=json.loads((O/'protocol.json').read_text());pb=sha(O/'protocol.json');rows=json.loads((O/'frozen_rows.json').read_text());ids=[r['question_id'] for r in rows];byid={r['question_id']:r for r in rows};expected=json.loads((O/'expected_bindings.json').read_text());baseline=json.loads((O/'baselines.json').read_text())
for f,k in [('frozen_rows.json','rows_sha256'),('expected_bindings.json','bindings_sha256'),('baselines.json','baselines_sha256'),('evaluation_config.json','evaluation_config_sha256'),('configuration.json','configuration_sha256')]:assert sha(O/f)==P[k]
for p,h in P['source_SHA256'].items():assert sha(L/'late9_sources'/p)==h,p
new={p.stem:json.loads(p.read_text()) for p in (O/'predictions').glob('*.json')};assert len(new)==5000 and set(new)==set(ids) and len(set(ids))==5000
assert [json.loads(x) for x in gzip.open(O/'references/TextVQA.jsonl.gz','rt')]==rows
D=Path('/home/featurize/ate-local-data/bundles/ate3090-textvqa-ocr');ann=D/'source_annotations/textvqa/TextVQA_0.5.1_val.json';assert sha(ann)=='4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d'
raw_rows=json.loads(ann.read_text())['data'];raw={str(x['question_id']):x for x in raw_rows};assert set(raw)==set(ids)
for r in rows:
 a=raw[r['question_id']];assert r['question']==a['question'] and r['gold']==a['answers'] and r['image_id']==a['image_id']
raw_annotation_same_order=[str(x['question_id']) for x in raw_rows]==ids
scorer=Scorer(rows,O/'references/official_textvqa_eval.py');tok=AutoTokenizer.from_pretrained('/home/featurize/ate-local-models/InternVL3-8B',trust_remote_code=True,local_files_only=True,use_fast=False)
pred=[];audits=[]
for q in ids:
 x=new[q];g=x['generation'];r=byid[q];b=x['input_binding_details']
 assert x['protocol_sha256']==pb and g['reaccess_layers']==list(range(14,28)) and g['cutoff']==28 and g['independent_cache']
 assert [a['layer'] for a in g['kv_layout'] if a['visual_kv']]==list(range(14,28))
 assert all(b[k]==expected[q][k] for k in expected[q])
 text=tok.decode(g['token_ids'],skip_special_tokens=True).split('<|im_end|>')[0].strip();assert text==g['text']
 assert len(g['token_ids'])==g['generated_tokens']<=32 and g['eos']==(g['token_ids'][-1]==151645)
 assert g['truncated']==(len(g['token_ids'])==32 and not g['eos']) and g['empty']==(not text)
 s=scorer.score(q,text);assert abs(s['official_score']-x['official_score'])<1e-12 and abs(s['independent_score']-x['independent_score'])<1e-12
 pred.append(dict(method=P['configuration_name'],question_id=q,image_sha256=r['image_sha256'],prompt_sha256=r['prompt_sha256'],input_binding_sha256=x['input_binding'],input_binding_details=b,text=text,token_ids=g['token_ids'],**s,eos=g['eos'],truncated=g['truncated'],empty=g['empty']))
 if q in P['smoke_ids']:
  a=g['audit'];assert a['non_target_queries_preserved'] and a['original_logical_positions'] and a['finite']
  assert a['hidden_exit']==[dict(before=r['input_tokens'],after=r['input_tokens']-256,removed=256)] and a['exit_stage']=='after_L27_before_final_norm'
  assert a['allowed_image_probability_samples'] and all(z[2]>0 for z in a['allowed_image_probability_samples'])
  assert all(z[2]==0 for z in a['blocked_image_probability_samples']+a['R_image_probability_samples'])
  audits.append(dict(question_id=q,audit=a,kv_layout=g['kv_layout']))
for m,rs in baseline.items():
 assert len(rs)==5000 and {x['question_id'] for x in rs}==set(ids)
 for x in rs:
  q=x['question_id'];r=byid[q];assert x['gold']==r['gold'] and x['image_sha256']==r['image_sha256'] and x['prompt_sha256']==r['prompt_sha256']
  s=scorer.score(q,x['text']);assert abs(x['official_score']-s['official_score'])<1e-12
data={'Native':baseline['native'],'Physical-D':baseline['physical_d'],P['configuration_name']:pred};imap={m:{x['question_id']:x for x in rs} for m,rs in data.items()}
table=[dict(method=m,coverage=len(rs),score_sum=sum(x['official_score'] for x in rs),accuracy_percent=100*sum(x['official_score'] for x in rs)/5000,empty=sum(x['empty'] for x in rs),truncated=sum(x['truncated'] for x in rs),source='NEW5000' if m==P['configuration_name'] else P['source_manifest_commit']) for m,rs in data.items()]
table.insert(2,dict(method='Late9',coverage=5000,score_sum=56.004*50,accuracy_percent=56.004,empty=None,truncated=None,source='PUBLISHED_AGGREGATE_ONLY@'+P['late9_summary_commit']))
clusters=sorted({r['image_sha256'] for r in rows});assert len(clusters)==3166
cluster_index={s:i for i,s in enumerate(clusters)};ci_idx=np.array([cluster_index[r['image_sha256']] for r in rows]);sizes=np.bincount(ci_idx,minlength=len(clusters));paired=[];cases=[];name=P['configuration_name']
for ref in ['Native','Physical-D']:
 diffs=np.array([imap[name][q]['official_score']-imap[ref][q]['official_score'] for q in ids]);sums=np.bincount(ci_idx,weights=diffs,minlength=len(clusters));rng=np.random.default_rng(20261113);rep=[]
 for start in range(0,10000,256):
  boot=rng.integers(0,len(clusters),size=(min(256,10000-start),len(clusters)));rep.extend(100*sums[boot].sum(1)/sizes[boot].sum(1))
 ci=np.quantile(rep,[.025,.975]);paired.append(dict(method=name,reference=ref,n=5000,delta_pp=100*float(diffs.mean()),soft_score_improved=int((diffs>1e-12).sum()),soft_score_declined=int((diffs< -1e-12).sum()),equal=int((abs(diffs)<=1e-12).sum()),zero_to_positive=sum(imap[ref][q]['official_score']==0 and imap[name][q]['official_score']>0 for q in ids),positive_to_zero=sum(imap[ref][q]['official_score']>0 and imap[name][q]['official_score']==0 for q in ids),ci95_low_pp=float(ci[0]),ci95_high_pp=float(ci[1]),image_clusters=3166,bootstrap_replicates=10000,seed=20261113))
 for q,d in zip(ids,diffs):
  if abs(d)>1e-12:cases.append(dict(reference=ref,question_id=q,question=byid[q]['question'],gold=json.dumps(byid[q]['gold'],ensure_ascii=False),reference_output=imap[ref][q]['text'],new_output=imap[name][q]['text'],reference_score=imap[ref][q]['official_score'],new_score=imap[name][q]['official_score'],score_change=float(d)))
pub=O/'publication/vlm_research/results/internvl_svhr_new_textvqa5000_22875_20261113';pub.mkdir(parents=True,exist_ok=True)
def csvout(name,rs):
 with (pub/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def gzout(name,rs):
 with gzip.GzipFile(str(pub/name),'wb',mtime=0) as f:
  for x in rs:f.write((json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n').encode())
csvout('scores.csv',table);csvout('paired_statistics.csv',paired);csvout('changed_score_cases.csv',cases)
csvout('per_question_scores.csv',[dict(question_id=q,image_sha256=byid[q]['image_sha256'],**{m:imap[m][q]['official_score'] for m in data}) for q in ids])
gzout('predictions_interVL_SVH_R_new.jsonl.gz',pred);gzout('baseline_predictions.jsonl.gz',[dict(method=m,**{k:v for k,v in x.items() if k!='gold'}) for m,rs in data.items() if m!=name for x in rs]);gzout('inputs_light.jsonl.gz',[{k:r[k] for k in ['question_id','question','image_id','image_sha256','prompt','prompt_sha256','input_ids','roles','max_new_tokens']} for r in rows]);gzout('smoke_physical_audits.jsonl.gz',audits)
atomic(pub/'gold.json',[dict(question_id=r['question_id'],gold=r['gold']) for r in rows])
atomic(pub/'dataset_audit.json',dict(status='PASS',questions=5000,unique_images=3166,all_id_question_image_gold_mappings_match=True,frozen_manifest_order_preserved=True,raw_annotation_same_order=raw_annotation_same_order,raw_annotation_sha256=sha(ann),original_manifest_gzip_sha256=sha(O/'references/TextVQA.jsonl.gz'),manifest_source_commit=P['source_manifest_commit']))
for f in ['protocol.json','configuration.json','evaluation_config.json','cpu_audit.json','smoke.json','environment.json']:shutil.copy2(O/f,pub/f)
shutil.copy2(O/'reference_receipt.json',pub/'reference_receipt.json');shutil.copy2(O/'references/history_inventory.json',pub/'historical_git_blob_inventory.json')
(pub/'execution').mkdir(exist_ok=True)
for f in ['prepare.py','run.py','analyze.py','scoring.py']:shutil.copy2(O/f,pub/'execution'/f)
(pub/'scoring').mkdir(exist_ok=True);shutil.copy2(O/'references/official_textvqa_eval.py',pub/'scoring/official_textvqa_eval.py')
atomic(pub/'scorer_sources.json',dict(repo='OpenGVLab/InternVL',commit='2410d1dbf208f0e799459aff9376e5747dbf41a2',path='internvl_chat/eval/vqa/textvqa_eval.py',sha256=sha(pub/'scoring/official_textvqa_eval.py'),official_method='TextVQAAccuracyEvaluator._compute_answer_scores',independent_method='Independent normalization; k matches among10 gold answers yields [0,.3,.6,.9,1,1,1,1,1,1,1][k].'))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.used,memory.free','--format=csv,noheader'],text=True).strip();score=100*sum(x['official_score'] for x in pred)/5000
summary=dict(status='COMPLETE_FULL_COVERAGE_WITH_NUMERICAL_WAIVER',configuration_name=name,benchmark='TextVQA v0.5.1 validation',coverage=5000,expected=5000,unique_images=3166,accuracy_percent=score,scores=table,paired=paired,delta_vs_published_late9_pp=score-56.004,independent_scoring_checks=15000,cpu_token_decode_checks=5000,empty=sum(x['empty'] for x in pred),truncated=sum(x['truncated'] for x in pred),new_generations=5000,baseline_generations=0,numerical_status=P['numerical_status'],GPU_after=gpu)
atomic(pub/'summary.json',summary)
report=[f'# {name}：TextVQA5000','',f'状态：COMPLETE_FULL_COVERAGE_WITH_NUMERICAL_WAIVER。5000/5000题、3166张唯一图片，VQA Soft Accuracy **{score:.3f}%**。仅新生成Late14，Native与Physical-D复用历史逐题结果；Late9引用已发布全集汇总。','','## 协议与结果','','TextVQA v0.5.1 validation，严格继承ATE冻结Image-only输入（无OCR文字）、单448×448图块/256视觉Token、原问题及短答指令、InternVL Chat Template、BF16/Eager、Greedy、max_new_tokens=32、EOS151645。使用10答案leave-one-out VQA软评分，不是二值Exact-match，也不是LLaVA Paper-OCR或动态24图块新协议。','','| 方法 | 覆盖 | VQA Soft Accuracy | 空输出 | 截断 |','|---|---:|---:|---:|---:|']
for x in table:report.append(f"| {x['method']} | {x['coverage']} | {x['accuracy_percent']:.3f}% | {x['empty'] if x['empty'] is not None else '历史汇总未载入'} | {x['truncated'] if x['truncated'] is not None else '历史汇总未载入'} |")
report+=['',f'Late14相对已发布Late9：{score-56.004:+.3f} pp。Late9为同冻结协议历史全集56.004%的汇总引用，本轮没有下载其40MB全Benchmark逐题表，也不把4370条changed-only案例误当全集；因此不提供Late9配对改善数量或置信区间。','','| 对照 | Late14−对照 pp | 软分提高 | 软分降低 | 相同 | 图片SHA簇Bootstrap95% CI pp |','|---|---:|---:|---:|---:|---|']
for x in paired:report.append(f"| {x['reference']} | {x['delta_pp']:+.3f} | {x['soft_score_improved']} | {x['soft_score_declined']} | {x['equal']} | [{x['ci95_low_pp']:.3f}, {x['ci95_high_pp']:.3f}] |")
report+=['','Bootstrap以3166个图片SHA簇配对采样，10000次、seed20261113，未做多重比较校正。按连续软分统计提高/降低，不能把它们称为二值错→对；探索性结果不构成非劣效或等价证明。','','## 物理与输入验收','','Late14正式名interVL-SVH-R-new：P−1/Decode仅L14–L27回看原始视觉KV，R在L0–L19读取图像，L2–L5答案Attention物理跳算，视觉Hidden于L27后/Final Norm前退出，选择性持久视觉KV与独立Cache保持冻结实现。仅外层执行及评分代码新增，59个冻结源码SHA在运行前后不变。',f"4题Smoke {P['smoke_ids']}通过结构检查且计入5000题：图像实际输入、允许/禁止视觉Attention、原始位置/RoPE、退出、KV层集合及有限性检查通过。原全局严格数值比较失败和EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER保留，不宣称全局数值等价。",'','原5000题ID、Question、Gold10答案、Image ID和顺序核对原始注释；3166图片沿用可信SHA记录，实际生成逐题核对image、prompt、input_ids和BF16 pixels哈希。Gold独立保存，不加入模型输入。全部5000条Token IDs CPU解码与原文一致，15000条新旧预测经官方评分及独立规范化/软分计算一致。错误、空输出和截断全部保留。','','## 环境与归档','','OpenGVLab/InternVL3-8B@853e3a797a661694b1b8ece0cb72dc2b23e3dac9，RTX3090，PyTorch2.6.0+cu124、Transformers4.49.0、batch1。模型仅加载一次，数据/模型/环境/运行文件在本地盘；进度每250题备份至持续盘，不重跑对照，不做效率实验或其他Benchmark。',f'GPU释放后：{gpu}。未关闭服务器。','原始预测、配置、源码与日志分别归档；Windows与持续盘真实SHA回执见backup_receipt.json。独立分支发布轻量全集预测、输入哈希、分数、统计及核验脚本。']
report+=['','历史证据：[Native/Physical-D全集及冻结Manifest](https://github.com/reyuyu/ATE/commit/3297559f21d3de55a6630a5491c9ecf66301c4b7)、[Late9全集汇总及冻结物理源码](https://github.com/reyuyu/ATE/commit/c60a9919112a3cf5bec8c494a533cba56a97f29b)、[interVL-SVH-R-new命名与前项OCRBench全量结果](https://github.com/reyuyu/ATE/commit/20043c577c6e08588becfe7470344b830b357640)。']
(pub/'report_zh.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
pr=json.loads((O/'progress.json').read_text());pr.update(status=summary['status'],independent_scoring_checks=15000,cpu_token_decode_checks=5000);atomic(O/'progress.json',pr);shutil.copy2(O/'progress.json',pub/'progress.json');atomic(O/'analysis_done.json',dict(status='PASS',accuracy_percent=score,coverage=5000,publication=str(pub)))
print(json.dumps(summary,ensure_ascii=False,indent=2))

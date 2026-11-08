"""CPU-only rescoring, independent arithmetic, and portable scientific export audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,json,gzip,hashlib,ast,math,sqlite3,re,importlib.util,subprocess
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
R=Path('/home/featurize/work/ATE');D=Path('/home/featurize/data/ate-recovery');O=R/'vlm_research/results/featurize_new3090_c3v2_reaccess_sevenbench'
sys.path[:0]=[str(R),str(R/'vlm_research/results/featurize_new3090_recovery/scripts')]
import torch
from types import SimpleNamespace
from vlm_research.information_horizon_alignment import analyze,score as frozen_score
# Extract only the exact frozen score wrapper; never import the GPU entry,
# change CUDA visibility, construct a model, or run inference in this process.
entry=ast.parse((O/'seven_runner.py').read_text())
score_node=next(n for n in entry.body if isinstance(n,ast.FunctionDef) and n.name=='score')
scope={'scorer':frozen_score};exec(compile(ast.Module(body=[score_node],type_ignores=[]),str(O/'seven_runner.py'),'exec'),scope)
runner=SimpleNamespace(torch=torch,scorer=frozen_score,score=scope['score'])
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def readgz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
m=read(O/'run_metadata.json');assert m['status']=='GENERATIONS_COMPLETE_PENDING_INDEPENDENT_SCORING'
runner.COMMIT=m['config']['code_commit'];runner.REV=m['config']['model_revision'];runner.COUNTS=m['config']['expected_counts']
parity=read(O/'gqa_current_smoke_parity.json');assert parity['same_first_logits_SHA256'] and parity['same_greedy_prefix']
assert sha(O/'seven_runner.py')==m['config']['runner_sha256']
rows=readgz(O/'mapped_manifest.jsonl.gz');samples={s['key']:s for s in rows};assert len(samples)==36256 and digest(rows)==m['config']['mapped_manifest_digest']
results=readgz(O/'predictions.jsonl.gz');lookup={r['key']:r for r in results};assert len(results)==len(lookup)==36256 and set(samples)==set(lookup)
db=sqlite3.connect((O/'predictions.sqlite').as_uri()+'?mode=ro',uri=True)
assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
for key,payload,h in db.execute('select key,payload,sha256 from results'):
 assert digest(json.loads(payload))==h and json.loads(payload)==lookup[key]
db.close()
runner.scorer.AUTHOR=D/'sources/Information-Horizon'
all_scored=defaultdict(list);flags=defaultdict(Counter);independent_rows=0
for r in results:
 s=samples[r['key']];b=s['benchmark'];text=r['generation']['text'];v=r['scoring']['score']
 assert r['binding']==m['binding'] and r['sample_sha256']==digest(s) and r['gold']==s['answer'] and r['image_id']==s['image_id']
 assert r['condition']==m['config']['condition'] and r['question_id']==s['question_id'] and r['question']==s['question']
 assert r['image_sha256']==s['image_sha256'] and r['input_view']==s['input_view']
 if b!='GQA':assert r['prompt_sha256']==digest(s['prompt'])
 assert r['scoring']==runner.score(s,text)
 assert r['generation']['all_effective_logits_finite'] and 0<=v<=1
 if b=='GQA':assert v==int(text.rstrip('.').lower()==s['answer']);independent_rows+=1
 elif b=='POPE':
  words=text.split('.')[0].replace(',','').split(' ');p='no' if set(words)&{'No','no','not'} else 'yes'
  assert v==int(p==s['answer']);independent_rows+=1
 elif b=='MME':
  t=text.lower();p=t if t in ['yes','no'] else 'yes' if 'yes' in t[:4] else 'no' if 'no' in t[:4] else 'other'
  assert v==int(p==s['answer']);independent_rows+=1
 elif b=='TextVQA':
  ap=runner.scorer.text_evaluator().answer_processor;gold=[ap(a) for a in s['answer']];assert len(gold)==10
  pred=ap(text);soft=sum(min(1,sum(a==pred for j,a in enumerate(gold) if j!=i)/3) for i in range(10))/10
  assert math.isclose(v,soft,abs_tol=1e-12);independent_rows+=1
 elif b=='ScienceQA-IMG':
  opt=text if re.fullmatch('[A-E]',text) else text[0] if re.match('[A-E]\\. ',text) else None
  if opt is None:
   matches=re.findall(r'The answer is ([A-Z]).',text);opt=matches[0] if len(matches)==1 else None
  p='ABCDE'.find(opt) if opt and opt in s['options'] else -1
  assert v==int(p==s['answer']);independent_rows+=1
 elif b=='MMBench':assert v==int(r['scoring']['parsed']==s['answer'])
 if b not in ['TextVQA']:assert v in [0,1]
 all_scored[b].append((s,v));flags[b].update({k:int(bool(r['generation'].get(k))) for k in ['eos','empty','truncated']})
assert {b:len(v) for b,v in all_scored.items()}==m['config']['expected_counts']
# Execute the exact official scoring loop AST, without loading its GPU worker imports.
official=D/'sources/ocrbench-v1/official/OCRBench/scripts/LLaVA1_5.py'
assert sha(official)=='c480b271cf71e18848aa637c9f6e5d5f8b22fdcd3b969cb63167b4fbeb450a68'
tree=ast.parse(official.read_text());main=next(n for n in tree.body if isinstance(n,ast.If) and '__main__' in ast.unparse(n.test))
loop=next(n for n in main.body if isinstance(n,ast.For) and 'data_type =' in ast.unparse(n))
ocr=read(D/'normalized/OCRBench-v1/OCRBench.json')
for i,row in enumerate(ocr):row['predict']=lookup[f'OCRBench-v1:{i:04d}']['generation']['text']
scope={'data':ocr};exec(compile(ast.Module(body=[loop],type_ignores=[]),str(official),'exec'),scope)
for i,r in enumerate(ocr):assert r['result']==lookup[f'OCRBench-v1:{i:04d}']['scoring']['score']
independent_rows+=1000
summary={}
for b,entries in all_scored.items():
 ss=[s for s,v in entries];vs=[v for s,v in entries]
 item=dict(status='COMPLETE',questions=len(entries),generation_flags=dict(flags[b]),
  answer_parse_failures=sum(bool(lookup[s['key']]['scoring'].get('parse_failed')) for s in ss))
 if b=='MME':
  total,tasks=analyze.mme_score(ss,vs);pairs=defaultdict(list);bytask=defaultdict(list)
  for s,v in entries:pairs[s['task'],s['pair_id']].append(v);bytask[s['task']].append(v)
  assert len(pairs)==1187 and all(len(p)==2 for p in pairs.values())
  manual={}
  for task,values in bytask.items():
   groups=[g for (t,k),g in pairs.items() if t==task];a=100*sum(values)/len(values);plus=100*sum(all(g) for g in groups)/len(groups)
   manual[task]={'accuracy':a,'accuracy_plus':plus,'score':a+plus}
  for t in tasks:
   for k in ['accuracy','accuracy_plus','score']:assert math.isclose(t[k],manual[t['task']][k],abs_tol=1e-10)
  assert math.isclose(total['Total'],sum(v['score'] for v in manual.values()),abs_tol=1e-10)
  item.update(metric='MME Total',scores=total,tasks=tasks,image_pairs=1187,known_single_image_SHA_difference_preserved=True)
 elif b=='MMBench':
  units,circular=analyze.circular_rows(ss,vs);groups=defaultdict(list)
  for s,v in entries:groups[s['circular_group']].append(v)
  assert len(groups)==len(units)==1176
  v=100*sum(all(g) for g in groups.values())/len(groups);assert math.isclose(v,100*float(circular.mean()),abs_tol=1e-10)
  item.update(metric='Circular Accuracy',value=v,unit='%',groups=len(groups),groups_correct=sum(all(g) for g in groups.values()),row_accuracy_secondary=100*sum(vs)/len(vs))
 elif b=='OCRBench-v1':
  cats=defaultdict(list)
  for s,v in entries:cats[s['category']].append(v)
  item.update(metric='Official OCRBench v1 raw score',value=int(sum(vs)),unit='/1000',categories={k:dict(score=int(sum(v)),questions=len(v)) for k,v in cats.items()})
 else:
  item.update(metric='VQA Soft Accuracy' if b=='TextVQA' else 'Accuracy',value=100*sum(vs)/len(vs),unit='%',score_sum=sum(vs))
  if b=='POPE':
   strategies={}
   for strategy in ['random','popular','adversarial']:
    rs=[lookup[s['key']] for s,v in entries if s['strategy']==strategy];conf=Counter()
    for r in rs:conf[('t' if r['scoring']['parsed']==r['gold'] else 'f')+('p' if r['scoring']['parsed']=='yes' else 'n')]+=1
    n=len(rs);assert n=={'random':2910,'popular':3000,'adversarial':3000}[strategy]
    strategies[strategy]=dict(questions=n,accuracy=100*(conf['tp']+conf['tn'])/n,**{k:conf[k] for k in ['tp','tn','fp','fn']})
   assert math.isclose(item['value'],sum(x['accuracy']*x['questions'] for x in strategies.values())/8910,abs_tol=1e-10)
   item['strategies']=strategies
 summary[b]=item
# Same-input original-C3 paired deltas, with explicit full tensor/Manifest equality.
oldout=R/'vlm_research/results/featurize_new3090_c3v2_evaluation';oldmeta=read(oldout/'run_metadata.json')
old_manifest={s['key']:s for s in readgz(oldout/'mapped_manifest.jsonl.gz')}
old_results={r['key']:r for r in readgz(oldout/'predictions.jsonl.gz')};assert len(old_results)==22678
assert oldmeta['config']['source_sha256']==m['config']['source_sha256']
paired={}
for key,old in old_results.items():
 new=lookup[key];assert samples[key]==old_manifest[key]
 assert new['gold']==old['gold'] and new['question']==old['question']
 for k in ['input_ids_sha256','prompt_sha256','pixel_values_sha256','expanded_length']:assert new[k]==old[k]
for b in ['MME','ScienceQA-IMG','POPE','TextVQA','MMBench']:
 oldscore=read(oldout/'score_summary.json')['benchmarks'][b];new=summary[b]
 oldvalue=oldscore['scores']['Total'] if b=='MME' else oldscore['value'];newvalue=new['scores']['Total'] if b=='MME' else new['value']
 paired[b]=dict(comparable=True,control='Previous original Physical-C3',questions=new['questions'],control_value=oldvalue,reaccess_value=newvalue,delta=newvalue-oldvalue,unit='MME points' if b=='MME' else 'percentage points',input_tensor_and_manifest_equality=True)
# Historical GQA full Native/C3 controls; no new control generation.
hist=D/'sources/gqa-reaccess-history';gqa_controls=Counter();gqa_by_q={};control_rows=defaultdict(dict)
for rank in [0,1]:
 db=sqlite3.connect((hist/f'worker{rank}/results.sqlite').as_uri()+'?mode=ro',uri=True)
 for q,label,payload,h in db.execute('select qid,condition,payload,sha256 from results'):
  r=json.loads(payload);ref=lookup['GQA:'+q]['historical_record'];assert digest(r)==h and r['input_binding']==ref['input_binding']
  gqa_controls[label]+=r['score'];gqa_by_q[q,label]=r['score']
  if label!='C3-V2+L11-L24':
   control_rows[q][label]=dict(text=r['generation']['text'],token_ids=r['generation']['token_ids'],parsed=r['generation']['text'].rstrip('.').lower(),
     score=r['score'],record_sha256=h,binding=r['binding'],input_binding=r['input_binding'])
 db.close()
assert gqa_controls['C3-V2+L11-L24']==7537
with gzip.GzipFile(filename=str(O/'gqa_historical_controls.jsonl.gz'),mode='wb',mtime=0) as stream:
 for q,controls in sorted(control_rows.items()):
  s=samples['GQA:'+q];r=dict(question_id=q,image_id=s['image_id'],image_sha256=s['image_sha256'],gold=s['answer'],controls=controls,source_commit='57e37ce024b1b2adf9fa24fc24aabec32a1fed08')
  stream.write((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode())
paired['GQA']=dict(comparable=True,source_commit='57e37ce024b1b2adf9fa24fc24aabec32a1fed08',questions=12578,correct=dict(gqa_controls),
 native_accuracy=100*gqa_controls['Native']/12578,original_c3_accuracy=100*gqa_controls['C3-V2']/12578,
 reaccess_accuracy=summary['GQA']['value'],reaccess_minus_native_pp=100*(7537-gqa_controls['Native'])/12578,
 reaccess_minus_c3_pp=100*(7537-gqa_controls['C3-V2'])/12578,all_control_input_bindings_identical=True)
paired['OCRBench-v1']=dict(comparable=False,reason='No complete matching Native/original-C3 control; only current B2 score reported.')
save('paired_comparability.json',dict(status='PASS',conditions=paired,no_new_control_generation=True,no_bootstrap_computation_requested=True))
save('score_summary.json',dict(status='COMPLETE',records=36256,new_gpu_records=23678,historical_reused=12578,benchmarks=summary))
save('scoring_audit.json',dict(status='PASS',records=36256,all_row_scorings_recomputed_with_frozen_scorers=True,
 independent_non_MM_row_checks=independent_rows,MMBench='Official frozen matcher result plus independent group-all-correct arithmetic for1176 complete groups.',
 TextVQA='All5000: independently enumerate ten leave-one-out score terms after frozen answer normalization; not binary accuracy.',
 OCRBench='Exact official scorer AST loop independently executed on preserved raw annotations/predictions; all1000 agree.',
 MME='Independent14-task Accuracy + both-pair-correct Accuracy+ arithmetic for1187 pairs.',
 mismatches=[],input_or_prompt_issue=False,raw_predictions_unchanged=True,previous_native15_audit='PASS15/15, retained in Windows previous-run archive; smoke is not a formal Native baseline.'))
physical=read(O/'physical_smoke.json');assert physical['status']=='PASS' and len(physical['cases'])==7
compact=[]
for c in physical['cases']:
 proof={k:c[k] for k in ['benchmark','key','status','numerical_equivalence','rope_exact','visual_hidden_removed_before_L25','answer_attention_physically_skipped','compact_kv_original_positions_exact','blocked_image_probability_zero']}
 proof['actual_trace']={k:c['trace'][k] for k in ['vision_calls','projector_calls','image_embedding_exact','norm_calls','post_norm_calls','mlp_calls','skip_residual_exact_checks']}
 proof['retained_image_layers']=c['actual']['kv_audit']['retained_image_layers']
 proof['skipped_layer_projection_shapes']=[r for r in c['projection_shapes'] if r['layer'] in [2,3,4,5]]
 proof['smoke_tokens_only']=c['actual']['token_ids'];compact.append(proof)
save('physical_smoke_summary.json',dict(status='PASS',cases=compact,proof='Actual operator, hidden-row, vision/projector, selective KV and RoPE hooks; same-B2 three-token dense numerical reference only; full original thresholds unchanged.'))
assert not runner.torch.cuda.is_initialized()
for name,h in m['config']['source_sha256'].items():assert sha(R/name)==h
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()==runner.COMMIT
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=R,text=True).strip()
m['status']='COMPLETE';m['independent_final_scoring']='PASS';m['final_scored_at']=datetime.now(timezone(timedelta(hours=8))).isoformat();save('run_metadata.json',m)
p=read(O/'progress.json');p['status']='COMPLETE';save('progress.json',p)
lines=['# RTX3090 Physical C3-V2 + L11–L24 七项正式评测','',
 '**状态：COMPLETE**。36,256/36,256 条：本轮新生成23,678；GQA12,578条严格校验后复用历史完整B2结果。',
 f"冻结模型：llava-hf/llava-1.5-7b-hf@{runner.REV}；计算源码Commit：{runner.COMMIT}。",
 f"环境：{m['runtime']}；FP16/Eager，Greedy，Batch1；本轮模型加载{m['model_loads_this_run']}次。",'',
 '| Benchmark | 题数 | 正式指标 |','|---|---:|---:|']
for b in runner.COUNTS:
 v=summary[b];value=v['scores']['Total'] if b=='MME' else v['value'];unit='' if b=='MME' else '/1000' if b=='OCRBench-v1' else '%'
 lines.append(f"| {b} | {v['questions']} | {value:.6f}{unit} |")
lines+=['','MME 分任务（Accuracy+ 为同图两题均正确的比例）：','',
 '| Task | Accuracy % | Accuracy+ % | 任务分 |','|---|---:|---:|---:|']
for t in summary['MME']['tasks']:lines.append(f"| {t['task']} | {t['accuracy']:.6f} | {t['accuracy_plus']:.6f} | {t['score']:.6f} |")
lines+=['','MME 分类总分：'+json.dumps(summary['MME']['scores'],ensure_ascii=False),'',
 'POPE 论文视图三策略：','','| Strategy | 题数 | Accuracy % |','|---|---:|---:|']
for k,v in summary['POPE']['strategies'].items():lines.append(f"| {k} | {v['questions']} | {v['accuracy']:.6f} |")
lines+=['','OCRBench v1 十子类（原始整数分，不换成 VQA 软分）：','',
 '| Category | 得分 | 满分 |','|---|---:|---:|']
for k,v in summary['OCRBench-v1']['categories'].items():lines.append(f"| {k} | {v['score']} | {v['questions']} |")
lines+=['','与前次同协议原始 C3 的正式配对结果（MME 单位为 points，其余为百分点）：','',
 '| Benchmark | 前次 C3 | Re-access | 差值 |','|---|---:|---:|---:|']
for b in ['MME','ScienceQA-IMG','POPE','TextVQA','MMBench']:
 v=paired[b];lines.append(f"| {b} | {v['control_value']:.6f} | {v['reaccess_value']:.6f} | {v['delta']:+.6f} |")
g=paired['GQA'];lines+=['',f"历史 GQA：Native {g['native_accuracy']:.6f}%；C3 {g['original_c3_accuracy']:.6f}%；Re-access {g['reaccess_accuracy']:.6f}%；Re-access−Native {g['reaccess_minus_native_pp']:+.6f} pp，Re-access−C3 {g['reaccess_minus_c3_pp']:+.6f} pp。本轮未重跑这些对照。"]
lines+=['','Physical正确性：七类真实图像Smoke通过。R仅L0–19写入；P−1/Decode仅L11–24读原始Image Key；L25前物理移除视觉Hidden；L2–5跳算答案Attention，Norm/MLP/Residual继续；SelectiveKV仅指定层保存图像，原始RoPE和位置索引精确。',
 '同B2三token Dense仅作数值参考，无正式Dense实验。冻结atol0.10/rtol0.003、hidden RMSE0.01/cosine1e-4未放宽。详见physical_smoke_summary.json。',
 'GQA保持原正式HF crop/USER模板与cap32；Paper五项逐字段复用前次Manifest、pad预处理和Gold；MME/POPE/TextVQA cap128，ScienceQA/MMBench cap1024。TextVQA是Paper-OCR，POPE为8910题（2910/3000/3000）。MMBench1176完整Circular组。',
 'OCRBench v1官方1000题、930图；固定官方Commit6ddb0a570419ce7ccb6913cf1c5ed84930320e7e，原始标注与GoogleDrive图像ZIP保留；十子类官方子串评分，HME100k大小写敏感去空格；总分0–1000。Prompt=Vicuna_v1+单词/短语后缀，pad/CLIP336，cap128，未注入外部OCR。',
 '官方原始id有143个跨dataset重复；dataset_name+id无重复。ATE使用官方行号0000–0999作为唯一键，原始id/Gold未改写。',
 'MME保留landmark/03d5e3bfc958be38.jpg已知单图SHA差异，两题仍保留；本轮与前次C3使用同一恢复图片。',
 '独立评分核验：36256条冻结评分重算通过；TextVQA5000条额外独立留一计算；OCR1000条运行原官方AST评分循环；MME14任务/1187对和MMBench1176组独立聚合一致。逐题得分不以Smoke正确率替代正式成绩。',
 '可比性：五项与前次原始C3全部Manifest及输入张量哈希一致，可报告配对差值；GQA历史Native/C3/B2同输入绑定可比；无完整Paper Native或OCR对照。详见paired_comparability.json。',
 'GQA历史Python3.11.15与本机3.11.8、驱动/GPU实例不同；Torch/Transformers/模型/算法/输入张量逐项一致；当前已存真实Physical Smoke的首步logits SHA及Greedy Token也与历史同题完全一致。保留历史来源，不宣称本轮重新生成GQA正式全量。',
 f"峰值/释放：{m.get('gpu')}；{m.get('release')}。最终资源检查另见shutdown_verification.json。",
 '原始预测及完整映射Manifest均压缩保存，包含Gold、原始输出、解析答案、逐题得分、配对/题组和源码/协议绑定；不必依赖SQLite重评分。Windows和GitHub实际归档状态由最终receipt确认；本脚本不提前宣布安全关机。']
(O/'report_zh.md').write_text('\n'.join(lines)+'\n')
print('FINAL_SCORING_PASS',json.dumps(summary,ensure_ascii=False),flush=True)

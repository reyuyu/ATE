import ast,copy,csv,gzip,json,hashlib,sys,collections,shutil
from pathlib import Path
from run_common import L,D,A,RAW,sha,atomic,digest
OUT=L/'publication/vlm_research/results/ocrbench_ate_frozen_completion_22875_20261113';OUT.mkdir(parents=True,exist_ok=True)
OFF=A/'official/Yuliang-Liu__MultimodalOCR/OCRBench/example.py'
tree=ast.parse(OFF.read_text());loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and "data[i]['result']" in ast.unparse(n) and 'data_type' in ast.unparse(n) and 'dataset_name' in ast.unparse(n));compiled=compile(ast.Module(body=[loop],type_ignores=[]),'<frozen official OCRBench v1>','exec')
all_records=[];summary=[];discrepancies=[];llava_cpu=json.loads((L/'results/llava/cpu_bindings.json').read_text())
def boolean(x):return x if isinstance(x,bool) else str(x).lower()=='true'
def add(model,method,q,text,ids,image,origin,source,old_score=None,eos=False,truncated=False,empty=None,input_hash=None,prompt_hash=None):
 assert q==f'{int(q):04d}' and 0<=int(q)<1000
 raw=RAW[int(q)];data=[dict(copy.deepcopy(raw),predict=text)];exec(compiled,{'data':data});official=int(data[0]['result'])
 # Independent implementation: do not call any ATE scorer or compiled official code.
 def norm(v):
  v=str(v).strip().replace('\n',' ')
  return ''.join(v.split(' ')) if raw['dataset_name']=='HME100k' else v.lower()
 answers=raw['answers'] if isinstance(raw['answers'],list) else [raw['answers']]
 independent=int(any(norm(a) in norm(text) for a in answers));assert independent==official,(model,method,q,text)
 if old_score is not None and float(old_score)!=official:discrepancies.append(dict(model=model,method=method,question_id=q,old_score=old_score,correct_score=official,text=text))
 r=dict(model=model,method=method,question_id=q,question=raw['question'],image_id=raw['image_path'],image_sha256=image,dataset_name=raw['dataset_name'],category=raw['type'],gold=answers,text=text,token_ids=ids,official_score=official,independent_score=independent,eos=boolean(eos),truncated=boolean(truncated),empty=(not text if empty is None else boolean(empty)),origin=origin,source_record=source,input_binding_sha256=input_hash,prompt_sha256=prompt_hash)
 all_records.append(r)
def current(model,name,method,sub=None):
 folder=L/'results'/name/'predictions'
 if sub:folder/=sub
 for p in sorted(folder.glob('*.json')):
  r=json.loads(p.read_text());g=r['generation'];q=r['question_id'];layout=r.get('input_layout',{});ib=r.get('input_binding_details',{})
  prompt_hash=r.get('prompt_sha256') or ib.get('prompt_sha256') or layout.get('prompt_SHA256')
  if name=='qwen':prompt_hash=hashlib.sha256(layout['prompt'].encode()).hexdigest()
  if name=='llava':prompt_hash=hashlib.sha256(llava_cpu[q]['prompt'].encode()).hexdigest()
  add(model,method,q,g['text'],g.get('token_ids',g.get('output_ids',[])),r['image_sha256'],r['origin'],str(p.relative_to(L)),r['official_score'],g.get('eos',False),g.get('truncated',False),g.get('empty'),r.get('input_binding') or digest(r.get('input_hashes',layout)),prompt_hash)
old=list(csv.DictReader(gzip.open(A/'historical/native_stopv2/OCRBench/per_question.csv.gz','rt')))
for r in old:
 assert r['method'] in ['native','physical_d']
 add('InternVL3-8B','Native' if r['method']=='native' else 'Physical-D',r['question_id'],r['text'],json.loads(r['token_ids']),r['image_sha256'],'REUSED_PUBLISHED_FULL',str(A/'historical/native_stopv2/OCRBench/per_question.csv.gz'),r['score'],r['eos'],r['truncated'],r['empty'],r['input_binding'],r['prompt_sha256'])
for r in json.loads((L/'results/internvl3/historical_late9_3.json').read_text()):
 add('InternVL3-8B','Late9',r['question_id'],r['late9_text'],json.loads(r['late9_token_ids']),r['image_sha256'],'REUSED_PUBLISHED_PARTIAL_3',str(A/'historical/late9/ocrbench_partial_per_question.csv'),r['late9_official_score'],r['late9_eos'],r['late9_truncated'],r['late9_empty'],r['late9_input_binding'],r['late9_prompt_sha256'])
current('InternVL3-8B','internvl3','Late9')
for r in json.loads((L/'history/llava_svhr_1000.json').read_text()):
 g=r['generation'];add('LLaVA-1.5-7B','SVH-R L11-L24',r['question_id'],g['text'],g['token_ids'],r['image_sha256'],'REUSED_PUBLISHED_FULL','history/llava_svhr_1000.json',r['scoring']['score'],g['eos'],g['truncated'],g['empty'],r['binding'],r['prompt_sha256'])
for m,rr in json.loads((L/'results/llava/verified_reuse.json').read_text()).items():
 if m=='Native':continue
 for q,r in rr.items():
  g=r['generation'];prompt=llava_cpu[q]['prompt'];add('LLaVA-1.5-7B',m,q,g['text'],g['token_ids'],r['image_sha256'],r['origin'],'results/llava/verified_reuse.json',r['official_score'],g['eos'],g['truncated'],g['empty'],digest(r['input_hashes']),hashlib.sha256(prompt.encode()).hexdigest())
for q,r in json.loads((L/'results/llava/verified_full_native_reuse.json').read_text()).items():
 g=r['generation'];add('LLaVA-1.5-7B','Native',q,g['text'],g['token_ids'],r['image_sha256'],r['origin'],'history/fastv_native1000.json',r['official_score'],g['eos'],g['truncated'],g['empty'],digest(r['input_hashes']),hashlib.sha256(llava_cpu[q]['prompt'].encode()).hexdigest())
for m in ['Native','Physical-C3']:current('LLaVA-1.5-7B','llava',m,m)
for m in ['Native','Physical-Original']:current('Qwen3-VL-8B','qwen',m,m)
groups=collections.defaultdict(list)
for r in all_records:groups[(r['model'],r['method'])].append(r)
for m in ['Native','Physical-Original']:groups[('Qwen3-VL-8B',m)]
images={r['image_path']:None for r in RAW}
frozen=json.loads((L/'results/internvl3/frozen_rows.json').read_text());frozen_images={r['image_id']:r['image_sha256'] for r in frozen}
assert len(frozen_images)==930
for r in all_records:assert r['image_sha256']==frozen_images[r['image_id']]
for (model,method),rs in groups.items():
 assert len({r['question_id'] for r in rs})==len(rs),(model,method,'duplicate completed request')
 rs.sort(key=lambda r:r['question_id']);n=len(rs);complete=n==1000
 if complete:assert [r['question_id'] for r in rs]==[f'{i:04d}' for i in range(1000)]
 s=dict(model=model,method=method,coverage=n,expected=1000,complete=complete,score_0_1000=sum(r['official_score'] for r in rs) if complete else None,partial_correct=sum(r['official_score'] for r in rs),new_generated=sum(r['origin']=='NEW_GPU_MISSING_ONLY' for r in rs),reused=sum(r['origin']!='NEW_GPU_MISSING_ONLY' for r in rs),empty=sum(r['empty'] for r in rs),truncated=sum(r['truncated'] for r in rs),eos=sum(r['eos'] for r in rs),independent_scores_match=True)
 summary.append(s)
 filename=model.replace('.','').replace('-','_')+'__'+method.replace(' ','_').replace('-','_')+'.jsonl.gz'
 with gzip.open(OUT/'predictions'/filename,'wt',encoding='utf-8') if (OUT/'predictions').exists() else __import__('contextlib').nullcontext() as f:
  if f:
   for r in rs:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
atomic(OUT/'score_summary.json',dict(status='COMPLETE' if len(summary)==8 and all(s['complete'] for s in summary) else 'PARTIAL',official_scorer_SHA256=sha(OFF),dataset_annotation_SHA256=sha(D/'OCRBench.json'),independent_method='separate normalization plus substring membership; HME100k keeps case and removes spaces; other sets casefold by lower()',scoring_discrepancies=discrepancies,results=summary))
assert not discrepancies,discrepancies[:3]
with (OUT/'score_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
with (OUT/'input_manifest.jsonl').open('w',encoding='utf-8') as f:
 for i,r in enumerate(RAW):f.write(json.dumps(dict(question_id=f'{i:04d}',question=r['question'],gold=r['answers'],image_id=r['image_path'],image_sha256=frozen_images[r['image_path']],category=r['type'],dataset_name=r['dataset_name']),ensure_ascii=False)+'\n')
for name in ['internvl3','llava','qwen']:
 p=OUT/'protocols'/name;p.mkdir(parents=True,exist_ok=True)
 for fn in ['protocol.json','progress.json','smoke.json','environment.json','failure.json']:
  src=L/'results'/name/fn
  if src.exists():shutil.copy2(src,p/fn)
 print(name,'STATE', json.loads((L/'results'/name/'progress.json').read_text()) if (L/'results'/name/'progress.json').exists() else 'NOT_STARTED')
print(json.dumps(summary,ensure_ascii=False,indent=2))

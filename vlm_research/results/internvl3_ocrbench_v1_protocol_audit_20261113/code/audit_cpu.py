from pathlib import Path
import ast,csv,gzip,json,hashlib,copy,collections,importlib.util,sys,types,logging,os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
A=Path('/home/featurize/ate-audits/internvl3-ocrbench-v1-20261113')
P=Path('/home/featurize/work/ATE-internvl3-svh-stop-policy-v2-3090/vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090')
B=Path('/home/featurize/ate-local-data/bundles/OCRBench-v1')
M=Path('/home/featurize/ate-local-models/InternVL3-8B')
W=Path('/home/featurize/ate-experiments/rtx3090-18conditions-20261113')
O=A/'cpu';O.mkdir(exist_ok=True)
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else Path(x).read_bytes()).hexdigest()
def writej(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def writecsv(n,rows):
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def ast_funcs(path,names,scope):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope);return scope
rawpath=A/'official/Yuliang-Liu__MultimodalOCR/OCRBench/OCRBench/OCRBench.json'
raw=json.loads(rawpath.read_text());localraw=json.loads((B/'OCRBench.json').read_text())
with gzip.open(P/'manifests/OCRBench.jsonl.gz','rt') as f:frozen=[json.loads(l) for l in f]
with gzip.open(P/'OCRBench/per_question.csv.gz','rt') as f:preds=list(csv.DictReader(f))
index={r['question_id']:r for r in frozen};ann_errors=[];image_rows=[];cache={}
# Reuse this audit's just-completed 930-image SHA pass; no repeated image hashing.
if (O/'image_integrity.csv').exists():
 for z in csv.DictReader((O/'image_integrity.csv').open()):
  cache[z['path']]=(z['actual_sha256'],int(z['bytes']),int(z['width']),int(z['height']))
from PIL import Image
for i,(r,x) in enumerate(zip(frozen,raw)):
 exp=dict(question_id=f'{i:04d}',official_id=x['id'],image_id=x['image_path'],question=x['question'],gold=x['answers'],category=x['type'],dataset_name=x['dataset_name'])
 for k,v in exp.items():
  if r[k]!=v:ann_errors.append(dict(qid=r['question_id'],field=k,expected=v,actual=r[k]))
 p=B/'OCRBench_Images'/x['image_path']
 if str(p) not in cache:
  if p.is_file():
   with Image.open(p) as im:size=im.size
   cache[str(p)]=(sha(p),p.stat().st_size,*size)
  else:cache[str(p)]=('',0,0,0)
 actual,sz,width,height=cache[str(p)]
 image_rows.append(dict(question_id=r['question_id'],image_id=r['image_id'],path=str(p),expected_sha256=r['image_sha256'],actual_sha256=actual,bytes=sz,width=width,height=height,status='PASS' if actual==r['image_sha256'] else ('MISSING' if not actual else 'SHA_MISMATCH')))
writecsv('image_integrity.csv',image_rows)
image_bad=[x for x in image_rows if x['status']!='PASS']
blocked=list(csv.DictReader((A/'historical/late9/blocked_requests.csv').open()))
blocked_ocr=[r for r in blocked if r['benchmark']=='OCRBench']
latepred=list(csv.DictReader((A/'historical/late9/ocrbench_partial_per_question.csv').open()))
completed=set(r['question_id'] for r in latepred);blockset={r['question_id'] for r in blocked_ocr}
block_bad=[r for r in blocked_ocr if r['image_sha256']!=index[r['question_id']]['image_sha256']]
writej('dataset_validation.json',dict(status='PASS' if not ann_errors and not image_bad else 'FAIL',official_annotation_sha256=sha(rawpath),local_annotation_sha256=sha(B/'OCRBench.json'),historical_manifest_sha256=sha(P/'manifests/OCRBench.jsonl.gz'),official_count=len(raw),frozen_count=len(frozen),local_annotation_equal_official=raw==localraw,annotation_field_errors=ann_errors,unique_expected_image_sha256=len(set(r['image_sha256'] for r in frozen)),unique_paths_hashed=len(cache),image_sha_pass_questions=sum(r['status']=='PASS' for r in image_rows),image_failures=image_bad,late9_blocked_ocr=len(blocked_ocr),late9_completed_ids=sorted(completed),late9_input_partition_exact=(blockset|completed==set(index) and not blockset&completed),late9_blocked_sha_errors=block_bad,current_recovered_blocked_questions=sum(image_rows[int(q)]['status']=='PASS' for q in blockset)))
# Independently execute the unchanged original OCRBench scoring loop AST. No model module is imported.
src=A/'official/Yuliang-Liu__MultimodalOCR/OCRBench/example.py';tree=ast.parse(src.read_text());loop=None
for n in ast.walk(tree):
 if isinstance(n,ast.For) and 'data_type' in ast.unparse(n) and "data[i]['result']" in ast.unparse(n) and 'dataset_name' in ast.unparse(n):loop=n;break
assert loop is not None
loopcode=compile(ast.Module(body=[loop],type_ignores=[]),str(src),'exec')
def originals_score(rows):
 data=[]
 for r in rows:
  x=copy.deepcopy(raw[int(r['question_id'])]);x['predict']=r['text'];data.append(x)
 exec(loopcode,{'data':data});return [x.get('result',0) for x in data]
# Exact frozen/current VLMEvalKit OCRBench_eval with in-memory I/O, not an approximation.
import pandas as pd
vlmfile=A/'official/VLMEvalKit_baseline_pinned/vlmeval/dataset/utils/ocrbench.py'
def vlm_score(rows):
 frame=pd.DataFrame([dict(prediction=r['text'],answer=repr(raw[int(r['question_id'])]['answers'] if isinstance(raw[int(r['question_id'])]['answers'],list) else [raw[int(r['question_id'])]['answers']]),category=raw[int(r['question_id'])]['type']) for r in rows])
 result={};scope=dict(load=lambda _:frame,dump=lambda v,p:result.update(v),get_intermediate_file_path=lambda *a:'memory',logger=logging.getLogger('audit'),tqdm=lambda x:x)
 ast_funcs(vlmfile,{'OCRBench_eval'},scope)['OCRBench_eval']('memory');return result
rescored=[];summary=[];categories=[]
late_long=[]
for r in latepred:
 for method in ['native','physical_d','late9']:
  z=dict(question_id=r['question_id'],method=method)
  for field in ['text','score','eos','truncated','token_ids']:
   z[field]=r[method+'_'+field]
  z['generated_tokens']=str(len(json.loads(z['token_ids'])))
  late_long.append(z)
for label,allrows in [('native_stop_v2',preds),('late9_partial',late_long)]:
 for method in sorted(set(r['method'] for r in allrows)):
  rows=[r for r in allrows if r['method']==method];scores=originals_score(rows);v=vlm_score(rows)
  mismatches=[];cc=collections.Counter();rawchanges=[];rawscores=[]
  for r,s in zip(rows,scores):
   old=float(r['score']);cc[raw[int(r['question_id'])]['type']]+=s
   if old!=s:mismatches.append(r['question_id'])
   if 'raw_decode' in r:
    rr=dict(r,text=r['raw_decode']);rs=originals_score([rr])[0];rawscores.append(rs)
    if s!=rs:rawchanges.append(r['question_id'])
   rescored.append(dict(source=label,method=method,question_id=r['question_id'],stored_score=old,original_official_score=s,raw_decode_score=rawscores[-1] if 'raw_decode' in r else '',match=old==s))
  summary.append(dict(source=label,method=method,N=len(rows),official_original_score=sum(scores),VLMEvalKit_score=v['Final Score'],stored_score=sum(float(r['score']) for r in rows),mismatch_ids=mismatches,raw_vs_scored_changes=rawchanges,all_eos=sum(r.get('eos')=='True' for r in rows),truncated=sum(r.get('truncated')=='True' for r in rows),max_tokens=max(int(r.get('generated_tokens',0)) for r in rows)))
  for cat,n in collections.Counter(x['type'] for x in raw).items():categories.append(dict(source=label,method=method,category=cat,full_denominator=n,evaluated_questions=sum(raw[int(r['question_id'])]['type']==cat for r in rows),score=cc[cat]))
writecsv('rescored_predictions.csv',rescored);writecsv('category_scores.csv',categories);writej('scoring_validation.json',dict(status='PASS' if all(not r['mismatch_ids'] and r['official_original_score']==r['VLMEvalKit_score'] for r in summary) else 'FAIL',summary=summary,normalization='Original HME100k is case-sensitive, removes spaces/newlines; all other data lowercases, trims, replaces newline by space; accepted-answer substring, one point maximum; no VQA soft scoring.'))
# CPU-only tokenizer, model-native conversation, official image-preprocessing Smoke.
import torch,torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
assert not torch.cuda.is_initialized()
torch.set_num_threads(1)
tok=AutoTokenizer.from_pretrained(M,trust_remote_code=True,use_fast=False,local_files_only=True)
spec=importlib.util.spec_from_file_location('audit_model_conversation',M/'conversation.py');convmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=convmod;spec.loader.exec_module(convmod)
conv=convmod.get_conv_template('internvl2_5');prompt_bad=[];decode_bad=[]
for r in frozen:
 c=conv.copy();c.append_message(c.roles[0],r['request']);c.append_message(c.roles[1],None);prompt=c.get_prompt().replace('<image>','<img>'+'<IMG_CONTEXT>'*256+'</img>',1)
 if prompt!=r['prompt'] or sha(prompt.encode())!=r['prompt_sha256'] or tok(prompt)['input_ids']!=r['input_ids']:prompt_bad.append(r['question_id'])
for r in preds:
 ids=json.loads(r['token_ids']);plain=tok.decode(ids,skip_special_tokens=True);text=plain.split(conv.sep.strip())[0].strip()
 if plain!=r['raw_decode'] or text!=r['text'] or len(ids)!=int(r['generated_tokens']) or (ids[-1]==tok.convert_tokens_to_ids(conv.sep.strip()))!=(r['eos']=='True'):decode_bad.append(dict(method=r['method'],question_id=r['question_id']))
ns=dict(Image=Image,T=T,InterpolationMode=InterpolationMode,torch=torch,IMAGENET_MEAN=(.485,.456,.406),IMAGENET_STD=(.229,.224,.225))
uf=A/'official/VLMEvalKit_baseline_pinned/vlmeval/vlm/internvl/utils.py';ast_funcs(uf,{'build_transform','find_closest_aspect_ratio','dynamic_preprocess','load_image'},ns)
# Use the official aspect-ratio selector on sizes only for all1000, without expensive image-resize or GPU.
ratios=sorted(set((i,j) for n in range(1,25) for i in range(1,n+1) for j in range(1,n+1) if i*j<=24),key=lambda x:x[0]*x[1])
plan=[]
for r in image_rows:
 if r['width']==0:continue
 ratio=ns['find_closest_aspect_ratio'](r['width']/r['height'],ratios,r['width'],r['height'],448);blocks=ratio[0]*ratio[1];tiles=blocks+(blocks!=1)
 plan.append(dict(question_id=r['question_id'],width=r['width'],height=r['height'],grid_x=ratio[0],grid_y=ratio[1],tiles_with_thumbnail=tiles,visual_tokens=tiles*256,current_visual_tokens=256))
writecsv('official_dynamic_tile_plan.csv',plan)
# Frozen deterministic geometry-based smoke selection, independent of answers/scores.
ordered=sorted(plan,key=lambda x:(x['tiles_with_thumbnail'],x['width']*x['height'],x['question_id']))
sel=list(dict.fromkeys([ordered[0]['question_id'],ordered[len(ordered)//2]['question_id'],ordered[-1]['question_id'],'0053']))
sys.path.insert(0,str(W));from vlm_research.internvl3_visual_writing_figure3_3000.engine import official_preprocessor,tensor_hash
oldload=official_preprocessor(M);pixel_smoke=[]
for q in sel:
 p=Path(image_rows[int(q)]['path']);old=oldload(p,max_num=1).to(torch.bfloat16);one=ns['load_image'](p,max_num=1).to(torch.bfloat16);dyn=ns['load_image'](p,max_num=24).to(torch.bfloat16)
 oldhash=tensor_hash(old);bind=json.loads(next(r for r in preds if r['method']=='native' and r['question_id']==q)['input_binding_details'])
 pixel_smoke.append(dict(question_id=q,legacy_shape=list(old.shape),official_shape=list(dyn.shape),legacy_vs_official_one_tile_equal=torch.equal(old,one),legacy_pixel_sha256=oldhash,historical_pixel_sha256=bind['pixels_bf16_sha256'],historical_pixel_match=oldhash==bind['pixels_bf16_sha256']))
writej('cpu_smoke.json',dict(status='PASS' if not prompt_bad and not decode_bad and all(x['legacy_vs_official_one_tile_equal'] and x['historical_pixel_match'] for x in pixel_smoke) else 'FAIL',prompt_and_input_ids_checked=len(frozen),prompt_or_tokenizer_failures=prompt_bad,raw_token_ids_and_text_checked=len(preds),decode_failures=decode_bad,chat_template='internvl2_5',system_message=conv.system_message,eos_id=tok.convert_tokens_to_ids(conv.sep.strip()),eos_token=conv.sep.strip(),preprocessor_samples=pixel_smoke,official_ocrbench_max_num=24,official_thumbnail=True,official_tile_distribution=dict(collections.Counter(x['tiles_with_thumbnail'] for x in plan)),questions_with_more_than_one_official_tile=sum(x['tiles_with_thumbnail']>1 for x in plan),cuda_initialized=torch.cuda.is_initialized(),gpu_generations=0))
print(json.dumps(dict(dataset=json.loads((O/'dataset_validation.json').read_text()),scoring=summary,cpu_smoke=json.loads((O/'cpu_smoke.json').read_text())),ensure_ascii=False,indent=2))

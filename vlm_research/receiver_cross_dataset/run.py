import os
os.environ['CUDA_VISIBLE_DEVICES']='GPU-b08489af-0a8a-eb48-55ba-8346733af417'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from .common import *
from .selection import firstword
from .scoring import score
import torch,numpy as np,time,traceback,gc,contextlib
torch.set_num_threads(1)
from vlm_common import load_model,prepare_inputs,gpu_memory
from knockout_probe import model_layers
from benchmarks.smoke import hook_footprint
from paper_rq1.engine import cache_fingerprint
from query_role_smoke_fix.run import native
import gqa3000_token_visual_writing.engine as engine
engine.assert_no_stop=guard
def pointer_set(cache):return {x.data_ptr() for v in [cache.key_cache,cache.value_cache] for x in v}
@contextlib.contextmanager
def observations(model):
 counts=[0]*32;features=[];handles=[]
 def vision(m,a,o):features.append(dict(shape=list(o.shape),sha256=hashlib.sha256(o.detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
 try:
  for k,l in enumerate(model_layers(model)):
   def mlp(m,a,o,k=k):counts[k]+=1
   handles.append(l.mlp.register_forward_hook(mlp))
  handles.append(model.multi_modal_projector.register_forward_hook(vision))
  yield counts,features
 finally:
  for h in handles:h.remove()
def inputs_for(model,proc,r,s):
 result=prepare_inputs(model,proc,r['image_path'],r['question']+'\nAnswer the question using a single word or phrase.')
 inputs=result[0] if isinstance(result,tuple) else result
 assert tensors_hash(inputs)==s['input_hashes'] and inputs['input_ids'][0].tolist()==s['prompt_ids']
 return inputs
def one(model,proc,inputs,s,w,capture=False,retain=False,noop=False):
 before=hook_footprint(model)
 with observations(model) as (counts,features):result=engine.generate(model,proc.tokenizer,inputs,w,noop=noop,capture=capture,retain_cache=retain)
 g,first,trace,values,cache=result
 assert hook_footprint(model)==before and counts==[g['length']]*32
 assert len(features)==1 and features[0]['shape'][1]==576
 assert g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1
 assert tensors_hash(inputs)==s['input_hashes']
 return result,dict(MLP_calls=counts,image_features=features,hook_cleanup=True,input_hashes_unchanged=True)
def smoke_ids(rows,specs):
 chosen=[]
 predicates=[lambda r:firstword(r['question'])=='what',lambda r:firstword(r['question'])=='which',lambda r:firstword(r['question'])=='how',lambda r:firstword(r['question']) in {'is','are','do','does'},lambda r:len(specs[r['question_id']]['Q'])<10,lambda r:len(specs[r['question_id']]['Q'])>=18,lambda r:r['answer_type']=='number',lambda r:any(len(a.split())>=2 for a in r['answers'])]
 missing=[]
 for i,p in enumerate(predicates):
  found=next((r['question_id'] for r in rows if p(r) and r['question_id'] not in chosen),None)
  if found:chosen.append(found)
  else:missing.append(i)
 assert 5<=len(chosen)<=8
 return chosen,missing
def smoke(model,proc,rows,specs,d,weights):
 ids,missing=smoke_ids(rows,specs);write(d/'smoke_selection.json',dict(UTC=utc(),question_ids=ids,rule='First frozen-manifest example for each predefined what/which/how/yesno/short/long/number/multiword-annotation category; no model outcomes',unavailable_categories=missing))
 indexed={r['question_id']:r for r in rows};records=[]
 for q in ids:
  r=indexed[q];s=specs[q];inputs=inputs_for(model,proc,r,s);foot=hook_footprint(model)
  with observations(model) as (counts,features):ids_hf,rec=native(model,inputs)
  (g,first,trace,values,parent),checks=one(model,proc,inputs,s,s['R'],capture=True,retain=True,noop=True)
  assert ids_hf==g['token_ids'] and len(rec.logits)==len(trace)
  assert all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,trace)),'Native/No-op logits not torch.equal'
  assert counts==[len(ids_hf)]*32 and features==checks['image_features']
  assert all(c.get('logits_to_keep')==1 for c in rec.calls)
  fp=cache_fingerprint(parent);parent_ptrs=pointer_set(parent);conditions=[]
  for w in {tuple(x) for x in s['writers'].values()}:
   result,audit=one(model,proc,inputs,s,list(w),retain=True)
   gg,ff,_,_,cache=result
   assert not parent_ptrs&pointer_set(cache) and cache_fingerprint(parent)==fp
   assert audit['image_features']==features and gg['eos'] and not gg['empty'] and not gg['truncated']
   conditions.append(dict(writers=list(w),answer=gg['text'],token_ids=gg['token_ids'],eos=gg['eos'],mask_audit=gg['audit'],**audit));del cache,result
  try:
   with engine.ReceiverMask(model,inputs,s['Q']):raise RuntimeError('deliberate hook cleanup probe')
  except RuntimeError as e:assert str(e)=='deliberate hook cleanup probe'
  assert hook_footprint(model)==foot and weights=={n:p._version for n,p in model.named_parameters()}
  records.append(dict(question_id=q,Native_Noop_all_logits_torch_equal=True,steps=len(trace),Native_features=features,cache_independent_and_parent_unchanged=True,exception_hook_cleanup=True,conditions=conditions))
  write(d/'smoke_validation.json',dict(passed=False,completed=len(records),planned=len(ids),cases=records));del parent,rec,trace,inputs;gc.collect();torch.cuda.empty_cache()
 write(d/'smoke_validation.json',dict(passed=True,UTC=utc(),n=len(records),cases=records))
 print('SMOKE_PASS',d.name,len(records),flush=True)
def stage(model,proc,dataset,weights):
 d=OUT/dataset;manifest=read(d/'sample_manifest.json');assert manifest['status']=='FROZEN'
 assert read(d/'CPU_validation.json')['passed']
 rows=manifest['rows'];specs={r['question_id']:read(d/'input_specs'/(r['question_id']+'.json')) for r in rows}
 # The frozen source and every manifest/spec are rechecked before any inference or reuse.
 src=read(OUT/'source_manifest.json')['files'];assert sources()==src
 cpu=read(d/'CPU_validation.json');assert sha(d/'sample_manifest.json')==cpu['manifest_sha256']
 assert all(sha(d/'input_specs'/(q+'.json'))==h for q,h in cpu['input_specs_sha256'].items())
 smoke(model,proc,rows,specs,d,weights)
 total=0;started=time.monotonic();index=[];logical=[]
 for j,r in enumerate(rows):
  guard();q=r['question_id'];s=specs[q];assert sha(r['image_path'])==r['image_sha256'];inputs=inputs_for(model,proc,r,s)
  cp={}
  for name,w in s['writers'].items():
   cid=digest(w)[:24];path=d/'checkpoints'/q/(cid+'.json')
   if cid not in cp:
    if path.exists():
     c=read(path);assert c['source_manifest_sha256']==sha(OUT/'source_manifest.json') and c['writers']==w and c['input_hashes']==s['input_hashes'] and c['image_sha256']==r['image_sha256']
     assert c['score']==score(dataset,c['generation']['text'],r['answers'])
    else:
     result,audit=one(model,proc,inputs,s,w);g,first,_,values,_=result
     assert weights=={n:p._version for n,p in model.named_parameters()}
     c=dict(UTC=utc(),question_id=q,dataset=dataset,image_sha256=r['image_sha256'],writers=w,input_hashes=s['input_hashes'],input_spec_sha256=sha(d/'input_specs'/(q+'.json')),source_manifest_sha256=sha(OUT/'source_manifest.json'),generation=g,score=score(dataset,g['text'],r['answers']),first_token=int(first.argmax()),first_logits_sha256=hashlib.sha256(first.tobytes()).hexdigest(),weights_unchanged=True,**audit)
     write(path,c);total+=1
    cp[cid]=c;index.append(dict(question_id=q,path=str(path.relative_to(OUT)),sha256=sha(path)))
   c=cp[cid];g=c['generation']
   logical.append(dict(dataset=dataset,question_id=q,image_id=r['image_id'],image_sha256=r['image_sha256'],question=r['question'],answers=json.dumps(r['answers']),answer_type=r['answer_type'],condition=name,answer=g['text'],score=c['score'],first_token=c['first_token'],token_ids=json.dumps(g['token_ids']),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],k=len(w),R=len(s['R']),R_ratio=len(w)/len(s['R']),Q=len(s['Q']),checkpoint=str(path.relative_to(OUT)),metadata_overlap=r['metadata_overlap'],SHA_overlap=r['SHA_overlap']))
  del inputs;gc.collect()
  if (j+1)%20==0:print('PROGRESS',dataset,j+1,len(rows),'new',total,'seconds',round(time.monotonic()-started),flush=True)
 assert len(logical)==5*len(rows) and len({(x['question_id'],x['condition']) for x in logical})==len(logical)
 for row in index:assert sha(OUT/row['path'])==row['sha256']
 csvwrite(d/'per_question.csv',logical);csvwrite(d/'checkpoint_index.csv',index)
 write(d/'validation.json',dict(passed=True,UTC=utc(),CPU_passed=True,GPU_smoke_passed=True,all_checkpoint_hashes_verified=True,all_inputs_verified=True,all_source_hashes_verified=True,weights_unchanged=True,all_masks_exact_zero=True,all_nontarget_masks_exact=True,independent_cache=True,all32_MLP=True,hooks_clean=True,logical_records=len(logical),unique_generations=len(index),no_GQA_answers_reused=True,abnormal=dict(empty=sum(r['empty'] for r in logical),truncated=sum(r['truncated'] for r in logical),repeated=sum(r['repeated_trigram'] for r in logical),non_EOS=sum(not r['eos'] for r in logical))))
 write(d/'completion.json',dict(status='GENERATED_AND_VERIFIED',UTC=utc(),n=len(rows),logical_records=len(logical),unique_generations=len(index),new_generations_this_run=total,seconds=time.monotonic()-started))
 print('STAGE_COMPLETE',dataset,len(logical),len(index),flush=True)
def main():
 guard();state=hardware();write(OUT/'resource_start.json',state)
 chosen=next(x for x in state['gpus'].splitlines() if UUID in x);parts=[x.strip() for x in chosen.split(',')]
 assert 'RTX 3090' in chosen and int(parts[-3].split()[0])<300 and int(parts[-2].split()[0])>23000
 assert UUID not in state['processes']
 assert read(OUT/'cpu_validation.json')['passed'] and sources()==read(OUT/'source_manifest.json')['files']
 # Load outside inference_mode: parameter version counters remain auditable.
 model=None
 try:
  model,proc,env=load_model(str(MODEL));write(OUT/'environment.json',env);weights={n:p._version for n,p in model.named_parameters()}
  for dataset in ['vqa_v2','textvqa']:
   if read(OUT/dataset/'sample_manifest.json')['status']!='FROZEN':
    write(OUT/dataset/'completion.json',dict(status='DATA_UNAVAILABLE',UTC=utc()));continue
   if dataset=='textvqa':assert read(OUT/'vqa_v2/completion.json')['status']=='GENERATED_AND_VERIFIED'
   stage(model,proc,dataset,weights)
  assert weights=={n:p._version for n,p in model.named_parameters()}
  write(OUT/'gpu_run_complete.json',dict(UTC=utc(),status='GENERATED_AND_VERIFIED',GPU_UUID=UUID,pid=os.getpid(),gpu_memory=gpu_memory()))
 except BaseException:
  write(OUT/'STOP.json',dict(UTC=utc(),traceback=traceback.format_exc(),pid=os.getpid(),resources=hardware()));raise
 finally:
  if model is not None:del model
  gc.collect();torch.cuda.empty_cache();write(OUT/'resource_worker_final.json',hardware())
if __name__=='__main__':main()

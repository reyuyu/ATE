import os,sys
worker=int(sys.argv[1]);assert worker in [0,1]
uuids=['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
os.environ['CUDA_VISIBLE_DEVICES']=uuids[worker];os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']='1'
from .common import *
from .engine import generate,VisualReadingMask
import torch,numpy as np,time,traceback,gc
from textvqa_layer_recovery.run import observations,inputs_for,pointers
# Historical runner sets its old GPU UUID on import; restore selected device BEFORE load.
os.environ['CUDA_VISIBLE_DEVICES']=UUIDS[worker]
from vlm_common import load_model,gpu_memory
from knockout_probe import model_layers
from benchmarks.smoke import hook_footprint
from paper_rq1.engine import cache_fingerprint
from query_role_smoke_fix.run import native
import textvqa_layer_recovery.engine as legacy
legacy.guard=guard
torch.set_num_threads(1)
WOUT=OUT/'workers'/str(worker)
def one(model,proc,inputs,s,c,weights,**kw):
 before=hook_footprint(model);start=time.monotonic()
 with observations(model) as (counts,features):result=generate(model,proc.tokenizer,inputs,c,**kw)
 g,first,trace,mass,cache=result
 assert hook_footprint(model)==before and counts==[g['length']]*32 and len(features)==1 and features[0]['shape']==[1,576,4096]
 assert tensors_hash(inputs)==s['input_hashes'] and weights=={n:p._version for n,p in model.named_parameters()}
 return result,dict(MLP_calls=counts,image_features=features,hook_cleanup=True,input_hashes_unchanged=True,weights_unchanged=True,seconds=time.monotonic()-start,worker=worker,GPU_UUID=UUIDS[worker])
def checkpoint(r,s,c,result,audit,origin):
 g,first,_,mass,_=result
 data=dict(UTC=utc(),question_id=r['question_id'],condition=c,image_sha256=r['image_sha256'],input_hashes=s['input_hashes'],input_spec_sha256=sha(OUT/'input_specs'/(r['question_id']+'.json')),source_manifest_sha256=sha(OUT/'source_manifest.json'),protocol_sha256=sha(OUT/'protocol.json'),allowed_layers=DESIGN['conditions'][c],generation=g,score=score(g['text'],r['answers']),first_token=int(first.argmax()),first_logits_sha256=hashlib.sha256(first.tobytes()).hexdigest(),attention_mass=mass,origin=origin,**audit)
 write(OUT/'checkpoints'/r['question_id']/(c+'.json'),data)
def verify(v,r,s,c):
 assert v['condition']==c and v['allowed_layers']==DESIGN['conditions'][c] and v['source_manifest_sha256']==sha(OUT/'source_manifest.json') and v['protocol_sha256']==sha(OUT/'protocol.json')
 assert v['worker']==r['worker']==worker and v['input_hashes']==s['input_hashes'] and v['image_sha256']==r['image_sha256'] and v['generation']['audit']['passed'] and abs(v['score']-score(v['generation']['text'],r['answers']))<1e-12
def smoke(model,proc,rows,specs,weights):
 by={r['question_id']:r for r in rows};records=[];pending=[];fingerprints=[]
 for q in read(OUT/'smoke_selection.json')['question_ids']:
  r=by[q];s=specs[q];inputs=inputs_for(model,proc,r,s);foot=hook_footprint(model)
  with observations(model) as (counts,features):hf,rec=native(model,inputs)
  (ng,nf,nt,_,parent),na=one(model,proc,inputs,s,'B',weights,noop=True,capture=True,retain=True)
  assert hf==ng['token_ids'] and len(rec.logits)==len(nt) and all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,nt))
  assert all(x.get('logits_to_keep')==1 for x in rec.calls) and counts==[len(hf)]*32 and features==na['image_features']
  fp=cache_fingerprint(parent);ptr=pointers(parent);cases=[]
  for c in CONDITIONS:
   result,audit=one(model,proc,inputs,s,c,weights,capture=True,retain=True);g,f,trace,mass,child=result
   assert not ptr&pointers(child) and cache_fingerprint(parent)==fp and audit['image_features']==features
   if c=='A':assert len(trace)==len(rec.logits) and g['token_ids']==hf and all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,trace))
   if c in 'ABCDE':
    old=oldcp(q,c);assert g['token_ids']==old['generation']['token_ids'] and g['text']==old['generation']['text'] and hashlib.sha256(f.tobytes()).hexdigest()==old['first_logits_sha256'] and features==old['image_features']
    og,of,ot,_,oc=legacy.generate(model,proc.tokenizer,inputs,c,capture=True,retain=True);assert og['token_ids']==g['token_ids'] and torch.equal(torch.from_numpy(ot),torch.from_numpy(trace));assert not pointers(child)&pointers(oc);del oc,og,of,ot
   elif r['worker']==worker and (DESIGN['conditions'][c]['scope']==300 or r['stage2_subset']):pending.append((r,s,c,(g,f,None,mass,None),audit))
   fingerprints.append(dict(question_id=q,condition=c,token_ids=g['token_ids'],valid_logits_sha256=hashlib.sha256(trace.tobytes()).hexdigest(),first_logits_sha256=hashlib.sha256(f.tobytes()).hexdigest()))
   cases.append(dict(condition=c,answer=g['text'],length=g['length'],eos=g['eos'],truncated=g['truncated'],audit=g['audit']));del child,result,trace,mass
  try:
   with VisualReadingMask(model,inputs,'Both-8'):raise RuntimeError('deliberate cleanup probe')
  except RuntimeError as e:assert str(e)=='deliberate cleanup probe'
  assert hook_footprint(model)==foot
  records.append(dict(question_id=q,Native_noop_all_valid_logits_torch_equal=True,A_all_open_native_exact=True,historical_A_E_token_ids_first_logits_sha_and_all_valid_old_engine_logits_exact=True,independent_KV=True,features_unchanged=True,exception_cleanup=True,conditions=cases))
  write(WOUT/'gpu_smoke_validation.json',dict(passed=False,completed=len(records),records=records));print('SMOKE_CASE_PASS',worker,q,flush=True);del inputs,parent,rec,nt;gc.collect();torch.cuda.empty_cache()
 write(WOUT/'smoke_fingerprints.json',fingerprints);write(WOUT/'gpu_smoke_validation.json',dict(passed=True,UTC=utc(),worker=worker,GPU_UUID=UUIDS[worker],n=len(records),records=records))
 for r,s,c,result,audit in pending:checkpoint(r,s,c,result,audit,'Fresh smoke reused after all local smoke gates passed')
 print('SMOKE_PASS',worker,len(records),flush=True)
def wait_both_smoke():
 start=time.monotonic()
 while not all((OUT/'workers'/str(j)/'gpu_smoke_validation.json').exists() and read(OUT/'workers'/str(j)/'gpu_smoke_validation.json')['passed'] for j in [0,1]):guard();assert time.monotonic()-start<1800,'Peer smoke wait timeout';time.sleep(2)
 # Exact cross-GPU full valid-logit trace hashes on every smoke condition.
 assert read(OUT/'workers/0/smoke_fingerprints.json')==read(OUT/'workers/1/smoke_fingerprints.json'),'Cross-GPU smoke exact comparison failed'
def stage(model,proc,weights):
 manifest=read(OUT/'sample_manifest.json');cpu=read(OUT/'cpu_validation.json');assert cpu['passed'] and sha(OUT/'sample_manifest.json')==cpu['manifest_sha256']
 assert sources()==read(OUT/'source_manifest.json')['files'] and sha(OUT/'protocol.json')==read(OUT/'source_manifest.json')['protocol_sha256']
 rows=manifest['rows'];specs={r['question_id']:read(OUT/'input_specs'/(r['question_id']+'.json')) for r in rows}
 assert all(sha(OUT/'input_specs'/(q+'.json'))==h for q,h in cpu['input_specs_sha256'].items())
 if not (WOUT/'gpu_smoke_validation.json').exists() or not read(WOUT/'gpu_smoke_validation.json')['passed']:smoke(model,proc,rows,specs,weights)
 wait_both_smoke();owned=[r for r in rows if r['worker']==worker];index=[]
 for number,conditions in [(1,STAGE1),(2,STAGE2)]:
  if number==2:
   start=time.monotonic()
   while not all((OUT/'workers'/str(j)/'stage1_done.json').exists() for j in [0,1]):guard();assert time.monotonic()-start<3600;time.sleep(2)
  rr=owned if number==1 else [r for r in owned if r['stage2_subset']];generated=0;start=time.monotonic()
  for j,r in enumerate(rr):
   guard();q=r['question_id'];s=specs[q];assert sha(r['image_path'])==r['image_sha256'];inputs=inputs_for(model,proc,r,s);expected_features=oldcp(q,'B')['image_features']
   for c in conditions:
    p=OUT/'checkpoints'/q/(c+'.json')
    if not p.exists():
     result,audit=one(model,proc,inputs,s,c,weights);assert audit['image_features']==expected_features;checkpoint(r,s,c,result,audit,'Fresh stage'+str(number));generated+=1;del result
    v=read(p);verify(v,r,s,c);index.append(dict(question_id=q,condition=c,worker=worker,path=str(p.relative_to(OUT)),sha256=sha(p)))
   del inputs;gc.collect()
   if (j+1)%10==0:print('PROGRESS',worker,'stage',number,j+1,len(rr),'new',generated,'seconds',round(time.monotonic()-start),flush=True)
  write(WOUT/('stage'+str(number)+'_done.json'),dict(passed=True,UTC=utc(),worker=worker,n=len(rr),conditions=conditions,logical_records=len(rr)*len(conditions),new_in_formal_stage=generated,seconds=time.monotonic()-start))
 assert len(index)==1770;csvwrite(WOUT/'checkpoint_index.csv',index);write(WOUT/'generation_validation.json',dict(passed=True,UTC=utc(),worker=worker,fresh_checkpoints=1770,all_hashes_inputs_sources_masks_verified=True,all32_MLP=True,features_weights_unchanged=True,independent_KV=True,hooks_clean=True))
def main():
 guard();assert UUIDS==uuids;state=hardware();write(WOUT/'resource_start.json',state)
 for u in UUIDS:
  line=next(x for x in state['gpus'].splitlines() if u in x);parts=line.split(',');assert int(parts[-3].split()[0])<300 and int(parts[-2].split()[0])>23000 and u not in state['processes'],'Both GPUs must be free before launch'
 write(WOUT/'preload_ready.json',dict(UTC=utc(),worker=worker,both_free=True))
 ready_start=time.monotonic()
 while not all((OUT/'workers'/str(j)/'preload_ready.json').exists() for j in [0,1]):guard();assert time.monotonic()-ready_start<180;time.sleep(1)
 start=time.monotonic();model=None
 try:
  model,proc,env=load_model(str(MODEL));write(WOUT/'environment.json',env);weights={n:p._version for n,p in model.named_parameters()};names={id(m):n for n,m in model.named_modules()};mapping=[dict(layer=k,module=names[id(b)]) for k,b in enumerate(model_layers(model))];assert len(mapping)==32 and all(x['module'].endswith('.layers.'+str(x['layer'])) for x in mapping);write(WOUT/'layer_mapping.json',mapping)
  stage(model,proc,weights)
  write(WOUT/'gpu_run_complete.json',dict(passed=True,UTC=utc(),worker=worker,pid=os.getpid(),GPU_UUID=UUIDS[worker],total_GPU_phase_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),memory=gpu_memory()))
 except BaseException:
  if not (OUT/'STOP.json').exists():write(OUT/'STOP.json',dict(UTC=utc(),worker=worker,pid=os.getpid(),traceback=traceback.format_exc(),resources=hardware()))
  raise
 finally:
  if model is not None:del model
  gc.collect();torch.cuda.empty_cache();write(WOUT/'resource_worker_final.json',hardware())
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (OUT/'STOP.json').exists():write(OUT/'STOP.json',dict(UTC=utc(),worker=worker,pid=os.getpid(),traceback=traceback.format_exc(),resources=hardware()))
  raise

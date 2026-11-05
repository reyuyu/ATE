"""Two UUID-bound resident workers with strict shared gates and atomic records."""
import os,argparse
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);A=p.parse_args();WORKER=A.worker
assert WORKER in [0,1]
from .policy import *
UUID=UUIDS[WORKER]
os.environ.update(CUDA_VISIBLE_DEVICES=UUID,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import torch,numpy as np,subprocess,hashlib,time,gc,json,contextlib,traceback,fcntl
from receiver_cross_dataset.common import read,write,sha,tensors_hash,hardware,MODEL,utc
from vlm_common import load_model,prepare_inputs
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from query_role_smoke_fix.run import native
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
from .engine import generate,VisualMask
assert not torch.cuda.is_initialized();os.environ['CUDA_VISIBLE_DEVICES']=UUID;torch.set_num_threads(1)
PROTO=read(OUT/'protocol.json');SOURCE=sha(OUT/'execution_source_manifest.json');PHASH=sha(OUT/'protocol.json');SHARD=read(OUT/'shards.json')
REUSE={(r['question_id'],r['condition']):r for r in read(OUT/'historical_reuse_audit.json')['checkpoints']}
ROWS=read(OUT/'sample_manifest.json')['rows'];BY={r['question_id']:r for r in ROWS};SPECS={q:read(INPUTS/(q+'.json')) for q in BY}
EV=TextVQAAccuracyEvaluator();START=time.monotonic();COUNTS=dict(generated=0,resumed=0);LAST_PROGRESS=0
def old(q,c):return read(REUSE[(q,c)]['checkpoint_path'])
def firstsha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def pointers(cache):return {v.data_ptr() for a in [cache.key_cache,cache.value_cache] for v in a}
def verify_sources():
 f=read(OUT/'execution_source_manifest.json');assert sha(OUT/'protocol.json')==f['protocol_sha256']
 assert all(sha(n)==v for n,v in f['files'].items())
 assert sha(OUT/'sample_manifest.json')==PROTO['manifest_sha256'] and sha(OUT/'shards.json')==PROTO['shards_sha256']
def score(text,answers):
 with open(os.devnull,'w') as z,contextlib.redirect_stderr(z):return EV.eval_pred_list([dict(pred_answer=text,gt_answers=answers)])
def inputs_for(model,proc,q):
 r=BY[q];s=SPECS[q];inputs=prepare_inputs(model,proc,r['image_path'],r['question']+'\nAnswer the question using a single word or phrase.')[0]
 assert inputs['input_ids'][0].tolist()==s['prompt_ids'] and tensors_hash(inputs)==s['input_hashes'];return inputs
@contextlib.contextmanager
def observe(model):
 counts=[0]*32;norms=[0]*32;blocks=[0]*32;features=[];calls=[];hs=[]
 def before(m,a,kw):
  assert kw.get('logits_to_keep')==1
  if not calls:assert kw.get('past_key_values') is None or kw['past_key_values'].get_seq_length()==0
  calls.append(1)
 def vision(m,a,out):features.append(dict(shape=list(out.shape),sha256=tensor_sha(out)))
 try:
  hs.append(model.register_forward_pre_hook(before,with_kwargs=True));hs.append(model.multi_modal_projector.register_forward_hook(vision))
  for k,b in enumerate(model_layers(model)):
   def mlp(m,a,o,k=k):counts[k]+=1
   def norm(m,a,o,k=k):norms[k]+=1
   def block(m,a,o,k=k):blocks[k]+=1
   hs.extend([b.mlp.register_forward_hook(mlp),b.input_layernorm.register_forward_hook(norm),b.register_forward_hook(block)])
  yield dict(MLP=counts,Norm=norms,Block=blocks,features=features,forward_calls=calls)
 finally:
  for h in hs:h.remove()
def one(model,proc,inputs,c,smoke=False,**kw):
 foot=hook_footprint(model);start=time.monotonic()
 if smoke:
  with observe(model) as audit:res=generate(model,proc.tokenizer,inputs,c,**kw)
  g=res[0];assert audit['MLP']==audit['Norm']==audit['Block']==[g['length']]*32 and len(audit['forward_calls'])==g['length']
  assert len(audit['features'])==1 and audit['features'][0]['shape']==[1,576,4096]
  audit={k:v for k,v in audit.items() if k!='forward_calls'}
 else:res=generate(model,proc.tokenizer,inputs,c,**kw);audit={}
 assert hook_footprint(model)==foot
 return res,dict(**audit,hook_cleanup=True,empty_cache_at_start=True,seconds=time.monotonic()-start)
def save(q,c,res,audit,origin):
 g,f,_,_=res;r=BY[q];s=SPECS[q]
 if CONDITIONS[c]['family'] in ['Donly','Middle-minus','Addback']:assert firstsha(f)==old(q,'Native')['first_logits_sha256'],'Native first vector differs'
 g['audit']['condition']=c
 record=dict(question_id=q,condition=c,definition=CONDITIONS[c],image_sha256=r['image_sha256'],input_hashes=s['input_hashes'],input_spec_sha256=r['input_spec_sha256'],protocol_sha256=PHASH,source_manifest_sha256=SOURCE,model_manifest_sha256=PROTO['model_manifest_sha256'],runtime_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),generation=g,score=score(g['text'],r['answers']),first_logits_sha256=firstsha(f),worker=WORKER,GPU_UUID=UUID,pid=os.getpid(),UTC=utc(),origin=origin,**audit)
 path=OUT/'checkpoints'/q/(c+'.json');assert not path.exists(),'Never overwrite existing record';write(path,record)
def verify_cp(q,c):
 path=OUT/'checkpoints'/q/(c+'.json');v=read(path)
 assert v['question_id']==q and v['condition']==c and v['definition']==CONDITIONS[c] and v['worker']==SHARD['by_question'][q]==WORKER
 assert v['protocol_sha256']==PHASH and v['source_manifest_sha256']==SOURCE and v['input_hashes']==SPECS[q]['input_hashes'] and v['image_sha256']==BY[q]['image_sha256']
 assert v['generation']['audit']['passed'] and v['hook_cleanup'];return v
def progress(stage,force=False):
 global LAST_PROGRESS
 now=time.monotonic()
 if force or now-LAST_PROGRESS>=1800:
  v=dict(UTC=utc(),worker=WORKER,stage=stage,**COUNTS,worker_seconds=now-START,generations_per_second=COUNTS['generated']/max(now-START,1),remaining_wall_seconds=PROTO['deadline_epoch']-time.time(),GPU_stop_remaining_seconds=PROTO['gpu_deadline_epoch']-time.time(),anomaly=False)
  write(OUT/f'progress_worker{WORKER}.json',v)
  with (OUT/f'progress_worker{WORKER}.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
  print('PROGRESS',json.dumps(v),flush=True);LAST_PROGRESS=now
def wait_gate(name):
 while not all((OUT/f'{name}_worker{i}.json').exists() for i in [0,1]):guard();time.sleep(2)
 for i in [0,1]:assert read(OUT/f'{name}_worker{i}.json')['passed']
def smoke(model,proc,versions):
 records=[];pending=[]
 for q in read(OUT/'smoke_selection.json')['question_ids']:
  print('SMOKE_BEGIN',WORKER,q,flush=True);inputs=inputs_for(model,proc,q);foot=hook_footprint(model)
  hf,rec=native(model,inputs);parentres,pa=one(model,proc,inputs,'Native',smoke=True,noop=True,audit=True,capture=True,retain=True)
  ng,nf,nt,parent=parentres;assert hf==ng['token_ids'] and len(rec.logits)==len(nt) and all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,nt))
  assert ng['token_ids']==old(q,'Native')['generation']['token_ids'] and firstsha(nf)==old(q,'Native')['first_logits_sha256']
  fp=cache_fingerprint(parent);ptr=pointers(parent);firsts={};cases=[]
  # Every layer and every frozen intervention is inspected on each smoke question.
  for c in ['Native','Middle','Decode-None']+NATIVE+[x for x in PRIMARY if x.startswith('Middle')]:
   guard();res,a=one(model,proc,inputs,c,smoke=True,audit=True,capture=c=='Native',retain=True);g,f,tr,child=res
   assert not ptr&pointers(child) and cache_fingerprint(parent)==fp and a['features']==pa['features']
   assert tensors_hash(inputs)==SPECS[q]['input_hashes'] and versions=={n:p._version for n,p in model.named_parameters()}
   fam=CONDITIONS[c]['family'];firsts[c]=firstsha(f)
   if c in ['Native','Middle','Decode-None']:
    v=old(q,c);assert g['token_ids']==v['generation']['token_ids'] and firstsha(f)==v['first_logits_sha256']
   if c=='Native':assert hf==g['token_ids'] and len(tr)==len(rec.logits) and all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,tr))
   if fam in ['Donly','Middle-minus','Addback','Middle','Decode-None']:assert torch.equal(torch.from_numpy(f),rec.logits[0][-1])
   if c in ['All-Text-L0','Ponly-L15','Donly-L31','Both-L8','Middle-minus-L11','Addback-L0']:
    fast,fa=one(model,proc,inputs,c,smoke=True,audit=False,capture=True)
    assert fast[0]['token_ids']==g['token_ids'] and firstsha(fast[1])==firstsha(f),'Formal vs smoke math changed'
   cases.append(dict(condition=c,token_ids=g['token_ids'],first_logits_sha256=firstsha(f),audit=g['audit'],all32_MLP_Norm_Block=True,independent_KV=True))
   if c in NATIVE+PRIMARY and SHARD['by_question'][q]==WORKER:
    pending.append((q,c,(g,f,None,None),a))
   del child,res,tr
  for k in range(32):assert firsts[f'Ponly-L{k}']==firsts[f'Both-L{k}']
  try:
   with VisualMask(model,inputs,'All-Text-L0',audit=True):raise RuntimeError('intentional cleanup test')
  except RuntimeError as e:assert str(e)=='intentional cleanup test'
  assert hook_footprint(model)==foot
  records.append(dict(question_id=q,Native_HF_noop_all_valid_logits_torch_equal=True,historical_Native_Middle_DecodeNone_exact=True,Donly_all32_first_logits_Native_torch_equal=True,Ponly_Both_all32_first_logits_exact=True,independent_KV_parent_unchanged=True,all32_MLP_Norm_Block=True,inputs_features_weights_unchanged=True,normal_exception_hooks_clean=True,cases=cases))
  del parent,parentres,rec,nt,inputs;gc.collect();torch.cuda.empty_cache();print('SMOKE_CASE_PASS',WORKER,q,flush=True)
 write(OUT/f'gpu_smoke_worker{WORKER}.json',dict(passed=True,UTC=utc(),worker=WORKER,UUID=UUID,records=records,n=3))
 wait_gate('gpu_smoke')
 for q,c,res,a in pending:
  path=OUT/'checkpoints'/q/(c+'.json')
  if not path.exists():save(q,c,res,a,'audited smoke on owning worker reused after both GPU gates');COUNTS['generated']+=1
 print('DUAL_GPU_SMOKE_PASS',WORKER,flush=True)
def execute_stage(model,proc,versions,stage,qs,conditions):
 progress(stage,True);start=time.monotonic();nnew=0
 for q in qs:
  if SHARD['by_question'][q]!=WORKER:continue
  guard();r=BY[q];assert sha(r['image_path'])==r['image_sha256'];inputs=inputs_for(model,proc,q)
  for c in conditions:
   guard();path=OUT/'checkpoints'/q/(c+'.json')
   if path.exists():verify_cp(q,c);COUNTS['resumed']+=1;continue
   res,a=one(model,proc,inputs,c);save(q,c,res,a,'fresh formal');COUNTS['generated']+=1;nnew+=1;del res
   progress(stage)
  assert tensors_hash(inputs)==SPECS[q]['input_hashes'] and versions=={n:p._version for n,p in model.named_parameters()}
  del inputs
  if nnew%1000<len(conditions):print('ROW_DONE',WORKER,stage,q,'fresh',nnew,'seconds',round(time.monotonic()-start),flush=True)
 for q in qs:
  if SHARD['by_question'][q]==WORKER:
   for c in conditions:verify_cp(q,c)
 write(OUT/f'{stage}_worker{WORKER}.json',dict(passed=True,UTC=utc(),n_owned=sum(SHARD['by_question'][q]==WORKER for q in qs),conditions=conditions,new=nnew,seconds=time.monotonic()-start))
 progress(stage+'_complete',True);wait_gate(stage)
def main():
 model=None;proc=None;lock=(OUT/f'worker{WORKER}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  guard();verify_sources();assert read(OUT/'cpu_validation.json')['passed']
  state=hardware();write(OUT/f'resource_start_worker{WORKER}.json',state);line=next(x for x in state['gpus'].splitlines() if UUID in x);parts=[x.strip() for x in line.split(',')]
  assert 'RTX 3090' in line and int(parts[-3].split()[0])<300 and int(parts[-2].split()[0])>23000 and UUID not in state['processes']
  model,proc,env=load_model(str(MODEL));assert env['transformers']=='4.49.0' and env['model_revision']==PROTO['model_revision'];write(OUT/f'environment_worker{WORKER}.json',dict(**env,worker=WORKER,UUID=UUID,pid=os.getpid()))
  versions={n:p._version for n,p in model.named_parameters()};assert len(model_layers(model))==32
  smoke(model,proc,versions)
  pilot=SHARD['pilot120'];allq=[r['question_id'] for r in sorted(ROWS,key=lambda r:r['image_sha256'])]
  execute_stage(model,proc,versions,'pilot120',pilot,NATIVE)
  execute_stage(model,proc,versions,'primary300',allq,PRIMARY)
  execute_stage(model,proc,versions,'secondary300',allq,SECONDARY)
  # Only start optional after complete139x300 and with conservative throughput reserve.
  rates=[]
  for i in [0,1]:
   v=read(OUT/f'primary300_worker{i}.json');rates.append(v['new']/max(v['seconds'],1))
  projected=(150*32)/max(min(rates),0.01)*1.4
  do_optional=time.time()+projected<PROTO['gpu_deadline_epoch']
  write(OUT/f'optional_decision_worker{WORKER}.json',dict(passed=True,execute=do_optional,projected_seconds=projected,seconds_left=PROTO['gpu_deadline_epoch']-time.time(),rule='all139x300 complete; conservative slow-worker observed throughput x1.4 projected optional fits before GPU cutoff'))
  wait_gate('optional_decision');do_optional=all(read(OUT/f'optional_decision_worker{i}.json')['execute'] for i in [0,1])
  if do_optional:
   for q in read(OUT/'smoke_selection.json')['question_ids']:
    inputs=inputs_for(model,proc,q)
    for c in OPTIONAL:
     res,a=one(model,proc,inputs,c,smoke=True,audit=True)
     assert firstsha(res[1])==old(q,'Native')['first_logits_sha256']
     if SHARD['by_question'][q]==WORKER:save(q,c,res,a,'optional smoke after all mandatory A/B complete');COUNTS['generated']+=1
    del inputs
   write(OUT/f'optional_smoke_worker{WORKER}.json',dict(passed=True,questions=read(OUT/'smoke_selection.json')['question_ids'],conditions=OPTIONAL));wait_gate('optional_smoke')
   execute_stage(model,proc,versions,'optional300',allq,OPTIONAL)
  verify_sources();write(OUT/f'worker_complete{WORKER}.json',dict(passed=True,status='ALL_MANDATORY_COMPLETE',UTC=utc(),worker=WORKER,**COUNTS,seconds=time.monotonic()-START,optional_complete=do_optional,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
 except TimeoutError:
  write(OUT/f'worker_deadline{WORKER}.json',dict(status='PRESPECIFIED_TIME_LIMIT',UTC=utc(),worker=WORKER,**COUNTS));print('TIME_LIMIT_SAFE_STOP',WORKER,flush=True)
 except BaseException:
  write(OUT/'STOP.json',dict(UTC=utc(),worker=WORKER,pid=os.getpid(),traceback=traceback.format_exc(),hardware=hardware()));raise
 finally:
  del model,proc;gc.collect();torch.cuda.empty_cache();write(OUT/f'resource_final_worker{WORKER}.json',hardware())
if __name__=='__main__':main()

"""UUID-bound fixed factorial experiment; shared two-card Smoke barrier."""
import os,argparse
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,required=True);args=p.parse_args();WORKER=args.worker
from .policy import *
assert WORKER in [0,1]
UUID=UUIDS[WORKER];os.environ.update(CUDA_VISIBLE_DEVICES=UUID,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import torch,numpy as np,subprocess,hashlib,gc,contextlib,traceback,fcntl
from receiver_cross_dataset.common import read,write,sha,tensors_hash,hardware,MODEL,utc
from vlm_common import load_model,prepare_inputs
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from query_role_smoke_fix.run import native
from textvqa_single_layer_visual_ablation.engine import generate as legacy_generate
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
from .engine import generate,VisualMask
assert not torch.cuda.is_initialized();torch.set_num_threads(1)
PROTO=read(OUT/'protocol.json');SOURCE=sha(OUT/'execution_source_manifest.json');PHASH=sha(OUT/'protocol.json');SHARD=read(OUT/'shards.json')
REUSE={(r['question_id'],r['condition']):r for r in read(OUT/'historical_reuse_audit.json')['checkpoints']}
ROWS=read(OUT/'sample_manifest.json')['rows'];BY={r['question_id']:r for r in ROWS};SPECS={q:read(INPUTS/(q+'.json')) for q in BY}
EV=TextVQAAccuracyEvaluator();START=time.monotonic();COUNTS=dict(generated=0,resumed=0);LAST_PROGRESS=0
def firstsha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def pointers(cache):return {v.data_ptr() for a in [cache.key_cache,cache.value_cache] for v in a}
def verify_sources():
 f=read(OUT/'execution_source_manifest.json');assert sha(OUT/'protocol.json')==f['protocol_sha256'];assert all(sha(n)==v for n,v in f['files'].items());assert sha(OUT/'sample_manifest.json')==PROTO['manifest_sha256'] and sha(OUT/'shards.json')==PROTO['shards_sha256']
def score(text,answers):
 with open(os.devnull,'w') as z,contextlib.redirect_stderr(z):return EV.eval_pred_list([dict(pred_answer=text,gt_answers=answers)])
def inputs_for(model,proc,q):
 r=BY[q];s=SPECS[q];inputs=prepare_inputs(model,proc,r['image_path'],r['question']+'\nAnswer the question using a single word or phrase.')[0];assert inputs['input_ids'][0].tolist()==s['prompt_ids'] and tensors_hash(inputs)==s['input_hashes'];return inputs
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
  with observe(model) as a:res=generate(model,proc.tokenizer,inputs,c,**kw)
  g=res[0];assert a['MLP']==a['Norm']==a['Block']==[g['length']]*32 and len(a['forward_calls'])==g['length'];assert len(a['features'])==1 and a['features'][0]['shape']==[1,576,4096];a={k:v for k,v in a.items() if k!='forward_calls'}
 else:res=generate(model,proc.tokenizer,inputs,c,**kw);a={}
 assert hook_footprint(model)==foot;return res,dict(**a,hook_cleanup=True,empty_cache_at_start=True,seconds=time.monotonic()-start)
def reference_first(q):
 if BY[q]['cohort']=='old300':return REUSE[(q,'Native')]['first_logits_sha256']
 return read(OUT/'checkpoints'/q/'Native.json')['first_logits_sha256']
def save(q,c,res,audit,origin):
 g,f,_,_=res;r=BY[q];s=SPECS[q]
 if c!='Native':assert firstsha(f)==reference_first(q),'Prefill first vector differs from Native'
 record=dict(question_id=q,cohort=r['cohort'],condition=c,allowed_decode_image_layers=CONDITIONS[c],all32_Pminus1_native=True,image_sha256=r['image_sha256'],input_hashes=s['input_hashes'],input_spec_sha256=r['input_spec_sha256'],protocol_sha256=PHASH,source_manifest_sha256=SOURCE,model_manifest_sha256=PROTO['model_manifest_sha256'],runtime_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),generation=g,score=score(g['text'],r['answers']),first_logits_sha256=firstsha(f),worker=WORKER,GPU_UUID=UUID,pid=os.getpid(),UTC=utc(),origin=origin,**audit)
 path=OUT/'checkpoints'/q/(c+'.json');assert not path.exists();write(path,record)
def verify_cp(q,c):
 v=read(OUT/'checkpoints'/q/(c+'.json'));assert v['question_id']==q and v['condition']==c and v['allowed_decode_image_layers']==CONDITIONS[c] and v['worker']==SHARD['by_question'][q]==WORKER;assert v['protocol_sha256']==PHASH and v['source_manifest_sha256']==SOURCE and v['input_hashes']==SPECS[q]['input_hashes'] and v['image_sha256']==BY[q]['image_sha256'];assert v['generation']['audit']['passed'] and v['hook_cleanup'];return v
def progress(stage,force=False):
 global LAST_PROGRESS
 now=time.monotonic()
 if force or now-LAST_PROGRESS>=120:
  v=dict(UTC=utc(),worker=WORKER,stage=stage,**COUNTS,worker_seconds=now-START,generations_per_second=COUNTS['generated']/max(now-START,1),remaining_GPU_seconds=PROTO['gpu_deadline_epoch']-time.time());write(OUT/f'progress_worker{WORKER}.json',v);print('PROGRESS',json.dumps(v),flush=True);LAST_PROGRESS=now
def wait_gate(name):
 while not all((OUT/f'{name}_worker{i}.json').exists() for i in [0,1]):guard();time.sleep(2)
 for i in [0,1]:assert read(OUT/f'{name}_worker{i}.json')['passed']
def smoke(model,proc,versions):
 records=[];pending=[]
 for q in read(OUT/'smoke_selection.json')['question_ids']:
  guard();print('SMOKE_BEGIN',WORKER,q,flush=True);inputs=inputs_for(model,proc,q);foot=hook_footprint(model)
  hf,rec=native(model,inputs);parentres,pa=one(model,proc,inputs,'Native',smoke=True,noop=True,audit=True,capture=True,retain=True)
  ng,nf,nt,parent=parentres;assert hf==ng['token_ids'] and len(rec.logits)==len(nt) and all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,nt));fp=cache_fingerprint(parent);ptr=pointers(parent);cases=[]
  for c in CONDITIONS:
   res,a=one(model,proc,inputs,c,smoke=True,audit=True,capture=True,retain=True);g,f,tr,child=res
   assert torch.equal(torch.from_numpy(f),rec.logits[0][-1]);assert not ptr&pointers(child) and cache_fingerprint(parent)==fp and a['features']==pa['features'];assert tensors_hash(inputs)==SPECS[q]['input_hashes'] and versions=={n:p._version for n,p in model.named_parameters()}
   if c=='Native':assert hf==g['token_ids'] and len(tr)==len(rec.logits) and all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,tr))
   replay=False
   if c in ['A','B']:
    legacy=legacy_generate(model,proc.tokenizer,inputs,'Middle-minus-L0' if c=='A' else 'Middle',capture=True);assert legacy[0]['token_ids']==g['token_ids'] and torch.equal(torch.from_numpy(legacy[2]),torch.from_numpy(tr));replay=True;del legacy
   if (q,c) in REUSE:
    old=read(REUSE[(q,c)]['checkpoint_path']);assert g['token_ids']==old['generation']['token_ids'] and firstsha(f)==old['first_logits_sha256']
   fast,fa=one(model,proc,inputs,c,smoke=True,capture=True);assert fast[0]['token_ids']==g['token_ids'] and torch.equal(torch.from_numpy(fast[2]),torch.from_numpy(tr));del fast
   cases.append(dict(condition=c,token_ids=g['token_ids'],first_logits_sha256=firstsha(f),all_valid_logits_sha256=firstsha(tr),audit=g['audit'],frozen_legacy_all_logits_equal=replay,formal_vs_audit_all_logits_equal=True,independent_KV=True,all32_MLP_Norm_Block=True))
   if SHARD['by_question'][q]==WORKER and (BY[q]['cohort']=='new_images' or c in ['C','D']):pending.append((q,c,(g,f,None,None),a))
   del child,res,tr
  try:
   with VisualMask(model,inputs,'A',audit=True):raise RuntimeError('intentional cleanup test')
  except RuntimeError as e:assert str(e)=='intentional cleanup test'
  assert hook_footprint(model)==foot
  records.append(dict(question_id=q,Native_HF_noop_all_valid_logits_torch_equal=True,all_six_first_logits_Native_torch_equal=True,actual_forbidden_probabilities_zero=True,non_target_masks_exact=True,all32_MLP_Norm_Block=True,positions_cache_features_weights_exact=True,hooks_normal_exception_clean=True,cases=cases))
  del parent,parentres,rec,nt,inputs;gc.collect();torch.cuda.empty_cache();print('SMOKE_CASE_PASS',WORKER,q,flush=True)
 write(OUT/f'gpu_smoke_worker{WORKER}.json',dict(passed=True,UTC=utc(),worker=WORKER,UUID=UUID,records=records));wait_gate('gpu_smoke')
 # Native must be saved before new-input interventions for the first-logit check.
 for q,c,res,a in pending:
  if not (OUT/'checkpoints'/q/(c+'.json')).exists():save(q,c,res,a,'fresh owning-worker Smoke retained only after both-card gate');COUNTS['generated']+=1
 print('DUAL_GPU_SMOKE_PASS',WORKER,flush=True)
def execute_stage(model,proc,versions,stage,qs,conditions):
 progress(stage,True);start=time.monotonic();nnew=0
 for q in qs:
  if SHARD['by_question'][q]!=WORKER:continue
  guard();assert sha(BY[q]['image_path'])==BY[q]['image_sha256'];inputs=inputs_for(model,proc,q)
  for c in conditions:
   guard();path=OUT/'checkpoints'/q/(c+'.json')
   if path.exists():verify_cp(q,c);COUNTS['resumed']+=1;continue
   res,a=one(model,proc,inputs,c);save(q,c,res,a,'fresh formal');COUNTS['generated']+=1;nnew+=1;del res;progress(stage)
  assert tensors_hash(inputs)==SPECS[q]['input_hashes'] and versions=={n:p._version for n,p in model.named_parameters()};del inputs
 for q in qs:
  if SHARD['by_question'][q]==WORKER:
   for c in conditions:verify_cp(q,c)
 write(OUT/f'{stage}_worker{WORKER}.json',dict(passed=True,UTC=utc(),n_owned=sum(SHARD['by_question'][q]==WORKER for q in qs),conditions=conditions,new=nnew,seconds=time.monotonic()-start));progress(stage+'_complete',True);wait_gate(stage)
def main():
 model=None;proc=None;lock=(OUT/f'worker{WORKER}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  guard();verify_sources();assert read(OUT/'cpu_validation.json')['passed'];state=hardware();write(OUT/f'resource_start_worker{WORKER}.json',state);line=next(x for x in state['gpus'].splitlines() if UUID in x);parts=[x.strip() for x in line.split(',')];assert 'RTX 3090' in line and int(parts[-3].split()[0])<300 and int(parts[-2].split()[0])>23000 and UUID not in state['processes']
  model,proc,env=load_model(str(MODEL));assert env['transformers']=='4.49.0' and env['model_revision']==PROTO['model_revision'];write(OUT/f'environment_worker{WORKER}.json',dict(**env,worker=WORKER,UUID=UUID,pid=os.getpid()));versions={n:p._version for n,p in model.named_parameters()};assert len(model_layers(model))==32
  smoke(model,proc,versions)
  for stage,cohort,cs in [('new_validation','new_images',list(CONDITIONS)),('old300_completion','old300',['C','D'])]:
   qs=[r['question_id'] for r in sorted(ROWS,key=lambda r:(r['image_sha256'],r['question_id'])) if r['cohort']==cohort];execute_stage(model,proc,versions,stage,qs,cs)
  verify_sources();write(OUT/f'worker_complete{WORKER}.json',dict(passed=True,UTC=utc(),worker=WORKER,**COUNTS,seconds=time.monotonic()-START,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
 except TimeoutError:write(OUT/f'worker_deadline{WORKER}.json',dict(status='PRESPECIFIED_TIME_LIMIT',UTC=utc(),worker=WORKER,**COUNTS));print('TIME_LIMIT_SAFE_STOP',WORKER,flush=True)
 except BaseException:write(OUT/'STOP.json',dict(UTC=utc(),worker=WORKER,pid=os.getpid(),traceback=traceback.format_exc(),hardware=hardware()));raise
 finally:
  del model,proc;gc.collect();torch.cuda.empty_cache();write(OUT/f'resource_final_worker{WORKER}.json',hardware())
if __name__=='__main__':main()

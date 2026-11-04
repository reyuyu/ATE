"""Single-GPU, strictly gated generation of the four frozen transfer conditions."""
import os,argparse
parser=argparse.ArgumentParser();parser.add_argument('--gpu-uuid',required=True)
ARGS=parser.parse_args();UUID=ARGS.gpu_uuid
os.environ['CUDA_VISIBLE_DEVICES']=UUID
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='1'
from .policy import ROOT,OUT as O,LAYER_SETS,NEW,guard
from .engine import bind
impl=bind()
import torch,numpy as np,pathlib,hashlib,subprocess,time,gc,traceback,contextlib,fcntl,json
from receiver_cross_dataset.common import read,write,sha,csvwrite,tensors_hash,hardware,MODEL,utc
from benchmarks._vendor.m4c_evaluator import TextVQAAccuracyEvaluator
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from paper_rq1.engine import cache_fingerprint
from query_role_smoke_fix.run import native
from vlm_common import load_model,prepare_inputs,gpu_memory
# An imported historical module must not select another device before loading.
assert not torch.cuda.is_initialized()
os.environ['CUDA_VISIBLE_DEVICES']=UUID
torch.set_num_threads(1)
H=ROOT/'results/textvqa_visual_reaccess_budget_exploration'
P=pathlib.Path('/data/lk/ATE-resources/experiments/textvqa_coco_middle_transfer')
REUSE={(r['question_id'],r['condition']):r for r in read(O/'historical_reuse_audit.json')['checkpoints']}
SCORER=TextVQAAccuracyEvaluator()
def score(text,answers):
 with open(os.devnull,'w') as null,contextlib.redirect_stderr(null):return SCORER.eval_pred_list([dict(pred_answer=text,gt_answers=answers)])
def old(q,c):return read(REUSE[(q,c)]['checkpoint_path'])
def pointers(cache):return {v.data_ptr() for a in [cache.key_cache,cache.value_cache] for v in a}
def tsha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def firstsha(first):return hashlib.sha256(first.tobytes()).hexdigest()
def verify_sources():
 f=read(O/'execution_source_manifest.json')
 assert all(sha(ROOT/k)==v for k,v in f['files'].items())
 assert all(sha(k)==v for k,v in f['external_files'].items())
 assert sha(O/'execution_protocol.json')==f['execution_protocol_sha256']
 assert sha(O/'execution_model_manifest.json')==f['model_manifest_sha256']
 model=read(O/'execution_model_manifest.json')
 assert all((MODEL/k).stat().st_size==v['bytes'] and (MODEL/k).stat().st_mtime_ns==v['mtime_ns'] for k,v in model['files'].items())
 assert sha(O/'layer_sets.json')==read(O/'protocol.json')['layer_sets_sha256']
 assert sha(O/'sample_manifest.json')==read(O/'protocol.json')['manifest_sha256']
 assert sha(O/'ocr_labels.csv')==read(O/'classification_freeze.json')['labels_sha256']
@contextlib.contextmanager
def observe(model,prefill=False):
 counts=[0]*32;features=[];hooks=[];probe=dict(attention_output={},block_hidden={},attention_weights={});calls=[]
 def before(m,a,kw):
  if not calls:
   cache=kw.get('past_key_values');assert cache is None or cache.get_seq_length()==0
  assert kw.get('logits_to_keep')==1
  calls.append(1)
 def vision(m,a,v):features.append(dict(shape=list(v.shape),sha256=tsha(v)))
 try:
  hooks.append(model.register_forward_pre_hook(before,with_kwargs=True))
  hooks.append(model.multi_modal_projector.register_forward_hook(vision))
  for k,layer in enumerate(model_layers(model)):
   def mlp(m,a,v,k=k):counts[k]+=1
   hooks.append(layer.mlp.register_forward_hook(mlp))
   if prefill:
    def attention(m,a,out,k=k):
     if out[0].shape[1]>1:
      assert str(k) not in probe['attention_output'];probe['attention_output'][str(k)]=tsha(out[0])
      if out[1] is not None:probe['attention_weights'][str(k)]=tsha(out[1])
    def block(m,a,out,k=k):
     if out[0].shape[1]>1:probe['block_hidden'][str(k)]=tsha(out[0])
    hooks.append(layer.self_attn.register_forward_hook(attention));hooks.append(layer.register_forward_hook(block))
  yield counts,features,probe,calls
 finally:
  for h in hooks:h.remove()
def inputs_for(model,proc,r,s):
 inputs=prepare_inputs(model,proc,r['image_path'],r['question']+'\nAnswer the question using a single word or phrase.')[0]
 assert inputs['input_ids'][0].tolist()==s['prompt_ids'] and tensors_hash(inputs)==s['input_hashes']
 return inputs
def one(model,proc,inputs,s,c,versions,probe=False,**kw):
 footprint=hook_footprint(model);start=time.monotonic()
 with observe(model,probe) as (counts,features,prefill,calls):result=impl.generate(model,proc.tokenizer,inputs,c,**kw)
 g,first,trace,mass,cache=result
 assert hook_footprint(model)==footprint and counts==[g['length']]*32 and len(calls)==g['length']
 assert len(features)==1 and features[0]['shape']==[1,576,4096]
 assert tensors_hash(inputs)==s['input_hashes'] and versions=={n:p._version for n,p in model.named_parameters()}
 assert g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1
 assert not any(i==2 for i in g['token_ids'][:-1]) and (g['eos'] or g['truncated'])
 if c in NEW or c in ['A','F']:
  assert all(not r['blocked'] for r in mass if r['phase']=='Pminus1')
  for k in LAYER_SETS[c]['Decode']:
   rr=[r for r in mass if r['phase']=='Decode' and r['layer']==k]
   assert all(not r['blocked'] for r in rr)
   if rr:assert max(r['image_mass_mean'] for r in rr)>0,'Retained layer has no observed image access'
 return result,dict(MLP_calls=counts,image_features=features,hook_cleanup=True,input_hashes_unchanged=True,weights_unchanged=True,empty_cache_at_start=True,seconds=time.monotonic()-start,prefill_fingerprints=prefill if probe else None)
def checkpoint(r,s,c,result,audit,origin):
 g,first,_,mass,_=result;q=r['question_id'];assert firstsha(first)==old(q,'A')['first_logits_sha256']
 data=dict(UTC=utc(),question_id=q,condition=c,condition_name=LAYER_SETS[c]['name'],image_sha256=r['image_sha256'],input_hashes=s['input_hashes'],input_spec_sha256=sha(H/'input_specs'/(q+'.json')),execution_protocol_sha256=sha(O/'execution_protocol.json'),execution_source_manifest_sha256=sha(O/'execution_source_manifest.json'),runtime_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),allowed_layers=LAYER_SETS[c],generation=g,score=score(g['text'],r['answers']),first_logits_sha256=firstsha(first),Native_A_first_logits_sha256=old(q,'A')['first_logits_sha256'],first_logits_exact_Native_A=True,first_token=int(first.argmax()),attention_mass=mass,origin=origin,GPU_UUID=UUID,pid=os.getpid(),**audit)
 write(O/'checkpoints'/q/(c+'.json'),data)
def verify_cp(v,r,s,c):
 assert v['condition']==c and v['allowed_layers']==LAYER_SETS[c]
 assert v['execution_protocol_sha256']==sha(O/'execution_protocol.json') and v['execution_source_manifest_sha256']==sha(O/'execution_source_manifest.json')
 assert v['input_hashes']==s['input_hashes'] and v['image_sha256']==r['image_sha256'] and v['input_spec_sha256']==sha(H/'input_specs'/(r['question_id']+'.json'))
 assert v['first_logits_sha256']==old(r['question_id'],'A')['first_logits_sha256'] and v['generation']['audit']['passed']
 assert abs(v['score']-score(v['generation']['text'],r['answers']))<1e-12
def smoke(model,proc,rows,specs,versions):
 by={r['question_id']:r for r in rows};records=[];pending=[]
 for q in read(O/'smoke_selection.json')['question_ids']:
  print('SMOKE_BEGIN',q,flush=True);r=by[q];s=specs[q];inputs=inputs_for(model,proc,r,s);foot=hook_footprint(model)
  hf,rec=native(model,inputs)
  result,na=one(model,proc,inputs,s,'A',versions,probe=True,noop=True,capture=True,retain=True)
  ng,nf,nt,_,parent=result
  assert hf==ng['token_ids'] and len(rec.logits)==len(nt) and all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,nt)),'Native/No-op valid logits differ'
  assert all(x.get('logits_to_keep')==1 for x in rec.calls)
  fp=cache_fingerprint(parent);ptr=pointers(parent);cases=[]
  for c in ['A','F','B']+NEW:
   result,audit=one(model,proc,inputs,s,c,versions,probe=True,capture=True,retain=True);g,f,trace,mass,child=result
   assert not ptr&pointers(child) and cache_fingerprint(parent)==fp and audit['image_features']==na['image_features']
   if c in ['A','F']+NEW:
    assert torch.equal(torch.from_numpy(f),rec.logits[0][-1]),'Prefill first valid logits differ'
    assert audit['prefill_fingerprints']==na['prefill_fingerprints'],'Full Prefill attention/hidden differs'
   if c=='A':assert hf==g['token_ids'] and len(rec.logits)==len(trace) and all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,trace))
   if c in ['A','F','B']:
    v=old(q,c);assert g['token_ids']==v['generation']['token_ids'] and g['text']==v['generation']['text'] and firstsha(f)==v['first_logits_sha256']
   if c in NEW:pending.append((r,s,c,(g,f,None,mass,None),audit))
   cases.append(dict(condition=c,answer=g['text'],token_ids=g['token_ids'],eos=g['eos'],truncated=g['truncated'],audit=g['audit'],first_logits_sha256=firstsha(f),prefill_fingerprints=audit['prefill_fingerprints']));del result,child,trace,mass
  try:
   with impl.VisualReadingMask(model,inputs,'C'):raise RuntimeError('intentional hook cleanup probe')
  except RuntimeError as e:assert str(e)=='intentional hook cleanup probe'
  assert hook_footprint(model)==foot
  records.append(dict(question_id=q,Native_noop_all_valid_logits_torch_equal=True,A_native_all_valid_logits_torch_equal=True,new_first_valid_logits_torch_equal_Native=True,new_entire_Prefill_attention_and_hidden_hashes_exact=True,historical_A_F_B_tokens_and_first_logits_exact=True,independent_KV=True,parent_cache_unchanged=True,exception_cleanup=True,cases=cases))
  write(O/'gpu_smoke_validation.json',dict(passed=False,UTC=utc(),completed=len(records),records=records));print('SMOKE_CASE_PASS',q,flush=True)
  del parent,inputs,ng,nf,rec,nt;gc.collect();torch.cuda.empty_cache()
 write(O/'gpu_smoke_validation.json',dict(passed=True,UTC=utc(),GPU_UUID=UUID,n=len(records),records=records))
 for r,s,c,result,audit in pending:checkpoint(r,s,c,result,audit,'Fresh smoke generation retained after all three strict smoke gates passed')
 print('GPU_SMOKE_PASS',len(records),flush=True)
def stage(model,proc,versions):
 verify_sources();rows=read(O/'sample_manifest.json')['rows'];specs={r['question_id']:read(H/'input_specs'/(r['question_id']+'.json')) for r in rows}
 smoke(model,proc,rows,specs,versions)
 index=[];start=time.monotonic();generated=0
 for j,r in enumerate(rows):
  guard();q=r['question_id'];s=specs[q];assert sha(r['image_path'])==r['image_sha256'];inputs=inputs_for(model,proc,r,s);expected_features=old(q,'A')['image_features']
  for c in NEW:
   path=O/'checkpoints'/q/(c+'.json')
   if not path.exists():
    result,audit=one(model,proc,inputs,s,c,versions);assert audit['image_features']==expected_features
    checkpoint(r,s,c,result,audit,'Fresh formal generation');generated+=1;del result
   v=read(path);verify_cp(v,r,s,c);index.append(dict(question_id=q,condition=c,path=str(path.relative_to(O)),sha256=sha(path)))
  del inputs;gc.collect()
  if (j+1)%10==0:print('PROGRESS',j+1,300,'new_formal',generated,'seconds',round(time.monotonic()-start),flush=True)
 assert len(index)==1200 and len({(r['question_id'],r['condition']) for r in index})==1200
 csvwrite(O/'checkpoint_index.csv',index);verify_sources()
 write(O/'generation_validation.json',dict(passed=True,UTC=utc(),question_count=300,new_checkpoints=1200,reused_baselines=900,all_checkpoints_hashes_inputs_protocol_sources_verified=True,all_new_first_logits_Native_exact=True,all32_MLP_native=True,all_non_target_masks_exact=True,all_forbidden_probabilities_exact_zero=True,retained_layers_actual_image_access=True,independent_KV_and_prefix=True,all_hooks_clean=True,weights_unchanged=True))
 write(O/'generation_completion.json',dict(status='GENERATED_AND_VERIFIED',UTC=utc(),new_checkpoints=1200,new_formal=generated,smoke_reused=1200-generated,formal_seconds=time.monotonic()-start))
def main():
 P.mkdir(exist_ok=True,parents=True);lock=(P/'single_gpu_worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 guard();verify_sources();assert read(O/'execution_cpu_validation.json')['passed']
 state=hardware();write(O/'resource_start.json',state)
 line=next(x for x in state['gpus'].splitlines() if UUID in x);parts=[x.strip() for x in line.split(',')]
 assert 'RTX 3090' in line and int(parts[-3].split()[0])<300 and int(parts[-2].split()[0])>23000 and UUID not in state['processes'],'Selected GPU is no longer truly idle'
 start=time.monotonic();model=None;proc=None
 try:
  model,proc,env=load_model(str(MODEL));assert env['transformers']=='4.49.0' and env['model_revision']=='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
  write(O/'environment.json',dict(**env,pid=os.getpid(),GPU_UUID=UUID,physical_gpu=2,execution_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip()))
  versions={n:p._version for n,p in model.named_parameters()};mapping=[dict(index=k,module=next(n for n,m in model.named_modules() if m is b)) for k,b in enumerate(model_layers(model))];assert len(mapping)==32 and all(r['module'].endswith('.layers.'+str(r['index'])) for r in mapping);write(O/'layer_mapping.json',mapping)
  stage(model,proc,versions)
  write(O/'gpu_run_complete.json',dict(status='GENERATED_AND_VERIFIED',UTC=utc(),pid=os.getpid(),GPU_UUID=UUID,total_GPU_phase_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),memory=gpu_memory()))
 except BaseException:
  write(O/'STOP.json',dict(UTC=utc(),pid=os.getpid(),traceback=traceback.format_exc(),resources=hardware()));raise
 finally:
  del model,proc;gc.collect();torch.cuda.empty_cache();write(O/'resource_worker_final.json',hardware())
if __name__=='__main__':main()

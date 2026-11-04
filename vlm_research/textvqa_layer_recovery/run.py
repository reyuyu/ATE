import os
os.environ['CUDA_VISIBLE_DEVICES']='GPU-b08489af-0a8a-eb48-55ba-8346733af417'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from .common import *
from .engine import generate,VisualReadingMask
import torch,numpy as np,time,traceback,gc,contextlib
from vlm_common import load_model,prepare_inputs,gpu_memory
from knockout_probe import model_layers
from benchmarks.smoke import hook_footprint
from paper_rq1.engine import cache_fingerprint
from query_role_smoke_fix.run import native
import gqa3000_token_visual_writing.engine as legacy
legacy.assert_no_stop=guard
torch.set_num_threads(1)

def pointers(cache):return {x.data_ptr() for group in [cache.key_cache,cache.value_cache] for x in group}
@contextlib.contextmanager
def observations(model):
 counts=[0]*32;features=[];handles=[]
 def vision(m,a,o):features.append(dict(shape=list(o.shape),sha256=hashlib.sha256(o.detach().cpu().contiguous().numpy().tobytes()).hexdigest()))
 try:
  for k,l in enumerate(model_layers(model)):
   def mlp(m,a,o,k=k):counts[k]+=1
   handles.append(l.mlp.register_forward_hook(mlp))
  handles.append(model.multi_modal_projector.register_forward_hook(vision));yield counts,features
 finally:
  for h in handles:h.remove()
def inputs_for(model,proc,r,s):
 inputs=prepare_inputs(model,proc,r['image_path'],r['question']+'\nAnswer the question using a single word or phrase.')[0]
 assert inputs['input_ids'][0].tolist()==s['prompt_ids'] and tensors_hash(inputs)==s['input_hashes']
 return inputs
def one(model,proc,inputs,s,condition,weights,**kw):
 before=hook_footprint(model);start=time.monotonic()
 with observations(model) as (counts,features):result=generate(model,proc.tokenizer,inputs,condition,**kw)
 g,first,trace,mass,cache=result
 assert hook_footprint(model)==before and counts==[g['length']]*32
 assert len(features)==1 and features[0]['shape']==[1,576,4096]
 assert tensors_hash(inputs)==s['input_hashes'] and weights=={n:p._version for n,p in model.named_parameters()}
 return result,dict(MLP_calls=counts,image_features=features,hook_cleanup=True,input_hashes_unchanged=True,weights_unchanged=True,seconds=time.monotonic()-start)
def checkpoint(r,s,c,result,audit,origin):
 g,first,_,mass,_=result
 data=dict(UTC=utc(),question_id=r['question_id'],condition=c,condition_name=NAMES[c],image_sha256=r['image_sha256'],input_hashes=s['input_hashes'],input_spec_sha256=sha(OUT/'input_specs'/(r['question_id']+'.json')),source_manifest_sha256=sha(OUT/'source_manifest.json'),protocol_sha256=sha(OUT/'protocol.json'),fixed_layers=LAYERS,generation=g,score=score(g['text'],r['answers']),first_token=int(first.argmax()),first_logits_sha256=hashlib.sha256(first.tobytes()).hexdigest(),attention_mass=mass,origin=origin,**audit)
 write(OUT/'checkpoints'/r['question_id']/(c+'.json'),data)
def smoke(model,proc,rows,specs,weights):
 index={r['question_id']:r for r in rows};records=[];pending=[]
 for q in read(OUT/'smoke_selection.json')['question_ids']:
  r=index[q];s=specs[q];inputs=inputs_for(model,proc,r,s);foot=hook_footprint(model)
  with observations(model) as (counts,features):hf,rec=native(model,inputs)
  (ng,nf,nt,_,parent),na=one(model,proc,inputs,s,'B',weights,noop=True,capture=True,retain=True,heads=True)
  assert hf==ng['token_ids'] and len(rec.logits)==len(nt)
  assert all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,nt)),'Native/noop all valid logits not torch.equal'
  assert all(x.get('logits_to_keep')==1 for x in rec.calls) and counts==[len(hf)]*32 and features==na['image_features']
  fp=cache_fingerprint(parent);ptr=pointers(parent);cases=[]
  for c in CONDITIONS:
   result,audit=one(model,proc,inputs,s,c,weights,capture=True,retain=True,heads=True);g,f,trace,mass,child=result
   assert not ptr&pointers(child) and cache_fingerprint(parent)==fp and audit['image_features']==features
   if c=='A':assert g['token_ids']==hf and all(torch.equal(x[-1],torch.from_numpy(y)) for x,y in zip(rec.logits,trace)) and len(trace)==len(rec.logits)
   if c=='B':
    old=baseline(q);assert g['token_ids']==old['generation']['token_ids'] and g['text']==old['generation']['text'] and g['eos']==old['generation']['eos']
    assert hashlib.sha256(f.tobytes()).hexdigest()==old['first_logits_sha256']
    with observations(model) as (oldcounts,oldfeatures):og,of,ot,_,oc=legacy.generate(model,proc.tokenizer,inputs,s['R'],capture=True,retain_cache=True)
    assert og['token_ids']==g['token_ids'] and np.array_equal(of,f) and torch.equal(torch.from_numpy(ot),torch.from_numpy(trace))
    assert oldfeatures==features and oldcounts==[g['length']]*32 and not pointers(child)&pointers(oc)
    del oc,og,of,ot
   else:pending.append((r,s,c,(g,f,None,mass,None),audit))
   cases.append(dict(condition=c,answer=g['text'],token_ids=g['token_ids'],audit=g['audit'],eos=g['eos'],truncated=g['truncated'],**audit));del result,child,trace,mass
  try:
   with VisualReadingMask(model,inputs,'E'):raise RuntimeError('deliberate hook cleanup probe')
  except RuntimeError as e:assert str(e)=='deliberate hook cleanup probe'
  assert hook_footprint(model)==foot
  records.append(dict(question_id=q,Native_noop_all_valid_logits_torch_equal=True,A_native_all_valid_logits_torch_equal=True,B_historical_ids_and_first_logits_sha_exact=True,B_frozen_engine_all_valid_logits_torch_equal=True,independent_caches=True,parent_cache_unchanged=True,exception_hook_cleanup=True,conditions=cases))
  write(OUT/'GPU_smoke_validation.json',dict(passed=False,completed=len(records),planned=len(read(OUT/'smoke_selection.json')['question_ids']),records=records))
  print('SMOKE_CASE_PASS',q,len(records),flush=True);del parent,rec,nt,inputs;gc.collect();torch.cuda.empty_cache()
 write(OUT/'GPU_smoke_validation.json',dict(passed=True,UTC=utc(),n=len(records),records=records))
 for r,s,c,result,audit in pending:checkpoint(r,s,c,result,audit,'Fresh smoke generation reused after all smoke gates passed')
 print('GPU_SMOKE_PASS',len(records),flush=True)
def verify_cp(c,r,s,condition):
 assert c['condition']==condition and c['source_manifest_sha256']==sha(OUT/'source_manifest.json') and c['protocol_sha256']==sha(OUT/'protocol.json')
 assert c['input_hashes']==s['input_hashes'] and c['image_sha256']==r['image_sha256'] and c['input_spec_sha256']==sha(OUT/'input_specs'/(r['question_id']+'.json'))
 assert c['generation']['audit']['passed'] and abs(c['score']-score(c['generation']['text'],r['answers']))<1e-12
def stage(model,proc,weights):
 m=read(OUT/'sample_manifest.json');cpu=read(OUT/'CPU_validation.json');assert cpu['passed'] and sha(OUT/'sample_manifest.json')==cpu['manifest_sha256']
 assert sources()==read(OUT/'source_manifest.json')['files'] and sha(OUT/'protocol.json')==read(OUT/'source_manifest.json')['protocol_sha256']
 rows=m['rows'];specs={r['question_id']:read(OUT/'input_specs'/(r['question_id']+'.json')) for r in rows}
 assert all(sha(OUT/'input_specs'/(q+'.json'))==h for q,h in cpu['input_specs_sha256'].items())
 if not (OUT/'GPU_smoke_validation.json').exists() or not read(OUT/'GPU_smoke_validation.json')['passed']:smoke(model,proc,rows,specs,weights)
 else:assert all((OUT/'checkpoints'/q/(c+'.json')).exists() for q in read(OUT/'smoke_selection.json')['question_ids'] for c in ['A','C','D','E'])
 generated=0;start=time.monotonic();index=[]
 for j,r in enumerate(rows):
  guard();q=r['question_id'];s=specs[q];assert sha(r['image_path'])==r['image_sha256'];inputs=inputs_for(model,proc,r,s)
  for c in ['A','C','D','E']:
   p=OUT/'checkpoints'/q/(c+'.json')
   if not p.exists():
    result,audit=one(model,proc,inputs,s,c,weights);checkpoint(r,s,c,result,audit,'Fresh formal generation');generated+=1;del result
   cp=read(p);verify_cp(cp,r,s,c);index.append(dict(question_id=q,condition=c,path=str(p.relative_to(OUT)),sha256=sha(p)))
  del inputs;gc.collect()
  if (j+1)%10==0:print('PROGRESS',j+1,300,'new_formal',generated,'elapsed_sec',round(time.monotonic()-start),flush=True)
 assert len(index)==1200
 csvwrite(OUT/'checkpoint_index.csv',index)
 write(OUT/'generation_validation.json',dict(passed=True,UTC=utc(),n=300,new_checkpoint_count=1200,reused_B_count=300,logical_records=1500,all_hashes_inputs_sources_verified=True,all32_MLP=True,weights_unchanged=True,hooks_clean=True,independent_real_caches=True,forbidden_probabilities_exact_zero=True,non_target_masks_exact=True))
 write(OUT/'generation_completion.json',dict(status='GENERATED_AND_VERIFIED',UTC=utc(),fresh_checkpoints=1200,reused_B=300,logical_records=1500,new_in_this_formal_session=generated,formal_seconds=time.monotonic()-start))
def main():
 guard();state=hardware();write(OUT/'resource_start.json',state);line=next(x for x in state['gpus'].splitlines() if UUID in x);parts=[x.strip() for x in line.split(',')]
 assert int(parts[-3].split()[0])<300 and int(parts[-2].split()[0])>23000 and UUID not in state['processes']
 assert read(OUT/'CPU_validation.json')['passed'] and sources()==read(OUT/'source_manifest.json')['files']
 start=time.monotonic();model=None
 try:
  model,proc,env=load_model(str(MODEL));write(OUT/'environment.json',env);weights={n:p._version for n,p in model.named_parameters()}
  names={id(m):n for n,m in model.named_modules()};mapping=[dict(index=k,module=names[id(b)]) for k,b in enumerate(model_layers(model))];assert len(mapping)==32
  assert all(x['module'].endswith('.layers.'+str(x['index'])) for x in mapping);write(OUT/'layer_mapping.json',mapping)
  stage(model,proc,weights)
  write(OUT/'gpu_run_complete.json',dict(status='GENERATED_AND_VERIFIED',UTC=utc(),GPU_UUID=UUID,pid=os.getpid(),total_GPU_phase_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),memory=gpu_memory()))
 except BaseException:
  write(OUT/'STOP.json',dict(UTC=utc(),pid=os.getpid(),traceback=traceback.format_exc(),resources=hardware()));raise
 finally:
  if model is not None:del model
  gc.collect();torch.cuda.empty_cache();write(OUT/'resource_worker_final.json',hardware())
if __name__=='__main__':main()

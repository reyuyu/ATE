import sys,gc,os,time,traceback,json
from run_common import *
sys.path[:0]=[str(L/'sources/llava'),'/home/featurize/work/ATE']
import torch
from vlm_research.information_horizon_alignment import engine as f
from vlm_research.mme_seed_quality import engine as loader
O=L/'results/llava';P=json.loads((O/'protocol.json').read_text());binding=sha(O/'protocol.json');start=time.time()
rows=json.loads((O/'rows.json').read_text());by={r['question_id']:r for r in rows};cpu=json.loads((O/'cpu_bindings.json').read_text());reuse=json.loads((O/'verified_reuse.json').read_text());current=None;adapter=None
records={m:load_records(O/'predictions'/m) for m in P['conditions']}
def progress(status,**kw):atomic(O/'progress.json',dict(status=status,complete={m:len(records[m])+len(reuse[m]) for m in records},expected_each=1000,new={m:len(records[m]) for m in records},elapsed_s=time.time()-start,**kw))
try:
 for p,h in P['source_SHA256'].items():assert sha(L/'sources/llava'/p)==h,p
 gpu=gpu_idle();progress('LOADING_MODEL',gpu=gpu);loader.MODELS['llava']=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf');adapter=f.load('B')
 assert all(p.dtype==torch.float16 and p.device.type=='cuda' for p in adapter.model.parameters())
 assert adapter.model.config.text_config._attn_implementation=='eager'
 atomic(O/'environment.json',dict(gpu=gpu,torch=torch.__version__,transformers=__import__('transformers').__version__,model_loads=1,all_gpu=True,fp16=True,eager=True))
 def infer(q,m,audit):
  global current
  current=dict(question_id=q,method=m)
  if q in records[m]:return records[m][q]
  assert q not in reuse[m],'Never regenerate verified completed request'
  s=by[q];layout,b=f.prepare(adapter,s);x=cpu[q]
  assert adapter.inputs['input_ids'][0].tolist()==x['input_ids']
  hashes={k:__import__('hashlib').sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in adapter.inputs.items()};assert hashes==x['input_hashes']
  torch.cuda.reset_peak_memory_stats();t=time.time()
  g,_=f.hf_generate(adapter,layout,'HF-Native' if m=='Native' else 'Physical-C3',128,audit=audit)
  # Large hidden/rotary observation tensors are not part of persistent output.
  for k in list(g):
   if k.startswith('_'):del g[k]
  assert g['independent_cache'] and g['all_effective_logits_finite']
  r=dict(model='LLaVA-1.5-7B',method=m,question_id=q,image_sha256=s['image_sha256'],input_hashes=hashes,protocol_sha256=binding,generation=g,**score(q,g['text']),origin='NEW_GPU_MISSING_ONLY',peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,elapsed_s=time.time()-t)
  atomic(O/'predictions'/m/(q+'.json'),r);records[m][q]=r;adapter.inputs={};progress('SMOKE' if audit else 'RUNNING',last=current)
  if audit or len(records[m])%50==0:print('LLAVA_PROGRESS',m,len(records[m])+len(reuse[m]),'/1000',q,flush=True)
  return r
 for m in P['conditions']:
  for q in P['smoke_ids']:infer(q,m,True)
 atomic(O/'smoke.json',dict(status='PASS',ids=P['smoke_ids'],conditions=P['conditions'],same_input_hashes=True,frozen_core_unmodified=True,reused_as_formal=True,no_completed_regeneration=True))
 for s in rows:
  for m in P['conditions']:
   if s['question_id'] not in reuse[m]:infer(s['question_id'],m,False)
 assert all(len(records[m])+len(reuse[m])==1000 for m in records);progress('COMPLETE_PENDING_INDEPENDENT_AUDIT')
except BaseException as e:
 atomic(O/'failure.json',dict(status='BLOCKED',current=current,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));progress('BLOCKED',error=str(e));traceback.print_exc();raise
finally:
 adapter=None;gc.collect();torch.cuda.empty_cache();print('LLAVA_GPU_RELEASE_ON_PROCESS_EXIT',flush=True)

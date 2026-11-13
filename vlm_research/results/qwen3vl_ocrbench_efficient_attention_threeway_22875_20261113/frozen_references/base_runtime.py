"""Shared new-backend request wrapper. Each request creates an independent cache."""
import sys,time,uuid,json
from pathlib import Path
from contextlib import nullcontext
import torch
from attention_backend import Backend,StageMonitor

def cache_audit(cache):
 if hasattr(cache,'audit'):return cache.audit()
 rows=[]
 for i,layer in enumerate(cache.layers):
  pair=[layer.keys,layer.values];rows.append(dict(layer=i,live_tokens=pair[0].shape[-2],capacity_tokens=pair[0].shape[-2],live_bytes=sum(t.numel()*t.element_size() for t in pair),capacity_bytes=sum(t.untyped_storage().nbytes() for t in pair),key_shape=list(pair[0].shape)))
 return dict(layers=rows,live_bytes=sum(x['live_bytes'] for x in rows),capacity_bytes=sum(x['capacity_bytes'] for x in rows))

def generate(model,processor,inputs,layout,condition='Native',profile=False,capture=False,execution=None,forced=None):
 assert condition in ['Native','Physical-Original','Dense-F'] and (condition=='Native')==(execution is None)
 device=next(model.parameters()).device;x={k:v.to(device) for k,v in inputs.items()};backend=Backend(model);backend.detail=profile
 monitor=StageMonitor(model,backend,profile);past=None;ids=[];scores=[];step_ms=[];first=None;initial_cache_none=past is None;cache_uuid=str(uuid.uuid4())
 eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
 assert initial_cache_none
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
 if execution is not None:backend.check=execution.check_attention
 try:
  with torch.inference_mode(),backend.context(),(execution.context() if execution is not None else nullcontext()),monitor.context():
   for t in range(128 if forced is None else len(forced)):
    backend.step=t
    kw=dict(x) if t==0 else dict(input_ids=token.reshape(1,1),attention_mask=torch.ones((1,layout['input_length']+t),device=device,dtype=torch.long))
    kw.update(use_cache=True,past_key_values=past,cache_position=torch.arange(layout['input_length'],device=device) if t==0 else torch.tensor([layout['input_length']+t-1],device=device),logits_to_keep=1)
    t0=time.perf_counter();out=model(**kw);torch.cuda.synchronize();step_ms.append((time.perf_counter()-t0)*1000)
    assert out.logits.shape[:2]==(1,1);vector=out.logits[0,-1];assert bool(torch.isfinite(vector).all()),'NONFINITE_LOGITS'
    if capture:scores.append(vector.detach().float().cpu().clone())
    greedy=vector.argmax();token=greedy if forced is None else torch.tensor(forced[t],device=device);ids.append(int(token));past=out.past_key_values
    if t==0:first=(time.perf_counter()-begin)*1000
    if forced is None and ids[-1] in eos:break
  torch.cuda.synchronize();memory=monitor.finish();kv=cache_audit(past);text=processor.tokenizer.decode(ids,skip_special_tokens=True)
  result=dict(condition=condition,text=text,token_ids=ids,eos=ids[-1] in eos,empty=not text.strip(),truncated=ids[-1] not in eos if forced is None else None,generated_tokens=len(ids),max_new_tokens=128,natural_eos=forced is None,teacher_forced_reference=forced is not None,finite_logits=True,input_length=layout['input_length'],image_tokens=layout['image_tokens'],backend='SDPA_EFFICIENT_ATTENTION_ONLY',independent_cache=True,initial_cache_none=initial_cache_none,request_cache_uuid=cache_uuid,kv=kv,memory=memory,timing=dict(e2e_ms=(time.perf_counter()-begin)*1000,first_token_ms=first,step_forward_ms=step_ms),attention_calls=backend.events)
  if execution is not None:result['physical_audit']=execution.audit()
  assert len(kv['layers'])==36
  if condition=='Native':assert all(r['live_tokens']==layout['input_length']+len(ids)-1 for r in kv['layers'])
  return result,scores
 except BaseException as error:
  error.request_partial=dict(condition=condition,token_ids=ids,last_attention=backend.last,stage_memory=monitor.rows,kernel_evidence=monitor.kernels)
  raise
 finally:
  past=None;x=None


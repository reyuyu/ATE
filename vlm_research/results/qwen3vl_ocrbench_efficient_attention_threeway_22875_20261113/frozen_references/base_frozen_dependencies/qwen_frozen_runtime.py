"""Isolated request wrapper for frozen Qwen C3 Execution.

Adaptations from qwen_quality_speed.generate: remove unused Repaired imports;
parameterize generation cap and fixed length (16/32). Transformer math is unchanged.
"""
import time
from unittest.mock import patch
import torch
from vlm_research.qwen3vl_c3v2_physical.engine import Execution

def generate(model,processor,inputs,layout,condition,audit=False,capture=False,
             noop=False,max_tokens=32,fixed_length=False,check_finite=True):
 device=next(model.parameters()).device
 kind={'Native':'N','Dense-F':'F','Physical-Original':'C'}[condition]
 assert max_tokens>0 and (not fixed_length or max_tokens in [16,32])
 scores=[];ids=[];events=[];vision=[];first_wall=None;cache=None
 eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
 original=model.model.get_image_features
 def image_features(*a,**kw):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  start.record();v=original(*a,**kw);end.record();vision.append((start,end));return v
 torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
 x={k:v.to(device) for k,v in inputs.items()}
 e=Execution(model,layout,kind,max_tokens=max_tokens,audit=audit,noop=noop)
 with torch.inference_mode(),e.context(),patch.object(model.model,'get_image_features',image_features):
  for t in range(max_tokens):
   kw=dict(x) if t==0 else dict(input_ids=token.reshape(1,1),attention_mask=torch.ones((1,layout['input_length']+t),device=device,dtype=torch.long))
   kw.update(use_cache=True,past_key_values=cache,cache_position=torch.arange(layout['input_length'],device=device) if t==0 else torch.tensor([layout['input_length']+t-1],device=device),logits_to_keep=1)
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();out=model(**kw);end.record();events.append((start,end))
   assert out.logits.shape[1]==1
   vector=out.logits[0,-1]
   if check_finite:assert bool(torch.isfinite(vector).all()),'NaN or Inf logits'
   token=vector.argmax();tid=int(token);ids.append(tid);cache=out.past_key_values
   if t==0:first_wall=(time.perf_counter()-begin)*1000
   if capture:scores.append(vector.detach().float().cpu())
   if tid in eos and not fixed_length:break
 torch.cuda.synchronize();total=(time.perf_counter()-begin)*1000
 peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
 if kind=='N':e.cache=cache
 a=e.audit();tm=[x.elapsed_time(y) for x,y in events];vm=sum(x.elapsed_time(y) for x,y in vision)
 text=processor.tokenizer.decode(ids,skip_special_tokens=True)
 result=dict(condition=condition,token_ids=ids,text=text,eos=ids[-1] in eos,first_eos_index=next((i for i,v in enumerate(ids) if v in eos),None),truncated=not fixed_length and ids[-1] not in eos,empty=not text.strip(),generated_tokens=len(ids),incremental_decode_steps=len(ids)-1,fixed_length=fixed_length,max_new_tokens=max_tokens,finite_checked=check_finite,first_token=ids[0],timing=dict(vision_ms=vm,prefill_language_ms=tm[0]-vm,ttft_gpu_ms=tm[0],ttft_wall_ms=first_wall,decode_gpu_ms=sum(tm[1:]),decode_gpu_ms_per_token=sum(tm[1:])/max(1,len(ids)-1),e2e_wall_ms=total),memory=dict(baseline_allocated=baseline,peak_allocated=peak,incremental_peak_allocated=peak-baseline,peak_reserved=reserved),kv=a['kv'])
 if audit:result['audit']=a
 return result,scores

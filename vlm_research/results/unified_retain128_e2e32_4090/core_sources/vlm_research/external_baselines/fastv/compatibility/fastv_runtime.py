"""Single-request greedy cached inference and measured live CUDA KV tensors."""
from contextlib import nullcontext
from unittest.mock import patch
import time,torch
from fastv_hf_adapter import FastV,cache_snapshot

@torch.inference_mode()
def generate(model,processor,inputs,source,method,length=128,fixed=False,trace=False,capture=False):
 assert method in ('Native','FastV','FastV-Off')
 ids=inputs['input_ids'];n=ids.shape[-1]
 image=tuple(torch.where(ids[0]==32000)[0].tolist());assert len(image)==576
 ctrl=FastV(model,source,image,n,enabled=method=='FastV',trace=trace) if method!='Native' else None
 positions=torch.arange(n+length-1,device=ids.device)
 masks=torch.ones((1,n+length-1),dtype=torch.long,device=ids.device)
 cache=None;tokens=[];finite=[];logits=[];vision_calls=[0];prefill_kv=None
 original=model.get_image_features
 def vision(*args,**kw):
  vision_calls[0]+=1;return original(*args,**kw)
 try:
  with ctrl.context() if ctrl else nullcontext(),patch.object(model,'get_image_features',vision):
   torch.cuda.synchronize();start=time.perf_counter()
   for step in range(length):
    kw=dict(inputs) if step==0 else dict(input_ids=tokens[-1].reshape(1,1),attention_mask=masks[:,:n+step])
    kw.update(use_cache=True,return_dict=True,past_key_values=cache,logits_to_keep=1,
              cache_position=positions[:n] if step==0 else positions[n+step-1:n+step])
    out=model(**kw);last=out.logits[:,-1];token=last.argmax(-1);tokens.append(token)
    finite.append(torch.isfinite(last).all());cache=out.past_key_values
    if step==0:
     torch.cuda.synchronize();ttft=(time.perf_counter()-start)*1000
     if trace:prefill_kv=cache_snapshot(cache,ctrl,n,1,'FastV' if method=='FastV' else 'Native')
    if capture:logits.append(last.detach().float().cpu())
    del out,last
    if not fixed and int(token)==processor.tokenizer.eos_token_id:break
   torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
  ids_out=torch.cat(tokens).tolist();assert torch.stack(finite).all().item(),'Nonfinite logits'
  assert vision_calls[0]==1
  if fixed:assert len(ids_out)==length
  kv=cache_snapshot(cache,ctrl,n,len(ids_out),'FastV' if method=='FastV' else 'Native')
  text=processor.tokenizer.decode(ids_out,skip_special_tokens=True).strip()
  record={'condition':method,'token_ids':ids_out,'text':text,'output_tokens':len(ids_out),'decode_steps':len(ids_out)-1,
          'eos':ids_out[-1]==processor.tokenizer.eos_token_id,'truncated':not fixed and len(ids_out)==length and ids_out[-1]!=processor.tokenizer.eos_token_id,
          'fixed_output_length':fixed,'all_logits_finite':True,'vision_forward_calls':vision_calls[0],
          'ttft_ms':ttft,'E2E_ms':elapsed,'TPOT_wall_ms':(elapsed-ttft)/(len(ids_out)-1) if len(ids_out)>1 else None,
          'cache':kv,'input_tokens':n,'image_start':image[0],'prefill_cache':prefill_kv}
  if ctrl:
   record['pruning_events']=ctrl.events
   if ctrl.enabled:
    record['kept_prefill_original_positions']=ctrl.active.tolist()
    record['selected_image_positions']=[p for p in record['kept_prefill_original_positions'] if p in image]
    assert len(record['selected_image_positions'])==192
   if trace:record['layer_trace']=ctrl.rows
  return record,logits
 finally:
  del cache,ctrl

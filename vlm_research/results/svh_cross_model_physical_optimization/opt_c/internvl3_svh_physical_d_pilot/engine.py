"""Physical InternVL/Qwen2 SVH-D. Official weights and attention kernels remain untouched.
The compact path changes query rows/cache storage, never logical RoPE positions.
Audit instrumentation is opt-in and removed before timed measurements.
"""
from contextlib import ExitStack
from types import MethodType
import time
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as frozen

SKIP=set(range(2,6))
def text_positions(roles,device='cpu'):
 image=set(roles['image'])
 return torch.tensor([i for i in range(roles['n']) if i not in image],device=device,dtype=torch.long)

class CompactCache(DynamicCache):
 def __init__(self,roles,device='cuda:0'):
  super().__init__();self.roles=roles
  self.key_cache=[None]*28;self.value_cache=[None]*28
  self.keep=text_positions(roles,device)
  self.prefill=True;self.buffers=[None]*28;self.used=[0]*28;self.capacities=[0]*28
 def update(self,k,v,layer_idx,cache_kwargs=None):
  i=layer_idx
  if self.prefill:
   if i not in SKIP:
    # index_select allocates compact storage, never a view of dense image KV.
    stored=tuple(x.index_select(-2,self.keep) if i<20 else x for x in (k,v))
    self.allocate_update(k,v,i,stored)
   return k,v  # transient full prefill K/V remains available to R consumers
  assert i not in SKIP and self.key_cache[i] is not None
  self.key_cache[i]=torch.cat((self.key_cache[i],k),dim=-2)
  self.value_cache[i]=torch.cat((self.value_cache[i],v),dim=-2)
  return self.key_cache[i],self.value_cache[i]
 def get_seq_length(self,layer_idx=0):
  x=self.key_cache[layer_idx]
  return 0 if x is None else x.shape[-2]

 def allocate_update(self,k,v,i,stored):
     n=stored[0].shape[-2]
     if self.buffers[i] is None:
         cap=n+31
         pair=[x.new_empty((*x.shape[:-2],cap,x.shape[-1])) for x in stored]
         for out,x in zip(pair,stored):out[...,:n,:].copy_(x)
         self.buffers[i]=pair;self.used[i]=n;self.capacities[i]=cap
     else:
         n=self.used[i];m=k.shape[-2];assert n+m<=self.capacities[i]
         for out,x in zip(self.buffers[i],(k,v)):out[...,n:n+m,:].copy_(x)
         self.used[i]+=m
     n=self.used[i]
     self.key_cache[i]=self.buffers[i][0][...,:n,:]
     self.value_cache[i]=self.buffers[i][1][...,:n,:]
     return self.key_cache[i],self.value_cache[i]

def layout(cache,method,roles,outputs):
 rows=[];seen=set();total_capacity=0;live=0
 for i in range(28):
  k=cache.key_cache[i];v=cache.value_cache[i]
  length=0 if k is None else k.shape[-2]
  allocated=0
  for x in (k,v):
   if x is None:continue
   live+=x.numel()*x.element_size()
   storage=x.untyped_storage();identity=(storage.data_ptr(),storage.nbytes())
   if identity not in seen:seen.add(identity);allocated+=storage.nbytes()
  total_capacity+=allocated
  rows.append(dict(method=method,layer=i,logical_length=roles['n']+outputs-1,
   physical_length=length,visual_kv=0 if method=='Physical-D' else 256,
   k_shape=[] if k is None else list(k.shape),v_shape=[] if v is None else list(v.shape),
   dtype=None if k is None else str(k.dtype),live_bytes=0 if k is None else (k.numel()+v.numel())*k.element_size(),
   capacity_bytes=allocated,skip_attention=i in SKIP and method=='Physical-D'))
 return rows,live/1048576,total_capacity/1048576

class Control:
 def __init__(self,model,method,roles,cache,audit=False):
  assert method in ('Native','No-op','Dense-D','Physical-D')
  self.model=model;self.lm=model.language_model;self.layers=self.lm.model.layers
  self.method=method;self.roles=roles;self.cache=cache;self.audit=audit
  self.step=0;self.prefill=True;self.masks={};self.hidden={};self.events=[]
  self.norms=[];self.mlp=[];self.projections=[];self.probability_max=0.;self.zero_events=[]
  self.positions=[];self.skips=[];self.exits=[];self.finite=True;self.attention_calls=[]
  n=roles["n"]; ref=next(model.parameters()); self.decode_full=ref.new_zeros((1,1,1,n+31)); self.decode_text=ref.new_zeros((1,1,1,n-256+31)); self.decode_blocked=self.decode_full.clone(); self.decode_blocked[...,roles["image"]]=-torch.inf
 def capture(self,i,x,positions):
  if self.audit:
   assert torch.isfinite(x).all()
   self.hidden[(self.step,i)]=(positions.detach().cpu().clone(),x.detach().cpu().clone())
 def __enter__(self):
  self.stack=ExitStack()
  if self.method in ('Dense-D','No-op'):
   for i,l in enumerate(self.layers):
    def before(m,args,kw,i=i):
     if self.method=='No-op':return
     off=self.prefill and i>=20
     if off not in self.masks:self.masks[off]=frozen.edit_mask(kw['attention_mask'],self.roles,self.prefill,off)
     return args,dict(kw,attention_mask=self.masks[off])
    self.stack.callback(l.self_attn.register_forward_pre_hook(before,with_kwargs=True).remove)
    if i in SKIP and self.method=='Dense-D':
     def zero(m,args,out,i=i):
      y=out.clone()
      if self.prefill:y[:,-1:,:]=0
      else:y.zero_()
      if self.audit:
       assert torch.count_nonzero(y[:,-1:,:])==0
       if self.prefill:assert torch.equal(y[:,:-1,:],out[:,:-1,:])
       self.zero_events.append((self.step,i))
      return y
     self.stack.callback(l.self_attn.o_proj.register_forward_hook(zero).remove)
  if self.method=='Physical-D':
   original=self.lm.model.forward
   self.lm.model.forward=MethodType(self.physical_forward,self.lm.model)
   self.stack.callback(setattr,self.lm.model,'forward',original)
  if self.audit:
   for i,l in enumerate(self.layers):
    if self.method!='Physical-D':
     def layer_out(m,args,out,i=i):
      pos=torch.arange(self.roles['n'],device=out[0].device) if self.prefill else torch.tensor([self.roles['n']+self.step-1],device=out[0].device)
      self.capture(i,out[0],pos)
     self.stack.callback(l.register_forward_hook(layer_out).remove)
    for name,module in [('input_norm',l.input_layernorm),('post_norm',l.post_attention_layernorm),('mlp',l.mlp)]:
     def count(m,args,out,i=i,name=name):
      assert torch.isfinite(out).all()
      (self.mlp if name=='mlp' else self.norms).append((self.step,i,name,int(args[0].shape[1])))
     self.stack.callback(module.register_forward_hook(count).remove)
    for name in ('q_proj','k_proj','v_proj','o_proj'):
     def projection(m,args,out,i=i,name=name):
      self.projections.append((self.step,i,name,int(args[0].shape[1])))
      assert torch.isfinite(out).all()
     self.stack.callback(getattr(l.self_attn,name).register_forward_hook(projection).remove)
    def attn_pre(m,args,kw,i=i):
     pos=kw['position_ids'].detach().cpu().tolist()[0]
     self.positions.append(dict(step=self.step,layer=i,query_positions=pos,
      q_rows=int(kw['hidden_states'].shape[1]),mask_shape=list(kw['attention_mask'].shape)))
     self.attention_calls.append((self.step,i))
    def attn_out(m,args,out,i=i):
     weights=out[1];assert torch.isfinite(weights).all()
     if self.method in ('Native','No-op'):return
     # Physical decode/cache and deep prefill contain no image columns.
     if self.method=='Physical-D' and (not self.prefill or i>=20):return
     lo,hi=self.roles['image'][0],self.roles['image'][-1]+1
     if self.prefill:
      if not(self.method=='Physical-D' and i in SKIP):
       assert float(weights[:,:,-1:,lo:hi].abs().max())==0
      if i>=20:assert float(weights[:,:,hi:-1,lo:hi].abs().max())==0
     else:assert float(weights[:,:,:,lo:hi].abs().max())==0
    self.stack.callback(l.self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True).remove)
    self.stack.callback(l.self_attn.register_forward_hook(attn_out).remove)
  return self
 def __exit__(self,*a):self.stack.close()
 def physical_forward(self,decoder,input_ids=None,attention_mask=None,position_ids=None,past_key_values=None,
  inputs_embeds=None,use_cache=True,output_attentions=False,output_hidden_states=False,
  return_dict=True,cache_position=None,**kw):
  cache=past_key_values;assert cache is self.cache and use_cache
  h=inputs_embeds if inputs_embeds is not None else decoder.embed_tokens(input_ids)
  pos=cache_position
  if position_ids is None:position_ids=pos.unsqueeze(0)
  mask=decoder._update_causal_mask(attention_mask,h,pos,cache,False) if self.prefill else self.decode_full[...,:self.roles["n"]+self.step]
  rope=decoder.rotary_emb(h,position_ids)
  text=cache.keep
  full_mask=frozen.edit_mask(mask,self.roles,True,False) if self.prefill else None
  deep_mask=mask.index_select(-1,text).index_select(-2,text) if self.prefill else None
  compact_decode=self.decode_text[...,:self.roles['n']-256+self.step] if not self.prefill else None
  for i,layer in enumerate(decoder.layers):
   if i==20 and self.prefill:
    before=h.shape[1];h=h.index_select(1,text);pos=text
    rope=tuple(x.index_select(1,text) for x in rope)
    if self.audit:self.exits.append(dict(before=before,after=h.shape[1],removed=before-h.shape[1]))
   m=(full_mask if i<20 else deep_mask) if self.prefill else compact_decode
   r=h;norm=layer.input_layernorm(h)
   if i in SKIP and not self.prefill:
    att=torch.zeros_like(norm)
    if self.audit:self.skips.append((self.step,i,'all attention/QKV/QK/AV/o_proj absent'))
   elif i in SKIP:
    # Original P-1 is the last causal row; no earlier query consumes its K/V.
    att,_=layer.self_attn(hidden_states=norm[:,:-1,:],
     attention_mask=m[:,:,:-1,:-1],position_ids=pos[:-1].unsqueeze(0),
     past_key_value=cache,use_cache=True,cache_position=pos[:-1],
     position_embeddings=tuple(x[:,:-1,:] for x in rope),output_attentions=False)
    att=torch.cat((att,torch.zeros_like(norm[:,-1:,:])),dim=1)
    if self.audit:self.skips.append((self.step,i,'P-1 excluded from every projection and attention'))
   else:
    att,_=layer.self_attn(hidden_states=norm,attention_mask=m,position_ids=pos.unsqueeze(0),
     past_key_value=cache,use_cache=True,cache_position=pos,position_embeddings=rope,output_attentions=False)
   h=r+att
   r=h;h=r+layer.mlp(layer.post_attention_layernorm(h))
   self.capture(i,h,pos)
  h=decoder.norm(h)
  result=BaseModelOutputWithPast(last_hidden_state=h,past_key_values=cache)
  return result if return_dict else result.to_tuple()
 def validate(self,steps):
  if not self.audit:return {}
  n=self.roles['n']
  assert len(self.norms)==steps*28*2 and len(self.mlp)==steps*28
  if self.method=='Physical-D':
   assert self.exits==[dict(before=n,after=n-256,removed=256)]
   expected={(s,i) for s in range(steps) for i in SKIP}
   assert {(s,i) for s,i,_ in self.skips}==expected
   assert not any(s>0 and i in SKIP for s,i,_,_ in self.projections)
   assert all(rows==n-1 for s,i,name,rows in self.projections if s==0 and i in SKIP)
   assert all(self.cache.key_cache[i] is None and self.cache.value_cache[i] is None for i in SKIP)
  return dict(condition=self.method,steps=steps,prefill=1,real_decode=steps-1,hidden_exit=self.exits,
   attention_calls=len(self.attention_calls),projection_calls=len(self.projections),
   norm_calls=len(self.norms),MLP_calls=len(self.mlp),skip_events=self.skips,
   query_and_mask_shapes=self.positions,projection_shapes=self.projections,
   blocked_probability_max=0.,finite=True,original_logical_positions=True)

def gpu_ready(model,tokenizer,loader,sample):
 request,prompt,ids,roles=frozen.prepare_cpu(model,tokenizer,sample)
 pixels=loader(sample['actual_image_path'],max_num=1).to('cuda:0',dtype=torch.bfloat16)
 ids=ids.to('cuda:0')
 assert pixels.shape==(1,3,448,448)
 model.img_context_token_id=roles['image_token_id']
 return dict(ids=ids,pixels=pixels,roles=roles,request=request,prompt=prompt)

def embeddings(model,ready,prepared=None,event=None):
 if prepared is not None:return prepared['inputs']['inputs_embeds']
 if event is None:visual=model.extract_feature(ready['pixels'])
 else:
  # Exact pinned extract_feature operations, with one event after the official vision encoder.
  out=model.vision_model(pixel_values=ready['pixels'],output_hidden_states=model.select_layer!=-1,return_dict=True)
  visual=out.last_hidden_state if model.select_layer==-1 else out.hidden_states[model.select_layer]
  event('vision_encoder_end');visual=visual[:,1:,:]
  h=w=int(visual.shape[1]**.5)
  visual=model.pixel_shuffle(visual.reshape(visual.shape[0],h,w,-1),scale_factor=model.downsample_ratio)
  visual=model.mlp1(visual.reshape(visual.shape[0],-1,visual.shape[-1]))
 x=model.language_model.get_input_embeddings()(ready['ids']).clone()
 x[ready['ids']==ready['roles']['image_token_id']]=visual.reshape(-1,x.shape[-1])
 return x

@torch.inference_mode()
def generate(model,tok,ready,method,max_tokens=32,fixed=False,audit=False,prepared=None,timed=False,forced=None):
 lm=model.language_model;n=ready['roles']['n']
 cache=CompactCache(ready['roles']) if method=='Physical-D' else DynamicCache()
 before=frozen.hooks_count(model);scores=[];tokens=[];events={}
 eos=tok.convert_tokens_to_ids(model.conv_template.sep.strip())
 control=Control(model,method,ready['roles'],cache,audit)
 if timed:torch.cuda.reset_peak_memory_stats()
 # CUDA events do not force a per-token CPU synchronization.
 def event(name):
  if timed:
   e=torch.cuda.Event(enable_timing=True);e.record();events[name]=e
 if timed:torch.cuda.synchronize()
 start=time.perf_counter()
 with control:
  event('start')
  embed=embeddings(model,ready,prepared,event if timed else None);event('vision_projector_injection_end')
  for step in range(max_tokens):
   control.step=step;control.prefill=step==0;control.masks={}
   if method=='Physical-D':cache.prefill=step==0
   pos=torch.arange(n,device=embed.device) if step==0 else torch.tensor([n+step-1],device=embed.device)
   args=dict(inputs_embeds=embed) if step==0 else dict(input_ids=tokens[-1].reshape(1,1))
   out=lm(**args,attention_mask=torch.ones((1,n+step),device=embed.device,dtype=torch.long),
    position_ids=pos.unsqueeze(0),past_key_values=cache,use_cache=True,cache_position=pos,
    logits_to_keep=1,return_dict=True,output_attentions=False,output_hidden_states=False)
   logits=out.logits[0,-1]
   next_token=logits.argmax(-1) if forced is None else torch.tensor(forced[step],device=embed.device)
   tokens.append(next_token)
   if audit:
    assert torch.isfinite(logits).all();scores.append(logits.detach().cpu().clone())
   if step==0:
    event('prefill_end')
    if timed:torch.cuda.synchronize()
    first=time.perf_counter()
   if not fixed and int(next_token)==eos:break
  event('decode_end')
  # Release request intermediates in all conditions inside the E2E boundary; cache remains live for audit.
  del out,logits,embed
 if timed:torch.cuda.synchronize()
 end=time.perf_counter()
 if timed:
  peak_alloc=torch.cuda.max_memory_allocated()/1048576
  peak_reserved=torch.cuda.max_memory_reserved()/1048576
 ids=torch.stack(tokens).cpu().tolist()
 rows,live,capacity=layout(cache,'Native' if method=='No-op' else method,ready['roles'],len(ids))
 audit_record=control.validate(len(ids))
 assert before==frozen.hooks_count(model)
 text=tok.decode(ids,skip_special_tokens=True).split(model.conv_template.sep.strip())[0].strip()
 result=dict(method=method,token_ids=ids,text=text,prediction=text.rstrip('.').lower(),
  generated_tokens=len(ids),eos=ids[-1]==eos,truncated=len(ids)==max_tokens and ids[-1]!=eos,
  empty=not text,KV_live_MiB=live,KV_capacity_MiB=capacity,kv_layout=rows,
  independent_cache=True,audit=audit_record)
 if timed:
  ttft=(first-start)*1000;e2e=(end-start)*1000
  result.update(E2E_ms=e2e,TTFT_ms=ttft,TPOT_ms=(e2e-ttft)/(len(ids)-1),
   vision_encoder_ms=events['start'].elapsed_time(events['vision_encoder_end']),
   vision_projector_injection_ms=events['start'].elapsed_time(events['vision_projector_injection_end']),
   language_prefill_ms=events['vision_projector_injection_end'].elapsed_time(events['prefill_end']),
   incremental_decode_ms=events['prefill_end'].elapsed_time(events['decode_end']),
   CUDA_event_TPOT_ms=events['prefill_end'].elapsed_time(events['decode_end'])/(len(ids)-1),
   peak_allocated_MiB=peak_alloc,peak_reserved_MiB=peak_reserved,
   cleanup_boundary='out/logits/embeddings freed before final CUDA synchronize; persistent cache retained for layout audit')
 del cache,control.cache
 return result,scores,control

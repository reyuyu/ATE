"""Dynamic cutoff including terminal cutoff28 afterL27, before final norm.
Parent core is unchanged; one terminal-exit branch is added in this independent file.
Dynamic physical cutoff and selective image-KV retention; frozen R boundary L20.
Derived from the frozen InternVL V2 engine; old files are never modified.
Physical lifetime checks use cutoff, while permission checks keep the constant20.
Physical selective image-KV retention, plus independent Dense-R reference.
Transient zero-only scratch preserves native masked-key reduction width at skipped
prefill and non-retained deep/decode attention. It is not original image KV and
is not persistent. Extra scatter/zero/matmul work is included in efficiency.
No answer QKV/attention is computed at skipped layers.
"""
from contextlib import ExitStack
from types import MethodType
import time
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from ..internvl3_svh_physical_d_pilot import engine as base
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as frozen
SKIP=set(range(2,6))
class CompactCache(DynamicCache):
 def __init__(self,roles,K,device,cutoff=20,audit=False):
  super().__init__();self.roles=roles;self.K=set(K);self.keep=base.text_positions(roles,device)
  self.cutoff=int(cutoff);self.audit=audit;self.original_positions=[[] for _ in range(28)]
  assert 20<=self.cutoff<=28 and self.K<=set(range(28)) and not self.K&SKIP
  assert not self.K or max(self.K)<self.cutoff
  self.key_cache=[None]*28;self.value_cache=[None]*28;self.prefill=True;self.buffers=[None]*28;self.used=[0]*28;self.capacities=[0]*28
 def zero_image_scratch(self,k,v):
  total=k.shape[-2]+256
  extra=k.shape[-2]-len(self.keep)
  positions=torch.cat((self.keep,torch.arange(self.roles["n"],self.roles["n"]+extra,device=k.device)))
  shape=(*k.shape[:-2],total,k.shape[-1])
  a=k.new_zeros(shape);b=v.new_zeros(shape)
  a.index_copy_(-2,positions,k);b.index_copy_(-2,positions,v)
  return a,b
 def update(self,k,v,layer_idx,cache_kwargs=None):
  i=layer_idx
  if self.audit and i not in SKIP:
   positions=cache_kwargs["cache_position"].detach().cpu().tolist()
   assert len(positions)==k.shape[-2] and len(set(positions))==len(positions)
   if self.prefill:self.original_positions[i]=positions if i in self.K or i>=self.cutoff else [positions[j] for j in self.keep.cpu().tolist()]
   else:
    assert not set(positions)&set(self.original_positions[i]),"DUPLICATE_KV_LOGICAL_POSITION"
    self.original_positions[i]+=positions
  if self.prefill:
   if i in SKIP:
    # The excluded P-1 is causal-future to every remaining query. These zeros
    # are never consumers' data or persistent storage, just reduction padding.
    z=k.new_zeros((*k.shape[:-2],1,k.shape[-1]))
    return torch.cat((k,z),dim=-2),torch.cat((v,torch.zeros_like(z)),dim=-2)
   stored=tuple(x if i in self.K or i>=self.cutoff else x.index_select(-2,self.keep) for x in (k,v))
   self.allocate_update(k,v,i,stored)
   return self.zero_image_scratch(k,v) if i>=self.cutoff else (k,v)
  assert i not in SKIP and self.key_cache[i] is not None
  pair=self.allocate_update(k,v,i,(k,v))
  return pair if i in self.K else self.zero_image_scratch(*pair)
 def get_seq_length(self,layer_idx=0):
  k=self.key_cache[layer_idx];return 0 if k is None else k.shape[-2]

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
def edit_mask(original,roles,prefill,layer,K):
 m=original.clone();lo,hi=roles["image"][0],roles["image"][-1]+1
 if layer not in K:m[:,:,-1 if prefill else 0,lo:hi]=-torch.inf
 if prefill and layer>=20:m[:,:,hi:roles["n"]-1,lo:hi]=-torch.inf
 return m
def layout(cache,physical,roles,steps,K):
 seen=set();rows=[];total=live=0
 for i in range(28):
  k=cache.key_cache[i];v=cache.value_cache[i];cap=0
  for x in (k,v):
   if x is None:continue
   live+=x.numel()*x.element_size();s=x.untyped_storage();identity=(s.data_ptr(),s.nbytes())
   if identity not in seen:seen.add(identity);cap+=s.nbytes()
  n=0 if k is None else k.shape[-2];visual=256 if not physical or i in K else 0
  rows.append(dict(layer=i,logical_length=roles["n"]+steps-1,physical_length=n,visual_kv=visual,k_shape=[] if k is None else list(k.shape),v_shape=[] if v is None else list(v.shape),dtype=None if k is None else str(k.dtype),live_bytes=0 if k is None else (k.numel()+v.numel())*k.element_size(),capacity_bytes=cap,skip_attention=physical and i in SKIP))
  total+=cap
 return rows,live/1048576,total/1048576
class Control(base.Control):
 def __init__(self,model,physical,roles,cache,K,audit=False,cutoff=20):
  super().__init__(model,"Physical-D" if physical else "Dense-D",roles,cache,audit)
  self.physical=physical;self.K=set(K);self.cutoff=int(cutoff);
  assert 20<=self.cutoff<=28 and not self.K&SKIP and (not self.K or max(self.K)<self.cutoff)
  self.allowed_image_probabilities=[];self.blocked=[];self.R_image_probabilities=[]
 def __enter__(self):
  self.stack=ExitStack()
  if self.physical:
   original=self.lm.model.forward;self.lm.model.forward=MethodType(self.physical_forward,self.lm.model)
   self.stack.callback(setattr,self.lm.model,"forward",original)
  else:
   for i,layer in enumerate(self.layers):
    def before(module,args,kw,i=i):
     key=(self.prefill,i in self.K,i>=20)
     if key not in self.masks:self.masks[key]=edit_mask(kw["attention_mask"],self.roles,self.prefill,i,self.K)
     if self.audit:
      changed=self.masks[key]!=kw["attention_mask"];expected=torch.zeros_like(changed)
      lo,hi=self.roles["image"][0],self.roles["image"][-1]+1
      if i not in self.K:expected[:,:,-1 if self.prefill else 0,lo:hi]=True
      if self.prefill and i>=20:expected[:,:,hi:self.roles["n"]-1,lo:hi]=True
      assert torch.equal(changed,expected),"NON_TARGET_DENSE_MASK_CHANGED"
     return args,dict(kw,attention_mask=self.masks[key])
    self.stack.callback(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True).remove)
    if i in SKIP:
     def zero(module,args,out,i=i):
      z=out.clone()
      if self.prefill:z[:,-1:]=0
      else:z.zero_()
      if self.audit:
       assert torch.count_nonzero(z[:,-1:])==0
       if self.prefill:assert torch.equal(z[:,:-1],out[:,:-1])
       self.zero_events.append((self.step,i))
      return z
     self.stack.callback(layer.self_attn.o_proj.register_forward_hook(zero).remove)
  if self.audit:
   for i,l in enumerate(self.layers):
    if not self.physical:
     def out_hook(m,args,out,i=i):
      pos=torch.arange(self.roles["n"],device=out[0].device) if self.prefill else torch.tensor([self.roles["n"]+self.step-1],device=out[0].device)
      self.capture(i,out[0],pos)
     self.stack.callback(l.register_forward_hook(out_hook).remove)
    for name,module in [("input_norm",l.input_layernorm),("post_norm",l.post_attention_layernorm),("mlp",l.mlp)]:
     def count(m,args,out,i=i,name=name):
      assert torch.isfinite(out).all()
      (self.mlp if name=="mlp" else self.norms).append((self.step,i,name,int(args[0].shape[1])))
     self.stack.callback(module.register_forward_hook(count).remove)
    for name in ("q_proj","k_proj","v_proj","o_proj"):
     def projection(m,args,out,i=i,name=name):
      assert torch.isfinite(out).all();self.projections.append((self.step,i,name,int(args[0].shape[1])))
     self.stack.callback(getattr(l.self_attn,name).register_forward_hook(projection).remove)
    def attn_pre(m,args,kw,i=i):
     p=kw["position_ids"].detach().cpu().tolist()[0]
     self.positions.append(dict(step=self.step,layer=i,query_positions=p,q_rows=int(kw["hidden_states"].shape[1]),mask_shape=list(kw["attention_mask"].shape)))
     self.attention_calls.append((self.step,i))
    def attn_out(m,args,out,i=i):
     weights=out[1];assert torch.isfinite(weights).all()
     lo,hi=self.roles["image"][0],self.roles["image"][-1]+1
     if not(self.physical and self.prefill and i in SKIP):
      a=float(weights[:,:,-1:,lo:hi].abs().max())
      if i not in self.K:assert a==0;self.blocked.append((self.step,i,a))
      else:self.allowed_image_probabilities.append((self.step,i,a));assert a>0,"OPEN_LAYER_NO_IMAGE_ATTENTION"
     if self.prefill and i>=20:
      if self.physical and i>=self.cutoff:
       text=[j for j in range(self.roles["n"]) if j not in self.roles["image"]]
       rel=[text.index(j) for j in self.roles["R"]]
       rv=float(weights[:,:,rel,:][...,lo:hi].abs().max());assert rv==0;self.R_image_probabilities.append((self.step,i,rv))
      else:
       rv=float(weights[:,:,hi:self.roles["n"]-1,lo:hi].abs().max());assert rv==0;self.R_image_probabilities.append((self.step,i,rv))
    self.stack.callback(l.self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True).remove)
    self.stack.callback(l.self_attn.register_forward_hook(attn_out).remove)
  return self
 def physical_forward(self,decoder,input_ids=None,attention_mask=None,position_ids=None,past_key_values=None,inputs_embeds=None,use_cache=True,output_attentions=False,output_hidden_states=False,return_dict=True,cache_position=None,**kw):
  cache=past_key_values;assert cache is self.cache and use_cache
  h=inputs_embeds if inputs_embeds is not None else decoder.embed_tokens(input_ids);pos=cache_position
  if position_ids is None:position_ids=pos.unsqueeze(0)
  mask=decoder._update_causal_mask(attention_mask,h,pos,cache,False) if self.prefill else self.decode_full[...,:self.roles["n"]+self.step]
  rope=decoder.rotary_emb(h,position_ids);text=cache.keep
  if self.prefill:
   blocked=edit_mask(mask,self.roles,True,0,set())
   native=mask
   late_blocked=edit_mask(mask,self.roles,True,20,set())
   late_open=late_blocked.clone()
   lo,hi=self.roles["image"][0],self.roles["image"][-1]+1
   late_open[:,:,-1,lo:hi]=mask[:,:,-1,lo:hi]
   deep=late_blocked.index_select(-2,text)
  else:
   compact=self.decode_blocked[...,:self.roles["n"]+self.step]
  for i,layer in enumerate(decoder.layers):
   if i==self.cutoff and self.prefill:
    before=h.shape[1];h=h.index_select(1,text);pos=text;rope=tuple(x.index_select(1,text) for x in rope)
    if self.audit:self.exits.append(dict(before=before,after=h.shape[1],removed=before-h.shape[1]))
   if self.prefill:
    m=deep if i>=self.cutoff else (late_open if i in self.K else late_blocked) if i>=20 else (native if i in self.K else blocked)
   else:m=mask if i in self.K else compact
   residual=h;norm=layer.input_layernorm(h)
   if i in SKIP and not self.prefill:
    att=torch.zeros_like(norm)
    if self.audit:self.skips.append((self.step,i,"all answer attention/QKV/QK/AV/o_proj absent"))
   elif i in SKIP:
    att,_=layer.self_attn(hidden_states=norm[:,:-1],attention_mask=m[:,:,:-1,:],position_ids=pos[:-1].unsqueeze(0),past_key_value=cache,use_cache=True,cache_position=pos[:-1],position_embeddings=tuple(x[:,:-1] for x in rope),output_attentions=False)
    att=torch.cat((att,torch.zeros_like(norm[:,-1:])),dim=1)
    if self.audit:self.skips.append((self.step,i,"P-1 projections absent; one transient masked zero key padding"))
   else:
    att,_=layer.self_attn(hidden_states=norm,attention_mask=m,position_ids=pos.unsqueeze(0),past_key_value=cache,use_cache=True,cache_position=pos,position_embeddings=rope,output_attentions=False)
   h=residual+att;h=h+layer.mlp(layer.post_attention_layernorm(h));self.capture(i,h,pos)
  if self.prefill and self.cutoff==len(decoder.layers):
   # Terminal exit: every language block has consumed real image Hidden.
   # Final RMSNorm is tokenwise; remove visual rows only afterL27.
   before=h.shape[1];h=h.index_select(1,text)
   if self.audit:self.exits.append(dict(before=before,after=h.shape[1],removed=before-h.shape[1]))
  h=decoder.norm(h);out=BaseModelOutputWithPast(last_hidden_state=h,past_key_values=cache)
  return out if return_dict else out.to_tuple()
 def validate(self,steps):
  result=super().validate(steps)
  if self.audit:
   for i,row in enumerate(self.positions):
    layer=row["layer"];step=row["step"]
    expected=list(range(self.roles["n"])) if step==0 else [self.roles["n"]+step-1]
    if self.physical and step==0:
     if layer in SKIP:expected=expected[:-1]
     elif layer>=self.cutoff:expected=[p for p in expected if p not in self.roles["image"]]
    assert row["query_positions"]==expected,"ROPE_LOGICAL_POSITION_MISMATCH"
   if self.physical:
    for step,i,name,rows in self.mlp+self.norms:
     expected_rows=(self.roles["n"] if i<self.cutoff else self.roles["n"]-256) if step==0 else 1
     assert rows==expected_rows,("NORM_MLP_PHYSICAL_ROWS",step,i,name,rows,expected_rows)
    for step,i,name,rows in self.projections:
     expected_rows=(self.roles["n"]-1 if i in SKIP else self.roles["n"] if i<self.cutoff else self.roles["n"]-256) if step==0 else 1
     assert rows==expected_rows,("ACTUAL_PROJECTION_ROWS",step,i,name,rows,expected_rows)
    for i,positions in enumerate(self.cache.original_positions):
     expected=[] if i in SKIP else list(range(self.roles["n"])) if i in self.K else [p for p in range(self.roles["n"]) if p not in self.roles["image"]]
     if i not in SKIP:expected+=list(range(self.roles["n"],self.roles["n"]+steps-1))
     assert positions==expected,("PERSISTENT_KV_ORIGINAL_POSITIONS",i)
   result.update(cutoff=self.cutoff,exit_stage="after_L27_before_final_norm" if self.cutoff==28 else "before_L"+str(self.cutoff),prefill_visual_rows_removed_from_language_blocks=256*(28-self.cutoff),R_image_permission_boundary=20,MLP_input_rows=self.mlp,norm_input_rows=self.norms,persistent_KV_original_positions=self.cache.original_positions if self.physical else None,reaccess_layers=sorted(self.K),allowed_image_probability_samples=self.allowed_image_probabilities,blocked_image_probability_samples=self.blocked,R_image_probability_samples=self.R_image_probabilities,transient_zero_padding_skipped_prefill_and_nonK_deep_decode=self.physical,attention_reduction_width_preserved_with_zero_scratch=self.physical,non_target_queries_preserved=True)
  return result
@torch.inference_mode()
def generate(model,tok,ready,name,K,max_tokens=16,fixed=False,audit=False,timed=False,forced=None,physical=True,cutoff=20):
 if name in ("Native","No-op"):
  return base.generate(model,tok,ready,name,max_tokens=max_tokens,fixed=fixed,audit=audit,timed=timed,forced=forced)
 n=ready["roles"]["n"];cache=CompactCache(ready["roles"],K,ready["ids"].device,cutoff=cutoff,audit=audit) if physical else DynamicCache()
 control=Control(model,physical,ready["roles"],cache,K,audit,cutoff=cutoff);before=frozen.hooks_count(model)
 tokens=[];scores=[];events={};lm=model.language_model;eos=tok.convert_tokens_to_ids(model.conv_template.sep.strip())
 finite=torch.ones((),device=ready["ids"].device,dtype=torch.bool) if not timed else None
 def event(name):
  if timed:e=torch.cuda.Event(enable_timing=True);e.record();events[name]=e
 if timed:torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
 start=time.perf_counter()
 with control:
  event("start");embed=base.embeddings(model,ready,event=event if timed else None);event("vision_projector_injection_end")
  for step in range(max_tokens):
   control.step=step;control.prefill=step==0;control.masks={}
   if physical:cache.prefill=step==0
   pos=torch.arange(n,device=embed.device) if step==0 else torch.tensor([n+step-1],device=embed.device)
   args=dict(inputs_embeds=embed) if step==0 else dict(input_ids=tokens[-1].reshape(1,1))
   out=lm(**args,attention_mask=torch.ones((1,n+step),device=embed.device,dtype=torch.long),position_ids=pos.unsqueeze(0),past_key_values=cache,use_cache=True,cache_position=pos,logits_to_keep=1,return_dict=True,output_attentions=False,output_hidden_states=False)
   logits=out.logits[0,-1]
   if finite is not None:finite=finite & torch.isfinite(logits).all()
   token=logits.argmax(-1) if forced is None else torch.tensor(forced[step],device=embed.device);tokens.append(token)
   if audit:scores.append(logits.detach().cpu().clone())
   if step==0:
    event("prefill_end")
    if timed:torch.cuda.synchronize()
    first=time.perf_counter()
   if not fixed and int(token)==eos:break
  event("decode_end");del out,logits,embed
 if timed:torch.cuda.synchronize()
 end=time.perf_counter()
 peak_alloc=torch.cuda.max_memory_allocated()/1048576 if timed else None;peak_reserved=torch.cuda.max_memory_reserved()/1048576 if timed else None
 if finite is not None:assert bool(finite),"NONFINITE_LOGITS"
 ids=torch.stack(tokens).cpu().tolist();rows,live,capacity=layout(cache,physical,ready["roles"],len(ids),K)
 for row in rows:
  i=row["layer"];expected=0 if physical and i in SKIP else n+len(ids)-1-(256 if physical and i not in K else 0)
  assert row["physical_length"]==expected
  if physical and i in SKIP:assert row["capacity_bytes"]==0
 assert before==frozen.hooks_count(model),"HOOK_LEAK"
 text=tok.decode(ids,skip_special_tokens=True).split(model.conv_template.sep.strip())[0].strip()
 result=dict(method=name,cutoff=cutoff,R_image_permission_boundary=20,reaccess_layers=list(K),token_ids=ids,text=text,prediction=text.rstrip(".").lower(),generated_tokens=len(ids),eos=ids[-1]==eos,truncated=len(ids)==max_tokens and ids[-1]!=eos,empty=not text,KV_live_MiB=live,KV_capacity_MiB=capacity,kv_layout=rows,independent_cache=True,audit=control.validate(len(ids)))
 if timed:
  e2e=(end-start)*1000;ttft=(first-start)*1000
  result.update(E2E_ms=e2e,TTFT_ms=ttft,TPOT_ms=(e2e-ttft)/(len(ids)-1),vision_encoder_ms=events["start"].elapsed_time(events["vision_encoder_end"]),vision_projector_injection_ms=events["start"].elapsed_time(events["vision_projector_injection_end"]),language_prefill_ms=events["vision_projector_injection_end"].elapsed_time(events["prefill_end"]),incremental_decode_ms=events["prefill_end"].elapsed_time(events["decode_end"]),CUDA_event_TPOT_ms=events["prefill_end"].elapsed_time(events["decode_end"])/(len(ids)-1),peak_allocated_MiB=peak_alloc,peak_reserved_MiB=peak_reserved)
 del cache,control.cache
 return result,scores,control

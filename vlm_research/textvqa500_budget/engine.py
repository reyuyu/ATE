"""Decode-only hard masks around unmodified native HF attention and DynamicCache.

No decoder/attention forward replacement, KV compaction, bypass, or visual replay.
The existing frozen greedy loop supplies the original complete native prefill.
"""
from contextlib import contextmanager,nullcontext
import torch
from transformers.cache_utils import DynamicCache
from ..middle10_attribution.engine import generate as native_generate
from ..p1_layerwise_accuracy.engine import tensor_hash

class DecodeMask:
 def __init__(self,adapter,layout,keep,audit=False):
  self.a,self.layout,self.keep,self.audit=adapter,layout,frozenset(keep),audit
  self.calls=[0]*32;self.masked=[0]*32;self.mlp=[0]*32;self.qkv=[[0]*3 for _ in range(32)]
  self.events=[];self.zero_checks=0;self.prefill_hashes={};self.cache_hashes=[];self.cache_steps=[]
  self.image=list(layout.image);self.nonimage=[i for i in range(layout.length) if i not in self.image]
 @contextmanager
 def context(self):
  handles=[]
  try:
   for i,layer in enumerate(self.a.layers):
    def before(module,args,kw,i=i):
     h=kw.get('hidden_states',args[0] if args else None);cp=kw['cache_position'];count=self.calls[i]
     prefill=count==0
     assert h.shape[1]==(self.layout.length if prefill else 1)
     assert cp.tolist()==(list(range(self.layout.length)) if prefill else [self.layout.length+count-1])
     self.calls[i]+=1
     if not prefill and i not in self.keep:
      old=kw.get('attention_mask')
      mask=h.new_zeros((1,1,1,int(cp[-1])+1)) if old is None else old.clone()
      mask[...,self.image]=-torch.inf
      if self.audit:
       text=[j for j in range(int(cp[-1])+1) if j not in self.image]
       original=torch.zeros_like(mask) if old is None else old
       assert torch.equal(mask[...,text],original[...,text])
       assert torch.isneginf(mask[...,self.image]).all() and (mask==0).any(-1).all()
      kw['attention_mask']=mask;self.masked[i]+=1
     # Prefill and permitted layers return their original masks without edits.
     if self.audit:kw['output_attentions']=True
     return args,kw
    handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True))
    if self.audit:
     def after(module,args,kw,out,i=i):
      weights=out[1];assert weights is not None and torch.isfinite(weights).all()
      if self.calls[i]>1 and i not in self.keep:
       assert torch.count_nonzero(weights[...,self.image])==0
       self.zero_checks+=1
     handles.append(layer.self_attn.register_forward_hook(after,with_kwargs=True))
     def mlp_hook(module,args,out,i=i):self.mlp[i]+=1
     handles.append(layer.mlp.register_forward_hook(mlp_hook))
     for j,name in enumerate(['q_proj','k_proj','v_proj']):
      def qkv_hook(module,args,out,i=i,j=j):
       self.qkv[i][j]+=1
       if self.calls[i]==1:self.prefill_hashes[f'{i}:{j}']=tensor_hash(out)
      handles.append(getattr(layer.self_attn,name).register_forward_hook(qkv_hook))
   if self.audit:
    def output(module,args,kw,out):
     cache=out.past_key_values;assert isinstance(cache,DynamicCache)
     expected=self.layout.length+len(self.cache_steps)
     assert len(cache.key_cache)==32 and all(cache.get_seq_length(i)==expected for i in range(32))
     self.cache_steps.append(expected)
     self.cache_hashes.append([tensor_hash(x) for pair in zip(cache.key_cache,cache.value_cache) for x in pair])
    handles.append(self.a.model.register_forward_hook(output,with_kwargs=True))
   yield self
  finally:
   for h in handles:h.remove()
 def summary(self):
  return dict(calls=self.calls,masked_decode_calls=self.masked,mlp_calls=self.mlp,qkv_calls=self.qkv,exact_zero_probability_checks=self.zero_checks,cache_steps=self.cache_steps,native_DynamicCache=True,prefill_masks_untouched=True,permitted_layer_masks_untouched=True,nonimage_mask_unchanged=True)

@torch.inference_mode()
def generate(adapter,layout,keep=None,audit=False,max_tokens=32,fixed=False):
 # keep=None is genuinely unhooked Native Full. keep=all32 is the no-op smoke.
 control=None if keep is None else DecodeMask(adapter,layout,keep,audit)
 with control.context() if control else nullcontext():
  result=native_generate(adapter,layout,'Native',max_tokens=max_tokens,fixed=fixed,capture='all' if audit else False)
 g,logits,valid,hidden=result
 assert len(g['kv']['layers'])==32 and all(x['used_tokens']==layout.length+len(g['token_ids'])-1 for x in g['kv']['layers'])
 if control:
  assert control.calls==[len(g['token_ids'])]*32
  assert control.masked==[0 if i in keep else len(g['token_ids'])-1 for i in range(32)]
  if audit:
   assert control.mlp==control.calls and all(x==[len(g['token_ids'])]*3 for x in control.qkv)
 adapter.verify()
 return g,logits,control

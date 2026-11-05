"""Dense accuracy-only interventions. No timing, FLOP or performance probes."""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask
from ..qwen3vl_early_windows.experiment import tensor_digest
from .common import SPECS,SKIP

def layer_mask(base,images,n,pre,layer,condition):
 spec=SPECS[condition];m=base.clone()
 allow=(spec['p1'] if pre else spec['decode']) and layer not in SKIP
 if not allow:m[...,-1,images]=-torch.inf
 if pre and layer>=22 and not spec['late_writer']:
  qi=torch.arange(int(images[-1])+1,n-1,device=m.device)
  m[0,0,qi[:,None],images[None,:]]=-torch.inf
 return m

class DenseExecution:
 def __init__(self,model,layout,condition,audit=False,noop=False):
  self.model=model;self.d=model.model.language_model;self.layout=layout;self.c=condition;self.audit_enabled=audit;self.noop=noop
  self.n=layout['input_length'];self.I=layout['image_positions'];self.device=next(model.parameters()).device
  self.images=torch.tensor(self.I,device=self.device);self.cache=DynamicCache(config=self.d.config);self.step=0;self.records=[];self.deepstack=[];self.features={};self.mlp=[0]*36;self.residual=[0]*36;self.scope=None
 def forward(self,input_ids=None,inputs_embeds=None,position_ids=None,cache_position=None,attention_mask=None,visual_pos_masks=None,deepstack_visual_embeds=None,**kwargs):
  h=inputs_embeds if inputs_embeds is not None else self.d.embed_tokens(input_ids);pre=self.step==0;cp=cache_position
  assert cp.tolist()==(list(range(self.n)) if pre else [self.n+self.step-1])
  textpos=position_ids[0]
  if position_ids.shape[0]==4:position_ids=position_ids[1:]
  rotary=self.d.rotary_emb(h,position_ids)
  base=create_causal_mask(config=self.d.config,input_embeds=h,attention_mask=attention_mask,cache_position=cp,past_key_values=self.cache,position_ids=textpos)
  if base is None:
   base=torch.full((1,1,self.n,self.n),torch.finfo(h.dtype).min,device=h.device,dtype=h.dtype).triu(1) if pre else h.new_zeros((1,1,1,self.n+self.step))
  # Same dense tensor shapes and mask arithmetic as the inherited F reference.
  # Reuse masks within each permission group; no physical image exit or compact KV.
  masks={}
  for i,layer in enumerate(self.d.layers):
   key=(i in SKIP,i>=22)
   if key not in masks:masks[key]=layer_mask(base,self.images,self.n,pre,i,self.c)
   mask=masks[key];self.scope=(i,pre,mask)
   if self.audit_enabled:
    assert h.shape[1]==(self.n if pre else 1)
    target=torch.zeros_like(mask,dtype=torch.bool)
    if not ((SPECS[self.c]['p1'] if pre else SPECS[self.c]['decode']) and i not in SKIP):target[...,-1,self.images]=True
    if pre and i>=22 and not SPECS[self.c]['late_writer']:
     qi=torch.arange(self.I[-1]+1,self.n-1,device=h.device);target[0,0,qi[:,None],self.images[None,:]]=True
    assert torch.equal(mask[~target],base[~target]),'Non-target mask changed'
    assert bool(torch.isneginf(mask[target]).all())
    self.records.append(dict(step=self.step,layer=i,hidden_rows=h.shape[1],absolute_positions=cp.tolist(),non_target_mask_exact=True,masked_entries=int(target.sum()),p1_or_decode_image_allowed=not bool(target[...,-1,self.images].all())))
   h=layer(h,attention_mask=mask,position_ids=textpos,cache_position=cp,position_embeddings=rotary,past_key_values=self.cache,use_cache=True)
   if deepstack_visual_embeds is not None and i<len(deepstack_visual_embeds):
    assert pre and i<3 and h.shape[1]==self.n
    h=self.d._deepstack_process(h,visual_pos_masks,deepstack_visual_embeds[i]);self.deepstack.append(i)
  h=self.d.norm(h);self.step+=1
  return BaseModelOutputWithPast(last_hidden_state=h,past_key_values=self.cache)
 @contextmanager
 def context(self):
  with ExitStack() as stack:
   if self.c!='Native':stack.enter_context(patch.object(self.d,'forward',self.forward))
   if self.audit_enabled:
    for i,merger in enumerate(self.model.model.visual.deepstack_merger_list):
     handle=merger.register_forward_hook(lambda m,a,o,i=i:self.features.__setitem__(i,tensor_digest(o)));stack.callback(handle.remove)
   for i,layer in enumerate(self.d.layers):
    attn=layer.self_attn;old=attn.forward
    def wrapped(*args,old=old,i=i,**kw):
     raw,w=old(*args,**kw);result=raw
     if self.c!='Native' and i in SKIP and not SPECS[self.c]['early']:
      result=raw.clone();result[:,-1,:]=0
      if self.audit_enabled:
       assert torch.count_nonzero(result[:,-1,:])==0 and torch.equal(raw[:,:-1,:],result[:,:-1,:])
     if self.audit_enabled and w is not None:
      assert bool(torch.isfinite(w).all()) and float(w.min())>=0
      row_error=float((w.float().sum(-1)-1).abs().max());assert row_error<.01
      mask=kw.get('attention_mask')
      if mask is not None:
       blocked=torch.isneginf(mask[0,0])
       if bool(blocked.any()):assert float(w[0,:,blocked].abs().max())==0
      self.records.append(dict(attention_layer=i,step=self.step,probability_row_error=row_error,forbidden_probability=0,image_probability_mass_last_query=float(w[0,:,-1,self.images].float().sum(-1).mean())))
     return (result,None) if self.c!='Native' else (result,w)
    if self.c!='Native' or self.noop or self.audit_enabled:stack.enter_context(patch.object(attn,'forward',wrapped))
    if self.audit_enabled:
     trace={}
     def start(m,a,t=trace):t['input']=a[0].detach()
     def attn_out(m,a,o,t=trace):t['attention']=o[0].detach()
     def mid(m,a,t=trace):t['mid']=a[0].detach();assert torch.equal(t['input']+t['attention'],t['mid'])
     def mlp(m,a,o,t=trace,i=i):t['mlp']=o.detach();self.mlp[i]+=1
     def end(m,a,o,t=trace,i=i):assert torch.equal(t['mid']+t['mlp'],o);self.residual[i]+=1;t.clear()
     for h in [layer.register_forward_pre_hook(start),attn.register_forward_hook(attn_out),layer.post_attention_layernorm.register_forward_pre_hook(mid),layer.mlp.register_forward_hook(mlp),layer.register_forward_hook(end)]:stack.callback(h.remove)
   yield self
 def audit(self,cache,steps):
  assert len(cache.layers)==36
  for l in cache.layers:assert l.keys.shape[-2]==l.values.shape[-2]==self.n+steps-1
  result=dict(dense_cache_all_layers=True,full_prefill_rows=self.n,all_image_hidden_retained=True,independent_cache=True)
  if self.audit_enabled:
   assert self.mlp==self.residual==[steps]*36
   if self.c!='Native':assert self.deepstack==[0,1,2]
   result.update(records=self.records,mlp_calls=self.mlp,residual_calls=self.residual,deepstack=self.deepstack,visual_features_sha256=self.features)
  return result

def generate(model,processor,inputs,layout,condition,audit=False,capture=False,noop=False,legacy=False):
 x={k:v.to(next(model.parameters()).device) for k,v in inputs.items()}
 if legacy:
  from ..qwen3vl_c3v2_physical.engine import Execution
  e=Execution(model,layout,'N' if condition=='Native' else 'F',audit=False,noop=noop)
 else:e=DenseExecution(model,layout,condition,audit=audit,noop=noop)
 ids=[];vectors=[];cache=None;eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
 with torch.inference_mode(),e.context():
  for t in range(32):
   kw=dict(x) if t==0 else dict(input_ids=token.reshape(1,1),attention_mask=torch.ones((1,layout['input_length']+t),device=next(model.parameters()).device,dtype=torch.long))
   kw.update(use_cache=True,past_key_values=cache,cache_position=torch.arange(layout['input_length'],device=next(model.parameters()).device) if t==0 else torch.tensor([layout['input_length']+t-1],device=next(model.parameters()).device),logits_to_keep=1)
   out=model(**kw);assert out.logits.shape[1]==1;v=out.logits[0,-1];assert bool(torch.isfinite(v).all())
   token=v.argmax();tid=int(token);ids.append(tid);cache=out.past_key_values
   if capture:vectors.append(v.detach().float().cpu())
   if tid in eos:break
 text=processor.tokenizer.decode(ids,skip_special_tokens=True)
 result=dict(condition=condition,token_ids=ids,text=text,eos=ids[-1] in eos,truncated=ids[-1] not in eos,empty=not text.strip(),first_token=ids[0],generated_tokens=len(ids),all_logits_finite=True)
 if not legacy:result['audit']=e.audit(cache,len(ids))
 return result,vectors

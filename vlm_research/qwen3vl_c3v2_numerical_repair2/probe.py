"""Read-only late Prefill and complete Decode probes. Native outputs returned verbatim.

Attention raw matrices retain only original text queries and frozen heads.
No full all-query/all-head attention matrix is archived.
"""
from contextlib import ExitStack,contextmanager
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from ..qwen3vl_c3v2_numerical_debug.common import SHADOW_LAYERS

ORDER=['layer_input','input_norm','q_proj','k_proj','v_proj','q_norm','k_norm','rope_q','rope_k',
       'scores_before_mask','effective_mask','scores_after_mask','attention_probabilities',
       'attention_AV_output','o_proj_raw','attention_effective_output','attention_residual',
       'post_attention_norm','mlp_output','layer_output','deepstack_output','final_norm','lm_head_input']
class Probe:
 def __init__(self,model,layout,condition):
  self.model=model;self.d=model.model.language_model;self.layout=layout;self.c=condition
  self.n=layout['input_length'];self.images=set(layout['image_positions']);self.text=[i for i in range(self.n) if i not in self.images]
  self.layer=-1;self.step=0;self.records={};self.shadow_inputs={};self.position_records=[];self.deepstack=[];self.attention_shadow={};self.cache_snapshots={};self.current_cache=None
 def positions(self,length):
  if self.step>0 and length==1:return [self.n+self.step-1]
  if length==self.n:return list(range(self.n))
  if length==self.n-1:return list(range(self.n-1))
  if length==len(self.text):return list(self.text)
  if self.step==0 and length==1:return [self.n-1]
  if self.step>0 and length==self.n+self.step:return list(range(self.n+self.step))
  if self.step>0 and length==len(self.text)+self.step:return self.text+list(range(self.n,self.n+self.step))
  raise AssertionError(('Unmapped physical positions',self.c,self.layer,self.step,length))
 def put(self,stage,t,axis=1,positions=None,key_positions=None):
  if self.step==0 and self.layer<22:return # Reuse early Prefill evidence; newly trace late Prefill and all Decode.
  positions=self.positions(t.shape[axis]) if positions is None else positions
  assert len(positions)==t.shape[axis]
  key=f'S{self.step}/L{self.layer:02d}/{stage}'
  assert key not in self.records,key
  self.records[key]=dict(layer=self.layer,stage=stage,axis=axis,positions=list(positions),
       key_positions=key_positions,shape=list(t.shape),dtype=str(t.dtype),value=t.detach().contiguous().cpu().clone())
 def linear_input(self,name,args):
  if self.c=='F' and (self.step>0 or self.layer in [22,23]):
   self.shadow_inputs[f'{self.step}/{self.layer}/{name}']=args[0].detach().contiguous().cpu().clone()
 @contextmanager
 def context(self):
  with ExitStack() as stack:
   def begin(m,args,kw):
    cp=kw['cache_position'];self.step=0 if int(cp[0])==0 else int(cp[0])-self.n+1
   h=self.model.register_forward_pre_hook(begin,with_kwargs=True);stack.callback(h.remove)
   for i,layer in enumerate(self.d.layers):
    def start(m,args,kw,i=i):
     self.layer=i;self.current_cache=kw.get('past_key_values');self.put('layer_input',args[0]);
     if self.step==0:self.position_records.append(dict(layer=i,cache_position=kw['cache_position'].detach().cpu().tolist(),position_ids=kw['position_ids'].detach().cpu().tolist(),cos=kw['position_embeddings'][0].detach().cpu().clone(),sin=kw['position_embeddings'][1].detach().cpu().clone()))
    h=layer.register_forward_pre_hook(start,with_kwargs=True);stack.callback(h.remove)
    modules=[(layer.input_layernorm,'input_norm'),(layer.self_attn.q_proj,'q_proj'),
      (layer.self_attn.k_proj,'k_proj'),(layer.self_attn.v_proj,'v_proj'),
      (layer.self_attn.q_norm,'q_norm'),(layer.self_attn.k_norm,'k_norm'),
      (layer.self_attn.o_proj,'o_proj_raw'),(layer.post_attention_layernorm,'post_attention_norm'),
      (layer.mlp,'mlp_output'),(layer,'layer_output')]
    for module,stage in modules:
     h=module.register_forward_hook(lambda m,a,out,stage=stage:self.put(stage,out));stack.callback(h.remove)
    h=layer.self_attn.register_forward_hook(lambda m,a,out:self.put('attention_effective_output',out[0]));stack.callback(h.remove)
    def cache_end(m,args,out,i=i):
     c=self.current_cache
     if c is None:return
     if hasattr(c,'entries'):
      pair=c.entries[i];used=c.used[i]
      if pair is not None:pair=tuple(t[...,:used,:] for t in pair)
     else:
      item=c.layers[i];pair=(item.keys,item.values);used=pair[0].shape[-2]
     self.cache_snapshots[f'{self.step}/{i}']=dict(layer=i,step=self.step,positions=[] if pair is None else self.positions(used),keys=None if pair is None else pair[0].detach().contiguous().cpu().clone(),values=None if pair is None else pair[1].detach().contiguous().cpu().clone())
    h=layer.register_forward_hook(cache_end);stack.callback(h.remove)
    h=layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:self.put('attention_residual',a[0]));stack.callback(h.remove)
    for name in ['q_proj','k_proj','v_proj','o_proj']:
     h=getattr(layer.self_attn,name).register_forward_pre_hook(lambda m,a,name=name:self.linear_input(name,a));stack.callback(h.remove)
    for name in ['gate_proj','up_proj','down_proj']:
     h=getattr(layer.mlp,name).register_forward_pre_hook(lambda m,a,name=name:self.linear_input(name,a));stack.callback(h.remove)
   original_rope=qm.apply_rotary_pos_emb
   def rope(q,k,*args,**kw):
    out=original_rope(q,k,*args,**kw)
    self.put('rope_q',out[0],axis=2);self.put('rope_k',out[1],axis=2)
    return out
   stack.enter_context(patch.object(qm,'apply_rotary_pos_emb',rope))
   eager=qm.eager_attention_forward
   def observe(module,q,k,v,attention_mask=None,**kw):
    out=eager(module,q,k,v,attention_mask,**kw)
    if not hasattr(module,'layer_idx'):return out
    qpos=self.positions(q.shape[-2]);kpos=(list(range(self.n-1))+[-1]) if self.step==0 and self.layer in range(1,7) and self.c in ['B','C'] and k.shape[-2]==self.n else self.positions(k.shape[-2]);text_indices=[i for i,p in enumerate(qpos) if p not in self.images]
    qi=torch.tensor(text_indices,device=q.device);heads=torch.arange(q.shape[1],device=q.device) if self.step>0 or self.layer in [22,23] else torch.tensor([0,16,31],device=q.device)
    if self.c=='F' and (self.step>0 or self.layer==22):
     self.attention_shadow[f'{self.step}/{self.layer}']=dict(query=q.detach().cpu().clone(),key=k.detach().cpu().clone(),value=v.detach().cpu().clone(),mask=None if attention_mask is None else attention_mask.detach().cpu().clone(),probabilities=out[1].detach().cpu().clone())
    # Additional observation only; never replaces native eager output.
    kr=qm.repeat_kv(k,module.num_key_value_groups)
    full_before=torch.matmul(q,kr.transpose(2,3))*kw['scaling']
    full_after=full_before if attention_mask is None else full_before+attention_mask[...,:k.shape[-2]]
    replay=torch.nn.functional.softmax(full_after,dim=-1,dtype=torch.float32).to(q.dtype)
    assert torch.equal(replay,out[1]),'Score replay does not reproduce actual attention probabilities'
    before=full_before.index_select(-2,qi)
    mask=before.new_zeros((1,1,len(qi),k.shape[-2])) if attention_mask is None else attention_mask[...,:k.shape[-2]].index_select(-2,qi)
    after=before+mask
    for stage,t in [('scores_before_mask',before),('scores_after_mask',after),('attention_probabilities',out[1].index_select(-2,qi))]:
     self.put(stage,t.index_select(1,heads),axis=2,positions=[qpos[i] for i in text_indices],key_positions=kpos)
    self.put('effective_mask',mask,axis=2,positions=[qpos[i] for i in text_indices],key_positions=kpos)
    self.put('attention_AV_output',out[0])
    return out
   stack.enter_context(patch.object(qm,'eager_attention_forward',observe))
   deep=self.d._deepstack_process
   def deepstack(h,visual_mask,features):
    out=deep(h,visual_mask,features)
    self.put('deepstack_output',out)
    if self.step==0:self.deepstack.append(dict(layer=self.layer,feature_shape=list(features.shape),visual_mask=visual_mask.detach().cpu().clone(),features=features.detach().cpu().clone()))
    return out
   stack.enter_context(patch.object(self.d,'_deepstack_process',deepstack))
   def final(m,a,out):
    previous=self.layer;self.layer=36;self.put('final_norm',out);self.layer=previous
   h=self.d.norm.register_forward_hook(final);stack.callback(h.remove)
   def head(m,a):
    previous=self.layer;self.layer=37;self.put('lm_head_input',a[0]);self.layer=previous
   h=self.model.lm_head.register_forward_pre_hook(head);stack.callback(h.remove)
   yield self

def align(a,b,scope='text'):
 """Align true logical query/key positions, never compare exited image rows."""
 apos=a['positions'];bpos=b['positions'];common=sorted(set(apos)&set(bpos))
 if scope=='p_minus_1':common=[max(apos+bpos)] if max(apos+bpos) in common else []
 elif scope=='text':
  # Image membership is attached by the analysis caller via retained_positions.
  common=[p for p in common if p in a.get('retained_positions',common)]
 if not common:return None,None,[]
 x=a['value'].index_select(a['axis'],torch.tensor([apos.index(p) for p in common]))
 y=b['value'].index_select(b['axis'],torch.tensor([bpos.index(p) for p in common]))
 if a.get('key_positions') is not None and b.get('key_positions') is not None:
  ak=a['key_positions'];bk=b['key_positions'];keys=sorted(set(ak)&set(bk))
  x=x.index_select(-1,torch.tensor([ak.index(p) for p in keys]));y=y.index_select(-1,torch.tensor([bk.index(p) for p in keys]))
 return x,y,common

def metrics(x,y):
 assert x.shape==y.shape
 x=x.float();y=y.float();same_nonfinite=(~torch.isfinite(x))&(~torch.isfinite(y))&(x==y)
 bad=(~torch.isfinite(x)|~torch.isfinite(y))&~same_nonfinite
 if bad.any():return dict(equal=False,max_abs=None,mean_abs=None,rmse=None,relative_l2=None,nonfinite_mismatch=int(bad.sum()))
 x=torch.where(same_nonfinite,0.,x);y=torch.where(same_nonfinite,0.,y);d=x-y
 return dict(equal=bool(torch.equal(x,y)),max_abs=float(d.abs().max()),mean_abs=float(d.abs().mean()),rmse=float(d.square().mean().sqrt()),relative_l2=float(d.norm()/x.norm().clamp_min(1e-12)),nonfinite_mismatch=0)

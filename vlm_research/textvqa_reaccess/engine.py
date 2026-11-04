"""Frozen historical mask observer; only permitted layer sets generalized."""
from contextlib import AbstractContextManager
import torch,numpy as np
from knockout_probe import model_layers,ROW_ATOL
from query_role_attention.engine import roles
from gqa3000_token_visual_writing.engine import forward
from query_role_formal.common import repeated_trigram
from .common import *
def blocked_layer(condition,layer,decode):
 assert condition in CONDITIONS and 0<=layer<32
 allowed=DESIGN['conditions'][condition]['Decode' if decode else 'Pminus1']
 return layer not in allowed

def blocked_edges(layout,condition,layer,past=0,device='cpu',noop=False):
 n=layout['prompt_length'];q=1 if past else n;length=past+1 if past else n
 assert not past or past>=n
 mask=torch.zeros(q,length,dtype=torch.bool,device=device)
 if not noop and blocked_layer(condition,layer,bool(past)):mask[0 if past else n-1,layout['I']]=True
 return mask
def apply_mask(native,layout,condition,layer,past=0,noop=False):
 mask=blocked_edges(layout,condition,layer,past,native.device,noop)
 return (native.clone().masked_fill(mask[None,None],-torch.inf) if mask.any() else native),mask
class VisualReadingMask(AbstractContextManager):
 def __init__(self,model,inputs,condition,noop=False,heads=False):
  self.model=model;self.layout=roles(inputs,model.config.image_token_index);self.condition=condition;self.noop=noop;self.heads=heads
  self.handles=[];self.metrics=[];self.meta=[];self.state={};self.native_masks={};self.edge_cache={};self.head_values=[]
 def __enter__(self):
  try:
   blocks=model_layers(self.model);assert len(blocks)==32
   for k,block in enumerate(blocks):
    def before(m,args,kw,k=k):
     h=kw['hidden_states'];cp=kw['cache_position'];kv=kw.get('past_key_value');past=kv.get_seq_length(k) if kv is not None else 0;n=self.layout['prompt_length']
     assert h.shape[:2]==(1,1 if past else n) and (not past or past>=n)
     old=kw['attention_mask'];assert old.shape==(1,1,h.shape[1],past+h.shape[1]) and old.dtype==torch.float16
     if k==0:guard();self.native_masks[past]=old.detach().clone()
     native=self.native_masks[past]
     key=(k,past)
     if key not in self.edge_cache:
      mask=blocked_edges(self.layout,self.condition,k,past,old.device,self.noop);self.edge_cache[key]=mask
     mask=self.edge_cache[key];active=not self.noop and blocked_layer(self.condition,k,bool(past));changed=old.clone().masked_fill(mask[None,None],-torch.inf) if active else old
     same=(old==native).all();non_target=(changed[0,0][~mask]==old[0,0][~mask]).all();legal=(old[0,0][mask]==0).all();position=(cp==torch.arange(past,past+h.shape[1],device=cp.device)).all()
     if kw.get('position_ids') is not None:position=position&(kw['position_ids']==cp[None]).all()
     restored=(changed==old).all() if not active else torch.ones((),device=old.device,dtype=torch.bool)
     checks=torch.stack([same,non_target,legal,position,restored]).float();self.state[k]=(past,mask,active,checks)
     return args,dict(kw,attention_mask=changed,output_attentions=True)
    def after(m,args,out,k=k):
     past,mask,active,checks=self.state[k];w=out[1];assert w is not None and w.shape[2:]==mask.shape
     idx=0 if past else self.layout['prompt_length']-1;row=w[0,:,idx,:].float();visual=row[:,self.layout['I']];mass=visual.sum(-1)
     maximum=visual.max() if active else torch.zeros((),device=row.device);error=(row.sum(-1)-1).abs().max();finite=torch.isfinite(row).all().float();minimum=row.min()
     measured=torch.stack([maximum,error,finite,minimum,mass.mean(),mass.median(),mass.max(),mass.min()]);self.metrics.append(torch.cat([measured,checks]));self.meta.append((k,past,int(row.shape[-1]),active))
     if self.heads:self.head_values.append(mass.detach().cpu().tolist())
     return (out[0],None)+tuple(out[2:])
    self.handles.append(block.self_attn.register_forward_pre_hook(before,with_kwargs=True));self.handles.append(block.self_attn.register_forward_hook(after))
   return self
  except BaseException:self.__exit__(None,None,None);raise
 def __exit__(self,*exc):
  for h in self.handles:h.remove()
  self.handles=[]
 def finalize(self,steps):
  assert not self.handles and len(self.metrics)==32*steps
  vals=torch.stack(self.metrics).detach().float().cpu().numpy();assert np.isfinite(vals).all()
  assert (vals[:,0]==0).all(),'Forbidden actual attention probability is nonzero'
  assert (vals[:,1]<=ROW_ATOL).all() and (vals[:,2]==1).all() and (vals[:,3]>=0).all() and (vals[:,8:]==1).all()
  rows=[]
  for j,(k,past,length,active) in enumerate(self.meta):
   assert k==j%32 and active==(not self.noop and blocked_layer(self.condition,k,bool(past)))
   r=dict(step=j//32,phase='Decode' if past else 'Pminus1',layer=k,absolute_query=past if past else self.layout['prompt_length']-1,key_length=length,original_image_keys=len(self.layout['I']),blocked=active,masked_connections=len(self.layout['I']) if active else 0,possible_target_connections=len(self.layout['I']),forbidden_probability_max=float(vals[j,0]),row_sum_error=float(vals[j,1]),image_mass_mean=float(vals[j,4]),image_mass_head_median=float(vals[j,5]),image_mass_head_max=float(vals[j,6]),image_mass_head_min=float(vals[j,7]),source='fresh_actual_attention')
   if self.heads:r['per_head_mass']=self.head_values[j]
   rows.append(r)
  return dict(passed=True,condition=self.condition,noop=self.noop,calls=len(rows),all_32_layers=True,all_forbidden_exact_zero=True,all_non_target_masks_exact=True,all_native_masks_equal=True,restored_masks_native_exact=True,absolute_positions_exact=True,original_R_all_open=True,ROW_ATOL=ROW_ATOL),rows
@torch.inference_mode()
def generate(model,tokenizer,inputs,condition,noop=False,capture=False,retain=False,heads=False):
 ids=[];trace=[];cache=None;lengths=[]
 with VisualReadingMask(model,inputs,condition,noop,heads) as mask:
  while len(ids)<32 and (not ids or ids[-1]!=2):
   guard();v,cache,step=forward(model,inputs,ids,cache,len(ids));lengths.append(step)
   if not ids:first=v.copy()
   if capture:trace.append(v.copy())
   ids.append(int(v.argmax()))
 audit,mass=mask.finalize(len(ids));text=tokenizer.decode(ids,skip_special_tokens=True).strip()
 g=dict(condition=condition,token_ids=ids,text=text,length=len(ids),non_EOS_length=len(ids)-int(ids[-1]==2),eos=ids[-1]==2,empty=not bool(text),truncated=len(ids)==32 and ids[-1]!=2,repeated_trigram=repeated_trigram(ids),own_cache=True,own_prefix=True,cache_lengths=lengths,query_layout=mask.layout,logits_to_keep=1,audit=audit)
 return g,first,np.stack(trace) if trace else None,mass,cache if retain else None

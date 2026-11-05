"""Native eager math; exact edge mask. Full probability auditing only in Smoke."""
from contextlib import AbstractContextManager
import torch,numpy as np
from knockout_probe import model_layers,ROW_ATOL
from query_role_attention.engine import roles
from gqa3000_token_visual_writing.engine import forward
from query_role_formal.common import repeated_trigram
from .policy import query_rows
def blocked_edges(layout,c,layer,past=0,device='cpu',noop=False):
 n=layout['prompt_length'];b=torch.zeros((1 if past else n,past+1 if past else n),dtype=torch.bool,device=device)
 rows=query_rows(layout,c,layer,past,noop)
 if rows:b[torch.tensor(rows,device=device)[:,None],torch.tensor(layout['I'],device=device)[None,:]]=True
 return b
class VisualMask(AbstractContextManager):
 def __init__(self,model,inputs,c,noop=False,audit=False):
  self.model=model;self.layout=roles(inputs,model.config.image_token_index);self.c=c;self.noop=noop;self.audit=audit
  self.handles=[];self.calls=[];self.measures=[];self.states={};self.native={};self.modified={}
 def __enter__(self):
  try:
   for k,block in enumerate(model_layers(self.model)):
    def before(m,args,kw,k=k):
     h=kw['hidden_states'];cp=kw['cache_position'];cache=kw.get('past_key_value');past=cache.get_seq_length(k) if cache is not None else 0
     n=self.layout['prompt_length'];assert h.shape[:2]==(1,1 if past else n) and (not past or past>=n)
     old=kw['attention_mask'];assert old.shape==(1,1,h.shape[1],past+h.shape[1]) and old.dtype==torch.float16
     rows=query_rows(self.layout,self.c,k,past,self.noop);active=bool(rows)
     if active:
      # Preserve every other original entry byte-for-byte; image keys are contiguous.
      changed=old.clone();changed[0,0,rows,self.layout['I'][0]:self.layout['I'][-1]+1]=-torch.inf
     else:changed=old
     self.calls.append((k,past,len(rows),len(rows)*len(self.layout['I'])))
     if self.audit:
      b=blocked_edges(self.layout,self.c,k,past,old.device,self.noop)
      if k==0:self.native[past]=old.detach().clone()
      checks=torch.stack([(old==self.native[past]).all(),(changed[0,0][~b]==old[0,0][~b]).all(),(old[0,0][b]==0).all(),(cp==torch.arange(past,past+h.shape[1],device=cp.device)).all(),(kw['position_ids']==cp[None]).all()]).float()
      self.states[k]=(past,b,checks)
      return args,dict(kw,attention_mask=changed,output_attentions=True)
     return args,dict(kw,attention_mask=changed)
    self.handles.append(block.self_attn.register_forward_pre_hook(before,with_kwargs=True))
    if self.audit:
     def after(m,args,out,k=k):
      past,b,checks=self.states[k];w=out[1];assert w is not None and w.shape[2:]==b.shape
      idx=[0] if past else self.layout['R']+[self.layout['prompt_length']-1]
      row=w[0,:,idx,:].float();maximum=w[0,:,b].float().max() if b.any() else torch.zeros((),device=w.device)
      vals=torch.stack([maximum,(row.sum(-1)-1).abs().max(),torch.isfinite(row).all().float(),row.min()])
      self.measures.append(torch.cat([vals,checks]));return (out[0],None)+tuple(out[2:])
     self.handles.append(block.self_attn.register_forward_hook(after))
   return self
  except BaseException:self.__exit__(None,None,None);raise
 def __exit__(self,*a):
  for h in self.handles:h.remove()
  self.handles=[]
 def finalize(self,steps):
  assert not self.handles and len(self.calls)==32*steps
  for j,(k,past,_,_) in enumerate(self.calls):assert k==j%32 and past==(0 if j<32 else self.layout['prompt_length']+j//32-1)
  out=dict(passed=True,all32_layers=True,layer_calls=len(self.calls),masked_edges_per_head=sum(x[3] for x in self.calls),prefill_masked_queries_by_layer=[x[2] for x in self.calls[:32]],decode_masked_queries_by_layer=[sum(x[2] for x in self.calls[32+k::32]) for k in range(32)],independent_cache=True,positions_and_cache_lengths_exact=True,full_probability_audit=self.audit)
  if self.audit:
   a=torch.stack(self.measures).float().cpu().numpy();assert np.isfinite(a).all() and (a[:,0]==0).all() and (a[:,1]<=ROW_ATOL).all() and (a[:,2]==1).all() and (a[:,3]>=0).all() and (a[:,4:]==1).all()
   out.update(forbidden_probability_max=float(a[:,0].max()),row_sum_error_max=float(a[:,1].max()),non_target_masks_exact=True,native_masks_exact=True,all_positions_exact=True)
  return out
@torch.inference_mode()
def generate(model,tok,inputs,c,noop=False,audit=False,capture=False,retain=False):
 ids=[];trace=[];cache=None;lengths=[]
 with VisualMask(model,inputs,c,noop,audit) as mask:
  while len(ids)<32 and (not ids or ids[-1]!=2):
   v,cache,step=forward(model,inputs,ids,cache,len(ids));lengths.append(step)
   if not ids:first=v.copy()
   if capture:trace.append(v.copy())
   ids.append(int(v.argmax()))
 a=mask.finalize(len(ids));text=tok.decode(ids,skip_special_tokens=True).strip()
 g=dict(condition=c,token_ids=ids,text=text,length=len(ids),eos=ids[-1]==2,empty=not bool(text),truncated=len(ids)==32 and ids[-1]!=2,repeated_trigram=repeated_trigram(ids),own_cache=True,own_prefix=True,cache_lengths=lengths,query_layout=mask.layout,logits_to_keep=1,audit=a)
 return g,first,np.stack(trace) if trace else None,cache if retain else None

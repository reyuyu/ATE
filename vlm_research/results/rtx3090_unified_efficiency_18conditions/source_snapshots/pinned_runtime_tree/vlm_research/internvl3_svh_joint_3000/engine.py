"""Functional o_proj output intervention; all native dense attention/cache work is retained."""
from unittest.mock import patch
import torch
from vlm_research.internvl3_visual_writing_figure3_3000 import engine as base
ORIGINAL_MASK=base.WriterMask
def writers(c):
 assert c in ('A','B','C','D')
 return list(range(20 if c in ('B','D') else 28))
def zero_answer(value,roles,prefill):
 assert value.ndim==3 and value.shape[0]==1 and value.shape[1]==(roles['n'] if prefill else 1)
 q=roles['n']-1 if prefill else 0
 result=value.clone();result[:,q:q+1,:]=0
 return result,q
class JointMask(ORIGINAL_MASK):
 def __enter__(self):
  super().__enter__()
  self.zero_events=[];self.zero_max=torch.zeros((),device='cuda:0')
  self.zero_input_finite=torch.ones((),device='cuda:0',dtype=torch.bool)
  if self.condition in ('C','D') and not self.noop:
   for index in range(2,6):
    def zero(module,args,value,i=index):
     prefill=len(self.steps)==1;step=len(self.steps)-1
     changed,q=zero_answer(value,self.roles,prefill)
     self.zero_input_finite=self.zero_input_finite & torch.isfinite(value).all()
     self.zero_max=torch.maximum(self.zero_max,changed[:,q:q+1,:].abs().max())
     if prefill:assert torch.equal(value[:,:q,:],changed[:,:q,:])
     self.zero_events.append((step,i))
     return changed
    # Runs before the inherited audit hook so residual checks see the actual zeroed o_proj output.
    self.stack.callback(self.layers[index].self_attn.o_proj.register_forward_hook(zero,prepend=True).remove)
  if self.trace:
   for index,layer in enumerate(self.layers):
    def prefix(module,args,out,i=index):
     if len(self.steps)==1:self.prefix['non_answer_L'+str(i)]=base.tensor_hash(out[0][:,:self.roles['n']-1,:])
    self.stack.callback(layer.register_forward_hook(prefix).remove)
  return self
 def validate(self,tokens):
  result=super().validate(tokens)
  active=self.condition in ('C','D') and not self.noop
  expected=[(s,i) for s in range(tokens) for i in range(2,6)] if active else []
  assert self.zero_events==expected and bool(self.zero_input_finite) and float(self.zero_max)==0
  result.update(answer_o_proj_zero_layers=list(range(2,6)) if active else [],
   zero_output_events=len(self.zero_events),zero_output_max=float(self.zero_max),
   non_answer_prefill_unchanged=True,functional_only=True,physical_attention_skip=False)
  return result
def generate(model,tok,prepared,condition='A',**kw):
 with patch.object(base,'writers',writers),patch.object(base,'WriterMask',JointMask):
  row,scores,trace=base.generate(model,tok,prepared,condition=condition,**kw)
 row['condition']=condition
 row['answer_attention_output_zero_layers']=list(range(2,6)) if condition in ('C','D') and not (kw.get('native') or kw.get('noop')) else []
 return row,scores,trace

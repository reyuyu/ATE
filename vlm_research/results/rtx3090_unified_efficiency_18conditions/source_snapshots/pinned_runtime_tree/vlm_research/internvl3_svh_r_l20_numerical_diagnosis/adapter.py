"""Minimal ephemeral projection padding; original model modules restored on exit."""
import torch
from contextlib import ExitStack
from types import MethodType
from vlm_research.internvl3_svh_r_middle_reaccess_search import engine as e
def pad_text(x,n,keep):
 assert x.shape[1]==keep.numel()
 z=x.new_zeros((x.shape[0],n,x.shape[2]));z.index_copy_(1,keep,x)
 return z
class ProjectionRepair:
 def __init__(self,model,ready,spec):
  self.model=model;self.ready=ready;self.spec=spec;self.events=[]
 def __enter__(self):
  self.stack=ExitStack();n=self.ready["roles"]["n"];keep=e.base.text_positions(self.ready["roles"],self.ready["ids"].device)
  for layer,names in self.spec.items():
   for name in names:
    module=getattr(self.model.language_model.model.layers[layer].self_attn,name);original=module.forward
    def forward(mod,x,original=original,layer=layer,name=name):
     if x.shape[1]!=keep.numel():return original(x)
     z=pad_text(x,n,keep);full=original(z);out=full.index_select(1,keep)
     self.events.append({"layer":layer,"operator":name,"compact_input_rows":x.shape[1],"actual_projection_rows":n,"returned_rows":out.shape[1],"zero_scratch":True})
     return out
    module.forward=MethodType(forward,module);self.stack.callback(setattr,module,"forward",original)
  return self
 def __exit__(self,*exc):return self.stack.__exit__(*exc)

"""Frozen C3 repairs, identical tensor math, optional read-only audit checks.

No new kernel, heuristic, precision, or algorithm is introduced here.
"""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from ..qwen3vl_c3v2_physical.engine import Execution,CompactKV,SKIP
from ..qwen3vl_c3v2_numerical_debug.repair import pad_dead_key
from ..qwen3vl_c3v2_numerical_repair2.repair import restore_dead_image_columns
from ..qwen3vl_c3v2_prefill_av_final.engine import restore_prefill_columns
from ..qwen3vl_c3v2_l22_kernel_diagnosis.candidates import LtPlan

class RepairedExecution(Execution):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw)
  self.text_list=[i for i in range(self.n) if i not in set(self.I)]
  self.plans={};self.workspace=None;self.repairs=[]
 @contextmanager
 def context(self):
  eager=qm.eager_attention_forward
  def aligned(module,q,k,v,attention_mask=None,**kw):
   layer=getattr(module,'layer_idx',-1);mode=None;ix=None
   if self.skip and layer in SKIP and self.step==0 and q.shape[-2]==self.n-1:
    kp,vp,mp=pad_dead_key(k,v,attention_mask);mode='early_dead_key'
   elif self.step>0 and isinstance(self.cache,CompactKV):
    kp,vp,mp,ix=restore_dead_image_columns(k,v,self.text_list,self.n,self.step);mode='decode_zero_image'
   elif self.physical and self.step==0 and 22<=layer<=35:
    kp,vp,mp,ix=restore_prefill_columns(k,v,attention_mask,self.text_list,self.n);mode='prefill_zero_image'
   else:return eager(module,q,k,v,attention_mask,**kw)
   out,w=eager(module,q,kp,vp,mp,**kw)
   if self.audit_enabled:
    dead=w[...,-1:] if mode=='early_dead_key' else w[...,self.image]
    assert float(dead.abs().max())==0.
    if ix is not None:
     assert torch.equal(kp.index_select(-2,ix),k) and torch.equal(vp.index_select(-2,ix),v)
     if mode=='prefill_zero_image':assert torch.equal(mp.index_select(-1,ix),attention_mask)
     assert bool(torch.isneginf(mp[...,self.image]).all())
    self.repairs.append(dict(layer=layer,step=self.step,mode=mode,query_shape=list(q.shape),original_key_rows=k.shape[-2],actual_key_rows=kp.shape[-2],persistent_dummy_KV=0,forbidden_probability=0,extra_QK_AV_flops=4*q.shape[0]*q.shape[1]*q.shape[-2]*(kp.shape[-2]-k.shape[-2])*q.shape[-1]))
   return out,w[...,:-1] if mode=='early_dead_key' else w.index_select(-1,ix)
  try:
   with ExitStack() as st:
    st.enter_context(patch.object(qm,'eager_attention_forward',aligned))
    for i in range(22,36):
     for name in ['k_proj','v_proj','gate_proj','up_proj']:
      mod=getattr(self.d.layers[i].self_attn if name in ['k_proj','v_proj'] else self.d.layers[i].mlp,name);fn=mod.forward
      def projected(x,mod=mod,fn=fn,i=i,name=name):
       if not(self.physical and self.step==0):return fn(x)
       assert x.dtype==torch.bfloat16 and x.shape[0]==1 and x.shape[1]==len(self.text) and mod.bias is None
       a=x.reshape(-1,x.shape[-1]).contiguous();m,k=a.shape;n=mod.weight.shape[0];key=(m,n,k)
       if self.workspace is None:self.workspace=torch.empty(8*2**20,device=x.device,dtype=torch.uint8)
       if key not in self.plans:
        dense=LtPlan(self.n,n,k,self.workspace);physical=LtPlan(m,n,k,self.workspace)
        assert physical.select(dense,0)==0
        self.plans[key]=(dense,physical)
       dense,physical=self.plans[key];y=physical.run(a,mod.weight).reshape(1,m,n)
       if self.audit_enabled:self.repairs.append(dict(layer=i,step=0,mode='fixed_Lt',operator=name,input_shape=list(x.shape),config=physical.config(),dense_descriptor_only=self.n,dense_projection_executed=False,matrix_flops=2*m*n*k))
       return y
      st.enter_context(patch.object(mod,'forward',projected))
    st.enter_context(super().context());yield self
  finally:
   for pair in self.plans.values():
    for p in pair:p.close()
   self.plans.clear();self.workspace=None
 def audit(self):
  a=super().audit();a['repairs']=self.repairs;return a

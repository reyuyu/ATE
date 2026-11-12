"""One Prefill dead-image-key repair. Genuine text queries and persistent KV unchanged."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from ..qwen3vl_c3v2_l22_kernel_diagnosis.full_model import FixedLtExecution
from ..qwen3vl_c3v2_physical import engine as original

def restore_prefill_columns(k,v,mask,text_positions,full_length):
 assert k.shape==v.shape and k.shape[-2]==len(text_positions)
 assert mask is not None and mask.shape[-2:]==(len(text_positions),len(text_positions))
 assert len(set(text_positions))==len(text_positions) and all(0<=p<full_length for p in text_positions)
 ix=torch.as_tensor(text_positions,device=k.device);kp=k.new_zeros((*k.shape[:-2],full_length,k.shape[-1]));vp=torch.zeros_like(kp)
 kp.index_copy_(-2,ix,k);vp.index_copy_(-2,ix,v)
 mp=mask.new_full((*mask.shape[:-1],full_length),-torch.inf);mp.index_copy_(-1,ix,mask)
 return kp,vp,mp,ix

class PrefillAVExecution(FixedLtExecution):
 def __init__(self,*a,prefill_zero=True,**kw):
  super().__init__(*a,repair_layers=tuple(range(22,36)),**kw);self.prefill_zero=prefill_zero;self.prefill_records=[]
 @contextmanager
 def context(self):
  eager=qm.eager_attention_forward
  def aligned(module,q,k,v,attention_mask=None,**kw):
   layer=getattr(module,'layer_idx',-1)
   if not(self.physical and self.step==0 and 22<=layer<=35):return eager(module,q,k,v,attention_mask,**kw)
   assert q.shape[-2]==len(self.text) and k.shape[-2]==len(self.text)
   # Persistent CompactKV has already been updated with genuine compact tensors.
   original_shapes=dict(query=list(q.shape),key=list(k.shape),value=list(v.shape));nbytes=0
   if self.prefill_zero:
    kp,vp,mp,ix=restore_prefill_columns(k,v,attention_mask,self.text.tolist(),self.n)
    assert torch.equal(kp.index_select(-2,ix),k) and torch.equal(vp.index_select(-2,ix),v) and torch.equal(mp.index_select(-1,ix),attention_mask)
    assert bool(torch.isneginf(mp[...,self.image]).all()) and torch.count_nonzero(kp[...,self.image,:])==0 and torch.count_nonzero(vp[...,self.image,:])==0
    nbytes=kp.numel()*kp.element_size()+vp.numel()*vp.element_size()+mp.numel()*mp.element_size()+ix.numel()*ix.element_size()
   else:kp,vp,mp=k,v,attention_mask;ix=None
   begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();out,w=eager(module,q,kp,vp,mp,**kw);end.record()
   if self.prefill_zero:assert float(w[...,self.image].abs().max())==0.
   b,h,m,d=q.shape;before=4*b*h*m*k.shape[-2]*d;after=4*b*h*m*kp.shape[-2]*d
   self.prefill_records.append(dict(layer=layer,step=0,original_shapes=original_shapes,actual_query_shape=list(q.shape),actual_key_shape=list(kp.shape),actual_value_shape=list(vp.shape),persistent_image_KV_added=0,image_hidden_rows_projected=0,temporary_zero_image_columns=len(self.I) if self.prefill_zero else 0,original_QK_AV_flops=before,actual_QK_AV_flops=after,extra_QK_AV_flops=after-before,allocated_restoration_bytes=nbytes,attention_probability_bytes=w.numel()*w.element_size(),start=begin,end=end,forbidden_image_probability=0. if self.prefill_zero else None,non_target_mask_strict=True))
   # Original compact observer must receive its own key indexing; returned context is native and unchanged.
   return out,w.index_select(-1,ix) if ix is not None else w
  with patch.object(qm,'eager_attention_forward',aligned),super().context():yield self
 def audit(self):
  a=super().audit();a['prefill_zero_key_enabled']=self.prefill_zero
  a['prefill_av_alignment']=[{**{k:v for k,v in r.items() if k not in ['start','end']},'single_call_gpu_ms':r['start'].elapsed_time(r['end']),'timing_note':'Single call GPU event, may include read-only probe overhead; use uninstrumented repeat for costs, not E2E benchmark'} for r in self.prefill_records]
  a['actual_language_matrix_flops']+=sum(r['extra_QK_AV_flops'] for r in self.prefill_records)
  return a

def generate(model,processor,inputs,layout,condition,prefill_zero=True,noop=False):
 class Bound(PrefillAVExecution):
  def __init__(self,*a,**k):super().__init__(*a,prefill_zero=prefill_zero,**k)
 with patch.object(original,'Execution',Bound):return original.generate(model,processor,inputs,layout,condition,True,True,noop=noop,max_tokens=32)

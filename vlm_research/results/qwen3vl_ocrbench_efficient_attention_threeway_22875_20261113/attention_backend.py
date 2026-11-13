"""Isolated Qwen attention implementation: official PyTorch SDPA, efficient CUDA only.

Compared with HF4.57.6 sdpa_attention_forward: explicitly expand grouped KV
instead of enable_gqa (unsupported by the efficient kernel), and align additive
mask storage stride to eight. Logical Q/K/V, scale, dropout and key sets agree.
No model projection, normalization, RoPE or Processor arithmetic is replaced.
"""
import collections,time
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

def repeat_kv(x,groups):
 if groups==1:return x
 b,h,n,d=x.shape
 return x[:,:,None,:,:].expand(b,h,groups,n,d).reshape(b,h*groups,n,d)

def align_mask(mask):
 if mask is None or mask.stride(-2)%8==0:return mask
 n,k=mask.shape[-2:];padded=(k+7)//8*8
 out=mask.new_empty((*mask.shape[:-1],padded));out[...,:k].copy_(mask)
 return out[...,:k]

def sdpa_compute(module,q,k,v,mask,scaling,dropout=0.0,is_causal=None):
 groups=getattr(module,'num_key_value_groups',1);k=repeat_kv(k,groups);v=repeat_kv(v,groups)
 if mask is not None:mask=align_mask(mask[...,:k.shape[-2]])
 causal=(q.shape[-2]>1 and mask is None and getattr(module,'is_causal',True)) if is_causal is None else bool(is_causal)
 out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=dropout,scale=scaling,is_causal=causal)
 return out.transpose(1,2).contiguous(),None

@contextmanager
def efficient_only():
 with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
  assert torch.backends.cuda.mem_efficient_sdp_enabled()
  assert not torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.cudnn_sdp_enabled()
  yield

class Backend:
 def __init__(self,model):
  self.model=model;self.mapping={id(b.attn):('Vision',i) for i,b in enumerate(model.model.visual.blocks)}
  self.mapping.update({id(b.self_attn):('Language',i) for i,b in enumerate(model.model.language_model.layers)})
  self.step=0;self.events=[];self.check=None;self.detail=True;self.last=None
 def forward(self,module,q,k,v,attention_mask=None,scaling=None,dropout=0.0,is_causal=None,**kw):
  assert q.dtype==k.dtype==v.dtype==torch.bfloat16 and q.device.type=='cuda'
  assert module.config._attn_implementation=='sdpa' and dropout==0
  assert torch.backends.cuda.mem_efficient_sdp_enabled() and not torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.cudnn_sdp_enabled()
  stage,layer=self.mapping[id(module)]
  event=dict(stage=stage,layer=layer,step=self.step,query=list(q.shape),key=list(k.shape),value=list(v.shape),mask_shape=list(attention_mask.shape) if attention_mask is not None else None,mask_dtype=str(attention_mask.dtype) if attention_mask is not None else None,backend='SDPA_EFFICIENT_ATTENTION_ONLY',gqa='explicit exact group expansion',zero_padded_KV=False)
  self.last=event
  if self.check is not None and stage=='Language':self.check(module,q,k,v,attention_mask,event)
  if self.detail or self.step<=1:self.events.append(event)
  return sdpa_compute(module,q,k,v,attention_mask,scaling,dropout,is_causal)
 @contextmanager
 def context(self):
  from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
  assert self.model.config.text_config._attn_implementation==self.model.config.vision_config._attn_implementation=='sdpa'
  with efficient_only(),patch.object(qm,'ALL_ATTENTION_FUNCTIONS',{'sdpa':self.forward}):yield self

class StageMonitor:
 """Stage memory and real CUDA kernel evidence, without saving a large trace."""
 def __init__(self,model,backend,profile=False):
  self.model=model;self.backend=backend;self.profile=profile;self.rows=[];self.peaks=[];self.kernels=[];self.features=[]
 def remember_peak(self):self.peaks.append((torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()))
 @contextmanager
 def stage(self,name):
  torch.cuda.synchronize();self.remember_peak();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();start=time.perf_counter()
  profiler=None
  if self.profile and (name!='Decode' or self.backend.step==1):
   profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False)
  try:
   if profiler is not None:profiler.__enter__()
   yield
  finally:
   if profiler is not None:profiler.__exit__(None,None,None)
   torch.cuda.synchronize();self.remember_peak()
   r=dict(stage=name,step=self.backend.step,baseline_allocated=base,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),elapsed_ms=(time.perf_counter()-start)*1000)
   if profiler is not None:
    events=profiler.events();ops=collections.Counter(e.name for e in events if 'scaled_dot_product' in e.name)
    kernels=collections.Counter(e.name for e in events if e.device_type==torch.autograd.DeviceType.CUDA)
    assert ops.get('aten::_scaled_dot_product_efficient_attention',0)>0,(name,dict(ops))
    assert not any('math' in k or 'flash' in k or 'cudnn' in k for k in ops),(name,dict(ops))
    self.kernels.append(dict(stage=name,step=self.backend.step,attention_operators=dict(ops),CUDA_kernel_counts=dict(kernels)))
   self.rows.append(r)
 def flatten(self,x):
  if isinstance(x,torch.Tensor):return [x]
  if isinstance(x,(list,tuple)):return [t for a in x for t in self.flatten(a)]
  return []
 @contextmanager
 def context(self):
  get=self.model.model.get_image_features;lm=self.model.model.language_model;language=lm.forward
  def vision(*a,**kw):
   with self.stage('Vision'):
    out=get(*a,**kw);ts=self.flatten(out)
    assert len(ts)==4 and all(bool(torch.isfinite(t).all()) for t in ts)
    self.features=[dict(shape=list(t.shape),dtype=str(t.dtype),finite=True) for t in ts]
    return out
  def text(*a,**kw):
   with self.stage('Language_Prefill' if self.backend.step==0 else 'Decode'):return language(*a,**kw)
  with patch.object(self.model.model,'get_image_features',vision),patch.object(lm,'forward',text):yield self
 def finish(self):
  self.remember_peak()
  return dict(stage_memory=self.rows,kernel_evidence=self.kernels,vision_outputs=self.features,peak_allocated=max(x[0] for x in self.peaks),peak_reserved=max(x[1] for x in self.peaks))

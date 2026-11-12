"""Old E2E32 timer reused unchanged for eight methods; same boundary for fusions."""
from contextlib import nullcontext
from unittest.mock import patch
from pathlib import Path
import importlib.util,time,torch
from e2e32_runtime import generate as original_generate
from e2e32_prepare import SOURCE
from vlm_research.fastv_svh import engine as fast
from vlm_research.sparsevlm_svh.engine import Fusion as SparseFusion
from apet_fusion import ApETFusion
fast.OFFICIAL=SOURCE
spec=importlib.util.spec_from_file_location('prefix_frozen',Path('/home/featurize/work/ATE/vlm_research/results/v2drop_prefix_svh_accuracy_3090/fusion_engine.py'))
prefix=importlib.util.module_from_spec(spec);spec.loader.exec_module(prefix)
METHODS=['Native','SVH','SVH-R','FastV','SparseVLM','V2Drop','ApET','DUET','FastV+SVH','SparseVLM+SVH','V2Drop+SVH','ApET+SVH']
NAMES=['Native Full','SVH','SVH-R','FastV','SparseVLM','V²Drop','ApET','DUET-VLM (C+all)','FastV+SVH','SparseVLM+SVH','V²Drop+SVH','ApET+SVH']
class Trace:
 def __init__(self,a,layout):self.a=a;self.layout=layout;self.rows=[];self.proj=[];self.step=0;self.handles=[]
 def layer(self,i,args,kw):
  if self.step not in (0,1,2,3,4,5,6,7,8,31):return
  h=args[0];pos=kw['position_ids'];expected=self.a.model.language_model.model.rotary_emb(h,pos)
  assert all(torch.equal(x,y) for x,y in zip(kw['position_embeddings'],expected))
  self.rows.append(dict(step=self.step,layer=i,hidden_rows=h.shape[1],positions=pos.tolist(),RoPE_exact=True,mask_shape=None if kw.get('attention_mask') is None else list(kw['attention_mask'].shape)))
 def projection(self,i,name,args):
  if self.step in (0,1,2,3,4,5,6,7,8,31):self.proj.append(dict(step=self.step,layer=i,op=name,rows=args[0].shape[-2]))
 def __enter__(self):
  for i,l in enumerate(self.a.layers):
   self.handles.append(l.register_forward_pre_hook(lambda m,args,kw,i=i:self.layer(i,args,kw),with_kwargs=True))
   for op in ('q','k','v','o'):self.handles.append(getattr(l.self_attn,op+'_proj').register_forward_pre_hook(lambda m,args,i=i,op=op:self.projection(i,op,args)))
  return self
 def __exit__(self,*args):
  for h in self.handles:h.remove()
 def audit(self,method):
  assert len(self.rows)==320
  n=self.layout.length;nt=len(self.layout.text)
  for r in self.rows:
   i,s=r['layer'],r['step'];rows=r['hidden_rows']
   if s==0:
    visual=(576 if i<3 else 192 if i<20 else 0) if method=='FastV+SVH' else (576 if i<3 else 288 if i<17 else 144 if i<20 else 0) if method=='V2Drop+SVH' else (576 if i<3 else 309 if i<7 else 204 if i<16 else 113 if i<20 else 0) if method=='SparseVLM+SVH' else (288 if i<16 else 96 if i<20 else 0)
    assert rows==nt+visual,(method,r,nt+visual)
   else:assert rows==1
   if s>0 and i in (2,3,4,5):assert not any(p['step']==s and p['layer']==i for p in self.proj),'Answer QKV or o_proj was called'
  return dict(layer_shapes=self.rows,projection_calls=self.proj,physical_visual_exit_before20=True,skip_decode_QKVO_layers=[2,3,4,5],RoPE_exact=True)
def combo_kv(cache,layout):
 rows=[];allstorage={}
 for i,pair in enumerate(cache.entries):
  if i in (2,3,4,5):
   assert pair is None and cache.used[i]==0;rows.append(dict(layer=i,used_tokens=0,capacity_tokens=0,visual_tokens=0,live_bytes=0,storage_bytes=0,K_shape=None,V_shape=None,dtype=None));continue
  used=cache.used[i];assert used==len(layout.text)+31,(i,used,len(layout.text)+31)
  k,v=pair;assert k.dtype==v.dtype==torch.float16
  storage={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in pair};allstorage.update(storage)
  live=sum(t.shape[0]*t.shape[1]*used*t.shape[-1]*t.element_size() for t in pair)
  rows.append(dict(layer=i,used_tokens=used,capacity_tokens=k.shape[-2],visual_tokens=0,live_bytes=live,storage_bytes=sum(storage.values()),K_shape=list(k.shape),V_shape=list(v.shape),dtype=str(k.dtype)))
 return dict(layers=rows,live_bytes=sum(r['live_bytes'] for r in rows),storage_bytes=sum(allstorage.values()),cached_decode_inputs=31,cache_class=type(cache).__name__)
@torch.inference_mode()
def generate(a,layout,method,smoke=False):
 if '+' not in method and method!='ApETDenseOracle':return original_generate(a,layout,method,smoke)
 events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(32)]
 n=layout.length;device=a.inputs['input_ids'].device;cache=ctrl=out=None;tokens=[];finite=[];logits=[];vision_calls=[0];orig=a.model.get_image_features
 def vision(*args,**kw):vision_calls[0]+=1;return orig(*args,**kw)
 observer=Trace(a,layout) if smoke else None
 torch.cuda.synchronize();start=time.perf_counter()
 try:
  pos=torch.arange(n+31,device=device);mask=torch.ones((1,n+31),dtype=torch.long,device=device)
  if method=='FastV+SVH':ctrl=fast.Fusion(a,layout,32,audit=smoke,verify_selector=smoke)
  elif method=='SparseVLM+SVH':ctrl=SparseFusion(a,layout,32,audit=smoke,verify_reference=smoke)
  elif method=='V2Drop+SVH':ctrl=prefix.PrefixSVH(a,layout,32,audit=smoke)
  else:ctrl=ApETFusion(a,layout,trace=smoke,dense=method=='ApETDenseOracle');observer=None if method=='ApETDenseOracle' else observer
  with ctrl.context(),observer if observer else nullcontext(),patch.object(a.model,'get_image_features',vision) if not method.startswith('ApET') else nullcontext():
   for step in range(32):
    if observer:observer.step=step
    kw=dict(a.inputs) if step==0 else dict(input_ids=tokens[-1].reshape(1,1),attention_mask=mask[:,:n+step])
    kw.update(use_cache=True,return_dict=True,past_key_values=cache,cache_position=pos[:n] if step==0 else pos[n+step-1:n+step],logits_to_keep=1)
    events[step][0].record();out=a.model(**kw);last=out.logits[:,-1];tokens.append(last.argmax(-1));finite.append(torch.isfinite(last).all());events[step][1].record();cache=out.past_key_values
    if step==0:torch.cuda.synchronize();first=time.perf_counter()
    if smoke and step<8:logits.append(last.detach().cpu().tolist())
    del out,last;out=None
  torch.cuda.synchronize();end=time.perf_counter()
  assert torch.stack(finite).all().item(),'Nonfinite logits'
  ids=torch.cat(tokens).tolist();kv=combo_kv(cache,layout);ms=[x.elapsed_time(y) for x,y in events]
  calls=ctrl.vision_calls if method.startswith('ApET') else vision_calls[0];assert calls==1
  g=dict(method=method,token_ids=ids,output_tokens=32,real_decode_calls=31,prefill_calls=1,vision_forward_calls=calls,all_logits_finite=True,TTFT_ms=(first-start)*1000,E2E_ms=(end-start)*1000,TPOT_ms=(end-first)*1000/31,CUDA_Event_Decode_TPOT_ms=sum(ms[1:])/31,step_event_ms=ms,cache=kv,logits_to_keep=1,use_cache=True,smoke=smoke,formal_layer_observer_hooks=smoke)
  if smoke:g.update(smoke_execution=None if observer is None else observer.audit(method),first_logits=logits[0],decode_logits=logits,pruning_events=getattr(ctrl,'selection',getattr(ctrl,'selection_events',getattr(ctrl,'events',None))))
  return g
 finally:del cache,ctrl,out,tokens,finite

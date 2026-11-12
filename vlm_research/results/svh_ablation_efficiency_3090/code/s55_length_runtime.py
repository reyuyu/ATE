"""Common 32-token request lifecycle; actual physical persistent cache audit."""
from contextlib import nullcontext,contextmanager
from dataclasses import replace
from unittest.mock import patch
import time,hashlib,json
import torch
from fastv_hf_adapter import FastV
from hf_adapter import V2Drop
from vlm_research.c3v2_middle_reaccess.engine import MiddleWriter
from e2e32_official_compat import OfficialCompat
from e2e32_prepare import SOURCE,VSOURCE,METHODS

def expected_visual(method,layer):
 if method in ('Native','BridgeNative'):return 576
 if method=='FastV':return 576 if layer<3 else 192
 if method=='V2Drop':return 576 if layer<3 else 288 if layer<17 else 144 if layer<22 else 1
 if method=='ApET':return 288 if layer<16 else 96
 if method=='DUET':return 307 if layer<16 else 153 if layer<24 else 0
 if method=='SparseVLM':return 576 if layer<3 else 309 if layer<7 else 204 if layer<16 else 113
 return 576 if method=='SVH-R' and 11<=layer<=24 else 0

def kv_snapshot(cache,ctrl,layout,method,steps):
 n=layout.length;image=set(layout.image);rows=[];storage_all={}
 special=method in ('SVH','SVH-R')
 if not special:
  assert len(cache.key_cache)==len(cache.value_cache)==32;pairs=list(zip(cache.key_cache,cache.value_cache));meta=None
 else:assert len(cache.entries)==32;meta=cache.audit()['layers'];pairs=cache.entries
 for i,pair in enumerate(pairs):
  absent=special and 2<=i<=5;visual=expected_visual(method,i)
  expected_tokens=0 if absent else n-576+visual+steps-1
  if absent:
   assert pair is None and cache.used[i]==0
   rows.append(dict(layer=i,used_tokens=0,capacity_tokens=0,visual_tokens=0,live_bytes=0,storage_bytes=0,K_shape=None,V_shape=None,dtype=None));continue
  k,v=pair;assert k.is_cuda and v.is_cuda and k.dtype==v.dtype==torch.float16
  used=int(k.shape[-2]) if meta is None else cache.used[i]
  assert used==expected_tokens,(method,i,used,expected_tokens)
  assert list(k.shape)==list(v.shape)==[1,32,int(k.shape[-2]),128]
  positions=None
  if meta is not None:positions=meta[i]['original_positions']
  elif method in ('Native','BridgeNative'):positions=list(range(n+steps-1))
  elif method=='V2Drop' and ctrl.trace and i in ctrl.kv_positions:
   positions=ctrl.kv_positions[i]
  elif method=='FastV':positions=(list(range(n)) if i<3 else ctrl.active.tolist())+list(range(n,n+steps-1))
  if positions is not None:assert len(positions)==used and sum(p in image for p in positions)==visual
  if method in ('ApET','DUET','SparseVLM'):assert ctrl.layer_visual[i]==visual
  live=sum(x.shape[0]*x.shape[1]*used*x.shape[-1]*x.element_size() for x in pair)
  storage={x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for x in pair};storage_all.update(storage)
  assert live==used*2*32*128*2
  rows.append(dict(layer=i,used_tokens=used,capacity_tokens=int(k.shape[-2]),visual_tokens=visual,live_bytes=live,storage_bytes=sum(storage.values()),K_shape=list(k.shape),V_shape=list(v.shape),dtype=str(k.dtype),position_convention='compact (merged/CLS visual representations)' if method in ('ApET','DUET') else 'original gathered positions',original_positions_sha256=None if positions is None else hashlib.sha256(json.dumps(positions).encode()).hexdigest()))
 return dict(cache_class=type(cache).__name__,layers=rows,live_bytes=sum(r['live_bytes'] for r in rows),storage_bytes=sum(storage_all.values()),visual_tokens_sum=sum(r['visual_tokens'] for r in rows),cached_decode_inputs=steps-1)

class Observer:
 """Only smoke: actual Hidden, mask, position/RoPE and QKV launch tracing."""
 def __init__(self,a,layout,method,ctrl):self.a=a;self.layout=layout;self.method=method;self.ctrl=ctrl;self.step=0;self.rows={};self.handles=[]
 def enabled(self):return self.step in (0,1,2,3,4,5,6,7,8,31)
 def layer(self,i,args,kw):
  if not self.enabled():return
  h=args[0];r=self.rows.setdefault((self.step,i),dict(step=self.step,layer=i,hidden_rows=int(h.shape[1]),q_rows=0,k_rows=0,v_rows=0))
  r['hidden_rows']=int(h.shape[1]);pos=kw['position_ids'][0].tolist();r['positions']=pos if self.step else [pos[0],pos[-1]]
  if self.method in ('ApET','DUET','SparseVLM','BridgeNative'):
   visual=expected_visual(self.method,i);length=self.layout.length-576+visual
   want=list(range(length)) if self.step==0 else [length+self.step-1]
  else:
   want=list(range(self.layout.length)) if self.step==0 else [self.layout.length+self.step-1]
   if self.step==0 and self.method=='FastV' and i>=3:want=self.ctrl.active.tolist()
   if self.step==0 and self.method=='V2Drop':want=self.ctrl.active.tolist()
   if self.step==0 and self.method in ('SVH','SVH-R') and i>=(20 if self.method=='SVH' else 25):want=list(self.layout.text)
  assert pos==want and h.shape[1]==len(want),(self.method,self.step,i,pos[:2],want[:2])
  expected=self.a.model.language_model.model.rotary_emb(h,kw['position_ids'])
  assert all(torch.equal(x,y) for x,y in zip(kw['position_embeddings'],expected))
  r['RoPE_exact']=True;r['mask_shape']=None if kw.get('attention_mask') is None else list(kw['attention_mask'].shape)
 def projection(self,i,kind,args):
  if self.enabled():
   # DUET ranks using an EXTRA next-layer QKV pass before its forward hook.
   r=self.rows.setdefault((self.step,i),dict(step=self.step,layer=i,hidden_rows=None,q_rows=0,k_rows=0,v_rows=0))
   r[kind+'_rows']+=int(args[0].shape[-2])
 @contextmanager
 def context(self):
  try:
   for i,l in enumerate(self.a.layers):
    self.handles.append(l.register_forward_pre_hook(lambda m,args,kw,i=i:self.layer(i,args,kw),with_kwargs=True))
    for op in ('q','k','v'):self.handles.append(getattr(l.self_attn,op+'_proj').register_forward_pre_hook(lambda m,args,i=i,op=op:self.projection(i,op,args)))
   yield self
  finally:
   for h in self.handles:h.remove()
 def finish(self):
  for (step,i),r in self.rows.items():
   assert r['hidden_rows'] is not None
   skip=self.method in ('SVH','SVH-R') and 2<=i<=5
   q=(self.layout.length-1 if step==0 else 0) if skip else r['hidden_rows']
   kv=(self.layout.length if step==0 else 0) if skip else r['hidden_rows']
   if self.method=='DUET' and step==0 and i in (16,24):
    extra=self.layout.length-576+(307 if i==16 else 153);q+=extra;kv+=extra
   assert r['q_rows']==q and r['k_rows']==r['v_rows']==kv,(self.method,step,i,r,q,kv)
  assert len(self.rows)==320;return list(self.rows.values())

@torch.inference_mode()
def generate(a,layout,method,smoke=False,length=32):
 assert method in METHODS+['BridgeNative','FastVNoop','V2DropNoop'] and len(layout.image)==576
 requested=method
 if requested.endswith('Noop'):method='Native'
 n=layout.length;device=a.inputs['input_ids'].device
 events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(length)]
 cache=ctrl=out=None;tokens=[];finite=[];vision_calls=[0];smoke_cache={};observer=None;first_logits=None
 original=a.model.get_image_features
 def vision(*args,**kwargs):vision_calls[0]+=1;return original(*args,**kwargs)
 torch.cuda.synchronize();start=time.perf_counter()
 try:
  positions=torch.arange(n+length-1,device=device);mask=torch.ones((1,n+length-1),dtype=torch.long,device=device)
  if method=='FastV' or requested=='FastVNoop':ctrl=FastV(a.model,SOURCE,layout.image,n,enabled=requested!='FastVNoop',trace=smoke)
  elif method=='V2Drop' or requested=='V2DropNoop':ctrl=V2Drop(a,layout.image,VSOURCE,enabled=requested!='V2DropNoop',trace=smoke,reference=smoke,selection_timing=True)
  elif method in ('SVH','SVH-R'):
   layout=replace(layout,cutoff=20 if method=='SVH' else 25);ctrl=MiddleWriter(a,layout,tuple(range(11,25)) if method=='SVH-R' else (),length,False,False)
  elif method in ('ApET','DUET','SparseVLM','BridgeNative'):ctrl=OfficialCompat(a,layout,method,trace=smoke)
  if smoke:observer=Observer(a,layout,method,ctrl)
  with ctrl.context() if ctrl else nullcontext():
   new=method in ('ApET','DUET','SparseVLM','BridgeNative')
   with observer.context() if observer else nullcontext(),nullcontext() if new else patch.object(a.model,'get_image_features',vision):
    for step in range(length):
     if observer:observer.step=step
     kw=dict(a.inputs) if step==0 else dict(input_ids=tokens[-1].reshape(1,1),attention_mask=mask[:,:n+step])
     kw.update(use_cache=True,return_dict=True,past_key_values=cache,cache_position=positions[:n] if step==0 else positions[n+step-1:n+step],logits_to_keep=1)
     events[step][0].record();out=a.model(**kw);last=out.logits[:,-1];token=last.argmax(-1);tokens.append(token);finite.append(torch.isfinite(last).all());events[step][1].record();cache=out.past_key_values
     if step==0:
      torch.cuda.synchronize();first=time.perf_counter()
      if smoke:first_logits=last.detach().cpu().tolist()
     if smoke and step in (0,1,8,31):smoke_cache[str(step+1)]=kv_snapshot(cache,ctrl,layout,method,step+1)
     del out,last;out=None
  torch.cuda.synchronize();end=time.perf_counter()
  assert torch.stack(finite).all().item(),'Nonfinite logits'
  calls=ctrl.vision_calls if new else vision_calls[0];ids=torch.cat(tokens).tolist();assert len(ids)==length and calls==1
  kv=kv_snapshot(cache,ctrl,layout,method,length);assert kv['storage_bytes']==kv['live_bytes']
  ms=[b.elapsed_time(e) for b,e in events]
  result=dict(method=requested,token_ids=ids,output_tokens=length,real_decode_calls=length-1,prefill_calls=1,vision_forward_calls=calls,all_logits_finite=True,
   TTFT_ms=(first-start)*1000,E2E_ms=(end-start)*1000,TPOT_ms=(end-first)*1000/(length-1),CUDA_Event_Decode_TPOT_ms=sum(ms[1:])/(length-1),
   step_event_ms=ms,cache=kv,logits_to_keep=1,use_cache=True,smoke=smoke,formal_layer_observer_hooks=smoke)
  if ctrl is not None:
   result['pruning_events']=getattr(ctrl,'events',None);result['selector_provenance']=getattr(ctrl,'provenance',None)
  if smoke:result.update(smoke_layer_shapes=observer.finish(),smoke_cache_checkpoints=smoke_cache,first_logits=first_logits)
  return result
 finally:del cache,ctrl,out,tokens,finite

"""Call the unchanged frozen32 timer; choose the same physical quality classes.

Only controller factory and post-timer KV/Smoke inspection are scoped patches.
No decoder/Attention/Greedy loop/timing code is copied or modified.
"""
from contextlib import ExitStack
from unittest.mock import patch
import hashlib,json,torch
import e2e32_runtime as ref
from s55_ablation_engine import NoSelectiveTransfer,NoAttentionBypass
ORIGINAL_FACTORY=ref.MiddleWriter
ORIGINAL_KV=ref.kv_snapshot
ORIGINAL_OBSERVER=ref.Observer
ORIGINAL_GENERATE=ref.generate

def restored_attention_kv(cache,ctrl,layout,method,steps):
 assert method=='SVH' and not ctrl.physical_skip and not ctrl.zero_reference
 assert not ctrl.cache.absent
 assert len(cache.entries)==32
 meta=cache.audit()['layers'];rows=[];storage_all={};want=list(layout.text)+list(range(layout.length,layout.length+steps-1))
 for i,(k,v) in enumerate(cache.entries):
  used=cache.used[i];positions=meta[i]['original_positions']
  assert positions==want and used==len(want)
  assert k.is_cuda and v.is_cuda and k.dtype==v.dtype==torch.float16
  assert list(k.shape)==list(v.shape)==[1,32,int(k.shape[-2]),128] and k.shape[-2]>=used
  live=sum(x.shape[0]*x.shape[1]*used*x.shape[-1]*x.element_size() for x in (k,v))
  storage={x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for x in (k,v)};storage_all.update(storage)
  assert live==used*2*32*128*2
  rows.append(dict(layer=i,used_tokens=used,capacity_tokens=int(k.shape[-2]),visual_tokens=0,live_bytes=live,storage_bytes=sum(storage.values()),K_shape=list(k.shape),V_shape=list(v.shape),dtype=str(k.dtype),position_convention='original gathered positions',original_positions_sha256=hashlib.sha256(json.dumps(positions).encode()).hexdigest()))
 result=dict(cache_class=type(cache).__name__,layers=rows,live_bytes=sum(r['live_bytes'] for r in rows),storage_bytes=sum(storage_all.values()),visual_tokens_sum=0,cached_decode_inputs=steps-1)
 return result

def observer_for(condition):
 class ComponentObserver(ORIGINAL_OBSERVER):
  def layer(self,i,args,kw):
   if not self.enabled():return
   h=args[0];r=self.rows.setdefault((self.step,i),dict(step=self.step,layer=i,hidden_rows=int(h.shape[1]),q_rows=0,k_rows=0,v_rows=0))
   r['hidden_rows']=int(h.shape[1]);pos=kw['position_ids'][0].tolist();r['positions']=pos if self.step else [pos[0],pos[-1]]
   want=list(range(self.layout.length)) if self.step==0 else [self.layout.length+self.step-1]
   if self.step==0 and condition=='wo_bypass' and i>=20:want=list(self.layout.text)
   assert pos==want and h.shape[1]==len(want),(condition,self.step,i)
   rotary=self.a.model.language_model.model.rotary_emb(h,kw['position_ids'])
   assert all(torch.equal(x,y) for x,y in zip(kw['position_embeddings'],rotary))
   r['RoPE_exact']=True;r['mask_shape']=None if kw.get('attention_mask') is None else list(kw['attention_mask'].shape)
  def finish(self):
   for (step,i),r in self.rows.items():
    assert r['hidden_rows'] is not None
    skip=condition=='wo_selective' and 2<=i<=5
    q=(self.layout.length-1 if step==0 else 0) if skip else r['hidden_rows']
    kv=(self.layout.length if step==0 else 0) if skip else r['hidden_rows']
    assert r['q_rows']==q and r['k_rows']==r['v_rows']==kv,(condition,step,i,r,q,kv)
   assert len(self.rows)==320
   return list(self.rows.values())
 return ComponentObserver

def generate(a,layout,condition,smoke=False):
 assert condition in ('Native','SVH','wo_selective','wo_bypass')
 try:
  with ExitStack() as stack:
   if condition in ('wo_selective','wo_bypass'):
    cls=NoSelectiveTransfer if condition=='wo_selective' else NoAttentionBypass
    stack.enter_context(patch.object(ref,'MiddleWriter',cls))
    if condition=='wo_bypass':stack.enter_context(patch.object(ref,'kv_snapshot',restored_attention_kv))
    if smoke:stack.enter_context(patch.object(ref,'Observer',observer_for(condition)))
   result=ORIGINAL_GENERATE(a,layout,'Native' if condition=='Native' else 'SVH',smoke=smoke)
  result['component_condition']=condition
  result['controller_factory']='None' if condition=='Native' else 'MiddleWriter' if condition=='SVH' else cls.__name__
  return result
 finally:
  assert ref.generate is ORIGINAL_GENERATE and ref.MiddleWriter is ORIGINAL_FACTORY
  assert ref.kv_snapshot is ORIGINAL_KV and ref.Observer is ORIGINAL_OBSERVER

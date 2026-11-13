"""Frozen Qwen SVH-R, with only its auxiliary capture dispatch bridged to SDPA.

The inherited Physical.forward, SelectedKV and Original.context are unchanged.
The historical builder hooks eager_attention_forward; HF SDPA bypasses that hook.
This scoped bridge routes SDPA through the historical capture, then the same
efficient-only backend used by Native/Original. No token/key set is modified.
"""
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from vlm_research.qwen3vl_svhr_search.physical import Physical
from vlm_research.qwen3vl_c3v2_physical.engine import SKIP

READ=tuple(range(19,36))

class EfficientSVHR(Physical):
 def __init__(self,model,layout,condition='C',max_tokens=128,audit=False):
  super().__init__(model,layout,condition,max_tokens,audit,layers=READ)
  self.building=False;self.checks=[];self.builder_checks=[];self.max_tokens=max_tokens
  self.text_list=self.text.tolist();self.Iset=set(self.I)

 def construct_late_KV(self,h,rotary,textpos,cp):
  assert not self.building and self.step==0
  backend=qm.ALL_ATTENTION_FUNCTIONS['sdpa']
  def route(module,q,k,v,attention_mask=None,**kw):
   return qm.eager_attention_forward(module,q,k,v,attention_mask,**kw)
  self.building=True
  try:
   with patch.object(qm,'eager_attention_forward',backend),patch.object(qm,'ALL_ATTENTION_FUNCTIONS',{'sdpa':route}):
    super().construct_late_KV(h,rotary,textpos,cp)
   assert set(self.cache.late)==set(range(22,36))
  finally:self.building=False

 def check_attention(self,module,q,k,v,mask,event):
  i=module.layer_idx;pre=self.step==0
  if self.building:
   assert pre and 22<=i<35
   qp=kp=list(range(self.I[-1]+1));role='VISUAL_KV_BUILDER'
  else:
   assert self.current[0]==i and self.current[1]==pre
   qp=(list(range(self.n-1)) if i in SKIP else list(range(self.n))) if pre and i<22 else self.text_list if pre else [self.n+self.step-1]
   kp=(list(range(self.n-1)) if i in SKIP else list(range(self.n))) if pre and i<22 else (list(range(self.n)) if i in self.read_layers else self.text_list) if pre else self.cache.positions[i]+list(range(self.n,self.n+self.step))
   role='MAIN'
   assert pre or i not in SKIP
   if i not in SKIP:
    pair=self.cache.entries[i];used=self.cache.used[i];assert used==len(self.cache.positions[i])+(0 if pre else self.step)
    assert pair[0].shape[-2]==len(self.cache.positions[i])+self.max_tokens-1
    if pre and i<22 and i not in self.read_layers:stored=[t.index_select(-2,self.text) for t in (k,v)]
    else:stored=[k,v]
    assert all(torch.equal(p[...,:used,:],t) for p,t in zip(pair,stored))
  assert q.shape[-2]==len(qp) and k.shape[-2]==v.shape[-2]==len(kp),(role,i,self.step,q.shape,k.shape,len(qp),len(kp))
  assert mask is not None and mask.shape[-2:]==(len(qp),len(kp))
  choose=[j for j,p in enumerate(qp) if p not in self.Iset]
  if not self.audit_enabled and len(choose)>8:choose=choose[:3]+choose[-5:]
  visual=[j for j,p in enumerate(qp) if p in self.Iset]
  if visual:choose += [visual[0],visual[len(visual)//2],visual[-1]]
  choose=sorted(set(choose));sel=torch.tensor(choose,device=q.device,dtype=torch.long)
  qs=torch.tensor([qp[j] for j in choose],device=q.device);ks=torch.tensor(kp,device=q.device)
  expected=ks[None,:]<=qs[:,None];ik=torch.isin(ks,self.image)
  if not self.building:
   blocked=(qs>=self.n-1) if i not in self.read_layers else torch.zeros_like(qs,dtype=torch.bool)
   if pre and i>=22:blocked |= (qs>self.I[-1])&(qs<self.n-1)
   expected &= ~(blocked[:,None]&ik[None,:])
  assert torch.equal(mask[0,0].index_select(0,sel)==0,expected),('KEY_PERMISSION_MISMATCH',role,i,self.step)
  event['role']=role
  row=dict(role=role,layer=i,step=self.step,query_rows=len(qp),key_rows=len(kp),sampled_queries=len(choose),key_permissions_exact=True,stored_KV_exact=role=='MAIN' and i not in SKIP,original_absolute_positions=True)
  (self.builder_checks if self.building else self.checks).append(row)

 def audit(self):
  a=super().audit();kv=a['kv'];assert not self.cache.late and self.deepstack==[0,1,2]
  assert [r['layer'] for r in self.builder_layers]==list(range(22,36))
  assert [r['layer'] for r in self.builder_checks]==list(range(22,35))
  assert len(self.checks)==30*self.step+6
  assert kv['persistent_image_bytes']==17*len(self.I)*2*8*128*2
  for r in kv['layers']:
   i=r['layer'];assert r['persistent_image_tokens']==(len(self.I) if i in self.read_layers else 0)
   assert r['persistent_text_tokens']==(0 if i in SKIP else len(self.text_list)+self.step-1)
  a.update(read_layers=list(READ),main_hidden_exit_before_layer=22,answer_attention_skipped_layers=list(SKIP),checks=self.checks,builder_checks=self.builder_checks,builder_layers=self.builder_layers,auxiliary_full_blocks=list(range(22,35)),auxiliary_last_layer_QKV_only=35,second_vision=False,second_deepstack=False,frozen_forward_and_SelectedKV_unchanged=True,capture_bridge_only=True,dense_attention_scores_materialized=False)
  return a

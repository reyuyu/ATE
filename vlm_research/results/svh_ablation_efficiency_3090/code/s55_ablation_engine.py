"""Two component switches around the unchanged physical MiddleWriter forward."""
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch
import torch
from vlm_research.c3v2_middle_reaccess import engine as middle
from vlm_research.ate_fulljoint_v2.engine import AnswerWriter,AnswerKV,LAYERS
from vlm_research.physical_handoff_fastpath import engine as base
from vlm_research.information_horizon_alignment import engine as quality

class NoSelectiveTransfer(middle.MiddleWriter):
 def __init__(self,a,layout,keep,cap,audit=False,dense=False):
  # Visual Hidden must remain through all32 Prefill layers. Persistent answer KV
  # is still physically text-only, with the frozen early answer bypass.
  super().__init__(a,replace(layout,cutoff=32),(),cap,audit,dense)
 def prefill_mask(self,layer):return self.prefill_early
 def record(self,out,weights,positions,shape,mask):
  AnswerWriter.record(self,out,weights,positions,shape,mask)
  if not self.collect or weights is None:return
  layer,prefill,_,original,_=self.current
  if prefill:
   assert shape[1]==self.layout.length
   if self.physical_skip and layer in LAYERS:
    assert weights.shape[-2]==self.layout.length-1
    answer_mass=0.
   else:
    visual=weights[0,:,-1,list(self.layout.image)];assert torch.count_nonzero(visual)==0
    answer_mass=float(visual.float().sum(-1).mean())
   relay_mass=float(weights[0,:,list(self.layout.relay),:][...,list(self.layout.image)].float().sum(-1).mean())
   assert relay_mass>0,'Relays must really read image at every Prefill layer'
  else:
   answer_mass=relay_mass=0.
   if self.dense:assert torch.count_nonzero(weights[...,list(self.layout.image)])==0
  self.visual_events.append(dict(layer=layer,prefill=prefill,answer_image_mass=answer_mass,relay_image_mass=relay_mass,image_hidden_present=prefill))

class NoAttentionBypass(middle.MiddleWriter):
 def __init__(self,a,layout,keep,cap,audit=False,dense=False):
  super().__init__(a,replace(layout,cutoff=20),(),cap,audit,dense)
  self.physical_skip=False;self.zero_reference=False
  # The restored layers need real text KV. Merely enabling their forward while
  # leaving SelectiveKV's absent-layer metadata would be incorrect.
  self.cache=(base.InplaceKV(len(a.layers),self.layout,'D1 Dense',cap,self.device)
              if dense else AnswerKV(len(a.layers),self.layout,'Fast-P2',cap,self.device,absent=()))

@contextmanager
def component(condition):
 if condition=='wo_selective':
  with patch.object(middle,'MiddleWriter',NoSelectiveTransfer):yield
 elif condition=='wo_bypass':
  with patch.object(middle,'MiddleWriter',NoAttentionBypass):yield
 else:yield

def generate(a,layout,condition,cap,capture=False,audit=False,dense=False,noop=False):
 assert condition in ('Native','SVH','wo_selective','wo_bypass')
 with component(condition):
  return quality.hf_generate(a,layout,'HF-Native' if condition=='Native' else 'Physical-C3',cap,capture=capture,noop=noop,audit=audit,dense=dense)

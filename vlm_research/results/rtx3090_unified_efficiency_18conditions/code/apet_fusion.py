"""Independent ApET+SVH coordinator. Official 288->96 kernels unchanged.
Physical removal at20; official compact RoPE at ApET stage16, unchanged upon exit.
Dense=True is solely a diagnostic oracle, never an efficiency method.
"""
from contextlib import ExitStack,contextmanager
from unittest.mock import patch
from dataclasses import replace
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from e2e32_official_compat import OfficialCompat
from vlm_research.ate_fulljoint_v2.engine import AnswerWriter,LAYERS
from vlm_research.fastv_svh.engine import TextOnlyKV
class ApETFusion(OfficialCompat):
 def __init__(self,a,layout,trace=False,dense=False):
  super().__init__(a,layout,'ApET',trace)
  self.dense=dense;self.cap=32;self.original_layout=layout
  left=layout.image[0];n=layout.length-288
  ids=tuple(range(n));im=tuple(range(left,left+288));text=tuple(v for v in ids if v not in set(im))
  self.compact_layout=replace(layout,length=n,image=im,text=text,relay=tuple(range(left+288,n-1)),question=(),cutoff=20)
  self.aw=AnswerWriter(a,self.compact_layout,'C3-V2',self.compact_layout.relay,32,trace)
  self.cache=TextOnlyKV(self.compact_layout,32,32);self.aw.cache=self.cache
  self.structure=[];self.shape_audit=[];self.visual=288
 def decoder_forward(self,decoder,input_ids=None,inputs_embeds=None,**kwargs):
  h=decoder.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
  pre=self.step==0;self.cache.prefill=pre;self.aw.step=self.step
  visual=288;pos=None;rope=None;mask=None
  for i,layer in enumerate(decoder.layers):
   if pre:
    if i==0 or i==16:
     n=h.shape[1];pos=torch.arange(n,device=h.device);rope=decoder.rotary_emb(h,pos[None])
     mask=h.new_full((1,1,n,n),torch.finfo(h.dtype).min).triu(1)
     mask[...,-1,self.original_layout.image[0]:self.original_layout.image[0]+visual]=-torch.inf
    if i==20:
     ti=torch.cat((torch.arange(self.layout.image[0],device=h.device),torch.arange(self.layout.image[0]+visual,h.shape[1],device=h.device)))
     if self.dense:
      relay=torch.arange(self.layout.image[0]+visual,h.shape[1]-1,device=h.device)
      mask[0,0,relay[:,None],torch.arange(self.layout.image[0],self.layout.image[0]+visual,device=h.device)[None,:]]=-torch.inf
     else:
      h=h.index_select(1,ti);pos=pos.index_select(0,ti);rope=tuple(v.index_select(1,ti) for v in rope);mask=mask.index_select(-2,ti).index_select(-1,ti);visual=0
    self.cache.text_index=torch.cat((torch.arange(self.layout.image[0],device=h.device),torch.arange(self.layout.image[0]+visual,h.shape[1],device=h.device))) if visual else torch.arange(h.shape[1],device=h.device)
   else:
    logical=self.original_layout.length-576+(288 if i<16 else 96)+self.step-1
    pos=torch.tensor([logical],device=h.device);rope=decoder.rotary_emb(h,pos[None]);mask=None
   self.aw.active_mask=None if mask is None else mask[...,:-1,:]
   self.aw.current=(i,pre,mask,pos,h.shape[1] if pre else self.cache.used[i]+1)
   if pre:self.structure.append(dict(layer=i,hidden_rows=h.shape[1],visual_rows=visual))
   if self.trace:self.shape_audit.append(dict(step=self.step,layer=i,positions=pos.tolist(),hidden_rows=h.shape[1],mask_shape=None if mask is None else list(mask.shape)))
   h=layer(h,attention_mask=mask,position_ids=pos[None],cache_position=pos,position_embeddings=rope,past_key_value=self.cache,use_cache=True)[0]
   if pre and i==15:
    pp,mm,hh,ll=self.prune(self.proxy,0,16,h,pos[None],None,None,None,None,None)
    assert hh.shape[1]==h.shape[1]-192 and torch.equal(pp,torch.arange(hh.shape[1],device=h.device)[None])
    self.events.append(dict(before_layer=16,visual_before=288,visual_after=96,official_AST_unmodified=True))
    h=hh;visual=96
  self.cache.logical_length=self.compact_layout.length+self.step;self.step+=1
  return BaseModelOutputWithPast(last_hidden_state=decoder.norm(h),past_key_values=self.cache)
 @contextmanager
 def context(self):
  with ExitStack() as stack:
   stack.enter_context(super().context())
   for i in LAYERS:
    att=self.a.layers[i].self_attn
    stack.enter_context(patch.object(att,'forward',lambda *args,att=att,**kw:self.aw.answer_rows(att,*args,**kw)))
   yield self

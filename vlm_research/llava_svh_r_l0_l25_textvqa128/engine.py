"""Independent terminal-cutoff extension; unchanged pinned MiddleWriter math."""
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from vlm_research.c3v2_middle_reaccess.engine import MiddleWriter
class TerminalMiddleWriter(MiddleWriter):
 def forward(self,input_ids=None,inputs_embeds=None,position_ids=None,cache_position=None,**kwargs):
  h=inputs_embeds if inputs_embeds is not None else self.decoder.embed_tokens(input_ids)
  prefill=self.step==0
  if prefill:self.physical_exit_events=[]
  positions=self.positions[:self.layout.length] if prefill else self.positions[self.layout.length+self.step-1:self.layout.length+self.step]
  if position_ids is None:position_ids=positions.unsqueeze(0)
  rotary=self.decoder.rotary_emb(h,position_ids)
  if self.audit_enabled:self.rotary.append(dict(position_ids=position_ids.detach().cpu().clone(),cos=rotary[0].detach().cpu().clone(),sin=rotary[1].detach().cpu().clone()))
  self.cache.prefill=prefill
  for i,layer in enumerate(self.a.layers):
   if prefill and self.physical and i==self.layout.cutoff:
    before=h.shape[1];h=h.index_select(1,self.text_index);positions=self.text_positions;position_ids=position_ids.index_select(-1,self.text_index);rotary=tuple(v.index_select(1,self.text_index) for v in rotary)
    self.physical_exit_events.append(dict(before_layer=i,before_rows=before,after_rows=h.shape[1],removed=before-h.shape[1],stage='before_language_block'))
   if prefill:mask=self.prefill_mask(i);key_count=h.shape[1]
   else:
    key_count=self.cache.used[i]+1
    mask=None if self.physical or i in self.keep else self.decode_mask[...,:key_count]
   self.current=(i,prefill,mask,positions,key_count)
   h=layer(h,attention_mask=mask,position_ids=position_ids,cache_position=positions,position_embeddings=rotary,past_key_value=self.cache,use_cache=True)[0]
   if self.audit_enabled and prefill:
    kept=self.text_index if h.shape[1]==self.layout.length else torch.arange(h.shape[1],device=h.device)
    self.hidden.append(h.index_select(1,kept).detach().float().cpu())
  # Cutoff32 has no following language block. Exit afterL31, before tokenwise norm.
  # All language blocks still execute their original visual Hidden computations.
  if prefill and self.physical and self.layout.cutoff==len(self.a.layers):
   before=h.shape[1];h=h.index_select(1,self.text_index)
   self.physical_exit_events.append(dict(before_layer=len(self.a.layers),before_rows=before,after_rows=h.shape[1],removed=before-h.shape[1],stage='after_L31_before_final_tokenwise_norm'))
  h=self.decoder.norm(h);self.cache.logical_length=self.layout.length+self.step;self.step+=1
  return BaseModelOutputWithPast(last_hidden_state=h,past_key_values=self.cache)

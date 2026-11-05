"""Direct historical quality-only execution, keeping historical controllers intact."""
import torch
from ..qwen3vl_keep_l0_21.engine import generate as historical_keep
from ..qwen3vl_visual_writing_necessity.engine import generate as historical_layer_writer
from ..qwen3vl_early_windows.experiment import hook_counts

def generate(model,processor,inputs,spec,condition,audit=False,noop=False):
 device=next(model.parameters()).device;x={k:v.to(device) for k,v in inputs.items()};before=hook_counts(model)
 if condition=='Keep-L0-L21':g,scores,observer,cache=historical_keep(model,processor,x,spec,noop=noop,audit=audit,trace=False)
 else:
  assert condition=='All-Writer';g,scores,observer,cache=historical_layer_writer(model,processor,x,spec,'All-Writer',noop=noop,audit=audit,trace=False)
 assert hook_counts(model)==before
 # Preserve historical token generation exactly; keep the current1000 scorer unchanged.
 raw=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True)
 result=dict(condition=condition,token_ids=g['token_ids'],text=raw,historical_stripped_text=g['text'],prediction=raw.rstrip('.').lower(),eos=g['eos'],truncated=g['truncated'],empty=not raw.strip(),generated_tokens=len(g['token_ids']),first_token=g['token_ids'][0],all_logits_finite=g['all_logits_finite'],source_audit=g['source_audit'],cache_lengths=g['cache_lengths'],writer_layers=g['writer_layers'],P1_Decode_image_off=g['P1_Decode_image_off'],hook_cleanup=True)
 return result,scores,cache

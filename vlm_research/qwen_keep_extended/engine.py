"""Same audited historical hard mask; only the inclusive final writer layer varies."""
import torch
from ..qwen3vl_visual_writing_necessity.engine import LayerWriterMask
from ..qwen3vl_query_role.engine import generate as native_generate
from ..qwen3vl_early_windows.experiment import hook_counts

def controller(model,spec,last_layer,noop=False):
 assert last_layer in [21,24,26]
 ctl=LayerWriterMask(model,spec,'All-Writer',noop=noop)
 ctl.condition=f'Keep-L0-L{last_layer}';ctl.open_layers=set(range(last_layer+1));return ctl

@torch.inference_mode()
def generate(model,processor,inputs,spec,last_layer,noop=False,audit=False):
 before=hook_counts(model);device=next(model.parameters()).device;x={k:v.to(device) for k,v in inputs.items()}
 try:
  with controller(model,spec,last_layer,noop) as ctl:
   g,scores,observer,cache=native_generate(model,processor,x,spec['roles'],native=True,audit=audit,trace=False)
  n,count=spec['layout']['input_length'],len(spec['roles']['I']);layers=list(range(last_layer+1))
  expected=[count*(1+(0 if i in layers else len(spec['roles']['R']))) for i in range(36)]
  assert len(ctl.positions)==len(g['token_ids']) and len(ctl.events)==36*len(g['token_ids']) and not ctl.handles
  for e in ctl.events:
   assert e['past']==(n+e['step']-1 if e['step'] else 0)
   assert e['blocked_connections_per_head']==(count if e['step'] else expected[e['layer']])
  raw=processor.tokenizer.decode(g['token_ids'],skip_special_tokens=True)
  g.update(condition=f'Keep-L0-L{last_layer}',writer_layers=layers,P1_Decode_image_off=not noop,text=raw,historical_stripped_text=g['text'],prediction=raw.rstrip('.').lower(),empty=not raw.strip(),first_token=g['token_ids'][0],source_audit=dict(blocked_probability_max=max(e['blocked_probability_max'] for e in ctl.events),row_sum_max_error=max(e['row_sum_max_error'] for e in ctl.events),layer_events=len(ctl.events),prefill_blocked_connections_per_layer=expected,decode_blocked_connections_per_layer=[count]*36,open_R_image_probability_max={str(e['layer']):e['open_R_image_probability_max'] for e in ctl.events if not e['step'] and e['layer'] in layers},actual_positions=ctl.positions,non_target_mask_exact=True,early_attention_normal=True,MLP_DeepStack_mrope_native=True,hook_cleanup=True,noop=noop))
  return g,scores,cache
 finally:assert hook_counts(model)==before

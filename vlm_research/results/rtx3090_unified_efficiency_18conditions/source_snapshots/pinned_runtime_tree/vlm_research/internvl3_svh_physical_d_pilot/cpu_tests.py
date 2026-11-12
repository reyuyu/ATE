"""Meaningful CPU equivalence test using a 28-layer tiny official Qwen2 in float64."""
import types,torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
from .engine import Control,CompactCache,layout
from .common import *
def main():
 torch.set_num_threads(2);torch.manual_seed(20261110)
 config=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=28,
  num_attention_heads=4,num_key_value_heads=2,attention_dropout=0,rope_theta=1000000.)
 config._attn_implementation='eager'
 lm=Qwen2ForCausalLM(config).double().eval()
 model=types.SimpleNamespace(language_model=lm)
 roles=dict(n=272,image=list(range(8,264)),R=list(range(264,271)),A_prefill=[271])
 ids=torch.randint(0,32,(1,272));embed=lm.get_input_embeddings()(ids)
 outputs={};controls={}
 with torch.inference_mode():
  for method in ['Native','No-op','Dense-D','Physical-D']:
   cache=CompactCache(roles,'cpu') if method=='Physical-D' else DynamicCache()
   scores=[];tokens=[];control=Control(model,method,roles,cache,True)
   with control:
    for step in range(9):
     control.step=step;control.prefill=step==0;control.masks={}
     if method=='Physical-D':cache.prefill=step==0
     pos=torch.arange(272) if step==0 else torch.tensor([272+step-1])
     args=dict(inputs_embeds=embed) if step==0 else dict(input_ids=tokens[-1].view(1,1))
     out=lm(**args,attention_mask=torch.ones((1,272+step),dtype=torch.long),position_ids=pos[None],
      past_key_values=cache,use_cache=True,cache_position=pos,logits_to_keep=1)
     scores.append(out.logits.clone());tokens.append(out.logits[0,-1].argmax(-1))
   control.validate(9);outputs[method]=scores;controls[method]=control
  assert all(torch.equal(a,b) for a,b in zip(outputs['Native'],outputs['No-op']))
  maximum=max(float((a-b).abs().max()) for a,b in zip(outputs['Dense-D'],outputs['Physical-D']))
  assert maximum<1e-12,maximum
  hd=0.
  for key,(pos,p) in controls['Physical-D'].hidden.items():
   dpos,d=controls['Dense-D'].hidden[key];look={int(x):i for i,x in enumerate(dpos)}
   aligned=d[:,[look[int(x)] for x in pos],:]
   hd=max(hd,float((aligned-p).abs().max()))
  assert hd<1e-12,hd
  pc=controls['Physical-D'];rows,live,capacity=layout(pc.cache,'Physical-D',roles,9)
  assert all(r['physical_length']==(0 if r['layer'] in range(2,6) else 24) for r in rows)
  assert live==capacity
  original=lm.model.forward
  try:
   with Control(model,'Physical-D',roles,CompactCache(roles,'cpu'),False):
    raise RuntimeError('intentional independent-request cleanup test')
  except RuntimeError:pass
  assert lm.model.forward==original
 result=dict(status='PASS',tiny_official_Qwen2_float64_layers=28,seed=20261110,
  native_noop_strict_equal=True,steps=9,real_decode=8,aligned_hidden_max_abs=hd,
  dense_physical_logits_max_abs=maximum,logical_RoPE_preserved=True,
  removed_image_rows=256,answer_only_skip=True,non_answer_prefill_equivalent=True,
  compact_live_equals_capacity=True,no_dense_backing_storage=True,
  skip_layer_KV_absent=True,fresh_cache_and_exception_cleanup=True)
 save(RUN/'cpu_tests.json',result);save(OUT/'cpu_tests.json',result);print(json.dumps(result))
if __name__=='__main__':main()

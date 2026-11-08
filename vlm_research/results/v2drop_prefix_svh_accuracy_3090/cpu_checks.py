"""Necessary synthetic HF 4.49 integration checks; no pretrained/GPU model."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1')
from pathlib import Path
from types import SimpleNamespace
import sys,json,torch
R=Path('/home/featurize/work/ATE');O=R/'vlm_research/results/v2drop_prefix_svh_accuracy_3090';sys.path[:0]=[str(O),str(R)]
from transformers import LlamaConfig,LlamaForCausalLM
from fusion_engine import PrefixSVH,DenseOracle,greedy,SELECT,FIXED_SELECT
from vlm_research.c3v2_middle_reaccess.engine import MiddleWriter
from vlm_research.physical_handoff_fastpath.engine import Layout
from vlm_research.physical_visual_handoff.engine import equivalence
torch.set_num_threads(1);torch.manual_seed(20261109)
class CPUModel(torch.nn.Module):
 def __init__(self):
  super().__init__();c=LlamaConfig(vocab_size=64,hidden_size=64,intermediate_size=128,num_hidden_layers=32,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=2048,attention_dropout=0.)
  c._attn_implementation='eager';self.language_model=LlamaForCausalLM(c).eval();self.generation_config=self.language_model.generation_config
 def forward(self,**kw):
  if 'inputs_embeds' in kw:kw.pop('input_ids',None)
  return self.language_model(**kw)
model=CPUModel();start=35;n=start+576+16;image=tuple(range(start,start+576));text=tuple(i for i in range(n) if i not in image);relay=tuple(range(start+576,n-1));layout=Layout(n,image,text,relay,relay,20)
ids=torch.randint(0,64,(1,n));embeds=torch.randn((1,n,64));a=SimpleNamespace(kind='llava',model=model,layers=model.language_model.model.layers,processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda t,**kw:' '.join(map(str,t)))),inputs=dict(input_ids=ids,inputs_embeds=embeds,attention_mask=torch.ones_like(ids)))
with torch.inference_mode():
 f=PrefixSVH(a,layout,3,True,True);fg,fl=greedy(a,layout,f,3,True,True)
 d=DenseOracle(a,layout,3,f.selection_events);dg,dl=greedy(a,layout,d,3,True,True)
 gate=equivalence('llava',dl,fl,d.hidden,f.hidden);assert gate['passed'] and fg['token_ids']==dg['token_ids'],gate
 disabled=PrefixSVH(a,layout,3,False,True);ug,ul=greedy(a,layout,disabled,3,True,True)
 frozen=MiddleWriter(a,layout,(),3,True);sg,sl=greedy(a,layout,frozen,3,True,True)
 assert ug['token_ids']==sg['token_ids'] and all(torch.equal(x,y) for x,y in zip(ul,sl))
 assert all(torch.equal(x,y) for x,y in zip(disabled.hidden,frozen.hidden))
 assert [e['visual_hidden'] for e in f.layer_events if e['prefill']]==[576]*3+[288]*14+[144]*3+[0]*12
 kv=f.cache.audit();assert kv['retained_image_layers']==[]
 for row in kv['layers']:
  assert row['original_positions']==([] if row['layer'] in (2,3,4,5) else list(text)+[n,n+1])
 assert [s['layer'] for s in f.selection_events]==[3,17]
 for st in [5,35]:
  h=torch.randn(1,st+576+7,32);prev=torch.randn_like(h)
  for layer in [3,17]:
   keep,sel,scores=SELECT(h,prev,st,576,layer)
   pad=h.new_zeros(1,35,32);_,rs,rv=FIXED_SELECT(torch.cat([pad,h[:,st:]],1),torch.cat([pad,prev[:,st:]],1),35,576,layer)
   assert torch.equal(rs-35,sel-st) and torch.equal(rv,scores)
   assert set(keep.tolist())-set(range(st,st+576))==set(range(st))|set(range(st+576,st+576+7))
assert not torch.cuda.is_initialized()
result=dict(status='PASS',scope='CPU synthetic randomly initialized 32-layer HF Llama, not GPU/7B validation',float32_dense_gate=gate,disabled_pruning_frozen_SVH_bit_exact=True,physical_hidden_576_288_144_0=True,original_positions_and_compact_KV=True,official_selector_starts_5_and_35_exact=True,CUDA_initialized=False)
(O/'cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

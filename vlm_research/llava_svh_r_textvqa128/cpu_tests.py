import torch
from types import SimpleNamespace
from contextlib import nullcontext
from transformers import LlamaConfig,LlamaForCausalLM
from common import KEEP
from vlm_research.physical_handoff_fastpath.engine import Layout,FastHandoff
from vlm_research.c3v2_middle_reaccess.engine import MiddleWriter
@torch.inference_mode()
def test():
 torch.manual_seed(41);torch.set_num_threads(1)
 cfg=LlamaConfig(vocab_size=97,hidden_size=64,intermediate_size=128,num_hidden_layers=32,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=512,attention_dropout=0.)
 cfg._attn_implementation='eager';m=LlamaForCausalLM(cfg).double().eval();ids=torch.arange(15).reshape(1,-1);ids[0,2:8]=3
 a=SimpleNamespace(kind='llava',layers=m.model.layers,model=SimpleNamespace(language_model=m),inputs={'input_ids':ids})
 layout=Layout(15,tuple(range(2,8)),tuple([0,1]+list(range(8,15))),tuple(range(8,14)),tuple(range(8,11)),25)
 def execute(control):
  logits=[];h=[];cache=None
  with control.context() if control else nullcontext():
   for step in range(9):
    out=m(input_ids=ids if not step else torch.tensor([[11]]),past_key_values=cache,use_cache=True,return_dict=True,cache_position=torch.arange(15) if not step else torch.tensor([14+step]),logits_to_keep=1)
    assert torch.isfinite(out.logits).all();logits.append(out.logits.clone());cache=out.past_key_values
  return logits,control.hidden if control else [],control.cache.audit() if control else None
 native,_,_=execute(None);noop,_,_=execute(FastHandoff(a,layout,'Native-Noop',32,False))
 assert all(torch.equal(x,y) for x,y in zip(native,noop))
 dc=MiddleWriter(a,layout,KEEP,32,True,True);pc=MiddleWriter(a,layout,KEEP,32,True,False)
 d,dh,dk=execute(dc);p,ph,pk=execute(pc)
 maxlog=max((x-y).abs().max().item() for x,y in zip(d,p));maxh=max((x-y).abs().max().item() for x,y in zip(dh,ph))
 assert maxlog<1e-6 and maxh<1e-10,(maxlog,maxh)
 assert len(pk['layers'])==32
 for i,r in enumerate(pk['layers']):
  if 2<=i<=5:assert r['used_tokens']==0 and pc.cache.entries[i] is None;continue
  positions=(list(range(15)) if i in KEEP else list(layout.text))+list(range(15,23))
  assert r['original_positions']==positions and r['used_tokens']==len(positions)
 assert len(pc.hidden)==32 and all(x.shape[1]==9 for x in pc.hidden)
 assert all(e['blocked_probability_max']==0 for e in pc.events)
 assert all(e['computed_queries']==0 for e in pc.events if 2<=e['layer']<=5 and not e['prefill'])
 return dict(status='PASS',official_32_layers=True,Native_NoOp_strict_equal=True,CPU_softmax_dtype='frozen official float32 softmax even for float64 mock; 1e-6 CPU tolerance',Dense_Physical_logits_MaxAbs=maxlog,Dense_Physical_textHidden_MaxAbs=maxh,prefill_plus_decode_calls=9,compact_KV_original_positions=True,cutoff25_text_rows=True,skip2to5_query_compute_absent=True,all_forbidden_probabilities_zero=True)
if __name__=='__main__':
 from common import write,OUT
 write(OUT/'cpu_tests.json',test());print((OUT/'cpu_tests.json').read_text())

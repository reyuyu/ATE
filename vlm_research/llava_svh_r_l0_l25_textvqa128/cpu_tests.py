from common import KEEP
import torch
from types import SimpleNamespace
from contextlib import nullcontext
from transformers import LlamaConfig,LlamaForCausalLM
from vlm_research.physical_handoff_fastpath.engine import Layout,FastHandoff
from vlm_research.c3v2_middle_reaccess.engine import MiddleWriter
from engine import TerminalMiddleWriter
@torch.inference_mode()
def test():
 torch.manual_seed(41);torch.set_num_threads(1)
 cfg=LlamaConfig(vocab_size=97,hidden_size=64,intermediate_size=128,num_hidden_layers=32,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=512,attention_dropout=0.);cfg._attn_implementation='eager'
 m=LlamaForCausalLM(cfg).double().eval();ids=torch.arange(15).reshape(1,-1);ids[0,2:8]=3
 a=SimpleNamespace(kind='llava',layers=m.model.layers,model=SimpleNamespace(language_model=m),inputs={'input_ids':ids})
 def layout(cut):return Layout(15,tuple(range(2,8)),tuple([0,1]+list(range(8,15))),tuple(range(8,14)),tuple(range(8,11)),cut)
 def execute(control):
  logits=[];cache=None
  with control.context() if control else nullcontext():
   for step in range(9):
    out=m(input_ids=ids if not step else torch.tensor([[11]]),past_key_values=cache,use_cache=True,return_dict=True,cache_position=torch.arange(15) if not step else torch.tensor([14+step]),logits_to_keep=1)
    assert torch.isfinite(out.logits).all();logits.append(out.logits.clone());cache=out.past_key_values
  return logits
 old=MiddleWriter(a,layout(25),tuple(range(11,25)),32,True,False);new=TerminalMiddleWriter(a,layout(25),tuple(range(11,25)),32,True,False)
 x,y=execute(old),execute(new);assert all(torch.equal(p,q) for p,q in zip(x,y)) and all(torch.equal(p,q) for p,q in zip(old.hidden,new.hidden))
 native=execute(None);noop=execute(FastHandoff(a,layout(32),'Native-Noop',32,False));assert all(torch.equal(p,q) for p,q in zip(native,noop))
 d=TerminalMiddleWriter(a,layout(26),KEEP,32,True,True);p=TerminalMiddleWriter(a,layout(26),KEEP,32,True,False)
 dl=execute(d);pl=execute(p);maxlog=max((x-y).abs().max().item() for x,y in zip(dl,pl));maxhidden=max((x-y).abs().max().item() for x,y in zip(d.hidden,p.hidden))
 assert maxlog<1e-6 and maxhidden<1e-10
 assert p.physical_exit_events==[dict(before_layer=26,before_rows=15,after_rows=9,removed=6,stage='before_language_block')]
 audit=p.cache.audit()
 for i,r in enumerate(audit['layers']):
  if 2<=i<=5:assert r['used_tokens']==0 and p.cache.entries[i] is None;continue
  positions=(list(range(15)) if i in KEEP else list(layout(26).text))+list(range(15,23))
  assert positions==r['original_positions']
 assert all(e['blocked_probability_max']==0 for e in p.events)
 assert all(e['computed_queries']==0 for e in p.events if 2<=e['layer']<=5 and not e['prefill'])
 return dict(status='PASS',legacy_cutoff25_regression_torch_equal=True,Native_NoOp_strict_equal=True,L0_L25_cutoff26_actual_exit=True,hidden_MaxAbs=maxhidden,logit_MaxAbs=maxlog,CPU_tolerance='1e-6 due frozen internalFP32 softmax',cache_positions_and_skip=True,all32_language_blocks_run=True)

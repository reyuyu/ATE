"""CPU-only byte-path regression for unchanged nonterminal cutoffs26/27."""
import os,json,types
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
from pathlib import Path
from transformers import Qwen2Config,Qwen2ForCausalLM
from vlm_research.internvl3_svh_r_deep_exit_textvqa import engine as old
from . import engine as new
def main():
 torch.set_num_threads(1);torch.manual_seed(20261112)
 cfg=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=28,num_attention_heads=4,num_key_value_heads=2,attention_dropout=0,rope_theta=1000000.);cfg._attn_implementation="eager"
 lm=Qwen2ForCausalLM(cfg).double().eval();model=types.SimpleNamespace(language_model=lm);roles=dict(n=272,image=list(range(8,264)),R=list(range(264,271)),A_prefill=[271]);ids=torch.randint(0,32,(1,272));outrows=[]
 with torch.inference_mode():
  embed=lm.get_input_embeddings()(ids)
  for K,cutoff in [(list(range(20,26)),26),(list(range(21,27)),27)]:
   runs=[]
   for e in [old,new]:
    cache=e.CompactCache(roles,K,"cpu",cutoff=cutoff,audit=True);control=e.Control(model,True,roles,cache,K,True,cutoff=cutoff);scores=[]
    with control:
     for step in range(9):
      control.step=step;control.prefill=step==0;control.masks={};cache.prefill=step==0
      pos=torch.arange(272) if step==0 else torch.tensor([272+step-1]);args=dict(inputs_embeds=embed) if step==0 else dict(input_ids=torch.tensor([[step%31+1]]))
      x=lm(**args,attention_mask=torch.ones((1,272+step),dtype=torch.long),position_ids=pos[None],past_key_values=cache,use_cache=True,cache_position=pos,logits_to_keep=1);scores.append(x.logits.clone())
    control.validate(9);runs.append((scores,cache.original_positions))
   assert all(torch.equal(a,b) for a,b in zip(runs[0][0],runs[1][0])) and runs[0][1]==runs[1][1]
   outrows.append(dict(cutoff=cutoff,keep=K,steps=9,real_decode=8,old_new_logits_torch_equal=True,old_new_KV_positions_equal=True))
 assert not torch.cuda.is_initialized()
 p=Path(os.environ["ATE_DEEP_RUN"])/"results/nonterminal_regression.json";p.write_text(json.dumps(dict(status="PASS",checks=outrows,CUDA_initialized=False),indent=2)+"\n");print("NONTERMINAL_REGRESSION_PASS")
if __name__=="__main__":main()

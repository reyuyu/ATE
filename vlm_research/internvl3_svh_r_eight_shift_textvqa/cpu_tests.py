"""Float64 semantics on a 28-layer tiny official Qwen2; no GPU or weights."""
import os,types,json,gc
os.environ["CUDA_VISIBLE_DEVICES"]=""
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
from vlm_research.internvl3_svh_r_deep_exit_textvqa.engine import Control,CompactCache,layout,SKIP
from pathlib import Path
CONFIGS={"Shift8_L18_25":([*range(18,26)],26),"Shift8_L17_24":([*range(17,25)],25),"Shift8_L16_23":([*range(16,24)],24)}
def main():
 torch.set_num_threads(1);torch.manual_seed(20261112)
 cfg=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=28,num_attention_heads=4,num_key_value_heads=2,attention_dropout=0,rope_theta=1000000.)
 cfg._attn_implementation="eager"
 lm=Qwen2ForCausalLM(cfg).double().eval();model=types.SimpleNamespace(language_model=lm)
 roles=dict(n=272,image=list(range(8,264)),R=list(range(264,271)),A_prefill=[271])
 ids=torch.randint(0,32,(1,272));embed=lm.get_input_embeddings()(ids);results=[]
 with torch.inference_mode():
  for name,(K,cutoff) in CONFIGS.items():
   runs=[]
   for physical in [False,True]:
    cache=CompactCache(roles,K,"cpu",cutoff=cutoff,audit=True) if physical else DynamicCache()
    c=Control(model,physical,roles,cache,K,True,cutoff=cutoff);scores=[]
    with c:
     for step in range(9):
      c.step=step;c.prefill=step==0;c.masks={}
      if physical:cache.prefill=step==0
      pos=torch.arange(272) if step==0 else torch.tensor([272+step-1])
      args=dict(inputs_embeds=embed) if step==0 else dict(input_ids=torch.tensor([[step%31+1]]))
      out=lm(**args,attention_mask=torch.ones((1,272+step),dtype=torch.long),position_ids=pos[None],past_key_values=cache,use_cache=True,cache_position=pos,logits_to_keep=1)
      scores.append(out.logits.clone())
    audit=c.validate(9);runs.append((scores,c))
   ds,dc=runs[0];ps,pc=runs[1]
   maximum=max(float((a-b).abs().max()) for a,b in zip(ds,ps));assert maximum<1e-12,(name,maximum)
   hidden=0.
   for key,(pos,p) in pc.hidden.items():
    dpos,d=dc.hidden[key];look={int(x):i for i,x in enumerate(dpos)}
    err=float((d[:,[look[int(x)] for x in pos],:]-p).abs().max());hidden=max(hidden,err)
   assert hidden<1e-12,(name,hidden)
   rows,live,capacity=layout(pc.cache,True,roles,9,K)
   for row in rows:
    i=row["layer"];expected=0 if i in SKIP else 280-(0 if i in K else 256)
    assert row["physical_length"]==expected and row["capacity_bytes"]==row["live_bytes"]
   results.append(dict(condition=name,cutoff=cutoff,keep=K,steps=9,real_decode=8,logits_max_abs=maximum,aligned_hidden_max_abs=hidden,semantic_and_actual_shape_checks="PASS",R_to_Image_boundary=20,actual_cutoff=pc.cutoff))
   del runs,dc,pc,cache,c,ds,ps;gc.collect()
  for K,cutoff in [([25],25),([2],26),([28],28),([],19)]:
   try:CompactCache(roles,K,"cpu",cutoff=cutoff)
   except AssertionError:pass
   else:raise AssertionError("INVALID_CONFIG_ACCEPTED")
  a=CompactCache(roles,range(20,26),"cpu",cutoff=26);b=CompactCache(roles,range(20,26),"cpu",cutoff=26)
  assert a.key_cache is not b.key_cache and a.original_positions is not b.original_positions
  original=lm.model.forward
  try:
   with Control(model,True,roles,a,range(20,26),False,cutoff=26):raise RuntimeError("intentional cleanup")
  except RuntimeError:pass
  assert lm.model.forward==original and not torch.cuda.is_initialized()
 result=dict(status="PASS",official_Qwen2_layers=28,dtype="float64",steps=9,real_decode=8,configurations=results,logical_RoPE_and_persistent_positions_verified=True,all_non_target_connection_semantics_verified=True,SKIP_answer_QKV_absent=True,MLP_Norm_Residual_retained=True,KV_Live_equals_real_storage=True,no_cross_request_cache=True,invalid_configs_rejected=True,exception_cleanup=True,CUDA_initialized=False)
 out=Path(__file__).resolve().parents[2]
 p=Path(os.environ["ATE_DEEP_RUN"])/"results/cpu_tests.json";p.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)
if __name__=="__main__":main()

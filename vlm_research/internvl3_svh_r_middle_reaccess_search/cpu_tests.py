"""Actual tiny 28-layer Qwen2 CPU float64 checks, plus cache/cleanup invariants."""
from .common import *
from . import engine as e
import torch,traceback
from unittest.mock import patch
import torch.nn.functional as F
from transformers import Qwen2Config,Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
torch.set_num_threads(2)
class Wrapper(torch.nn.Module):
 def __init__(self,lm):super().__init__();self.language_model=lm
@torch.inference_mode()
def run(model,emb,roles,K,mode):
 cache=e.CompactCache(roles,K,"cpu") if mode=="physical" else DynamicCache()
 control=e.Control(model,mode=="physical",roles,cache,K,True) if mode in ("physical","dense") else e.base.Control(model,mode,roles,cache,True)
 scores=[];layouts=[];before=e.frozen.hooks_count(model)
 with control:
  for step in range(4):
   control.step=step;control.prefill=step==0;control.masks={}
   if mode=="physical":cache.prefill=step==0
   n=roles["n"];pos=torch.arange(n) if step==0 else torch.tensor([n+step-1])
   args={"inputs_embeds":emb} if step==0 else {"input_ids":torch.tensor([[7+step]])}
   out=model.language_model(**args,attention_mask=torch.ones(1,n+step,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
   scores.append(out.logits[0,-1].clone())
   if mode in ("physical","dense"):
    rows,live,cap=e.layout(cache,mode=="physical",roles,step+1,K);layouts.append((rows,live,cap))
    for x in rows:
     expected=0 if mode=="physical" and x["layer"] in e.SKIP else n+step-(256 if mode=="physical" and x["layer"] not in K else 0)
     assert x["physical_length"]==expected
     assert x["live_bytes"]<=x["capacity_bytes"]
 control.validate(4);assert before==e.frozen.hooks_count(model)
 return scores,control,layouts
def main():
 try:
  torch.manual_seed(SEED)
  cfg=Qwen2Config(vocab_size=80,hidden_size=32,intermediate_size=64,num_hidden_layers=28,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024,tie_word_embeddings=False)
  cfg._attn_implementation="eager";cfg.sliding_window=None
  native_softmax=F.softmax
  def mock_softmax(x,dim=None,_stacklevel=3,dtype=None):
   return native_softmax(x,dim=dim,_stacklevel=_stacklevel,dtype=x.dtype if x.dtype==torch.float64 else dtype)
  softmax_patch=patch("transformers.models.qwen2.modeling_qwen2.nn.functional.softmax",mock_softmax);softmax_patch.start()
  model=Wrapper(Qwen2ForCausalLM(cfg).double().eval())
  roles={"n":268,"image":list(range(5,261)),"R":list(range(261,267)),"A_prefill":[267],"image_token_id":79}
  emb=torch.randn(1,268,32,dtype=torch.float64)
  native,_,_=run(model,emb,roles,[],"Native");noop,_,_=run(model,emb,roles,[],"No-op")
  assert all(torch.equal(a,b) for a,b in zip(native,noop))
  records=[]
  for name,K in SPACE.items():
   dense,dc,_=run(model,emb,roles,K,"dense");physical,pc,layouts=run(model,emb,roles,K,"physical")
   maxlog=max(float((a-b).abs().max()) for a,b in zip(dense,physical));hidden=0.
   for key,(positions,pv) in pc.hidden.items():
    dp,dv=dc.hidden[key];idx=[dp.tolist().index(p) for p in positions.tolist()]
    hidden=max(hidden,float((dv[:,idx,:]-pv).abs().max()))
   assert maxlog<1e-10 and hidden<1e-10,(name,maxlog,hidden)
   records.append({"condition":name,"max_logits_abs":maxlog,"max_aligned_hidden_abs":hidden,"steps":4,"real_decode":3,"final_KV_live_MiB":layouts[-1][1],"final_KV_capacity_MiB":layouts[-1][2],"all_layer_permissions_cache_positions_projection_skips":"PASS"})
   print("CPU_PASS",name,maxlog,hidden,flush=True)
  forward=model.language_model.model.forward;count=e.frozen.hooks_count(model)
  try:
   with e.Control(model,True,roles,e.CompactCache(roles,[],"cpu"),[],False):raise RuntimeError("EXPECTED_TEST_EXCEPTION")
  except RuntimeError as x:assert str(x)=="EXPECTED_TEST_EXCEPTION"
  assert model.language_model.model.forward==forward and e.frozen.hooks_count(model)==count
  assert e.CompactCache(roles,[],"cpu") is not e.CompactCache(roles,[],"cpu")
  result={"status":"PASS","Native_noop_torch_equal":True,"Qwen2_actual_cpu_float64_28_layers":True,"all7_conditions":records,"independent_cache_exception_cleanup":"PASS","no_CUDA_initialized":not torch.cuda.is_initialized(),"Frozen_search_space_sha256":sha(OUT/"search_space.json"),"engine_sha256":sha(RUN/"code"/PKG/"engine.py")}
  result["CPU_math_mock_softmax"]="Explicit Float64; official backend otherwise forces FP32 even for Float64 weights. This mock-only change does not modify real GPU attention or numerical gates."
  softmax_patch.stop()
  assert result["no_CUDA_initialized"];writej(OUT/"cpu_tests.json",result);state("CPU_TESTS_PASS")
 except Exception as x:
  writej(OUT/"cpu_tests.json",{"status":"FAILED","error":str(x),"traceback":traceback.format_exc()});raise
if __name__=="__main__":main()

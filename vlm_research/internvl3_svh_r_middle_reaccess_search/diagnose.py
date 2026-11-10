"""One frozen-sample operator diagnostic after numerical failure; no search/accuracy."""
from .common import *
from . import engine as e
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
import torch,gc,traceback
def main():
 model=None;inputs=None
 try:
  w.gpu_idle();model,tok=w.load_model();inputs=w.Inputs(model,tok)
  sm=readj(OUT/"smoke_sample_manifest.json");q=sm["ids"][0];r=next(x for x in readj(OUT/"sample_manifest.json")["Discovery"] if x["question_id"]==q)
  ready,ib,binding=inputs.ready(r);data={};tokens={};layer=model.language_model.model.layers[20]
  for physical,label in [(False,"Dense-R0"),(True,"Physical-R0")]:
   saved={};handles=[]
   def hook(name,which="out"):
    def f(m,args,out):
     if name in saved:return
     value=args[0] if which=="input" else out
     if isinstance(value,tuple):value=value[1]
     saved[name]=value.detach().cpu().clone()
    return f
   modules=[("input_norm",layer.input_layernorm),("q_proj",layer.self_attn.q_proj),("k_proj",layer.self_attn.k_proj),("v_proj",layer.self_attn.v_proj),("o_proj",layer.self_attn.o_proj),("post_norm",layer.post_attention_layernorm),("gate_proj",layer.mlp.gate_proj),("up_proj",layer.mlp.up_proj),("down_proj",layer.mlp.down_proj)]
   for name,module in modules:handles.append(module.register_forward_hook(hook(name)))
   handles.append(layer.self_attn.o_proj.register_forward_hook(hook("o_proj_input","input")))
   handles.append(layer.self_attn.register_forward_hook(hook("attention_probabilities")))
   try:
    result,_,_=e.generate(model,tok,ready,"R0",[],max_tokens=1,audit=False,physical=physical)
   finally:
    for h in handles:h.remove()
   data[label]=saved;tokens[label]=result["token_ids"]
  keep=e.base.text_positions(ready["roles"],"cpu");rows=[]
  names=["input_norm","q_proj","k_proj","v_proj","attention_probabilities","o_proj_input","o_proj","post_norm","gate_proj","up_proj","down_proj"]
  for name in names:
   a=data["Dense-R0"][name];b=data["Physical-R0"][name]
   a=a.index_select(1 if a.ndim==3 else -2,keep).float();b=b.float();diff=(a-b).abs()
   rows.append({"operator":name,"dense_aligned_shape":list(a.shape),"physical_shape":list(b.shape),"torch_equal":bool(torch.equal(a,b)),"different_elements":int((diff!=0).sum()),"max_abs":float(diff.max()),"mean_abs":float(diff.mean())})
  first=next((x["operator"] for x in rows if not x["torch_equal"]),None)
  result={"status":"DIAGNOSTIC_ONLY_NUMERICAL_GATE_REMAINS_FAILED","question_id":q,"condition":"R0","first_recorded_divergent_operator":first,"L0_L19_shared_hidden_strict_equal_in_prior_smoke":True,"layer20_operators":rows,"tokens":tokens,"input_binding":binding,"causal_statement":"Operator-local differences are measured; no automatic claim that BF16 differences are acceptable or harmless.","formal_discovery":"NOT_RUN","formal_holdout":"NOT_RUN","formal_efficiency":"NOT_RUN"}
  writej(OUT/"numerical_diagnosis.json",result);(RUN/"diagnostics").mkdir(exist_ok=True);torch.save(data,RUN/"diagnostics/L20_component_tensors.pt")
  print("FIRST_L20_OPERATOR",first,rows,flush=True)
 except Exception as x:
  writej(OUT/"numerical_diagnosis.json",{"status":"DIAGNOSTIC_FAILED","error":str(x),"traceback":traceback.format_exc()});print(traceback.format_exc(),flush=True)
 finally:
  if inputs is not None:inputs.pixels.cache_clear()
  model=None;inputs=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  print("DIAG_GPU_RELEASED",flush=True)
if __name__=="__main__":main()

"""A/B on frozen R0 input; saves immutable raw BF16 tensors and backend traces."""
from .common import *
from .ops import *
from .adapter import pad_text
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
import traceback,subprocess,gc
@torch.inference_mode()
def main():
 model=None;inputs=None
 try:
  assert readj(OUT/"cpu_tests.json")["status"]=="PASS"
  frozen=readj(OUT/"source_manifest.json")
  for p,h in frozen["source_sha256"].items():assert sha(RUN/"code"/p)==h
  w.gpu_idle();model,tok=w.load_model();inputs=w.Inputs(model,tok);writej(OUT/"environment.json",w.environment(model,tok))
  q=readj(OUT/"sample_binding.json");r=q["frozen_question"];ready,ib,bind=inputs.ready(r);assert bind==q["prior_input_tensor_binding"]
  n=ready["roles"]["n"];keep=e.base.text_positions(ready["roles"],"cpu");assert n==321 and len(keep)==65
  writej(OUT/"model_manifest.json",{"revision":REV,"path":"/home/featurize/ate-local-models/InternVL3-8B","previous_model_manifest":readj(RUN/"references/model_manifest.json"),"previous_integrity_receipt_sha256":sha(RUN/"references/reused_model_integrity_receipt.json"),"actual_environment_sha256":sha(OUT/"environment.json"),"all_float_weights_BF16":True,"model_weights_changed":False})
  native,ns,nc=e.generate(model,tok,ready,"Native",[],max_tokens=16,audit=True)
  noop,zs,zc=e.generate(model,tok,ready,"No-op",[],max_tokens=16,audit=True)
  assert native["token_ids"]==noop["token_ids"] and len(ns)==len(zs) and all(torch.equal(a,b) for a,b in zip(ns,zs))
  writej(OUT/"native_noop.json",{"status":"PASS","tokens":native["token_ids"],"all_effective_logits_torch_equal":True,"input_binding":bind})
  del native,ns,nc,noop,zs,zc
  dense,ds,dc,d,dm,_=capture(model,tok,ready,False,layers=[20],first_only=True)
  physical,ps,pc,p,pm,_=capture(model,tok,ready,True,layers=[20],first_only=True)
  assert dense["token_ids"]==physical["token_ids"]==[65714,652,151645]
  d=d[(20,0)];p=p[(20,0)]
  assert torch.equal(d["hidden_input"].index_select(1,keep),p["hidden_input"])
  assert torch.equal(d["input_norm"].index_select(1,keep),p["input_norm"])
  for name in ["q_proj","k_proj","v_proj"]:
   assert torch.equal(d[name+"_input"].index_select(1,keep),p[name+"_input"])
  rec=save_raw("AB_v001/captured_L20_Dense_Physical_full_BF16.pt",{"Dense":d,"Physical":p,"dense_scores":ds,"physical_scores":ps,"dense_hidden_all_layers":dc.hidden,"physical_hidden_all_layers":pc.hidden},"Actual frozen R0 full generation; capture first Prefill L20 only; no repair")
  layer=model.language_model.model.layers[20];device=layer.input_layernorm.weight.device;kg=keep.to(device)
  full=d["input_norm"].to(device);compact=p["input_norm"].to(device);padded=pad_text(compact,n,kg)
  assert torch.equal(full.index_select(1,kg),compact) and torch.equal(padded.index_select(1,kg),compact)
  img=torch.tensor(ready["roles"]["image"],device=device);assert torch.count_nonzero(padded.index_select(1,img))==0
  # Zero raw Hidden followed by native RMSNorm gives the same normalized padded input.
  rawpad=pad_text(p["hidden_input"].to(device),n,kg);assert torch.equal(layer.input_layernorm(rawpad),padded)
  modes={"A1_Dense":full,"A2_Compact":compact,"A3_ZeroPad":padded};outs={};descriptors={};weights={};kernels=[]
  for name in ["q_proj","k_proj","v_proj"]:
   m=getattr(layer.self_attn,name);weights[name]={"weight":info(m.weight),"bias":info(m.bias),"module_type":str(type(m)),"module_identity":id(m)}
  for label,x in modes.items():
   outs[label]={};descriptors[label]=info(x)
   for name in ["q_proj","k_proj","v_proj"]:
    m=getattr(layer.self_attn,name);y=m(x);aligned=y if label=="A2_Compact" else y.index_select(1,kg)
    outs[label][name]=aligned.cpu()
    row={"variant":label,"operator":name,"input":info(x),"full_output":info(y),"aligned_output":info(aligned),"native_module_same_identity":id(m),"weight_sha256":weights[name]["weight"]["sha256"],"bias_sha256":weights[name]["bias"]["sha256"],"backend":"torch.nn.Linear / aten::linear with native BF16 CUDA"}
    try:
     from torch.profiler import profile,ProfilerActivity
     with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
      yy=m(x);torch.cuda.synchronize()
     path=RUN/"raw/AB_v001"/(label+"_"+name+"_cuda_trace.json")
     assert not path.exists();prof.export_chrome_trace(str(path))
     trace=readj(path);ks=[{"name":v["name"],"duration_us":v.get("dur"),"args":v.get("args")} for v in trace["traceEvents"] if v.get("cat")=="kernel"]
     row.update(cuda_kernels=ks,profiler_trace_sha256=sha(path),profiler_trace=str(path),profiled_output_equal=bool(torch.equal(y,yy)))
    except Exception as ex:row["profiler_unavailable"]=repr(ex)
    kernels.append(row)
  comp=[]
  for pair in [("A1_Dense","A2_Compact"),("A1_Dense","A3_ZeroPad"),("A2_Compact","A3_ZeroPad")]:
   for name in ["q_proj","k_proj","v_proj"]:comp.append(dict(comparison=" vs ".join(pair),operator=name,**metric(outs[pair[0]][name],outs[pair[1]][name]),a_sha256=tensor_sha(outs[pair[0]][name]),b_sha256=tensor_sha(outs[pair[1]][name])))
  csvonce(OUT/"l20_projection_comparison.csv",comp)
  save_raw("AB_v001/A1_A2_A3_projection_BF16.pt",{"inputs":{k:v.cpu() for k,v in modes.items()},"outputs":outs,"text_positions":keep,"image_positions":img.cpu()},"Same native L20 projection module, BF16, same frozen weights; A3 image rows strictly zero")
  pad_exact=all(x["torch_equal"] for x in comp if x["comparison"]=="A1_Dense vs A3_ZeroPad")
  mismatch=[x["operator"] for x in comp if x["comparison"]=="A1_Dense vs A2_Compact" and not x["torch_equal"]]
  writej(OUT/"l20_projection_inputs.json",{"actual_capture_raw_receipt":rec["file"]+".json","actual_CUDA_capture_metadata":{"Dense":dm,"Physical":pm},"projection_input_values_strict_equal_on_shared_text":True,"all_projection_inputs_equal":True,"same_weights":weights,"variants":descriptors,"backend_evidence":kernels,"no_QK_norm_in_actual_Qwen2_layer":not hasattr(layer.self_attn,"q_norm") and not hasattr(layer.self_attn,"k_norm"),"bf16_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,"allow_tf32":torch.backends.cuda.matmul.allow_tf32,"CUDA":torch.version.cuda,"cuDNN":torch.backends.cudnn.version(),"zero_padded_rows_have_no_original_image_data":True,"A1_A3_all_projections_torch_equal":pad_exact,"A1_A2_mismatch":mismatch,"input_binding":bind})
  replay_d=replay(layer,d,False,kg,n);replay_p=replay(layer,p,True,kg,n)
  exact_replay={}
  for label,a,orig in [("Dense",replay_d,d),("Physical",replay_p,p)]:
   exact_replay[label]={}
   for name in ["input_norm","q_proj","k_proj","v_proj","o_proj","post_norm","gate_proj","up_proj","down_proj","layer_hidden"]:
    exact_replay[label][name]=bool(torch.equal(a[name],orig[name]))
   assert all(exact_replay[label].values()),("OPERATOR_REPLAY_DIFFERS_FROM_NATIVE_PATH",label,exact_replay[label])
  operators=compare_operator_traces(replay_d,replay_p,keep,d["attention_mask"],"B_unrepaired")
  csvonce(OUT/"l20_operator_comparison.csv",operators)
  save_raw("AB_v001/independent_L20_operator_replay_BF16.pt",{"Dense":replay_d,"Physical":replay_p},"Exact native-operator replay, all recorded modules strictly match actual captured outputs")
  first=next(x["operator"] for x in operators if not x["torch_equal"])
  writej(OUT/"AB_validation.json",{"status":"PASS" if pad_exact and mismatch==["k_proj","v_proj"] else "DIAGNOSIS_INCONCLUSIVE","A1_A3_all_projections_equal":pad_exact,"A1_A2_differing_projections":mismatch,"first_B_divergence":first,"replay_outputs_strict_match_actual":exact_replay,"same_hidden_norm_inputs_weights":True,"same_text_original_positions":True,"C1_may_run":pad_exact and mismatch==["k_proj","v_proj"],"no_gate_relaxation":True})
  # Local projection-only cost, not request E2E; native K/V vs compact vs zero-pad repair.
  def calc(label):
   x=modes[label]
   if label=="A3_ZeroPad":x=pad_text(compact,n,kg)
   k=layer.self_attn.k_proj(x);v=layer.self_attn.v_proj(x)
   if label!="A2_Compact":k=k.index_select(1,kg);v=v.index_select(1,kg)
   return k,v
  for label in modes:
   for _ in range(25):tmp=calc(label)
  del tmp;torch.cuda.synchronize();times={k:[] for k in modes};peaks={}
  for j in range(100):
   order=list(modes);order=order[j%3:]+order[:j%3]
   for label in order:
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
    st=torch.cuda.Event(enable_timing=True);ed=torch.cuda.Event(enable_timing=True)
    st.record();tmp=calc(label);ed.record();ed.synchronize()
    times[label].append(st.elapsed_time(ed));peaks[label]=max(peaks.get(label,0),torch.cuda.max_memory_allocated()-baseline);del tmp
  import statistics
  cost=[]
  for label,ts in times.items():
   rows=n if label!="A2_Compact" else len(keep);flops=2*rows*(layer.self_attn.k_proj.in_features*layer.self_attn.k_proj.out_features+layer.self_attn.v_proj.in_features*layer.self_attn.v_proj.out_features)
   cost.append(dict(question_id=r["question_id"],variant=label,input_rows=n,text_rows=len(keep),actual_projection_rows=rows,CUDA_event_mean_ms=statistics.mean(ts),CUDA_event_median_ms=statistics.median(ts),warmup=25,repeats=100,peak_incremental_allocated_bytes=peaks[label],zero_hidden_scratch_bytes=2*n*layer.self_attn.k_proj.in_features if label=="A3_ZeroPad" else 0,projection_multiply_add_FLOPs=flops,scope="L20 K/V only, includes gather; repair includes allocation/zero/scatter; excludes CPU audit; not request E2E"))
  csvonce(OUT/"projection_cost.csv",cost)
  save_raw("AB_v001/projection_cost_raw_events.pt",times,"100 CUDA Event K/V costs per variant; no CPU copies inside measured region")
  writej(OUT/"AB_complete.json",{"status":"COMPLETE","first_divergence":first,"A1_A3_equal":pad_exact,"C1_ready":pad_exact and mismatch==["k_proj","v_proj"]})
  print("AB_COMPLETE",pad_exact,mismatch,"FIRST_OPERATOR",first,"COST",cost,flush=True)
 except Exception as ex:
  writej(OUT/"AB_failure_v001.json",{"status":"FAILED","error":str(ex),"traceback":traceback.format_exc()});print(traceback.format_exc(),flush=True)
 finally:
  if inputs is not None:inputs.pixels.cache_clear()
  model=None;inputs=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  print("AB_WORKER_GPU_RELEASED",flush=True)
if __name__=="__main__":main()

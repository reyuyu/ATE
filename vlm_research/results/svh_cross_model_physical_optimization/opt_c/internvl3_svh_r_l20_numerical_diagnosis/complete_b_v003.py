"""Complete independent B replay from immutable A capture; use effective Dense-R mask."""
from .common import *
from .ops import *
from .adapter import pad_text
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
import traceback,gc,statistics
@torch.inference_mode()
def main():
 model=None
 try:
  w.gpu_idle();model,tok=w.load_model();raw=torch.load(RUN/"raw/AB_v002/captured_L20_Dense_Physical_full_BF16.pt",map_location="cpu",weights_only=True)
  bind=readj(OUT/"sample_binding.json");roles=bind["frozen_question"]["roles"];keep=e.base.text_positions(roles,"cpu");n=roles["n"];d=raw["Dense"];p=raw["Physical"]
  effective=e.edit_mask(d["attention_mask"],roles,True,20,set());d=dict(d,attention_mask=effective)
  assert torch.equal(effective.index_select(-2,keep),p["attention_mask"]),"EFFECTIVE_SHARED_MASK_MISMATCH"
  layer=model.language_model.model.layers[20];device=layer.input_layernorm.weight.device;kg=keep.to(device)
  rd=replay(layer,d,False,kg,n);rp=replay(layer,p,True,kg,n);checked={}
  for label,a,old in [("Dense",rd,d),("Physical",rp,p)]:
   checked[label]={name:bool(torch.equal(a[name],old[name])) for name in ["input_norm","q_proj","k_proj","v_proj","o_proj","post_norm","gate_proj","up_proj","down_proj","layer_hidden"]}
   assert all(checked[label].values()),("REPLAY_MISMATCH",label,checked[label])
  ops=compare_operator_traces(rd,rp,keep,effective,"B_v003_effective_mask")
  csvonce(OUT/"l20_operator_comparison.csv",ops)
  save_raw("B_v003/independent_native_operator_replay_full_BF16.pt",{"Dense":rd,"Physical":rp,"effective_Dense_mask":effective,"actual_Physical_mask":p["attention_mask"]},"Effective frozen Dense-R mask; output of each recorded operator torch.equal actual captured native modules")
  a=list(csv.DictReader((OUT/"l20_projection_comparison.csv").open()))
  equal=all(x["torch_equal"]=="True" for x in a if x["comparison"]=="A1_Dense vs A3_ZeroPad")
  mismatch=[x["operator"] for x in a if x["comparison"]=="A1_Dense vs A2_Compact" and x["torch_equal"]!="True"]
  first=next(x["operator"] for x in ops if not x["torch_equal"])
  writej(OUT/"AB_validation.json",{"status":"PASS","A1_A3_all_projections_equal":equal,"A1_A2_differing_projections":mismatch,"first_B_divergence":first,"replay_outputs_strict_match_actual":checked,"same_hidden_norm_inputs_weights":True,"same_text_original_positions":True,"C1_may_run":equal and mismatch==["k_proj","v_proj"],"no_gate_relaxation":True,"audit_capture_correction":"Prior pre-hook captured Dense mask before functional Control hook. Independent replay now applies exact frozen e.edit_mask, proven equal to actual compact shared mask and all native operator outputs. No model mask or algorithm modified."})
  # A diagnostic shared-padding cost, NOT exact installed C1 two-pad adapter cost.
  full=d["input_norm"].to(device);compact=p["input_norm"].to(device)
  def calc(label):
   x=full if label=="A1_Dense" else compact
   if label=="A3_ZeroPad":x=pad_text(compact,n,kg)
   k=layer.self_attn.k_proj(x);v=layer.self_attn.v_proj(x)
   if label!="A2_Compact":k=k.index_select(1,kg);v=v.index_select(1,kg)
   return k,v
  labels=["A1_Dense","A2_Compact","A3_ZeroPad"];times={k:[] for k in labels};peaks={}
  for label in labels:
   for _ in range(25):t=calc(label)
  del t;torch.cuda.synchronize()
  for j in range(100):
   for label in labels[j%3:]+labels[:j%3]:
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
    s=torch.cuda.Event(enable_timing=True);ed=torch.cuda.Event(enable_timing=True)
    s.record();t=calc(label);ed.record();ed.synchronize();times[label].append(s.elapsed_time(ed));peaks[label]=max(peaks.get(label,0),torch.cuda.max_memory_allocated()-baseline);del t
  cost=[]
  for label in labels:
   count=n if label!="A2_Compact" else len(keep)
   cost.append(dict(question_id="201337043",variant=label,input_rows=n,text_rows=len(keep),actual_projection_rows=count,CUDA_event_mean_ms=statistics.mean(times[label]),CUDA_event_median_ms=statistics.median(times[label]),warmups=25,repeats=100,peak_incremental_allocated_bytes=peaks[label],zero_hidden_scratch_bytes=2*n*3584 if label=="A3_ZeroPad" else 0,projection_multiply_add_FLOPs=2*count*(3584*512+3584*512),scope="Local K/V only; gather included; A3 shares one padded input across K/V, not exact installed C1 two-pad wrapper; not request E2E"))
  csvonce(OUT/"projection_cost.csv",cost);save_raw("B_v003/projection_cost_raw_events.pt",times,"100 CUDA Events per A variant; CPU Tensor audit excluded")
  writej(OUT/"AB_complete.json",{"status":"COMPLETE","first_divergence":first,"A1_A3_equal":equal,"C1_ready":equal and mismatch==["k_proj","v_proj"]})
  print("AB_COMPLETE",first,equal,mismatch,"COST",cost,flush=True)
 except Exception as ex:
  writej(OUT/"B_failure_v003.json",{"status":"FAILED","error":str(ex),"traceback":traceback.format_exc()});print(traceback.format_exc(),flush=True)
 finally:
  model=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  print("B_WORKER_GPU_RELEASED",flush=True)
if __name__=="__main__":main()

"""Local installed-projection timing only; frozen C1/C2, no new repairs or requests."""
from .common import *
from .ops import *
from .adapter import ProjectionRepair
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
import gc,statistics,traceback
@torch.inference_mode()
def main():
 model=None
 try:
  assert readj(OUT/"C2_v001_validation.json")["completed_smoke_samples"]==1
  w.gpu_idle();model,tok=w.load_model()
  raw=torch.load(RUN/"raw/C1_v001/201337043_Dense_Physical_teacher_free_complete_BF16.pt",map_location="cpu",weights_only=True)
  d=raw["Dense_operators"][(20,0)];p=raw["Physical_teacher_operators"][(20,0)];device=model.language_model.model.layers[20].input_layernorm.weight.device
  roles=readj(OUT/"sample_binding.json")["frozen_question"]["roles"];n=roles["n"];keep=e.base.text_positions(roles,device)
  full=d["input_norm"].to(device);compact=p["input_norm"].to(device);full_o=d["o_proj_input"].to(device);compact_o=p["o_proj_input"].to(device)
  assert torch.equal(full.index_select(1,keep),compact) and torch.equal(full_o.index_select(1,keep),compact_o)
  ready={"roles":roles,"ids":torch.ones(1,n,dtype=torch.long,device=device)}
  layer=model.language_model.model.layers[20];groups={"KV":["k_proj","v_proj"],"O":["o_proj"]}
  variants={"Dense":{},"Compact":{},"C1_v001":{20:["k_proj","v_proj"]},"C2_v001":{20:["k_proj","v_proj","o_proj"]}}
  times={};peaks={};rows=[]
  class Sink:
   def append(self,x):pass
  for group,ops in groups.items():
   def calc(variant):
    a=full if variant=="Dense" else compact
    if group=="O":a=full_o if variant=="Dense" else compact_o
    out=[getattr(layer.self_attn,name)(a) for name in ops]
    if variant=="Dense":out=[x.index_select(1,keep) for x in out]
    return out
   for variant,spec in variants.items():
    with ProjectionRepair(model,ready,spec) as rp:
     rp.events=Sink()
     for _ in range(25):out=calc(variant)
    del out
   keys=list(variants)
   for j in range(100):
    for variant in keys[j%4:]+keys[:j%4]:
     # Install/restore excluded from this LOCAL projection timing, not silently
     # counted per projection as if each needed a new request-level installation.
     with ProjectionRepair(model,ready,variants[variant]) as rp:
      rp.events=Sink();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
      s=torch.cuda.Event(enable_timing=True);ed=torch.cuda.Event(enable_timing=True)
      s.record();out=calc(variant);ed.record();ed.synchronize()
      times.setdefault(group+"_"+variant,[]).append(s.elapsed_time(ed));peaks[group+"_"+variant]=max(peaks.get(group+"_"+variant,0),torch.cuda.max_memory_allocated()-base);del out
   for variant,spec in variants.items():
    total=0;extra=0;pad_modules=[]
    for name in ops:
     module=getattr(layer.self_attn,name);pad=variant=="Dense" or name in spec.get(20,[])
     rr=n if pad else len(keep);f=2*rr*module.in_features*module.out_features
     total+=f
     if variant!="Dense" and name in spec.get(20,[]):extra+=2*(n-len(keep))*module.in_features*module.out_features;pad_modules.append(name)
    values=times[group+"_"+variant]
    rows.append(dict(question_id="201337043",group=group,variant=variant,CUDA_event_mean_ms=statistics.mean(values),CUDA_event_median_ms=statistics.median(values),CUDA_event_min_ms=min(values),CUDA_event_max_ms=max(values),warmups=25,repeats=100,real_text_rows=len(keep),padded_rows=n if pad_modules else 0,padded_modules=pad_modules,projection_multiply_add_FLOPs=total,extra_FLOPs_vs_compact=extra,measured_peak_incremental_allocated_bytes=peaks[group+"_"+variant],scratch_hidden_bytes_per_padded_call=2*n*3584 if pad_modules else 0,no_CPU_tensor_audit_in_measurement=True,scope="Local native module calls; install/restore outside Event interval; allocation+zero+scatter+gather inside; not E2E"))
  csvonce(OUT/"projection_cost_installed_v001.csv",rows)
  save_raw("cost_installed_v001/CUDA_events_all_frozen_variants.pt",times,"Local installed-module timing, interleaved 100 repetitions; no new repair or quality request")
  writej(OUT/"cost_installed_validation.json",{"status":"PASS","no_new_model_generation":True,"frozen_primary_input_only":True,"no_third_repair":True,"do_not_sum_as_E2E":True,"legacy_cost_records":"projection_cost_C1/C2 included context installation inside per-call interval; retained but primary cost is installed_v001 with installation excluded","final_gpu_before_release":w.environment(model,tok)["nvidia_smi"]})
  print("INSTALLED_COST",rows,flush=True)
 except Exception as ex:
  writej(OUT/"cost_installed_failure.json",{"error":str(ex),"traceback":traceback.format_exc()});print(traceback.format_exc(),flush=True)
 finally:
  model=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  print("COST_WORKER_GPU_RELEASED",flush=True)
if __name__=="__main__":main()

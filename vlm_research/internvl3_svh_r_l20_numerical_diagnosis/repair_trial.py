"""One frozen local repair; strict full-generation R0 gates, never launches search."""
from .common import *
from .ops import *
from .adapter import ProjectionRepair
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
import sys,traceback,gc,statistics
def compare_all(q,label,ds,ps,dc,pc,free_equal):
 gate=readj(OUT/"protocol.json")["numerical_gate"]["physical_dense"];rows=[];good=free_equal
 assert len(ds)==len(ps)
 for step,(a,b) in enumerate(zip(ds,ps)):
  m=metric(a,b);top=bool(a.argmax()==b.argmax())
  passed=m["max_abs"]<=gate["per_step_logits_max_abs_le"] and m["mean_abs"]<=gate["per_step_logits_mean_abs_le"] and top
  rows.append(dict(question_id=q,candidate=label,kind="logits",step=step,layer=27,**m,top1_equal=top,passed=passed));good &=passed
 for (step,layer),(pos,b) in pc.hidden.items():
  dp,a=dc.hidden[(step,layer)];idx={v:i for i,v in enumerate(dp.tolist())};a=a[:,[idx[i] for i in pos.tolist()]]
  m=metric(a,b);passed=m["relative_RMS"]<=gate["per_layer_shared_hidden_relative_RMS_le"];good &=passed
  rows.append(dict(question_id=q,candidate=label,kind="hidden",step=step,layer=layer,**m,top1_equal=None,passed=passed))
 return bool(good),rows
def operator_compare(d,p,q,label):
 rows=[]
 order=["hidden_input","input_norm","q_proj","k_proj","v_proj","o_proj_input","o_proj","post_norm_input","post_norm","gate_proj","up_proj","down_proj_input","down_proj","layer_hidden"]
 for (layer,step),pv in p.items():
  dv=d[(layer,step)]
  for op in order:
   a=dv[op];b=pv[op]
   if step==0:
    pos=dv["position_ids"][0] if op=="hidden_input" else None
    # All Dense rows retain original order, compact rows are frozen text positions.
    keep=torch.tensor([x for x in range(a.shape[1]) if x not in _ROLES["image"]])
    a=a.index_select(1,keep)
   rows.append(dict(question_id=q,candidate=label,layer=layer,step=step,operator=op,**metric(a,b)))
 return rows
@torch.inference_mode()
def main():
 global _ROLES
 phase=sys.argv[1];specpath=OUT/(phase+"_frozen.json");doc=readj(specpath);spec={int(i):ops for i,ops in doc["layer_modules"].items()}
 model=None;inputs=None;all_rows=[];op_rows=[];records=[];layouts=[]
 try:
  assert readj(OUT/"AB_validation.json")["C1_may_run"]
  assert doc["source_sha256"]==sha(Path(__file__)) and doc["adapter_sha256"]==sha(RUN/"code"/PKG/"adapter.py")
  w.gpu_idle();model,tok=w.load_model();inputs=w.Inputs(model,tok)
  binding=readj(OUT/"sample_binding.json");primary=binding["frozen_question"];original=readj(RUN/"references/sample_manifest.json")
  lookup={r["question_id"]:r for r in original["Discovery"]}
  ids=[primary["question_id"]]+[q for q in binding["frozen_smoke_ids"] if q!=primary["question_id"]]
  for q in ids:
   r=lookup[q];ready,ib,bind=inputs.ready(r);_ROLES=ready["roles"]
   if q==primary["question_id"]:assert bind==binding["prior_input_tensor_binding"]
   dense,ds,dc,d,dm,_=capture(model,tok,ready,False)
   forced,ps,pc,p,pm,events=capture(model,tok,ready,True,spec,forced=dense["token_ids"])
   with ProjectionRepair(model,ready,spec) as repair:
    free,fs,fc=e.generate(model,tok,ready,"R0",[],max_tokens=16,audit=True,physical=True)
   equal=free["token_ids"]==dense["token_ids"];good,rows=compare_all(q,phase,ds,ps,dc,pc,equal)
   all_rows+=rows;local=operator_compare(d,p,q,phase);op_rows+=local
   first=next((x for x in local if x["step"]==0 and not x["torch_equal"]),None)
   layer_first=next((x["layer"] for x in rows if x["kind"]=="hidden" and x["step"]==0 and not x["torch_equal"]),None)
   rec={"candidate":phase,"question_id":q,"numerical_pass":good,"full_tokens_equal":equal,"Dense_tokens":dense["token_ids"],"Physical_free_tokens":free["token_ids"],"Dense_text":dense["text"],"Physical_text":free["text"],"max_logit_abs":max(x["max_abs"] for x in rows if x["kind"]=="logits"),"max_mean_logit_abs":max(x["mean_abs"] for x in rows if x["kind"]=="logits"),"max_hidden_relative_RMS":max(x["relative_RMS"] for x in rows if x["kind"]=="hidden"),"first_prefill_hidden_divergent_layer":layer_first,"first_prefill_recorded_operator_divergence":first,"physical_audit":forced["audit"],"actual_projection_padding_events":events,"input_binding":bind,"KV_live_MiB":forced["KV_live_MiB"],"KV_capacity_MiB":forced["KV_capacity_MiB"],"KV_output_tokens":len(forced["token_ids"]),"all_visual_KV_zero":all(x["visual_kv"]==0 for x in forced["kv_layout"])}
   records.append(rec);layouts+=list(dict(candidate=phase,question_id=q,**x) for x in forced["kv_layout"])
   save_raw(phase+"/"+q+"_Dense_Physical_teacher_free_complete_BF16.pt",{"Dense_scores":ds,"Physical_teacher_scores":ps,"Physical_free_scores":fs,"Dense_token_ids":dense["token_ids"],"Physical_free_token_ids":free["token_ids"],"Dense_hidden":dc.hidden,"Physical_teacher_hidden":pc.hidden,"Physical_free_hidden":fc.hidden,"Dense_operators":d,"Physical_teacher_operators":p},"Full actual per-step BF16 tensors; same teacher prefix and independent free generation; "+phase)
   writej(OUT/(phase+"_"+q+"_record.json"),rec)
   print("REPAIR",phase,q,"PASS" if good else "FAIL","logits",rec["max_logit_abs"],rec["max_mean_logit_abs"],"FIRST",first,flush=True)
   if not good:break
   del dense,ds,dc,d,dm,forced,ps,pc,p,pm,free,fs,fc;gc.collect()
  csvonce(OUT/("repair_numerical_comparison_"+phase+".csv"),all_rows)
  csvonce(OUT/("repair_operator_comparison_"+phase+".csv"),op_rows)
  csvonce(OUT/("kv_layout_"+phase+".csv"),layouts)
  writej(OUT/(phase+"_validation.json"),{"status":"PASS" if len(records)==8 and all(x["numerical_pass"] for x in records) else "PARTIAL_REPAIR","candidate":phase,"planned_smoke_samples":8,"completed_smoke_samples":len(records),"records":records,"original_gate_unchanged":True,"no_quality_scoring_or_search":True})
  # Actual installed adapter cost, distinct from A3's shared-padding diagnostic.
  primary_ready,_,_=inputs.ready(primary);raw=torch.load(RUN/"raw/AB_v002/captured_L20_Dense_Physical_full_BF16.pt",map_location="cpu",weights_only=True)
  device=model.language_model.model.layers[20].input_layernorm.weight.device
  full=raw["Dense"]["input_norm"].to(device);compact=raw["Physical"]["input_norm"].to(device)
  keep=e.base.text_positions(primary_ready["roles"],device);layer=model.language_model.model.layers[20]
  def calc(label):
   x=full if label=="Dense_L20_KV" else compact
   if label=="Actual_"+phase:
    with ProjectionRepair(model,primary_ready,spec) as rp:
     class Sink:
      def append(self,x):pass
     rp.events=Sink()
     a=layer.self_attn.k_proj(x);b=layer.self_attn.v_proj(x)
   else:a=layer.self_attn.k_proj(x);b=layer.self_attn.v_proj(x)
   if label=="Dense_L20_KV":a=a.index_select(1,keep);b=b.index_select(1,keep)
   return a,b
  labels=["Dense_L20_KV","Compact_L20_KV","Actual_"+phase];times={l:[] for l in labels};peak={}
  for label in labels:
   for _ in range(25):t=calc(label)
  del t;torch.cuda.synchronize()
  for j in range(100):
   order=labels[j%3:]+labels[:j%3]
   for label in order:
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();t=calc(label);b.record();b.synchronize();times[label].append(a.elapsed_time(b));peak[label]=max(peak.get(label,0),torch.cuda.max_memory_allocated()-base);del t
  cost=[]
  for label in labels:
   rows=321 if label!="Compact_L20_KV" else 65;flops=2*rows*(3584*512+3584*512)
   cost.append(dict(question_id=primary["question_id"],variant=label,CUDA_event_mean_ms=statistics.mean(times[label]),CUDA_event_median_ms=statistics.median(times[label]),warmups=25,repeats=100,actual_KV_projection_rows=rows,KV_projection_multiply_add_FLOPs=flops,peak_incremental_allocated_bytes=peak[label],scope="Actual L20 K/V module adapter, two independent pad allocations for repair; no full request E2E; CPU Tensor audit disabled",C2_other_operator_cost_included=False))
  csvonce(OUT/("projection_cost_"+phase+".csv"),cost);save_raw(phase+"/actual_projection_cost_events.pt",times,"Local actual adapter CUDA Event samples")
  print("ACTUAL_COST",phase,cost,flush=True)
 except Exception as ex:
  writej(OUT/(phase+"_failure.json"),{"status":"FAILED","error":str(ex),"traceback":traceback.format_exc()});print(traceback.format_exc(),flush=True)
 finally:
  if inputs is not None:inputs.pixels.cache_clear()
  model=None;inputs=None;gc.collect()
  if torch.cuda.is_initialized():torch.cuda.empty_cache()
  print("REPAIR_WORKER_GPU_RELEASED",phase,flush=True)
if __name__=="__main__":main()

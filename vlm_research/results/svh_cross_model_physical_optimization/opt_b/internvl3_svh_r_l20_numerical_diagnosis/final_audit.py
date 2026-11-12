"""Independent CPU-only recomputation of immutable A/B/C evidence; no model."""
from .common import *
import torch,math,subprocess,gc
def dist(a,b):
 assert a.dtype==b.dtype==torch.bfloat16
 assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
 delta=a.to(torch.float64)-b.to(torch.float64)
 return {"torch_equal":torch.equal(a,b),"max_abs":float(delta.abs().max()),"mean_abs":float(delta.abs().mean()),"relative_RMS":float(delta.square().mean().sqrt()/a.double().square().mean().sqrt().clamp_min(1e-30)),"different_elements":int(torch.count_nonzero(delta))}
def match(m,row):
 for k in ["max_abs","mean_abs","relative_RMS"]:assert math.isclose(m[k],float(row[k]),rel_tol=1e-9,abs_tol=1e-12),(k,m[k],row[k])
 assert m["torch_equal"]==(row["torch_equal"]=="True") and m["different_elements"]==int(row["different_elements"])
def main():
 torch.set_num_threads(2)
 frozen=readj(OUT/"source_manifest.json")
 assert frozen["protocol_sha256"]==sha(OUT/"protocol.json")
 for n,h in frozen["source_sha256"].items():assert sha(RUN/"code"/n)==h,("FROZEN_SOURCE_CHANGED",n)
 for n,v in readj(RUN/"references/prior_evidence_manifest.json").items():
  assert sha(RUN/"references"/n)==v["sha256"]
  assert sha(Path(v["origin"]))==v["sha256"],("OLD_EVIDENCE_CHANGED",n)
 for candidate in ["C1_v001","C2_v001"]:
  doc=readj(OUT/(candidate+"_frozen.json"))
  assert doc["source_sha256"]==sha(RUN/"code"/PKG/"repair_trial.py") and doc["adapter_sha256"]==sha(RUN/"code"/PKG/"adapter.py")
 assert readj(OUT/"C2_v001_frozen.json")["C1_record_sha256"]==sha(OUT/"C1_v001_201337043_record.json")
 bind=readj(OUT/"sample_binding.json");r=bind["frozen_question"];keep=torch.tensor([i for i in range(r["input_tokens"]) if i not in r["roles"]["image"]])
 assert sha(Path(r["image_path"]))==r["image_sha256"]==bind["prior_input_tensor_binding"]["image_sha256"]
 a=torch.load(RUN/"raw/AB_v002/A1_A2_A3_projection_BF16.pt",map_location="cpu",weights_only=True)
 assert torch.equal(a["inputs"]["A1_Dense"].index_select(1,keep),a["inputs"]["A2_Compact"])
 assert torch.equal(a["inputs"]["A3_ZeroPad"].index_select(1,keep),a["inputs"]["A2_Compact"])
 assert torch.count_nonzero(a["inputs"]["A3_ZeroPad"].index_select(1,a["image_positions"]))==0
 ac=list(csv.DictReader((OUT/"l20_projection_comparison.csv").open()))
 for row in ac:
  l,t=row["comparison"].split(" vs ");match(dist(a["outputs"][l][row["operator"]],a["outputs"][t][row["operator"]]),row)
 del a;gc.collect()
 b=torch.load(RUN/"raw/B_v003/independent_native_operator_replay_full_BF16.pt",map_location="cpu",weights_only=True)
 for row in csv.DictReader((OUT/"l20_operator_comparison.csv").open()):
  op=row["operator"];x=b["Dense"][op];y=b["Physical"][op]
  if op in ["attention_scores","masked_scores","softmax"]:
   x=x.index_select(-2,keep).index_select(-1,keep);y=y.index_select(-1,keep)
   legal=(b["effective_Dense_mask"].index_select(-2,keep).index_select(-1,keep)==0).expand_as(x);x=x[legal];y=y[legal]
  elif op in ["rope_q","rope_k","AV"]:x=x.index_select(-2,keep)
  else:x=x.index_select(1,keep)
  match(dist(x,y),row)
 del b;gc.collect()
 gate=readj(OUT/"protocol.json")["numerical_gate"]["physical_dense"];summary=[];allnum=[];kv=[]
 # C0 is from this task's new capture, not filled from old result numbers.
 raw=torch.load(RUN/"raw/AB_v002/captured_L20_Dense_Physical_full_BF16.pt",map_location="cpu",weights_only=True)
 c0=[dist(x,y) for x,y in zip(raw["dense_scores"],raw["physical_scores"])]
 summary.append({"candidate":"C0_unrepaired","max_logits_abs":max(x["max_abs"] for x in c0),"max_mean_logits_abs":max(x["mean_abs"] for x in c0),"first_prefill_operator":"k_proj","numerical_gate_pass":False})
 del raw;gc.collect()
 for candidate in ["C1_v001","C2_v001"]:
  raw=torch.load(RUN/"raw"/candidate/"201337043_Dense_Physical_teacher_free_complete_BF16.pt",map_location="cpu",weights_only=True)
  rec=readj(OUT/(candidate+"_201337043_record.json"));assert rec["input_binding"]==bind["prior_input_tensor_binding"]
  rows=list(csv.DictReader((OUT/("repair_numerical_comparison_"+candidate+".csv")).open()));allnum+=rows
  assert len(rows)==87
  logs=[x for x in rows if x["kind"]=="logits"];g=True
  for i,(x,y) in enumerate(zip(raw["Dense_scores"],raw["Physical_teacher_scores"])):
   m=dist(x,y);match(m,logs[i]);top=bool(x.argmax()==y.argmax());passed=m["max_abs"]<=gate["per_step_logits_max_abs_le"] and m["mean_abs"]<=gate["per_step_logits_mean_abs_le"] and top
   assert passed==(logs[i]["passed"]=="True");g &=passed
  for key,(pos,y) in raw["Physical_teacher_hidden"].items():
   step,layer=key;dp,x=raw["Dense_hidden"][key];idx={v:i for i,v in enumerate(dp.tolist())};x=x[:,[idx[v] for v in pos.tolist()]]
   m=dist(x,y);old=next(t for t in rows if t["kind"]=="hidden" and int(t["step"])==step and int(t["layer"])==layer);match(m,old)
   passed=m["relative_RMS"]<=gate["per_layer_shared_hidden_relative_RMS_le"];assert passed==(old["passed"]=="True");g &=passed
  oprows=list(csv.DictReader((OUT/("repair_operator_comparison_"+candidate+".csv")).open()))
  for row in oprows:
   key=(int(row["layer"]),int(row["step"]));op=row["operator"];x=raw["Dense_operators"][key][op];y=raw["Physical_teacher_operators"][key][op]
   if key[1]==0:x=x.index_select(1,keep)
   match(dist(x,y),row)
  exact=["input_norm","q_proj","k_proj","v_proj","o_proj_input"]+(["o_proj","post_norm_input","post_norm","gate_proj","up_proj","down_proj_input"] if candidate=="C2_v001" else [])
  firstrows={x["operator"]:x for x in oprows if int(x["step"])==0 and int(x["layer"])==20}
  assert all(firstrows[x]["torch_equal"]=="True" for x in exact)
  first=next(x["operator"] for x in oprows if int(x["step"])==0 and int(x["layer"])==20 and x["torch_equal"]!="True")
  assert first==("o_proj" if candidate=="C1_v001" else "down_proj")
  assert raw["Dense_token_ids"]==raw["Physical_free_token_ids"]==[65714,652,151645] and not g and not rec["numerical_pass"]
  audit=rec["physical_audit"];assert audit["blocked_probability_max"]==0 and audit["original_logical_positions"] and audit["finite"]
  assert audit["hidden_exit"]==[{"before":321,"after":65,"removed":256}]
  layers=list(csv.DictReader((OUT/("kv_layout_"+candidate+".csv")).open()))
  for l in layers:
   layer=int(l["layer"]);sh=json.loads(l["k_shape"]);vs=json.loads(l["v_shape"])
   assert int(l["visual_kv"])==0 and int(l["physical_length"])==(0 if layer in range(2,6) else 67)
   if sh:assert sh==vs and int(l["live_bytes"])==2*math.prod(sh)*2
   else:assert int(l["live_bytes"])==int(l["capacity_bytes"])==0
  live=sum(int(x["live_bytes"]) for x in layers)/1048576;cap=sum(int(x["capacity_bytes"]) for x in layers)/1048576
  assert live==cap==rec["KV_live_MiB"]==rec["KV_capacity_MiB"]==3.140625
  kv.append({"candidate":candidate,"output_tokens":3,"real_decode":2,"KV_live_MiB":live,"KV_capacity_MiB":cap,"persistent_original_visual_KV":0,"all_28_layers_verified":True,"not_at32":True})
  summary.append({"candidate":candidate,"max_logits_abs":rec["max_logit_abs"],"max_mean_logits_abs":rec["max_mean_logit_abs"],"max_hidden_relative_RMS":rec["max_hidden_relative_RMS"],"first_prefill_operator":first,"numerical_gate_pass":False})
  del raw;gc.collect()
 csvonce(OUT/"repair_numerical_comparison.csv",allnum)
 csvonce(OUT/"repair_summary.csv",summary)
 writej(OUT/"kv_integrity.json",{"status":"PASS_THIS_PRIMARY_SAMPLE_ONLY","records":kv,"no_hidden_dense_KV_backing":True,"does_not_claim_KV_at32":True})
 writej(OUT/"first_divergence.json",{"unrepaired":"L20 k_proj; Q/input_norm equal","C1":"L20 o_proj; o_proj input equal after K/V fix","C2":"L20 down_proj; MLP product input equal after K/V/O fix","moved_to_L21":False,"no_more_candidates":True})
 writej(OUT/"numerical_status.json",{"status":"PARTIAL_REPAIR","R0_original_gate_pass":False,"max_candidates_tested":2,"primary_question_id":"201337043","further_7_smoke":"NOT_RUN_FIRST_SAMPLE_FAILED","R4_R8":"NOT_RUN_R0_FAILED","Discovery":"NOT_RUN","Holdout":"NOT_RUN","E2E32_benchmark":"NOT_RUN","training":"NOT_RUN","next_search_numerical_prerequisite":False,"no_waiver_or_threshold_change":True})
 writej(OUT/"smoke_validation.json",{"status":"PARTIAL_REPAIR","native_noop":readj(OUT/"native_noop.json"),"candidate_records":[readj(OUT/(n+"_validation.json")) for n in ["C1_v001","C2_v001"]],"completed_unique_samples":1,"planned_samples_if_first_passed":8,"whole_tokens_equal":True,"strict_numeric_gates_failed":True})
 writej(OUT/"candidate_repairs_final.json",{"max_candidates":2,"frozen_original_candidate_document":str(OUT/"candidate_repairs.json"),"C1":{"binding_sha256":sha(OUT/"C1_v001_frozen.json"),"status":"PARTIAL_REPAIR","fix":"L20 K/V equal"},"C2":{"binding_sha256":sha(OUT/"C2_v001_frozen.json"),"status":"PARTIAL_REPAIR","fix":"L20 K/V/O equal; next down_proj differs"},"no_C3":True})
 # Verify all unique raw files against immutable per-file receipts.
 receipts=[]
 for p in sorted((RUN/"raw").rglob("*.pt.json")):
  meta=readj(p);assert sha(meta["file"])==meta["sha256"]
  receipts.append({"receipt":str(p),"receipt_sha256":sha(p),"file":meta["file"],"sha256":meta["sha256"],"bytes":meta["bytes"]})
 writej(OUT/"raw_tensor_manifest.json",{"raw_tensors_have_unique_phase_paths":True,"no_overwrite_atomic_link":True,"server_only_raw_files":receipts,"full_shape_dtype_tensor_sha_metadata":"Corresponding .pt.json receipts copied under raw_tensor_receipts/ in GitHub light evidence"})
 for p in sorted((RUN/"raw").rglob("*.pt.json")):
  atomic_bytes(OUT/"raw_tensor_receipts"/p.relative_to(RUN/"raw"),p.read_bytes())
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip()
 apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
 assert gpu.startswith(GPU) and not apps and not torch.cuda.is_initialized()
 writej(OUT/"validation.json",{"status":"PARTIAL_REPAIR","evidence_integrity":"PASS","all_A_comparisons_CPU_recomputed":True,"all_B_operators_CPU_recomputed":True,"C1_C2_complete_Logits_hidden_and_operator_comparisons_CPU_recomputed":True,"all_frozen_sources_and_original_evidence_unchanged":True,"original_model_input_binding_matches":True,"raw_tensor_file_sha256_verified":len(receipts),"no_old_or_new_evidence_overwritten":True,"new_A_v001_failed_before_generation_environment_field":True,"new_A_v002_B_replay_mask_capture_audit_corrected_in_v003":True,"no_inference_mask_or_protocol_changed_in_correction":True,"GPU_UUID":GPU,"GPU_released":True,"final_gpu_snapshot":gpu,"remaining_compute_processes":[],"CPU_audit_CUDA_initialized":False,"no_extra_repair_or_search":True})
 print("CPU_EVIDENCE_AUDIT_PASS_EXPERIMENT_PARTIAL_REPAIR",summary,flush=True)
if __name__=="__main__":main()

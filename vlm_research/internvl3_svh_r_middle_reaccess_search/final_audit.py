"""Independent CPU audit of failed Smoke; does not load model or initialize CUDA."""
from .common import *
import torch,gzip,collections,math,subprocess
def main():
 p=readj(OUT/"protocol.json");sm=readj(OUT/"smoke_validation.json");src=readj(OUT/"source_manifest.json")
 cp=readj(OUT/"cpu_tests.json");samples=readj(OUT/"sample_manifest.json")
 assert sm["status"]=="FAILED" and cp["status"]=="PASS"
 for name,key in [("protocol.json","protocol_sha256"),("sample_manifest.json","sample_manifest_sha256"),("search_space.json","search_space_sha256"),("selection_rule.json","selection_rule_sha256")]:
  assert sha(OUT/name)==src[key],("FROZEN_ARTIFACT_CHANGED",name)
 source_checks=[]
 for name,want in src["inference_source_SHA256"].items():
  actual=sha(RUN/"code"/name);assert actual==want,("SOURCE_CHANGED",name)
  source_checks.append({"path":name,"sha256":actual,"matches_frozen":True})
 with gzip.open(RUN/"references/GQA_full_manifest.jsonl.gz","rt") as f:original={x["question_id"]:x for x in map(json.loads,f)}
 assert len(original)==12578 and len({x["image_sha256"] for x in original.values()})==398
 assert sha(RUN/"references/GQA_full_manifest.jsonl.gz")==samples["source_manifest_sha256"]
 for split in ["Discovery","Holdout"]:
  assert len(samples[split])==240 and len({r["question_id"] for r in samples[split]})==240
  assert collections.Counter(r["category"] for r in samples[split])==dict(query=160,choose=20,compare=20,logical=20,verify=20)
  for r in samples[split]:assert canonical(r)==canonical(original[r["question_id"]]),("MANIFEST_BINDING_CHANGED",r["question_id"])
 a={r["image_sha256"] for r in samples["Discovery"]};b={r["image_sha256"] for r in samples["Holdout"]}
 assert not a&b
 eff=[original[q] for q in samples["efficiency_ids"]]
 assert len(eff)==20 and len({r["image_sha256"] for r in eff})==20 and sum(r["category"]=="query" for r in eff)==12
 numerical=list(csv.DictReader((OUT/"numerical_comparison.csv").open()))
 layout=list(csv.DictReader((OUT/"kv_layout.csv").open()))
 independent=[];summaries=[]
 gate=p["numerical_gate"]["physical_dense"]
 for ck in sm["checks"]:
  name=ck["condition"];q=ck["question_id"];K=SPACE[name]
  assert ck["native_noop_official_logits_strict_equal"] and ck["historical_native_input_tokens_equal"]
  r=original[q];binds=ck["input_binding_details"]
  assert binds["image_sha256"]==r["image_sha256"] and binds["prompt_sha256"]==r["prompt_sha256"]
  assert sha(Path(r["image_path"]))==r["image_sha256"]
  raw_path=RUN/"diagnostics"/(name+"_"+q+"_FAILED_scores.pt")
  raw=torch.load(raw_path,map_location="cpu",weights_only=True)
  assert raw["input_binding"]==binds
  assert raw["dense_tokens"]==ck["dense_tokens"] and raw["physical_tokens"]==ck["physical_tokens"]
  logs=[x for x in numerical if x["question_id"]==q and x["condition"]==name and x["kind"]=="logits"]
  hidden=[x for x in numerical if x["question_id"]==q and x["condition"]==name and x["kind"]=="hidden"]
  ds=raw["dense_scores"];ps=raw["same_prefix_physical_scores"]
  assert len(ds)==len(ps)==len(logs)==3
  good=True
  for step,(x,y) in enumerate(zip(ds,ps)):
   assert x.device.type==y.device.type=="cpu" and torch.isfinite(x).all() and torch.isfinite(y).all()
   d=(x.float()-y.float()).abs();mx=float(d.max());mean=float(d.mean());top=bool(x.argmax()==y.argmax())
   old=next(r for r in logs if int(r["step"])==step)
   assert mx==float(old["max_abs"]) and math.isclose(mean,float(old["mean_abs"]),rel_tol=1e-6,abs_tol=1e-7) and top==(old["top1_equal"]=="True")
   passed=mx<=gate["per_step_logits_max_abs_le"] and mean<=gate["per_step_logits_mean_abs_le"] and top
   assert passed==(old["passed"]=="True");good &=passed
   independent.append(dict(condition=name,question_id=q,step=step,max_abs=mx,mean_abs=mean,top1_equal=top,gate_pass=passed,raw_sha256=sha(raw_path)))
  assert len(hidden)==28*3
  for row in hidden:
   assert (float(row["relative_RMS"])<=gate["per_layer_shared_hidden_relative_RMS_le"])==(row["passed"]=="True")
  assert all(float(x["max_abs"])==0 for x in hidden if int(x["step"])==0 and int(x["layer"])<20)
  first=min(int(x["layer"]) for x in hidden if int(x["step"])==0 and float(x["max_abs"])>0)
  assert first==20
  layers=[r for r in layout if r["question_id"]==q and r["condition"]==name]
  assert len(layers)==28
  for row in layers:
   i=int(row["layer"]);n=int(row["physical_length"]);visual=int(row["visual_kv"]);kshape=json.loads(row["k_shape"]);vshape=json.loads(row["v_shape"])
   assert int(row["logical_length"])==r["input_tokens"]+2
   assert n==(0 if i in range(2,6) else r["input_tokens"]+2-(0 if i in K else 256))
   assert visual==(256 if i in K else 0)
   if n:
    assert kshape==vshape and kshape[-2]==n and row["dtype"]=="torch.bfloat16"
    assert int(row["live_bytes"])==2*math.prod(kshape)*2
   else:assert not kshape and not vshape and int(row["live_bytes"])==int(row["capacity_bytes"])==0
  audit=ck["physical_execution"]
  assert audit["steps"]==3 and audit["prefill"]==1 and audit["real_decode"]==2
  assert audit["hidden_exit"]==[{"before":r["input_tokens"],"after":r["input_tokens"]-256,"removed":256}]
  assert audit["attention_calls"]==76 and audit["projection_calls"]==304 and audit["norm_calls"]==168 and audit["MLP_calls"]==84
  assert audit["blocked_probability_max"]==0 and audit["finite"] and audit["original_logical_positions"]
  assert len(audit["skip_events"])==12 and set(audit["reaccess_layers"])==set(K)
  for pos in audit["query_and_mask_shapes"]:
   layer=pos["layer"];step=pos["step"];n=r["input_tokens"]
   expected=list(range(n)) if step==0 else [n+step-1]
   if step==0 and layer in range(2,6):expected=expected[:-1]
   elif step==0 and layer>=20:expected=[p for p in expected if p not in r["roles"]["image"]]
   assert pos["query_positions"]==expected
  assert len(audit["allowed_image_probability_samples"])==len(K)*3
  assert all(v[2]>0 for v in audit["allowed_image_probability_samples"])
  full=raw["dense_tokens"]==raw["physical_tokens"];maxhidden=max(float(x["relative_RMS"]) for x in hidden)
  actual_gate=good and maxhidden<=gate["per_layer_shared_hidden_relative_RMS_le"] and full
  assert actual_gate==ck["numerical_gate_pass"]==False
  summaries.append(dict(condition=name,reaccess_layers=K,numerical_status="FAILED",max_logits_abs=max(float(x["max_abs"]) for x in logs),mean_logits_abs_max=max(float(x["mean_abs"]) for x in logs),max_shared_hidden_relative_RMS=maxhidden,full_tokens_equal=full,first_prefill_divergent_layer=first,smoke_output_tokens=3,smoke_real_decode_steps=2,smoke_KV_live_MiB=sum(int(x["live_bytes"]) for x in layers)/1048576,smoke_KV_capacity_MiB=sum(int(x["capacity_bytes"]) for x in layers)/1048576,smoke_extra_capacity_vs_R0_MiB=None,NOT_KV_at32=True))
  del raw,ds,ps
 for row in summaries:row["smoke_extra_capacity_vs_R0_MiB"]=row["smoke_KV_capacity_MiB"]-summaries[0]["smoke_KV_capacity_MiB"]
 csvwrite(OUT/"independent_logit_validation.csv",independent)
 csvwrite(OUT/"smoke_condition_summary.csv",summaries)
 # Independent first-operator recheck of tensors, CPU only.
 data=torch.load(RUN/"diagnostics/L20_component_tensors.pt",map_location="cpu",weights_only=True)
 q=sm["checks"][0]["question_id"];roles=original[q]["roles"];keep=torch.tensor([i for i in range(roles["n"]) if i not in roles["image"]])
 diagnosis=readj(OUT/"numerical_diagnosis.json");operators=[]
 for old in diagnosis["layer20_operators"]:
  name=old["operator"];x=data["Dense-R0"][name];y=data["Physical-R0"][name]
  x=x.index_select(1 if x.ndim==3 else -2,keep).float();y=y.float();diff=(x-y).abs()
  assert float(diff.max())==old["max_abs"] and math.isclose(float(diff.mean()),old["mean_abs"],rel_tol=1e-6,abs_tol=1e-7) and bool(torch.equal(x,y))==old["torch_equal"]
  operators.append(dict(operator=name,torch_equal=bool(torch.equal(x,y)),max_abs=float(diff.max()),mean_abs=float(diff.mean())))
 del data
 assert next(x["operator"] for x in operators if not x["torch_equal"])=="k_proj"
 assert not list((RUN/"checkpoints").glob("*")),"UNEXPECTED_FORMAL_CHECKPOINT"
 assert not (OUT/"selected_candidates.json").exists(),"UNEXPECTED_SELECTION"
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu,clocks.sm,power.draw,power.limit,temperature.gpu","--format=csv,noheader"],text=True).strip()
 apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
 assert gpu.startswith(GPU) and not apps
 final={"status":"FAILED","failure":"NUMERICAL_EQUIVALENCE_FAILED","evidence_integrity":"PASS","frozen_numerical_thresholds_unchanged":True,"old_waiver_inherited":False,"sources_match_before_smoke":source_checks,"sample_manifest_matches_parent":True,"Discovery":{"questions":240,"image_clusters":len(a),"query":160,"nonquery":80},"Holdout":{"questions":240,"image_clusters":len(b),"query":160,"nonquery":80},"image_overlap":0,"efficiency_inputs_frozen":20,"efficiency_unique_images":20,"efficiency_query_inputs":12,"strict_official_native_noop":"PASS_ON_ONE_SAMPLE","full_8_sample_smoke":"NOT_COMPLETED_STOPPED_AFTER_FAILURE","gpu_smoke_samples_completed":1,"gpu_smoke_samples_planned":8,"all_7_configurations_numeric_failed":True,"CPU_mathematical_mock":"PASS_FLOAT64_SOFTMAX_MOCK_NOT_GPU_BF16_PROOF","independent_full_vocab_logit_recalculation":"PASS_21_STEP_COMPARISONS_FROM_CURRENT_V2_RAW_CPU_TENSORS","independent_mean_reduction_check":"CPU parallel reduction order may differ by <1e-7; only validation of archived means uses abs1e-7/rel1e-6. GPU acceptance thresholds unchanged.","shared_hidden_metrics":"Frozen CSV threshold assertions audited; full per-layer hidden tensors not retained","first_divergent_prefill_layer":20,"operator_diagnosis_CPU_recalculation":operators,"formal_discovery":"NOT_RUN","formal_holdout":"NOT_RUN","formal_efficiency":"NOT_RUN","formal_generation_count":0,"holdout_candidate_predictions_accessed":False,"best_cost_candidate":None,"best_query_candidate":None,"accuracy_bootstrap":"NOT_RUN_NO_FORMAL_PREDICTIONS","diagnostic_panel":"NOT_RUN","server_raw_retention_limitations":["V1 full Logit tensor .pt filenames were reused by V2; V1 per-step CSV, tokens, source, protocol and logs remain, but V1 full Logit tensors are not available. Current raw .pt files are V2 only."],"GPU_UUID":GPU,"final_gpu_snapshot":gpu,"final_compute_processes":[],"GPU_released":True,"CUDA_initialized_by_this_CPU_audit":torch.cuda.is_initialized(),"checked_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
 assert not final["CUDA_initialized_by_this_CPU_audit"]
 writej(OUT/"validation.json",final)
 print("INDEPENDENT_CPU_AUDIT_PASS_EVIDENCE; EXPERIMENT_FAILED; NO_FORMAL_REQUESTS",summaries,flush=True)
if __name__=="__main__":main()

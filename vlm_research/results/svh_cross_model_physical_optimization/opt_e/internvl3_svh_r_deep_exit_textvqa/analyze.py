"""Independent CPU-only post-run rescoring, KV audit and paired analysis."""
from .common import *
import subprocess,math,collections
from vlm_research.internvl3_svh_seven_benchmarks import scoring
scoring.RUN=RUN
METHODS=["Native","R0","R6","R14","Deep6_L20_25","Deep8_L19_26"]
def categories(x,r):
 text=x["text"].strip().lower();p=x["parsed_answer"]
 bad=x["score"]<=1e-10 and not x["flags"]["meta"]
 return dict(unanswerable=bool(re.search(r"\bunanswerable\b",text)),
 wrong_digits_heuristic=bad and bool(re.search(r"\d",text)),
 wrong_brand_heuristic=bad and bool(re.search(r"\b(brand|company|manufacturer|logo)\b",r["question"],re.I)) and bool(re.search(r"[a-z]",text)),
 wrong_word_heuristic=bad and bool(re.search(r"[a-z]",text)) and not bool(re.search(r"\d",text)),
 question_echo_heuristic=len(p.split())>=3 and (p in r["question"].lower() or r["question"].lower().rstrip("?") in text))
def main():
 mf=read(OUT/"sample_manifest.json");lookup={r["question_id"]:r for r in mf["Main"]}
 ph=sha(OUT/"protocol.json");sh=sha(OUT/"source_manifest.json");bind=read(OUT/"input_bindings.json")
 db=sqlite3.connect("file:"+str(RUN/"checkpoints/predictions.sqlite")+"?mode=ro",uri=True)
 assert db.execute("PRAGMA integrity_check").fetchone()==("ok",)
 mdict=dict(db.execute("select key,value from metadata"))
 assert mdict==dict(protocol_sha256=ph,source_manifest_sha256=sh,sample_manifest_sha256=sha(OUT/"sample_manifest.json"))
 records=[];kv=[];flat=[];independent=[]
 for phase,qid,m,origin,h,raw in db.execute("select phase,qid,method,origin,checksum,record from predictions order by phase,qid,method"):
  assert hashlib.sha256(raw.encode()).hexdigest()==h
  x=json.loads(raw);r=lookup[qid]
  assert x["phase"]==phase and x["question_id"]==qid and x["method"]==m and x["generation_origin"]==origin
  assert x["sample_sha256"]==r["sample_sha256"] and x["image_sha256"]==r["image_sha256"] and x["gold"]==r["gold"]
  assert x["input_binding"]==bind[qid]["binding"] and x["protocol_sha256"]==ph and x["source_manifest_sha256"]==sh
  assert 1<=len(x["token_ids"])<=32 and len(x["token_ids"])==x["generated_tokens"]
  assert x["eos"]==(x["token_ids"][-1]==151645)
  assert x["truncated"]==(len(x["token_ids"])==32 and x["token_ids"][-1]!=151645)
  parsed,official=scoring.score("TextVQA",x["text"],r);ind=scoring.independent_score("TextVQA",x["text"],r)
  assert abs(official-ind)<1e-10 and abs(official-x["score"])<1e-10 and parsed==x["parsed_answer"]
  assert x["flags"]==anomaly(x)
  keep=list(range(28)) if m=="Native" else SPACE[m];skip=[] if m=="Native" else [2,3,4,5]
  lengths=[];live=0;capacity=0
  assert [a["layer"] for a in x["kv_layout"]]==list(range(28))
  for a in x["kv_layout"]:
   i=a["layer"];expected=0 if i in skip else r["input_tokens"]+len(x["token_ids"])-1-(0 if i in keep else 256)
   assert a["physical_length"]==expected and a["visual_kv"]==(256 if i in keep else 0)
   if expected:
    assert a["k_shape"]==[1,4,expected,128] and a["v_shape"]==a["k_shape"] and a["dtype"]=="torch.bfloat16"
   else:assert not a["k_shape"] and not a["v_shape"]
   expected_bytes=2*1*4*expected*128*2
   assert a["live_bytes"]==expected_bytes and a["capacity_bytes"]==expected_bytes
   live+=a["live_bytes"];capacity+=a["capacity_bytes"]
   kv.append({**a,**dict(phase=phase,question_id=qid,method=m,origin=origin,generated_tokens=len(x["token_ids"]),input_tokens=r["input_tokens"],cutoff=ALL_CONFIGS.get(m,{}).get("cutoff","none" if m=="Native" else 20))})
  assert abs(live/1048576-x["KV_live_MiB"])<1e-10 and abs(capacity/1048576-x["KV_capacity_MiB"])<1e-10
  if phase.startswith("speed"):
   assert len(x["token_ids"])==32
   assert math.isclose(x["TPOT_ms"],(x["E2E_ms"]-x["TTFT_ms"])/31,abs_tol=1e-8)
  x["diagnostic_categories"]=categories(x,r);x["record_sha256"]=h
  independent.append(dict(phase=phase,question_id=qid,method=m,official_score=official,independent_score=ind,record_sha256=h))
  flat.append({k:v for k,v in x.items() if k not in ("audit","kv_layout")} | dict(question=r["question"]))
  records.append(x)
 db.close()
 mainrows=[x for x in records if x["phase"]=="Main"];controls=[x for x in records if x["phase"]=="Controls"];speed=[x for x in records if x["phase"].startswith("speed")]
 assert len(mainrows)==768 and len(controls)==96 and len(speed)==96
 assert sum(x["generation_origin"]=="NEW" for x in mainrows+controls)==352
 for m in METHODS:assert [x["question_id"] for x in mainrows if x["method"]==m]==sorted(lookup)
 for m in CONTROLS:assert set(x["question_id"] for x in controls if x["method"]==m)==set(mf["control_ids"])
 summ={x["method"]:x for x in summarize(mainrows,"Main128")};summary=[summ[m] for m in METHODS]
 for row in summary:
  m=row["method"];row["origin"]="REUSED_PREVIOUS_GPU" if m in METHODS[:4] else "NEW_CURRENT_GPU"
  row["native_gap_pp"]=summ["Native"]["soft_accuracy_pct"]-row["soft_accuracy_pct"]
  row["recovered_native_minus_R0_pct"]=100*(row["soft_accuracy_pct"]-summ["R0"]["soft_accuracy_pct"])/(summ["Native"]["soft_accuracy_pct"]-summ["R0"]["soft_accuracy_pct"])
 pairs=[pair(mainrows,a,b,"Main128") for a,b in [("R0","Deep6_L20_25"),("R0","Deep8_L19_26"),("R6","Deep6_L20_25"),("R6","Deep8_L19_26"),("R14","Deep6_L20_25"),("R14","Deep8_L19_26"),("Deep6_L20_25","Deep8_L19_26"),("Native","Deep6_L20_25"),("Native","Deep8_L19_26")]]
 subset=[x for x in mainrows if x["question_id"] in set(mf["control_ids"])]
 control_summary=summarize(subset+controls,"Controls48");control_pairs=[pair(subset+controls,a,b,"Controls48") for a,b in [("Exit26_NoLateRead","Deep6_L20_25"),("Exit27_NoLateRead","Deep8_L19_26"),("R0","Exit26_NoLateRead"),("R0","Exit27_NoLateRead")]]
 changes=[]
 for phase,rr,pp in [("Main128",mainrows,pairs),("Controls48",subset+controls,control_pairs)]:
  for p in pp:
   a={x["question_id"]:x for x in rr if x["method"]==p["base"]};b={x["question_id"]:x for x in rr if x["method"]==p["candidate"]}
   for q in sorted(set(a)&set(b)):
    x,y=a[q],b[q];d=y["score"]-x["score"]
    changes.append(dict(phase=phase,question_id=q,image_sha256=x["image_sha256"],question=lookup[q]["question"],base=x["method"],candidate=y["method"],base_answer=x["text"],candidate_answer=y["text"],gold=x["gold"],base_score=x["score"],candidate_score=y["score"],difference=d,change="improved" if d>1e-10 else "declined" if d< -1e-10 else "unchanged",first_token_changed=x["token_ids"][0]!=y["token_ids"][0],full_text_changed=x["text"]!=y["text"],base_token_ids=x["token_ids"],candidate_token_ids=y["token_ids"],base_meta=x["flags"]["meta"],candidate_meta=y["flags"]["meta"],base_record_sha256=x["record_sha256"],candidate_record_sha256=y["record_sha256"]))
 meta=[];length=[]
 for phase,rr in [("Main128",mainrows),("Controls48",subset+controls)]:
  for m in dict.fromkeys(x["method"] for x in rr):
   xx=[x for x in rr if x["method"]==m];nn=[x["generated_tokens"] for x in xx]
   meta.append(dict(phase=phase,method=m,N=len(xx),**{k:sum(x["flags"][k] for x in xx) for k in ["meta","answering_prefix","repeat","empty","truncated"]},**{k:sum(x["diagnostic_categories"][k] for x in xx) for k in xx[0]["diagnostic_categories"]},scored_nonmeta=sum(x["score"]>0 and not x["flags"]["meta"] for x in xx)))
   length.append(dict(phase=phase,method=m,N=len(xx),mean=float(np.mean(nn)),P50=float(np.percentile(nn,50)),P95=float(np.percentile(nn,95)),maximum=max(nn),EOS=sum(x["eos"] for x in xx),truncated=sum(x["truncated"] for x in xx),empty=sum(x["empty"] for x in xx)))
 audit=read(OUT/"physical_cutoff_audit.json");smoke=read(OUT/"smoke_validation.json");sr=[];pr=[];probs=[]
 for a in audit["checks"]:
  cfg=ALL_CONFIGS[a["condition"]];p=a["physical"];d=a["dense"]
  assert p["cutoff"]==cfg["cutoff"] and p["hidden_exit"][0]["removed"]==256
  assert p["R_image_permission_boundary"]==20 and p["non_target_queries_preserved"]
  assert p["blocked_probability_max"]==0 and p["finite"]
  assert all(v==0 for _,_,v in p["blocked_image_probability_samples"]+p["R_image_probability_samples"])
  assert all(v>0 for _,_,v in p["allowed_image_probability_samples"])
  for variant,aa in [("Physical",p),("Dense",d)]:
   for step,i,operator,rows in aa["projection_shapes"]+aa["MLP_input_rows"]+aa["norm_input_rows"]:
    pr.append(dict(question_id=a["question_id"],method=a["condition"],variant=variant,step=step,layer=i,operator=operator,actual_input_rows=rows))
   for name in ["allowed_image_probability_samples","blocked_image_probability_samples","R_image_probability_samples"]:
    for step,i,v in aa[name]:probs.append(dict(question_id=a["question_id"],method=a["condition"],variant=variant,kind=name,step=step,layer=i,max_abs_image_probability=v))
 for m in ALL_CONFIGS:
  xx=[x for x in smoke["checks"] if x["condition"]==m];ll=[v for x in xx for v in x["aligned_prefix_logits"]]
  sr.append(dict(method=m,questions=len(xx),full_tokens_equal=sum(x["full_tokens_equal"] for x in xx),normalized_answers_equal=sum(x["normalized_answer_equal"] for x in xx),aligned_steps=len(ll),aligned_logit_max_abs=max(x["max_abs"] for x in ll),mean_aligned_step_mean_abs=float(np.mean([x["mean_abs"] for x in ll])),aligned_top1_equal=sum(x["top1_equal"] for x in ll),strict_numerical_equivalence=False))
 speed_summary=list(csv.DictReader((OUT/"speed_summary.csv").open()));ss={x["method"]:{k:(v if k=="method" else float(v)) for k,v in x.items()} for x in speed_summary}
 native=ss["Native"];speed_deltas=[]
 for m in ss:
  x=ss[m];d=dict(method=m,KV_reduction_pct=100*(1-x["KV_live_MiB"]/native["KV_live_MiB"]),E2E_reduction_pct=100*(1-x["E2E_ms"]/native["E2E_ms"]),E2E_speedup=native["E2E_ms"]/x["E2E_ms"],TPOT_reduction_pct=100*(1-x["TPOT_ms"]/native["TPOT_ms"]))
  deltas=[]
  for q in mf["speed_ids"]:
   a=np.mean([v["E2E_ms"] for v in speed if v["method"]=="Native" and v["question_id"]==q]);b=np.mean([v["E2E_ms"] for v in speed if v["method"]==m and v["question_id"]==q]);deltas.append(b-a)
  d.update(paired_E2E_difference_ms=float(np.mean(deltas)),paired_E2E_difference_median_ms=float(np.median(deltas)))
  speed_deltas.append(d)
 for name,rows in [("quality_accuracy.csv",summary),("paired_soft_changes.csv",pairs),("cutoff_only_controls.csv",control_pairs),("cutoff_control_accuracy.csv",control_summary),("meta_response_analysis.csv",meta),("per_question_changes.csv",changes),("per_question.csv",flat),("generation_length_summary.csv",length),("kv_layer_audit.csv",kv),("operator_input_rows.csv",pr),("attention_probability_audit.csv",probs),("numerical_smoke_summary.csv",sr),("independent_scoring.csv",independent),("speed_relative_to_native.csv",speed_deltas)]:csvout(name,rows)
 proc=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip()
 assert not proc and GPU in gpu
 write(OUT/"gpu_release.json",dict(status="RELEASED",compute_processes=proc,GPU=gpu,checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),server_kept_running=True))
 checks=dict(status="PASS_COMPLETENESS_STRUCTURAL_AND_SCORING",numerical_status="EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER",historical_numerical_status="NUMERICAL_EQUIVALENCE_FAILED",all_960_record_checksums_verified=True,quality_records=864,new_formal_quality=352,reused_quality=512,speed_formal_requests=96,independent_official_scores_all_equal=True,all_26880_layer_layouts_verified=True,actual_KV_live_and_storage_verified=True,input_source_protocol_hashes_verified=True,Native_NoOp_strict_equal=smoke["Native_NoOp_strict_logits"],main_unique_images=len(set(r["image_sha256"] for r in mf["Main"])),all_source_sha256_unchanged=all(sha(RUN/"code"/p)==h for p,h in read(OUT/"source_manifest.json")["runtime_source_sha256"].items()),model_loads=read(OUT/"run_completion.json")["model_loads"],GPU_released=True,limitations=["historical same-question reuse across explicitly authorized different GPU","128 historical questions are NOT independent Holdout","BF16 Dense/Physical numerical equivalence NOT passed","unadjusted exploratory 95% CI; multiple comparisons","speed 8 images x3 not full benchmark","scratch retains attention reduction width; cache savings do not equal FLOPs savings"])
 assert checks["all_source_sha256_unchanged"]
 write(OUT/"validation.json",checks)
 write(OUT/"analysis_digest.json",dict(quality=summary,pairs=pairs,controls=control_summary,control_pairs=control_pairs,numerical=sr,speed_relative=speed_deltas))
 print(json.dumps(dict(status=checks["status"],quality=[dict(method=x["method"],accuracy=x["soft_accuracy_pct"]) for x in summary],pairs=pairs,control_pairs=control_pairs,speed_relative=speed_deltas,numerical=sr),ensure_ascii=False),flush=True)
if __name__=="__main__":main()

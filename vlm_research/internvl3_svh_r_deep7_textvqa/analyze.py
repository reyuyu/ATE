"""Independent CPU-only rescoring and same-question paired statistics for Deep7."""
from .common import *
import math,subprocess
from vlm_research.internvl3_svh_seven_benchmarks import scoring
from vlm_research.internvl3_svh_r_deep_exit_textvqa.analyze import categories
scoring.RUN=RUN
METHODS=["Native","R0","R6","R14","Deep6_L20_25","Deep7_L20_26","Deep8_L19_26"]
def main():
 mf=read(OUT/"sample_manifest.json");lookup={r["question_id"]:r for r in mf["Main"]};bind=read(OUT/"input_bindings.json")
 db=sqlite3.connect("file:"+str(RUN/"checkpoints/predictions.sqlite")+"?mode=ro",uri=True);assert db.execute("pragma integrity_check").fetchone()==("ok",)
 ph=sha(OUT/"protocol.json");sh=sha(OUT/"source_manifest.json");mainrows=[];controls=[];flat=[];kv=[]
 assert dict(db.execute("select key,value from metadata"))==dict(protocol_sha256=ph,source_manifest_sha256=sh,sample_manifest_sha256=sha(OUT/"sample_manifest.json"))
 for phase,qid,m,origin,h,raw in db.execute("select phase,qid,method,origin,checksum,record from predictions order by phase,qid,method"):
  assert hashlib.sha256(raw.encode()).hexdigest()==h;x=json.loads(raw);r=lookup[qid]
  assert x["phase"]==phase and x["method"]==m and x["generation_origin"]==origin and x["question_id"]==qid
  assert x["protocol_sha256"]==ph and x["source_manifest_sha256"]==sh and x["input_binding"]==bind[qid]["binding"]
  assert x["sample_sha256"]==r["sample_sha256"] and x["image_sha256"]==r["image_sha256"] and x["gold"]==r["gold"]
  ids=x["token_ids"];assert 1<=len(ids)<=32 and len(ids)==x["generated_tokens"] and x["eos"]==(ids[-1]==151645) and x["truncated"]==(len(ids)==32 and ids[-1]!=151645)
  parsed,score=scoring.score("TextVQA",x["text"],r);ind=scoring.independent_score("TextVQA",x["text"],r)
  assert abs(score-ind)<1e-10 and abs(score-x["score"])<1e-10 and parsed==x["parsed_answer"] and x["flags"]==anomaly(x)
  keep=list(range(28)) if m=="Native" else SPACE[m];skip=[] if m=="Native" else [2,3,4,5];live=0;capacity=0
  assert [a["layer"] for a in x["kv_layout"]]==list(range(28))
  for a in x["kv_layout"]:
   i=a["layer"];n=0 if i in skip else r["input_tokens"]+len(ids)-1-(0 if i in keep else 256);b=2*1*4*n*128*2
   assert a["physical_length"]==n and a["live_bytes"]==b and a["capacity_bytes"]==b and a["visual_kv"]==(256 if i in keep else 0)
   if n:assert a["k_shape"]==a["v_shape"]==[1,4,n,128] and a["dtype"]=="torch.bfloat16"
   else:assert a["k_shape"]==a["v_shape"]==[]
   live+=b;capacity+=b;kv.append({**a,**dict(phase=phase,question_id=qid,method=m,generated_tokens=len(ids),origin=origin)})
  assert live/1048576==x["KV_live_MiB"] and capacity/1048576==x["KV_capacity_MiB"]
  x["diagnostic_categories"]=categories(x,r);x["record_sha256"]=h
  (mainrows if phase=="Main" else controls).append(x);flat.append({k:v for k,v in x.items() if k not in ("audit","kv_layout")} | dict(question=r["question"]))
 db.close();assert len(mainrows)==896 and len(controls)==48
 assert sum(x["generation_origin"]=="NEW" for x in mainrows+controls)==128
 for m in METHODS:assert len([x for x in mainrows if x["method"]==m])==128
 assert set(x["question_id"] for x in controls)==set(mf["control_ids"]) and all(x["method"]=="Exit27_NoLateRead" for x in controls)
 sm={x["method"]:x for x in summarize(mainrows,"Main128")};summary=[sm[m] for m in METHODS]
 for x in summary:
  m=x["method"];x["native_gap_pp"]=sm["Native"]["soft_accuracy_pct"]-x["soft_accuracy_pct"]
  x["recovery_of_native_minus_R0_pct"]=100*(x["soft_accuracy_pct"]-sm["R0"]["soft_accuracy_pct"])/(sm["Native"]["soft_accuracy_pct"]-sm["R0"]["soft_accuracy_pct"])
  x["source"]="NEW_CURRENT_GPU" if m=="Deep7_L20_26" else "REUSED_OLD_GPU" if m in METHODS[:4] else "REUSED_SAME_CURRENT_GPU"
 pairs=[pair(mainrows,b,"Deep7_L20_26","Main128") for b in METHODS if b!="Deep7_L20_26"]
 subset=[x for x in mainrows if x["question_id"] in set(mf["control_ids"])]
 cp=pair(subset+controls,"Exit27_NoLateRead","Deep7_L20_26","Controls48");cs=summarize(subset+controls,"Controls48")
 changes=[]
 for p,rr in [(p,mainrows) for p in pairs]+[(cp,subset+controls)]:
  a={x["question_id"]:x for x in rr if x["method"]==p["base"]};b={x["question_id"]:x for x in rr if x["method"]=="Deep7_L20_26"}
  for q in sorted(set(a)&set(b)):
   x,y=a[q],b[q];delta=y["score"]-x["score"]
   changes.append(dict(phase=p["phase"],question_id=q,image_sha256=x["image_sha256"],question=lookup[q]["question"],base=x["method"],candidate=y["method"],base_text=x["text"],candidate_text=y["text"],gold=x["gold"],base_score=x["score"],candidate_score=y["score"],difference=delta,change="improved" if delta>1e-10 else "declined" if delta< -1e-10 else "unchanged",base_token_ids=x["token_ids"],candidate_token_ids=y["token_ids"],base_record_sha256=x["record_sha256"],candidate_record_sha256=y["record_sha256"]))
 meta=[];length=[]
 for m in METHODS:
  rr=[x for x in mainrows if x["method"]==m];nn=[x["generated_tokens"] for x in rr]
  meta.append(dict(method=m,N=128,**{k:sum(x["flags"][k] for x in rr) for k in ["meta","answering_prefix","repeat","empty","truncated"]},**{k:sum(x["diagnostic_categories"][k] for x in rr) for k in rr[0]["diagnostic_categories"]},scored_nonmeta=sum(x["score"]>0 and not x["flags"]["meta"] for x in rr)))
  length.append(dict(method=m,P50=float(np.percentile(nn,50)),P95=float(np.percentile(nn,95)),maximum=max(nn),EOS=sum(x["eos"] for x in rr),empty=sum(x["empty"] for x in rr),truncated=sum(x["truncated"] for x in rr)))
 smoke=read(OUT/"smoke_validation.json");audit=read(OUT/"physical_cutoff_audit.json");prows=[];prob=[]
 for a in audit["checks"]:
  p=a["physical"];assert p["cutoff"]==27 and p["hidden_exit"][0]["removed"]==256 and p["R_image_permission_boundary"]==20
  assert p["blocked_probability_max"]==0 and p["finite"] and p["non_target_queries_preserved"]
  assert all(v==0 for _,_,v in p["blocked_image_probability_samples"]+p["R_image_probability_samples"])
  assert all(v>0 for _,_,v in p["allowed_image_probability_samples"])
  for variant,aa in [("Physical",p),("Dense",a["dense"])]:
   for step,i,op,n in aa["projection_shapes"]+aa["MLP_input_rows"]+aa["norm_input_rows"]:prows.append(dict(question_id=a["question_id"],variant=variant,step=step,layer=i,operator=op,input_rows=n))
   for name in ["allowed_image_probability_samples","blocked_image_probability_samples","R_image_probability_samples"]:
    for step,i,v in aa[name]:prob.append(dict(question_id=a["question_id"],variant=variant,kind=name,step=step,layer=i,probability_max=v))
 ll=[v for x in smoke["checks"] for v in x["aligned_logits"]]
 num=dict(questions=3,full_tokens_equal=sum(x["full_tokens_equal"] for x in smoke["checks"]),normalized_answers_equal=sum(x["normalized_answers_equal"] for x in smoke["checks"]),aligned_steps=len(ll),logits_max_abs=max(x["max_abs"] for x in ll),mean_step_mean_abs=float(np.mean([x["mean_abs"] for x in ll])),strict_equivalence=False)
 for name,rr in [("quality_accuracy.csv",summary),("paired_soft_changes.csv",pairs),("cutoff_only_controls.csv",[cp]),("cutoff_control_accuracy.csv",cs),("per_question.csv",flat),("per_question_changes.csv",changes),("kv_layer_audit.csv",kv),("meta_response_analysis.csv",meta),("generation_length_summary.csv",length),("operator_input_rows.csv",prows),("attention_probability_audit.csv",prob),("numerical_smoke_summary.csv",[num])]:csvout(name,rr)
 proc=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip();assert not proc
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip();assert GPU in gpu
 write(OUT/"gpu_release.json",dict(status="RELEASED",GPU=gpu,compute_processes=proc,server_kept_running=True))
 valid=dict(status="PASS_COMPLETENESS_STRUCTURE_SCORING",numerical_status="EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER",historical_status="NUMERICAL_EQUIVALENCE_FAILED",records_verified=944,new_formal_quality=128,reused=816,new_speed=0,all_record_checksums_inputs_source_protocol_and_independent_scores_verified=True,all_real_KV_live_capacity_and_positions_verified=True,Native_NoOp_strict_equal=smoke["Native_NoOp_strict_equal"],model_loads=read(OUT/"run_completion.json")["model_loads"],GPU=GPU,GPU_released=True)
 assert all(sha(RUN/"code"/p)==h for p,h in read(OUT/"source_manifest.json")["runtime_source_sha256"].items())
 write(OUT/"validation.json",valid);write(OUT/"analysis_digest.json",dict(quality=summary,pairs=pairs,controls=cs,control_pair=cp,numerical=num,meta=meta,length=length))
 print(json.dumps(dict(quality=summary,pairs=pairs,control_pair=cp,numerical=num),ensure_ascii=False),flush=True)
if __name__=="__main__":main()

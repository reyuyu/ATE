"""Single-load fixed TextVQA R0/R6/R14 validation; A reuses R0/R6, B regenerates all three."""
from .common import *
import torch,gc,subprocess,traceback
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
from vlm_research.internvl3_svh_r_middle_reaccess_search import engine as e
def telemetry():return subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip()
def main():
 mf=read(OUT/"sample_manifest.json");ph=sha(OUT/"protocol.json");source=read(OUT/"source_manifest.json");sh=sha(OUT/"source_manifest.json")
 for p,h in source["runtime_source_sha256"].items():assert sha(RUN/"code"/p)==h,("SOURCE_HASH",p)
 assert read(OUT/"cpu_tests.json")["status"]=="PASS" and read(OUT/"input_reuse_validation.json")["status"].startswith("PASS")
 assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()==GPU
 assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(),"GPU_BUSY"
 bindings=read(RUN/"prepared/input_bindings.json");reuseA=read(RUN/"prepared/reused_A_records.json");reuseN=read(RUN/"prepared/reused_native_B_smoke.json");byq={r["question_id"]:r for r in mf["GroupA"]+mf["GroupB"]+mf["Smoke"]}
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");db.execute("PRAGMA journal_mode=WAL");db.execute("CREATE TABLE IF NOT EXISTS predictions(phase TEXT,qid TEXT,method TEXT,origin TEXT,record TEXT,PRIMARY KEY(phase,qid,method))");db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)");meta=dict(db.execute("SELECT key,value FROM meta"))
 for k,v in dict(protocol_sha256=ph,source_manifest_sha256=sh).items():
  if k in meta:assert meta[k]==v,("CHECKPOINT_HASH",k)
  else:db.execute("INSERT INTO meta VALUES(?,?)",(k,v))
 db.commit()
 def records(phase=None):return [json.loads(x[0]) for x in db.execute("SELECT record FROM predictions"+(" WHERE phase=?" if phase else ""),(phase,) if phase else ())]
 def get(phase,q,m):
  x=db.execute("SELECT record FROM predictions WHERE phase=? AND qid=? AND method=?",(phase,q,m)).fetchone();return json.loads(x[0]) if x else None
 def count():return db.execute("SELECT count(*) FROM predictions WHERE phase in ('A','B') AND origin='NEW'").fetchone()[0]
 def save(phase,r,m,x,origin):
  x.update(phase=phase,method=m,generation_origin=origin,protocol_sha256=ph,source_manifest_sha256=sh,input_binding=bindings[r["question_id"]]["binding"],sample_sha256=r["sample_sha256"])
  db.execute("INSERT INTO predictions VALUES(?,?,?,?,?)",(phase,r["question_id"],m,origin,json.dumps(x,ensure_ascii=False)))
 def status(phase,**kw):
  j=dict(phase=phase,new_quality_generations=count(),updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),GPU=telemetry(),**kw);write(OUT/"progress.json",j);print(json.dumps(j),flush=True)
 status("MODEL_LOADING");model,tok=w.load_model();write(OUT/"environment.json",w.environment(model,tok));hooks=e.frozen.hooks_count(model)
 def ready(r):
  pix=torch.load(RUN/"prepared"/(r["image_sha256"]+".pt"),map_location="cpu",weights_only=True);assert w.tensor_sha(pix)==bindings[r["question_id"]]["details"]["pixels_bf16_sha256"]
  c=model.conv_template.copy();c.system_message=model.system_message;c.append_message(c.roles[0],r["request"]);c.append_message(c.roles[1],None);p=c.get_prompt().replace("<image>","<img>"+"<IMG_CONTEXT>"*256+"</img>",1);assert p==r["prompt"]
  return dict(ids=torch.tensor([r["input_ids"]],device="cuda:0",dtype=torch.long),pixels=pix.to("cuda:0"),roles=r["roles"],request=r["request"],prompt=p)
 def call(r,rd,m,physical=True,audit=False,capture=False,timed=False,fixed=False):
  collected=[];h=None
  if capture and not audit:
   def hook(module,args,out):collected.append(out.logits[0,-1].detach().float().cpu())
   h=model.language_model.register_forward_hook(hook)
  try:
   x,s,ctl=e.generate(model,tok,rd,m,SPACE.get(m,[]),max_tokens=32,fixed=fixed,audit=audit,timed=timed,physical=physical)
   if collected:s=collected
   assert len(x["token_ids"])<=32 and x["independent_cache"] and e.frozen.hooks_count(model)==hooks+(1 if h else 0)
   return quality(x,r),s,ctl
  finally:
   if h:h.remove()
 def import_record(phase,r,m,info):
  x=json.loads(json.dumps(info["record"]));assert x["input_binding"]==bindings[r["question_id"]]["binding"]
  x["reused_from"]=dict(parent_record_sha256=info["original_record_sha256"],protocol_sha256=x["protocol_sha256"],source_manifest_sha256=x["source_manifest_sha256"],checkpoint="parent fast TextVQA" if phase=="A" else "frozen full TextVQA Native")
  x["KV_capacity_MiB"]=sum(v["capacity_bytes"] for v in x["kv_layout"])/1048576;save(phase,r,m,quality(x,r),"REUSED")
 if not (OUT/"smoke_validation.json").exists():
  checks=[];audits=[]
  for i,r in enumerate(mf["Smoke"]):
   rd=ready(r)
   if i==0:
    nx,ns,nc=call(r,rd,"Native",audit=True);no,ss,cc=call(r,rd,"No-op",audit=True)
    assert nx["token_ids"]==no["token_ids"]==reuseN[r["question_id"]]["record"]["token_ids"]
    assert len(ns)==len(ss) and all(torch.equal(a,b) for a,b in zip(ns,ss)),"NATIVE_NOOP_LOGITS"
    del nx,ns,nc,no,ss,cc
   heavy=i in (0,len(mf["Smoke"])-1);dx,ds,dc=call(r,rd,"R14",physical=False,audit=heavy,capture=True);px,ps,pc=call(r,rd,"R14",physical=True,audit=heavy,capture=True)
   assert len(px["kv_layout"])==28
   for v in px["kv_layout"]:
    l=v["layer"];assert v["visual_kv"]==(256 if l in SPACE["R14"] else 0)
    assert v["physical_length"]==(0 if l in e.SKIP else r["input_tokens"]+px["generated_tokens"]-1-(0 if l in SPACE["R14"] else 256))
   diffs=[]
   for j,(a,b) in enumerate(zip(ds,ps)):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    if dx["token_ids"][:j]!=px["token_ids"][:j]:break
    d=(a.float()-b.float()).abs();diffs.append(dict(step=j,max_abs=float(d.max()),mean_abs=float(d.mean()),top1_equal=int(a.argmax())==int(b.argmax())))
   checks.append(dict(question_id=r["question_id"],input_binding=bindings[r["question_id"]]["binding"],dense={k:v for k,v in dx.items() if k!="audit"},physical={k:v for k,v in px.items() if k!="audit"},aligned_prefix_logits=diffs,full_tokens_equal=dx["token_ids"]==px["token_ids"],normalized_answer_equal=dx["parsed_answer"]==px["parsed_answer"],structural_audit=heavy))
   if heavy:audits.append(dict(question_id=r["question_id"],dense=dx["audit"],physical=px["audit"]))
   write(OUT/"smoke_checks_partial.json",checks);write(OUT/"smoke_structure_audits.json",audits);del dx,ds,dc,px,ps,pc,rd;gc.collect();torch.cuda.empty_cache();status("SMOKE",completed_inputs=i+1,total_inputs=6)
  write(OUT/"smoke_validation.json",dict(status="PASS_WITH_NUMERICAL_WAIVER",Native_NoOp_strict_logits=True,Native_historical_tokens_match=True,questions=6,condition="Dense-R14 vs Physical-R14",R14_image_KV_layers=SPACE["R14"],full_token_equal=sum(x["full_tokens_equal"] for x in checks),normalized_answer_equal=sum(x["normalized_answer_equal"] for x in checks),checks=checks,large_raw_tensors_saved=False,numerical_equivalence_claimed=False,structural_errors=0))
 else:assert read(OUT/"smoke_validation.json")["status"]=="PASS_WITH_NUMERICAL_WAIVER"
 j=read(OUT/"input_reuse_validation.json");j.update(status="PASS",same_GPU_environment_and_Native_NoOp_checked=True);write(OUT/"input_reuse_validation.json",j)
 status("A",new_requests=96)
 for i,r in enumerate(mf["GroupA"]):
  q=r["question_id"]
  for m in ["Native","R0","R6"]:
   if not get("A",q,m):import_record("A",r,m,reuseA[q][m])
  if not get("A",q,"R14"):
   assert count()<480;rd=ready(r);x,s,c=call(r,rd,"R14");save("A",r,"R14",x,"NEW");del x,s,c,rd
  db.commit()
  if (i+1)%8==0:
   status("A",completed_inputs=i+1,total_inputs=96);csvout("group_A_accuracy.csv",summarize(records("A"),"A_historical_followup96"))
 csvout("group_A_accuracy.csv",summarize(records("A"),"A_historical_followup96"));status("A_COMPLETE")
 status("B",new_requests=384)
 methods=["R0","R6","R14"]
 for i,r in enumerate(mf["GroupB"]):
  q=r["question_id"];missing=[m for m in methods if not get("B",q,m)]
  if missing:
   rd=ready(r);offset=i%3
   for m in methods[offset:]+methods[:offset]:
    if m not in missing:continue
    assert count()<480;x,s,c=call(r,rd,m);save("B",r,m,x,"NEW");del x,s,c
   del rd
  if not get("B",q,"Native"):import_record("B",r,"Native",reuseN[q])
  db.commit()
  if (i+1)%8==0:
   status("B",completed_inputs=i+1,total_inputs=128);csvout("group_B_accuracy.csv",summarize(records("B"),"B_independent128"))
 B=records("B");csvout("group_B_accuracy.csv",summarize(B,"B_independent128"));p0=pair(B,"R0","R14","B_independent128");p6=pair(B,"R6","R14","B_independent128")
 positive=p0["CI95_low_pp"]>0 and p6["CI95_low_pp"]>0
 gate=dict(quality_status="EXPLORATORY_R14_GAIN_OVER_R0_AND_R6" if positive else "NO_CONFIRMED_R14_GAIN_OVER_BOTH_BASELINES",speed_status="TO_RUN" if positive else "NOT_RUN",frozen_gate="B R14-R0 and R14-R6 ordinary95%CI lower both>0",R14_minus_R0=p0,R14_minus_R6=p6)
 write(OUT/"quality_decision.json",gate);status("B_COMPLETE",speed_eligible=positive)
 speed=[]
 if positive:
  methods=["Native","R0","R6","R14"]
  for m in methods:
   for q in mf["speed_ids"][:2]:
    rd=ready(byq[q]);x,s,c=call(byq[q],rd,m,timed=True,fixed=True);del x,s,c,rd
  for repeat in range(3):
   for i,q in enumerate(mf["speed_ids"]):
    r=byq[q];rd=ready(r);offset=(i+repeat)%4;phase="speed_"+str(repeat)
    for m in methods[offset:]+methods[:offset]:
     x=get(phase,q,m)
     if not x:
      x,s,c=call(r,rd,m,timed=True,fixed=True);x["repeat"]=repeat;save(phase,r,m,x,"NEW");del s,c
     speed.append(x)
    db.commit();del rd
    if (i+1)%4==0:status("SPEED",repeat=repeat+1,completed_inputs=i+1)
  metrics=["E2E_ms","TTFT_ms","TPOT_ms","incremental_decode_ms","vision_encoder_ms","vision_projector_injection_ms","language_prefill_ms","CUDA_event_TPOT_ms","KV_live_MiB","KV_capacity_MiB","peak_allocated_MiB","peak_reserved_MiB"];ss=[]
  for m in methods:
   vals=[x for x in speed if x["method"]==m];row=dict(method=m,requests=len(vals),images=8,repeats=3)
   for k in metrics:row[k]=float(np.mean([x[k] for x in vals]));row[k+"_median"]=float(np.median([x[k] for x in vals]))
   ss.append(row)
  csvout("speed_summary.csv",ss);csvout("speed_per_input.csv",[{k:v for k,v in x.items() if k not in ("audit","kv_layout")} for x in speed]);gate["speed_status"]="COMPLETE";write(OUT/"quality_decision.json",gate)
 else:csvout("speed_summary.csv",[dict(status="NOT_RUN",reason="R14 failed the frozen GroupB gain-versus-both-R0-and-R6 criterion; no timing or KV@32 inferred from old data")])
 assert count()==480 and e.frozen.hooks_count(model)==hooks
 write(OUT/"run_completion.json",dict(status="COMPLETE",quality_generated=count(),group_A_new_R14=96,group_A_reused=288,group_B_new=384,group_B_Native_reused=128,formal_speed_requests=len(speed),speed_status=gate["speed_status"],quality_status=gate["quality_status"],model_loads=1,hooks_restored=True))
 del model,tok;gc.collect();torch.cuda.empty_cache();status("COMPLETE_GPU_RELEASED");db.close();write(OUT/"gpu_release.json",dict(GPU=telemetry(),worker_exits_after_receipt=True))
if __name__=="__main__":
 try:main()
 except Exception as ex:write(OUT/"failure.json",dict(status="FAILED_OR_PAUSED",exception=repr(ex),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True);raise

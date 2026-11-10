"""One resident model, TextVQA-only staged quality search, gated real speed."""
from .common import *
import torch,gc,subprocess,traceback
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
from vlm_research.internvl3_svh_r_middle_reaccess_search import engine as e
def telemetry():return subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip()
def main():
 manifest=read(OUT/"sample_manifest.json");protocol=read(OUT/"protocol.json");ph=sha(OUT/"protocol.json");source=read(OUT/"source_manifest.json");sh=sha(OUT/"source_manifest.json")
 assert read(OUT/"image_only_input_audit.json")["status"]=="PASS" and read(OUT/"cpu_tests.json")["status"]=="PASS"
 for path,h in source["runtime_source_sha256"].items():assert sha(RUN/"code"/path)==h,("SOURCE_MISMATCH",path)
 assert subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()==GPU
 assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(),"GPU_BUSY"
 bindings=read(RUN/"prepared/input_bindings.json");allrows={r["question_id"]:r for r in manifest["Discovery"]+manifest["Holdout"]+manifest["Smoke"]};hold_ids={r["question_id"] for r in manifest["Holdout"]}
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");db.execute("PRAGMA journal_mode=WAL");db.execute("CREATE TABLE IF NOT EXISTS predictions(phase TEXT,qid TEXT,method TEXT,record TEXT,PRIMARY KEY(phase,qid,method))");db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)")
 old=dict(db.execute("SELECT key,value FROM meta"))
 for key,v in {"protocol_sha256":ph,"source_manifest_sha256":sh}.items():
  if key in old:assert old[key]==v,("CHECKPOINT_BINDING",key)
  else:db.execute("INSERT INTO meta VALUES(?,?)",(key,v))
 db.commit();historical=sqlite3.connect("file:"+str(BASE/"checkpoints/TextVQA.sqlite")+"?mode=ro",uri=True)
 def records(phase=None):return [json.loads(x[0]) for x in db.execute("SELECT record FROM predictions"+(" WHERE phase=?" if phase else ""),(phase,) if phase else ())]
 def get(phase,q,m):
  x=db.execute("SELECT record FROM predictions WHERE phase=? AND qid=? AND method=?",(phase,q,m)).fetchone();return json.loads(x[0]) if x else None
 def save(phase,r,m,x):
  x.update(phase=phase,method=m,protocol_sha256=ph,source_manifest_sha256=sh,input_binding=bindings[r["question_id"]]["binding"],sample_sha256=r["sample_sha256"])
  db.execute("INSERT INTO predictions VALUES(?,?,?,?)",(phase,r["question_id"],m,json.dumps(x,ensure_ascii=False)))
 def count():return db.execute("SELECT count(*) FROM predictions WHERE phase in ('A','B','Holdout') AND method!='Native'").fetchone()[0]
 def status(phase,**kw):
  x=dict(phase=phase,quality_generated=count(),updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),GPU=telemetry(),**kw);write(OUT/"progress.json",x);print(json.dumps(x),flush=True)
 def native(r):
  if r["question_id"] in hold_ids:
   assert (OUT/"selection_freeze_receipt.json").exists(),"HOLDOUT_OPENED_BEFORE_SELECTION"
   assert sha(OUT/"selection.json")==read(OUT/"selection_freeze_receipt.json")["selection_sha256"]
  x=json.loads(historical.execute("SELECT record FROM predictions WHERE qid=? AND method='native'",(r["question_id"],)).fetchone()[0])
  assert x["input_binding"]==bindings[r["question_id"]]["binding"] and x["input_binding_details"]==bindings[r["question_id"]]["details"] and x["max_new_tokens"]==32
  return x
 status("MODEL_LOADING");model,tok=w.load_model();write(OUT/"environment.json",w.environment(model,tok));hooks=e.frozen.hooks_count(model)
 def ready(r):
  pix=torch.load(RUN/"prepared"/(r["image_sha256"]+".pt"),map_location="cpu",weights_only=True);assert w.tensor_sha(pix)==bindings[r["question_id"]]["details"]["pixels_bf16_sha256"]
  c=model.conv_template.copy();c.system_message=model.system_message;c.append_message(c.roles[0],r["request"]);c.append_message(c.roles[1],None);prompt=c.get_prompt().replace("<image>","<img>"+"<IMG_CONTEXT>"*256+"</img>",1);assert prompt==r["prompt"]
  rd=dict(ids=torch.tensor([r["input_ids"]],device="cuda:0",dtype=torch.long),pixels=pix.to("cuda:0"),roles=r["roles"],request=r["request"],prompt=prompt);assert set(rd)=={"ids","pixels","roles","request","prompt"}
  return rd
 def call(r,rd,m,physical=True,audit=False,capture=False,fixed=False,timed=False):
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
 def compact(x):return {k:v for k,v in x.items() if k!="audit"}
 if not (OUT/"smoke_validation.json").exists():
  checks=[];structures=[];native_ids=[];noop=False
  for i,q in enumerate(manifest["smoke_ids"]):
   r=allrows[q];rd=ready(r);heavy=i in (0,7)
   nx,ns,nc=call(r,rd,"Native",audit=not noop,capture=True);oldn=native(r)
   assert nx["token_ids"]==oldn["token_ids"] and nx["text"]==oldn["text"],("NATIVE_HISTORICAL_MISMATCH",q)
   native_ids.append(q)
   if not noop:
    no,ss,cc=call(r,rd,"No-op",audit=True);assert nx["token_ids"]==no["token_ids"] and len(ns)==len(ss) and all(torch.equal(a,b) for a,b in zip(ns,ss)),"NATIVE_NOOP_LOGITS";noop=True;del no,ss,cc
   del nx,ns,nc,oldn
   for m in ["R0","R4-M","R8"]:
    dx,ds,dc=call(r,rd,m,physical=False,audit=heavy,capture=True);px,ps,pc=call(r,rd,m,physical=True,audit=heavy,capture=True);diffs=[]
    for row in px["kv_layout"]:
     layer=row["layer"];assert row["visual_kv"]==(256 if layer in SPACE[m] else 0)
     assert row["physical_length"]==(0 if layer in e.SKIP else r["input_tokens"]+px["generated_tokens"]-1-(0 if layer in SPACE[m] else 256))
    for step,(a,b) in enumerate(zip(ds,ps)):
     assert torch.isfinite(a).all() and torch.isfinite(b).all()
     if dx["token_ids"][:step]!=px["token_ids"][:step]:break
     d=(a.float()-b.float()).abs();diffs.append(dict(step=step,max_abs=float(d.max()),mean_abs=float(d.mean()),top1_equal=int(a.argmax())==int(b.argmax())))
    assert diffs
    checks.append(dict(question_id=q,condition=m,input_binding=bindings[q]["binding"],dense=compact(dx),physical=compact(px),aligned_prefix_logits=diffs,full_tokens_equal=dx["token_ids"]==px["token_ids"],normalized_answer_equal=dx["parsed_answer"]==px["parsed_answer"],structural_audit=heavy))
    if heavy:structures.append(dict(question_id=q,condition=m,dense=dx["audit"],physical=px["audit"]))
    write(OUT/"smoke_checks_partial.json",checks);write(OUT/"smoke_structure_audits.json",structures);del dx,ds,dc,px,ps,pc
   del rd;gc.collect();torch.cuda.empty_cache();status("SMOKE",completed_inputs=i+1,total_inputs=8)
  summary={"status":"PASS_WITH_NUMERICAL_WAIVER","Native_NoOp_strict_logits_equal":True,"historical_Native_full_tokens_equal":native_ids,"smoke_discovery_holdout_image_overlap":0,"checks":checks,"structural_errors":0,"large_raw_tensors_saved":False,"known_meta_output_not_stop_condition":True,"numerical_equivalence_claimed":False}
  for m in ["R0","R4-M","R8"]:
   rr=[x for x in checks if x["condition"]==m];summary[m]=dict(N=8,full_tokens_equal=sum(x["full_tokens_equal"] for x in rr),normalized_answer_equal=sum(x["normalized_answer_equal"] for x in rr),physical_meta=sum(x["physical"]["flags"]["meta"] for x in rr),dense_meta=sum(x["dense"]["flags"]["meta"] for x in rr))
  write(OUT/"smoke_validation.json",summary)
 else:assert read(OUT/"smoke_validation.json")["status"]=="PASS_WITH_NUMERICAL_WAIVER"
 nr=read(OUT/"native_reuse_validation.json");nr.update(status="PASS",GPU_smoke_Native_exact_on_all_8=True,Holdout_predictions_not_read=True);write(OUT/"native_reuse_validation.json",nr)
 def phase(name,ids,methods):
  status(name,total=len(ids)*len(methods))
  # Each phase contains one frozen question per image; methods rotate within that image's prepared GPU input.
  for i,q in enumerate(ids):
   r=allrows[q];missing=[m for m in methods if not get(name,q,m)]
   if missing:
    rd=ready(r);rot=methods[i%len(methods):]+methods[:i%len(methods)]
    for m in rot:
     if m not in missing:continue
     assert count()<816,"FORMAL_QUALITY_BUDGET"
     x,ss,cc=call(r,rd,m);save(name,r,m,x);del x,ss,cc
    del rd
   if not get(name,q,"Native"):
    x=native(r);x["provenance"]={"source_manifest_sha256":x["source_manifest_sha256"],"protocol_sha256":x["protocol_sha256"],"historical_native":True};x["method"]="Native";x["KV_capacity_MiB"]=sum(a["capacity_bytes"] for a in x["kv_layout"])/1048576;x=quality(x,r);save(name,r,"Native",x)
   db.commit() # One small atomic batch per image, not heavy tensor trajectories.
   if (i+1)%8==0 or i+1==len(ids):
    status(name,completed_inputs=i+1,total_inputs=len(ids))
    csvout("stage_A_accuracy.csv" if name=="A" else "discovery_accuracy.csv" if name=="B" else "holdout_accuracy.csv",summarize(records("A") if name=="A" else records("A")+records("B") if name=="B" else records("Holdout"),name))
 phase("A",manifest["stage_A_ids"],ORDER);a=selections([x for x in records("A") if x["method"]!="Native"],"A");write(OUT/"selection_stage_A.json",a);status("A_COMPLETE",selected=a["selected"])
 phase("B",manifest["stage_B_ids"],a["selected"]);disc=[x for x in records("A")+records("B") if x["method"] in a["selected"]];selection=selections(disc,"B");selection.update(stage_A=a,protocol_sha256=ph,source_manifest_sha256=sh,selection_time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),Holdout_predictions_read_before_selection=0)
 if (OUT/"selection.json").exists():
  prior=read(OUT/"selection.json");assert prior["best_quality"]==selection["best_quality"] and prior["best_efficiency"]==selection["best_efficiency"];selection=prior
 else:write(OUT/"selection.json",selection)
 selsha=sha(OUT/"selection.json");write(OUT/"selection_freeze_receipt.json",dict(selection_sha256=selsha,before_any_Holdout_prediction_payload=True,holdout_methods=selection["holdout_methods"]))
 csvout("discovery_accuracy.csv",summarize(disc+[x for x in records("A")+records("B") if x["method"]=="Native"],"Discovery96"));status("B_COMPLETE",best_quality=selection["best_quality"],best_efficiency=selection["best_efficiency"])
 phase("Holdout",[r["question_id"] for r in manifest["Holdout"]],selection["holdout_methods"]);assert sha(OUT/"selection.json")==selsha
 hold=records("Holdout");csvout("holdout_accuracy.csv",summarize(hold,"Holdout96"));comparisons=[pair(hold,"R0",m,"Holdout96") for m in dict.fromkeys([selection["best_quality"],selection["best_efficiency"]]) if m!="R0"]
 confirmed=[p["candidate"] for p in comparisons if p["difference_pp"]>0 and p["CI95_low_pp"]>0]
 precedence=list(dict.fromkeys([selection["best_quality"],selection["best_efficiency"]]))
 speed_candidate=next((m for m in precedence if m in confirmed),None)
 gate={"quality_status":"CONFIRMED_EXPLORATORY_TEXTVQA_RECOVERY" if confirmed else "NO_CONFIRMED_TEXTVQA_RECOVERY","speed_status":"TO_RUN" if speed_candidate else "NOT_RUN","frozen_gate":"gain>0 and ordinary95%CI lower>0, best_quality-before-best_efficiency precedence","paired_holdout":comparisons,"confirmed_preselected_candidates":confirmed,"speed_candidate":speed_candidate,"selection_sha256":selsha,"Holdout_reselection":False}
 write(OUT/"holdout_decision.json",gate);status("HOLDOUT_COMPLETE",decision=gate["quality_status"],speed_candidate=speed_candidate)
 speed=[]
 if speed_candidate:
  methods=["Native","R0",speed_candidate]
  for m in methods:
   for q in manifest["speed_ids"][:2]:
    rd=ready(allrows[q]);x,ss,cc=call(allrows[q],rd,m,fixed=True,timed=True);del x,rd,ss,cc
  for repeat in range(3):
   for i,q in enumerate(manifest["speed_ids"]):
    r=allrows[q];rd=ready(r);offset=(i+repeat)%len(methods)
    for m in methods[offset:]+methods[:offset]:
     phasekey="speed_"+str(repeat);x=get(phasekey,q,m)
     if not x:
      x,ss,cc=call(r,rd,m,fixed=True,timed=True);x["repeat"]=repeat;save(phasekey,r,m,x);del ss,cc
     speed.append(x)
    db.commit();del rd
    if (i+1)%4==0:status("SPEED",repeat=repeat+1,completed_inputs=i+1)
  metrics=["E2E_ms","TTFT_ms","TPOT_ms","incremental_decode_ms","vision_encoder_ms","vision_projector_injection_ms","language_prefill_ms","CUDA_event_TPOT_ms","KV_live_MiB","KV_capacity_MiB","peak_allocated_MiB","peak_reserved_MiB"];srows=[]
  for m in methods:
   rr=[x for x in speed if x["method"]==m];row=dict(method=m,requests=len(rr),images=12,repeats=3)
   for k in metrics:row[k]=float(np.mean([x[k] for x in rr]));row[k+"_median"]=float(np.median([x[k] for x in rr]))
   srows.append(row)
  csvout("speed_summary.csv",srows);csvout("speed_per_input.csv",[{k:v for k,v in x.items() if k not in ("audit","kv_layout")} for x in speed]);gate["speed_status"]="COMPLETE";write(OUT/"holdout_decision.json",gate)
 else:csvout("speed_summary.csv",[{"status":"NOT_RUN","reason":"No preselected candidate passed frozen positive Holdout gain and CI gate; no TextVQA timing measured or historical GQA time reused"}])
 assert count()<=816 and e.frozen.hooks_count(model)==hooks
 write(OUT/"run_completion.json",dict(status="COMPLETE",quality_generated=count(),Native_historical_reused=192,new_Native_quality=0,formal_speed_requests=len(speed),speed_status=gate["speed_status"],model_loads=1,selection_sha256=selsha,hooks_restored=True,quality_status=gate["quality_status"]))
 historical.close();del model,tok;gc.collect();torch.cuda.empty_cache();status("COMPLETE_GPU_RELEASED");db.close()
 write(OUT/"gpu_release.json",dict(GPU=telemetry(),worker_exits_after_receipt=True))
if __name__=="__main__":
 try:main()
 except Exception as ex:
  write(OUT/"failure.json",{"status":"FAILED_OR_PAUSED","exception":repr(ex),"traceback":traceback.format_exc()});print(traceback.format_exc(),flush=True);raise

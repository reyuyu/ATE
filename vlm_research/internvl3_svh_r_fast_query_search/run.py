"""One model load: structural smoke -> A -> B -> sealed holdout -> small real speed."""
from .common import *
import torch,gc,subprocess,traceback,sys
from collections import defaultdict
from vlm_research.internvl3_svh_seven_benchmarks import worker as w,scoring
from vlm_research.internvl3_svh_r_middle_reaccess_search import engine as e
def telemetry():
 return subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip()
def main():
 protocol=read(OUT/"protocol.json");manifest=read(OUT/"sample_manifest.json");ph=sha(OUT/"protocol.json")
 assert read(OUT/"cpu_tests.json")["status"]=="PASS"
 sm=read(OUT/"source_manifest.json")
 for path,expected in sm["runtime_source_sha256"].items():assert sha(RUN/"code"/path)==expected,("SOURCE_HASH",path)
 sourcesha=sha(OUT/"source_manifest.json");assert manifest["image_overlap"]==0
 uuid=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip();assert uuid==GPU
 assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(),"GPU_BUSY"
 binding=read(RUN/"prepared/input_bindings.json")
 allrows={r["question_id"]:r for r in manifest["Discovery"]+manifest["Holdout"]}
 native=read(RUN/"prepared/historical_native.json")
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");db.execute("PRAGMA journal_mode=WAL")
 db.execute("CREATE TABLE IF NOT EXISTS predictions(phase TEXT,qid TEXT,method TEXT,record TEXT,PRIMARY KEY(phase,qid,method))")
 db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)")
 meta=dict(db.execute("SELECT key,value FROM meta"))
 for k,v in {"protocol_sha256":ph,"source_manifest_sha256":sourcesha}.items():
  if k in meta:assert meta[k]==v,("CHECKPOINT_BINDING",k)
  else:db.execute("INSERT INTO meta VALUES(?,?)",(k,v))
 db.commit()
 def records(phase=None):
  q="SELECT record FROM predictions"+(" WHERE phase=?" if phase else "")
  return [json.loads(x[0]) for x in db.execute(q,(phase,) if phase else ())]
 def get(phase,q,m):
  v=db.execute("SELECT record FROM predictions WHERE phase=? AND qid=? AND method=?",(phase,q,m)).fetchone();return json.loads(v[0]) if v else None
 def save(phase,r,m,x):
  x.update(phase=phase,method=m,protocol_sha256=ph,source_manifest_sha256=sourcesha,input_binding=binding[r["question_id"]]["binding"],sample_sha256=r["sample_sha256"])
  db.execute("INSERT INTO predictions VALUES(?,?,?,?)",(phase,r["question_id"],m,json.dumps(x,ensure_ascii=False)));db.commit()
 def status(phase,**kw):
  state={"phase":phase,"quality_generated":db.execute("SELECT count(*) FROM predictions WHERE phase in ('A','B','Holdout') AND method!='Native'").fetchone()[0],"updated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"GPU":telemetry(),**kw};write(OUT/"progress.json",state);print(json.dumps(state),flush=True)
 status("MODEL_LOADING")
 model,tok=w.load_model();write(OUT/"environment.json",w.environment(model,tok));hookbaseline=e.frozen.hooks_count(model)
 # All image IO/processing done on CPU. Loading small BF16 SSD cache once per question is outside inference/timing.
 def ready(r):
  pix=torch.load(RUN/"prepared"/(r["image_sha256"]+".pt"),map_location="cpu",weights_only=True)
  assert w.tensor_sha(pix)==binding[r["question_id"]]["details"]["pixels_bf16_sha256"]
  c=model.conv_template.copy();c.system_message=model.system_message;c.append_message(c.roles[0],r["request"]);c.append_message(c.roles[1],None)
  prompt=c.get_prompt().replace("<image>","<img>"+"<IMG_CONTEXT>"*256+"</img>",1);assert prompt==r["prompt"],"NATIVE_CHAT_TEMPLATE_MISMATCH"
  return dict(ids=torch.tensor([r["input_ids"]],device="cuda:0",dtype=torch.long),pixels=pix.to("cuda:0"),roles=r["roles"],request=r["request"],prompt=prompt)
 def call(r,rd,m,physical=True,audit=False,fixed=False,timed=False,capture=False):
  captured=[];handle=None
  if capture and not audit:
   def logits_hook(module,args,out):captured.append(out.logits[0,-1].detach().float().cpu())
   handle=model.language_model.register_forward_hook(logits_hook)
  try:
   x,s,ctl=e.generate(model,tok,rd,m,SPACE.get(m,[]),max_tokens=32 if fixed else 16,fixed=fixed,audit=audit,timed=timed,physical=physical)
   if captured:s=captured
   assert len(x["token_ids"])<= (32 if fixed else 16)
   assert x["independent_cache"] and e.frozen.hooks_count(model)==(hookbaseline+1 if handle else hookbaseline)
   parsed,score=scoring.score("GQA",x["text"],r);assert score==scoring.independent_score("GQA",x["text"],r)
   x=quality(x,r);assert x["score"]==score
   return x,s,ctl
  finally:
   if handle:handle.remove()
 def compact(x):
  return {k:v for k,v in x.items() if k!="audit"}
 # Minimal structure audits on shortest/longest; light logits only for remaining frozen smoke inputs.
 if not (OUT/"smoke_validation.json").exists():
  checks=[];noop_done=False;native_match=[];audits=[]
  for i,q in enumerate(manifest["smoke_ids"]):
   r=allrows[q];rd=ready(r);deep=i in (0,7)
   nx,ns,nc=call(r,rd,"Native",audit=not noop_done,capture=True)
   assert nx["token_ids"]==native[q]["token_ids"] and nx["text"]==native[q]["text"],("HISTORICAL_NATIVE_REUSE_FAILURE",q)
   native_match.append(q)
   if not noop_done:
    no,ss,cc=call(r,rd,"No-op",audit=True)
    assert no["token_ids"]==nx["token_ids"] and len(ns)==len(ss) and all(torch.equal(a,b) for a,b in zip(ns,ss)),"NATIVE_NOOP_LOGITS"
    noop_done=True;del no,ss,cc
   del nx,ns,nc
   for m in ["R0","R4-L","R8"]:
    dx,ds,dc=call(r,rd,m,physical=False,audit=deep,capture=True)
    px,ps,pc=call(r,rd,m,physical=True,audit=deep,capture=True)
    assert len(px["kv_layout"])==28
    for row in px["kv_layout"]:
     layer=row["layer"];assert row["visual_kv"]==(256 if layer in SPACE[m] else 0)
     assert row["physical_length"]==(0 if layer in e.SKIP else r["input_tokens"]+px["generated_tokens"]-1-(0 if layer in SPACE[m] else 256))
    diffs=[]
    for j,(a,b) in enumerate(zip(ds,ps)):
     # Only compare effective logits under exactly the same generated prefix.
     if dx["token_ids"][:j]!=px["token_ids"][:j]:break
     d=(a.float()-b.float()).abs();assert torch.isfinite(a).all() and torch.isfinite(b).all()
     diffs.append(dict(step=j,max_abs=float(d.max()),mean_abs=float(d.mean()),top1_equal=int(a.argmax())==int(b.argmax())))
    assert diffs
    checks.append(dict(question_id=q,condition=m,input_binding=binding[q]["binding"],dense=compact(dx),physical=compact(px),aligned_prefix_logits=diffs,answer_equal=dx["parsed_answer"]==px["parsed_answer"],full_tokens_equal=dx["token_ids"]==px["token_ids"],structural_audit=deep))
    if deep:audits.append(dict(question_id=q,condition=m,physical=px["audit"],dense=dx["audit"]))
    write(OUT/"smoke_checks_partial.json",checks);write(OUT/"smoke_structure_audits.json",audits)
    del dx,ds,dc,px,ps,pc
   del rd;gc.collect();torch.cuda.empty_cache();status("SMOKE",completed_inputs=i+1,total=8)
  result={"status":"PASS_WITH_NUMERICAL_WAIVER","Native_NoOp_strict_logits":True,"Native_historical_full_tokens_equal":native_match,"smoke_inputs":8,"conditions":["R0","R4-L","R8"],"checks":checks,"large_tensor_saved":False,"known_BF16_numerical_equivalence_claimed":False}
  failed=[]
  for m in ["R0","R4-L","R8"]:
   rr=[x for x in checks if x["condition"]==m];bad=sum(x["physical"]["flags"]["meta"] or x["physical"]["flags"]["empty"] or x["physical"]["flags"]["repeat"] for x in rr);disagree=sum(not x["answer_equal"] for x in rr)
   result[m]={"answer_agreement":8-disagree,"N":8,"severe_outputs":bad}
   if bad>=3 or disagree>=4:failed.append(m)
  result["multitoken_observed"]=any(len(x["physical"]["token_ids"])>=3 and not any(x["physical"]["flags"].values()) for x in checks)
  if failed or not result["multitoken_observed"]:result["status"]="PAUSED_SMOKE_SEMANTIC_ANOMALY"
  write(OUT/"smoke_validation.json",result);assert result["status"]=="PASS_WITH_NUMERICAL_WAIVER",result["status"]
 else:assert read(OUT/"smoke_validation.json")["status"]=="PASS_WITH_NUMERICAL_WAIVER"
 reuse=read(OUT/"native_reuse_validation.json");reuse["status"]="PASS";reuse["GPU_smoke_Native_matches_all_8"]=True;write(OUT/"native_reuse_validation.json",reuse)
 def monitor(method):
  rr=[x for x in records() if x["phase"] in ("A","B","Holdout") and x["method"]==method];last=rr[-10:]
  if len(last)==10 and all(x["flags"]["empty"] or x["flags"]["repeat"] for x in last):raise RuntimeError("SYSTEMATIC_10_EMPTY_REPEAT "+method)
  if len(rr)>=64 and len(rr)%32==0:
   windows=[rr[-64:-32],rr[-32:]]
   def bad(v):return sum(x["flags"]["meta"] for x in v)>=8 or sum(x["flags"]["empty"] or x["flags"]["repeat"] for x in v)>=4 or sum(x["truncated"] for x in v)>=16
   if all(bad(v) for v in windows):raise RuntimeError("SYSTEMATIC_OUTPUT_ANOMALY "+method)
 def phase(name,ids,methods):
  status(name,total=len(ids)*len(methods))
  for i,q in enumerate(ids):
   r=allrows[q];missing=[m for m in methods if not get(name,q,m)]
   if missing:
    rd=ready(r);rot=methods[i%len(methods):]+methods[:i%len(methods)]
    for m in rot:
     if m not in missing:continue
     x,ss,cc=call(r,rd,m);save(name,r,m,x);del ss,cc,x;monitor(m)
    del rd
   # Native holdout predictions become available here, strictly after selection is frozen.
   if not get(name,q,"Native"):
    rec=dict(native[q]);rec["method"]="Native";rec["KV_capacity_MiB"]=sum(x["capacity_bytes"] for x in rec["kv_layout"])/1048576
    rec=quality(rec,r);save(name,r,"Native",rec)
   if (i+1)%8==0 or i+1==len(ids):status(name,completed_inputs=i+1,total_inputs=len(ids));csvout("discovery_accuracy.csv" if name!="Holdout" else "holdout_accuracy.csv",summarize(records("A")+records("B"),"Discovery120_or_partial") if name!="Holdout" else summarize(records("Holdout"),"Holdout160_or_partial"))
  return records(name)
 phase("A",manifest["StageA_ids"],ORDER)
 a=selections([x for x in records("A") if x["method"]!="Native"],"A");write(OUT/"selection_stage_A.json",a);csvout("stage_A_accuracy.csv",summarize(records("A"),"A64"));status("A_COMPLETE",selected=a["selected"])
 phase("B",manifest["StageB_ids"],a["selected"])
 discovery=[x for x in records("A")+records("B") if x["method"] in a["selected"]]
 final=selections(discovery,"B");final.update(stage_A=a,protocol_sha256=ph,source_manifest_sha256=sourcesha,selection_time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),holdout_predictions_generated_before_selection=0)
 if (OUT/"selection.json").exists():
  prior=read(OUT/"selection.json");assert prior["best_quality"]==final["best_quality"] and prior["best_efficiency"]==final["best_efficiency"];final=prior
 else:write(OUT/"selection.json",final)
 selection_sha=sha(OUT/"selection.json");write(OUT/"selection_freeze_receipt.json",{"selection_sha256":selection_sha,"before_holdout":True,"holdout_methods":final["holdout_methods"]})
 csvout("discovery_accuracy.csv",summarize(discovery+[x for x in records("A")+records("B") if x["method"]=="Native"],"Discovery120"));status("B_COMPLETE",best_quality=final["best_quality"],best_efficiency=final["best_efficiency"])
 phase("Holdout",[r["question_id"] for r in manifest["Holdout"]],final["holdout_methods"])
 assert sha(OUT/"selection.json")==selection_sha
 csvout("holdout_accuracy.csv",summarize(records("Holdout"),"Holdout160"));status("HOLDOUT_COMPLETE")
 methods=list(dict.fromkeys(["Native","R0",final["best_efficiency"]]))
 for m in methods:
  for j,q in enumerate(manifest["speed_ids"][:2]):
   rd=ready(allrows[q]);x,ss,cc=call(allrows[q],rd,m,fixed=True,timed=True);del rd,x,ss,cc
 speed=[]
 for repeat in range(3):
  for i,q in enumerate(manifest["speed_ids"]):
   r=allrows[q];rd=ready(r);offset=(i+repeat)%len(methods)
   for m in methods[offset:]+methods[:offset]:
    key="speed_"+str(repeat);saved=get(key,q,m)
    if saved:x=saved
    else:
     x,ss,cc=call(r,rd,m,fixed=True,timed=True);x["repeat"]=repeat;save(key,r,m,x);del ss,cc
    speed.append(x)
   del rd
   if (i+1)%4==0:status("SPEED",repeat=repeat+1,completed_inputs=i+1)
 csvout("speed_per_input.csv",[{k:v for k,v in x.items() if k not in ("audit","kv_layout")} for x in speed])
 rows=[]
 for m in methods:
  rr=[x for x in speed if x["method"]==m];out={"method":m,"requests":len(rr),"images":12,"repeat":3}
  for k in ["E2E_ms","TTFT_ms","TPOT_ms","incremental_decode_ms","vision_encoder_ms","vision_projector_injection_ms","language_prefill_ms","CUDA_event_TPOT_ms","KV_live_MiB","KV_capacity_MiB","peak_allocated_MiB","peak_reserved_MiB"]:
   out[k]=float(np.mean([x[k] for x in rr]));out[k+"_median"]=float(np.median([x[k] for x in rr]))
  rows.append(out)
 csvout("speed_summary.csv",rows)
 csvout("kv_layout_speed.csv",[dict(question_id=x["question_id"],method=x["method"],repeat=x["repeat"],**{k:v for k,v in row.items() if k!="method"}) for x in speed for row in x["kv_layout"]])
 assert e.frozen.hooks_count(model)==hookbaseline
 write(OUT/"run_completion.json",{"status":"COMPLETE","quality_generated":db.execute("SELECT count(*) FROM predictions WHERE phase in ('A','B','Holdout') AND method!='Native'").fetchone()[0],"new_Native_quality":0,"historical_Native_reused":len(manifest["Discovery"])+len(manifest["Holdout"]),"speed_formal":len(speed),"model_loads":1,"selection_sha256":selection_sha,"hooks_restored":True,"smoke_numerical_status":"not equivalent; authorized waiver","runtime_code_unmodified":True})
 db.close();del model,tok;gc.collect();torch.cuda.empty_cache();status("COMPLETE_GPU_RELEASED");write(OUT/"gpu_release.json",{"GPU":telemetry(),"process_exits_after_this_receipt":True})
if __name__=="__main__":
 try:main()
 except Exception as ex:
  write(OUT/"failure.json",{"status":"FAILED_OR_PAUSED","exception":repr(ex),"traceback":traceback.format_exc()});print(traceback.format_exc(),flush=True);raise

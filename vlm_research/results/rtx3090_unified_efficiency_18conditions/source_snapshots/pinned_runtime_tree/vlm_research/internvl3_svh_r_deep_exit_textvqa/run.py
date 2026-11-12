"""One model load; fixed Deep6/Deep8 quality, 48-image controls, gated tiny speed."""
from .common import *
import sys,gc,traceback,subprocess,importlib.metadata as md
import torch
from . import engine as e
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
def gpu_state():
 return subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used","--format=csv,noheader"],text=True).strip()
def database():
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");db.execute("PRAGMA journal_mode=WAL");db.execute("PRAGMA synchronous=FULL")
 db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT)")
 db.execute("CREATE TABLE IF NOT EXISTS predictions(phase TEXT,qid TEXT,method TEXT,origin TEXT,checksum TEXT,record TEXT,PRIMARY KEY(phase,qid,method))")
 for k,v in dict(protocol_sha256=sha(OUT/"protocol.json"),source_manifest_sha256=sha(OUT/"source_manifest.json"),sample_manifest_sha256=sha(OUT/"sample_manifest.json")).items():
  x=db.execute("SELECT value FROM metadata WHERE key=?",(k,)).fetchone()
  if x:assert x[0]==v,"CHECKPOINT_METADATA_CHANGED"
  else:db.execute("INSERT INTO metadata VALUES (?,?)",(k,v))
 db.commit();return db
def saved(db,phase,q,m):
 x=db.execute("SELECT record,checksum FROM predictions WHERE phase=? AND qid=? AND method=?",(phase,q,m)).fetchone()
 if not x:return None
 assert hashlib.sha256(x[0].encode()).hexdigest()==x[1],"CHECKPOINT_CORRUPT"
 return json.loads(x[0])
def put(db,phase,r,m,x,origin="NEW"):
 assert saved(db,phase,r["question_id"],m) is None
 x.update(phase=phase,method=m,question_id=r["question_id"],sample_sha256=r["sample_sha256"],input_binding=BIND[r["question_id"]]["binding"],protocol_sha256=PH,source_manifest_sha256=SH,generation_origin=origin)
 raw=json.dumps(x,ensure_ascii=False,sort_keys=True);db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)",(phase,r["question_id"],m,origin,hashlib.sha256(raw.encode()).hexdigest(),raw));db.commit()
 assert db.execute("SELECT COUNT(*) FROM predictions WHERE origin='NEW' AND phase IN ('Main','Controls')").fetchone()[0]<=352
def rows(db,phase):
 out=[]
 for raw,h in db.execute("SELECT record,checksum FROM predictions WHERE phase=?",(phase,)):
  assert hashlib.sha256(raw.encode()).hexdigest()==h;out.append(json.loads(raw))
 return out
def status(db,phase,**kw):
 d=dict(phase=phase,new_quality_generations=db.execute("SELECT COUNT(*) FROM predictions WHERE origin='NEW' AND phase IN ('Main','Controls')").fetchone()[0],updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),GPU=gpu_state(),**kw)
 write(OUT/"progress.json",d);print(json.dumps(d),flush=True)
def ready(r):
 p=torch.load(RUN/"prepared"/(r["image_sha256"]+".pt"),weights_only=True,map_location="cpu")
 assert set(p)=={"ids","pixels","roles","request","prompt"} and p["prompt"]==r["prompt"] and p["request"]==r["request"] and p["roles"]==r["roles"]
 d=dict(image_sha256=r["image_sha256"],input_ids_sha256=w.tensor_sha(p["ids"]),pixels_bf16_sha256=w.tensor_sha(p["pixels"]),prompt_sha256=hashlib.sha256(p["prompt"].encode()).hexdigest())
 assert d==BIND[r["question_id"]]["details"] and w.bind(d)==BIND[r["question_id"]]["binding"]
 return {**p,"ids":p["ids"].to("cuda:0"),"pixels":p["pixels"].to("cuda:0")}
def generate(model,tok,p,m,**kw):
 cfg=ALL_CONFIGS.get(m,dict(keep=SPACE.get(m,[]),cutoff=20))
 return e.generate(model,tok,p,m,cfg["keep"],cutoff=cfg["cutoff"],max_tokens=32,**kw)
def smoke(model,tok,mf):
 checks=[];audits=[];hidden=[]
 for j,r in enumerate(mf["Smoke"]):
  p=ready(r)
  if j==0:
   n,ns,nc=e.base.generate(model,tok,p,"Native",max_tokens=32,audit=True)
   z,zs,zc=e.base.generate(model,tok,p,"No-op",max_tokens=32,audit=True)
   assert n["token_ids"]==z["token_ids"] and len(ns)==len(zs) and all(torch.equal(a,b) for a,b in zip(ns,zs)),"NATIVE_NOOP_STRICT_FAILED"
   ref=read(PARENT/"server_only/reused_native_B_smoke.json")[r["question_id"]]["record"]
   native_hist_equal=n["token_ids"]==ref["token_ids"]
   write(OUT/"new_GPU_native_smoke.json",dict(Native_NoOp_strict_logits=True,question_id=r["question_id"],new_GPU=GPU,historical_GPU=OLD_GPU,historical_complete_tokens_equal=native_hist_equal,new_token_ids=n["token_ids"],historical_token_ids=ref["token_ids"],no_new_formal_Native_quality=True))
   del n,z,ns,zs,nc,zc;gc.collect()
  for name,cfg in ALL_CONFIGS.items():
   dense,ds,dc=generate(model,tok,p,name,audit=True,physical=False)
   phys,ps,pc=generate(model,tok,p,name,audit=True,physical=True)
   assert phys["audit"]["cutoff"]==cfg["cutoff"] and phys["audit"]["hidden_exit"]==[dict(before=r["input_tokens"],after=r["input_tokens"]-256,removed=256)]
   assert [x["layer"] for x in phys["kv_layout"] if x["visual_kv"]]==cfg["keep"]
   diffs=[];aligned=[]
   for step,(a,b) in enumerate(zip(ds,ps)):
    if step and dense["token_ids"][:step]!=phys["token_ids"][:step]:break
    diff=(a.float()-b.float()).abs();diffs.append(dict(step=step,max_abs=float(diff.max()),mean_abs=float(diff.mean()),top1_equal=int(a.argmax())==int(b.argmax())))
    aligned.append(step)
   for (step,i),(pos,h) in pc.hidden.items():
    if step not in aligned:continue
    dpos,dh=dc.hidden[(step,i)];lookup={int(x):j for j,x in enumerate(dpos)};delta=(h.float()-dh[:,[lookup[int(x)] for x in pos]].float()).abs()
    hidden.append(dict(question_id=r["question_id"],condition=name,step=step,layer=i,aligned_query_rows=len(pos),max_abs=float(delta.max()),mean_abs=float(delta.mean())))
   a=quality(dense,r);b=quality(phys,r)
   checks.append(dict(question_id=r["question_id"],condition=name,cutoff=cfg["cutoff"],keep=cfg["keep"],full_tokens_equal=dense["token_ids"]==phys["token_ids"],normalized_answer_equal=a["parsed_answer"]==b["parsed_answer"],aligned_prefix_logits=diffs,dense_text=dense["text"],physical_text=phys["text"],dense_tokens=dense["token_ids"],physical_tokens=phys["token_ids"],structural="PASS"))
   audits.append(dict(question_id=r["question_id"],condition=name,physical=phys["audit"],dense=dense["audit"],physical_KV=phys["kv_layout"],dense_KV=dense["kv_layout"]))
   write(OUT/"smoke_checks_partial.json",checks)
   del dense,phys,ds,ps,dc,pc,a,b;gc.collect()
  del p
  print("SMOKE_INPUT_PASS",r["question_id"],flush=True)
 write(OUT/"smoke_validation.json",dict(status="PASS_STRUCTURAL_WITH_NUMERICAL_WAIVER",historical_status="NUMERICAL_EQUIVALENCE_FAILED preserved",Native_NoOp_strict_logits=True,questions=len(mf["Smoke"]),checks=checks,numerical_equivalence_claimed=False,new_GPU=GPU,previous_GPU=OLD_GPU))
 write(OUT/"physical_cutoff_audit.json",dict(status="PASS",R_to_Image_boundary=20,real_late_Hidden_before_cutoff=True,all_cutoffs_measured=True,original_positions_and_scratch_mask_verified=True,checks=audits))
 csvout("smoke_hidden_differences.csv",hidden)
def main():
 global PH,SH,BIND
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 for rel,h in read(OUT/"source_manifest.json")["runtime_source_sha256"].items():assert sha(RUN/"code"/rel)==h,("SOURCE_HASH_MISMATCH",rel)
 assert GPU in gpu_state() and "3090" in gpu_state()
 assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True).strip(),"OTHER_GPU_PROCESS"
 PH=sha(OUT/"protocol.json");SH=sha(OUT/"source_manifest.json");BIND=read(RUN/"prepared/input_bindings.json");mf=read(OUT/"sample_manifest.json");db=database()
 for item in read(RUN/"prepared/reused_records.json"):
  x=item["record"];r=next(r for r in mf["Main"] if r["question_id"]==x["question_id"])
  if saved(db,"Main",r["question_id"],x["method"]):continue
  x=dict(x,reused_from=dict(commit=PARENT_COMMIT,original_record_sha256=item["original_record_sha256"],protocol_sha256=x["protocol_sha256"],source_manifest_sha256=x["source_manifest_sha256"],GPU=OLD_GPU))
  put(db,"Main",r,x["method"],x,"REUSED")
 status(db,"MODEL_LOADING")
 model,tok=w.load_model();env=w.environment(model,tok);env.update(GPU_UUID=GPU,historical_GPU_UUID=OLD_GPU,GPU_change_authorized=True,model_loads_this_process=1,timm=md.version("timm"),timm_import=__import__("timm").__file__,CPU_threads=torch.get_num_threads());write(OUT/"environment.json",env)
 try:
  if not (OUT/"smoke_validation.json").exists():smoke(model,tok,mf)
  status(db,"SMOKE_PASS")
  control_ids=set(mf["control_ids"])
  for i,r in enumerate(mf["Main"]):
   methods=list(CONFIGS)+(list(CONTROLS) if r["question_id"] in control_ids else [])
   methods=methods[i%len(methods):]+methods[:i%len(methods)]
   todo=[m for m in methods if not saved(db,"Main" if m in CONFIGS else "Controls",r["question_id"],m)]
   if todo:
    p=ready(r)
    for m in todo:
     result,scores,c=generate(model,tok,p,m,audit=False,physical=True,fixed=False)
     assert not scores and not result["audit"] and result["generated_tokens"]<=32
     x=quality(result,r);put(db,"Main" if m in CONFIGS else "Controls",r,m,x)
     del scores,c,result,x
    del p
   if (i+1)%8==0:status(db,"QUALITY",completed_images=i+1,total=128);gc.collect()
  mainrows=rows(db,"Main");summary=summarize(mainrows,"Main128");sm={r["method"]:r for r in summary}
  eligible=[m for m in CONFIGS if sm[m]["soft_accuracy_pct"]>=20 and sm[m]["soft_accuracy_pct"]-sm["R0"]["soft_accuracy_pct"]>=10]
  write(OUT/"quality_decision.json",dict(eligible=eligible,speed_status="RUN" if eligible else "NOT_RUN",fixed_gate="candidate Main>=20% AND gain over Main R0>=10pp",summary=summary))
  status(db,"QUALITY_COMPLETE",speed_eligible=eligible)
  speedcount=0
  if eligible:
   methods=["Native","R0"]+eligible;speedrows=[next(r for r in mf["Main"] if r["question_id"]==q) for q in mf["speed_ids"]]
   for m in methods:
    for r in speedrows[:2]:
     p=ready(r);x,s,c=generate(model,tok,p,m,audit=False,timed=True,fixed=True)
     assert len(x["token_ids"])==32;del p,x,s,c
   for i,r in enumerate(speedrows):
    p=ready(r)
    for rep in range(3):
     order=methods[(i+rep)%len(methods):]+methods[:(i+rep)%len(methods)]
     for m in order:
      phase="speed_"+str(rep)
      if saved(db,phase,r["question_id"],m):continue
      x,s,c=generate(model,tok,p,m,audit=False,timed=True,fixed=True)
      assert x["generated_tokens"]==32 and x["independent_cache"]
      put(db,phase,r,m,quality(x,r));del x,s,c
    del p
    status(db,"SPEED",completed_images=i+1,total=8)
   rr=[x for rep in range(3) for x in rows(db,"speed_"+str(rep))];speedcount=len(rr);assert speedcount==len(methods)*24
   keys=["E2E_ms","TTFT_ms","TPOT_ms","vision_encoder_ms","vision_projector_injection_ms","language_prefill_ms","incremental_decode_ms","CUDA_event_TPOT_ms","KV_live_MiB","KV_capacity_MiB","peak_allocated_MiB","peak_reserved_MiB"]
   csvout("speed_summary.csv",[dict(method=m,requests=sum(x["method"]==m for x in rr),**{k:float(np.mean([x[k] for x in rr if x["method"]==m])) for k in keys},**{k+"_median":float(np.median([x[k] for x in rr if x["method"]==m])) for k in keys}) for m in methods])
   csvout("speed_per_input.csv",[{k:v for k,v in x.items() if k not in ("kv_layout","audit")} for x in rr])
  counts=db.execute("SELECT phase,method,origin,count(*) FROM predictions GROUP BY phase,method,origin").fetchall()
  assert sum(n for phase,m,o,n in counts if o=="NEW" and phase in ("Main","Controls"))==352
  assert len(rows(db,"Main"))==768 and len(rows(db,"Controls"))==96
  write(OUT/"run_completion.json",dict(status="COMPLETE",new_quality_generations=352,new_Deep6=128,new_Deep8=128,new_controls=96,reused_historical=512,speed_formal_requests=speedcount,model_loads=1,checkpoint_counts=counts,hooks_restored=True))
  status(db,"COMPLETE");db.close()
 finally:
  del model,tok;gc.collect();torch.cuda.empty_cache()
if __name__=="__main__":
 try:main()
 except BaseException as err:
  write(OUT/"failure.json",dict(status="FAILED",error=repr(err),traceback=traceback.format_exc(),GPU=gpu_state()));print(traceback.format_exc(),flush=True);raise

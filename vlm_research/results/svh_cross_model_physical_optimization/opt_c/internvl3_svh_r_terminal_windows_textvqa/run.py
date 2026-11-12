"""Exactly three user-accepted late windows,384 new formal quality answers; no speed."""
from .common import *
import torch,gc,traceback,subprocess,importlib.metadata as md
from . import engine as e
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
PREPARED=Path("/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090/prepared")
def gpu():return subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,clocks.sm,power.draw,temperature.gpu,memory.used","--format=csv,noheader"],text=True).strip()
def ready(r):
 p=torch.load(PREPARED/(r["image_sha256"]+".pt"),weights_only=True,map_location="cpu")
 assert set(p)=={"ids","pixels","roles","request","prompt"} and p["prompt"]==r["prompt"] and p["request"]==r["request"] and p["roles"]==r["roles"]
 details=dict(image_sha256=r["image_sha256"],input_ids_sha256=w.tensor_sha(p["ids"]),pixels_bf16_sha256=w.tensor_sha(p["pixels"]),prompt_sha256=hashlib.sha256(p["prompt"].encode()).hexdigest())
 assert details==BIND[r["question_id"]]["details"] and w.bind(details)==BIND[r["question_id"]]["binding"]
 return {**p,"ids":p["ids"].to("cuda:0"),"pixels":p["pixels"].to("cuda:0")}
def database():
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");db.execute("pragma journal_mode=WAL");db.execute("pragma synchronous=FULL")
 db.execute("create table if not exists metadata(key text primary key,value text)");db.execute("create table if not exists predictions(phase text,qid text,method text,origin text,checksum text,record text,primary key(phase,qid,method))")
 for k,v in dict(protocol_sha256=PH,source_manifest_sha256=SH,sample_manifest_sha256=sha(OUT/"sample_manifest.json")).items():
  x=db.execute("select value from metadata where key=?",(k,)).fetchone()
  if x:assert x[0]==v,"CHECKPOINT_HASH_CHANGED"
  else:db.execute("insert into metadata values (?,?)",(k,v))
 db.commit();return db
def saved(db,q,m):
 row=db.execute("select checksum,record from predictions where phase='Main' and qid=? and method=?",(q,m)).fetchone()
 if row:assert hashlib.sha256(row[1].encode()).hexdigest()==row[0]
 return row is not None
def put(db,r,m,x,origin):
 assert not saved(db,r["question_id"],m)
 x=dict(x,phase="Main",method=m,question_id=r["question_id"],sample_sha256=r["sample_sha256"],input_binding=BIND[r["question_id"]]["binding"],protocol_sha256=PH,source_manifest_sha256=SH,generation_origin=origin)
 raw=json.dumps(x,ensure_ascii=False,sort_keys=True);db.execute("insert into predictions values (?,?,?,?,?,?)",("Main",r["question_id"],m,origin,hashlib.sha256(raw.encode()).hexdigest(),raw));db.commit()
 assert db.execute("select count(*) from predictions where origin='NEW'").fetchone()[0]<=384
def generate(model,tok,p,m,**kw):
 cfg=CONFIGS[m];return e.generate(model,tok,p,m,cfg["keep"],cutoff=cfg["cutoff"],max_tokens=32,**kw)
def smoke(model,tok,mf):
 checks=[];audits=[];hidden=[]
 for j,r in enumerate(mf["Smoke"]):
  p=ready(r)
  if j==0:
   n,ns,nc=e.base.generate(model,tok,p,"Native",max_tokens=32,audit=True);z,zs,zc=e.base.generate(model,tok,p,"No-op",max_tokens=32,audit=True)
   assert n["token_ids"]==z["token_ids"] and len(ns)==len(zs) and all(torch.equal(a,b) for a,b in zip(ns,zs)),"NATIVE_NOOP_FAILED"
   old=read(PARENT/"native_smoke.json");write(OUT/"native_smoke.json",dict(strict_Native_NoOp_tokens_and_logits_equal=True,same_GPU_previous_Native_tokens_equal=n["token_ids"]==old["new_tokens"],new_tokens=n["token_ids"],previous_tokens=old["new_tokens"],question_id=r["question_id"]))
   del n,z,ns,zs,nc,zc;gc.collect()
  for m,cfg in CONFIGS.items():
   dense,ds,dc=generate(model,tok,p,m,audit=True,physical=False);phys,ps,pc=generate(model,tok,p,m,audit=True,physical=True)
   assert phys["audit"]["hidden_exit"]==[dict(before=r["input_tokens"],after=r["input_tokens"]-256,removed=256)] and phys["audit"]["cutoff"]==cfg["cutoff"]
   assert [x["layer"] for x in phys["kv_layout"] if x["visual_kv"]]==cfg["keep"]
   diffs=[];aligned=[]
   for step,(a,b) in enumerate(zip(ds,ps)):
    if step and dense["token_ids"][:step]!=phys["token_ids"][:step]:break
    delta=(a.float()-b.float()).abs();diffs.append(dict(step=step,max_abs=float(delta.max()),mean_abs=float(delta.mean()),top1_equal=int(a.argmax())==int(b.argmax())));aligned.append(step)
   for (step,i),(pos,h) in pc.hidden.items():
    if step not in aligned:continue
    dpos,dh=dc.hidden[(step,i)];lookup={int(x):j for j,x in enumerate(dpos)};delta=(h.float()-dh[:,[lookup[int(x)] for x in pos]].float()).abs();hidden.append(dict(condition=m,question_id=r["question_id"],step=step,layer=i,aligned_rows=len(pos),max_abs=float(delta.max()),mean_abs=float(delta.mean())))
   a=quality(dense,r);b=quality(phys,r)
   checks.append(dict(condition=m,cutoff=cfg["cutoff"],keep=cfg["keep"],question_id=r["question_id"],full_tokens_equal=dense["token_ids"]==phys["token_ids"],normalized_answers_equal=a["parsed_answer"]==b["parsed_answer"],dense_tokens=dense["token_ids"],physical_tokens=phys["token_ids"],dense_text=dense["text"],physical_text=phys["text"],aligned_logits=diffs))
   audits.append(dict(question_id=r["question_id"],condition=m,physical=phys["audit"],dense=dense["audit"],physical_KV=phys["kv_layout"],dense_KV=dense["kv_layout"]))
   write(OUT/"smoke_checks_partial.json",checks)
   del dense,phys,ds,ps,dc,pc,a,b;gc.collect()
  del p;print("SMOKE_INPUT_PASS",r["question_id"],flush=True)
 write(OUT/"smoke_validation.json",dict(status="PASS_STRUCTURAL_WITH_NUMERICAL_WAIVER",Native_NoOp_strict_equal=True,checks=checks,strict_numerical_equivalence=False,historical_status="NUMERICAL_EQUIVALENCE_FAILED"))
 write(OUT/"physical_cutoff_audit.json",dict(status="PASS",R_to_Image_boundary=20,configs=CONFIGS,checks=audits));csvout("smoke_hidden_differences.csv",hidden)
def main():
 global PH,SH,BIND
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 assert GPU in gpu() and "3090" in gpu()
 assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True).strip(),"OTHER_GPU_PROCESS"
 assert read(OUT/"cpu_tests.json")["status"]=="PASS" and read(OUT/"input_reuse_validation.json")["status"]=="PASS_CPU"
 assert read(OUT/"nonterminal_regression.json")["status"]=="PASS"
 for rel,h in read(OUT/"source_manifest.json")["runtime_source_sha256"].items():assert sha(RUN/"code"/rel)==h
 PH=sha(OUT/"protocol.json");SH=sha(OUT/"source_manifest.json");BIND=read(OUT/"input_bindings.json");mf=read(OUT/"sample_manifest.json");lookup={r["question_id"]:r for r in mf["Main"]};db=database()
 old=sqlite3.connect("file:"+str(PARENT/"server_only/predictions.sqlite")+"?mode=ro&immutable=1",uri=True)
 for q,m,origin,h,raw in old.execute("select qid,method,origin,checksum,record from predictions where phase='Main'"):
  assert hashlib.sha256(raw.encode()).hexdigest()==h
  if saved(db,q,m):continue
  x=json.loads(raw);x["reused_from_current_parent"]=dict(commit=PARENT_COMMIT,record_sha256=h,original_protocol_sha256=x["protocol_sha256"],original_source_manifest_sha256=x["source_manifest_sha256"],original_origin=origin)
  put(db,lookup[q],m,x,"REUSED")
 old.close();assert db.execute("select count(*) from predictions where origin='REUSED'").fetchone()[0]==1280
 model,tok=w.load_model();env=w.environment(model,tok);env.update(GPU_UUID=GPU,model_loads=1,quality_only=True,timm=md.version("timm"));write(OUT/"environment.json",env)
 try:
  smoke(model,tok,mf)
  for i,r in enumerate(mf["Main"]):
   methods=list(CONFIGS);methods=methods[i%3:]+methods[:i%3];todo=[m for m in methods if not saved(db,r["question_id"],m)]
   if todo:
    p=ready(r)
    for m in todo:
     x,ss,c=generate(model,tok,p,m,audit=False,physical=True,fixed=False)
     assert not ss and x["generated_tokens"]<=32 and x["independent_cache"];put(db,r,m,quality(x,r),"NEW");del x,ss,c
    del p
   if (i+1)%8==0:
    write(OUT/"progress.json",dict(phase="QUALITY",completed_images=i+1,total_images=128,new_generations=db.execute("select count(*) from predictions where origin='NEW'").fetchone()[0],GPU=gpu()));print("QUALITY_IMAGES",i+1,128,flush=True);gc.collect()
  assert db.execute("select count(*) from predictions where origin='NEW'").fetchone()[0]==384 and db.execute("select count(*) from predictions").fetchone()[0]==1664
  write(OUT/"run_completion.json",dict(status="COMPLETE",new_formal_quality=384,new_each_condition=128,reused=1280,new_speed=0,model_loads=1,GPU=GPU,configs=CONFIGS));db.close()
 finally:
  del model,tok;gc.collect();torch.cuda.empty_cache()
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  write(OUT/"failure.json",dict(status="FAILED",error=repr(ex),traceback=traceback.format_exc(),GPU=gpu()));print(traceback.format_exc(),flush=True);raise

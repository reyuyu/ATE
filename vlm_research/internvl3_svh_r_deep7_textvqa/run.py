"""One user-requested seven-layer follow-up; exactly128 new formal answers, no timing."""
from .common import *
import torch,gc,traceback,subprocess,importlib.metadata as md
from vlm_research.internvl3_svh_r_deep_exit_textvqa import engine as e
from vlm_research.internvl3_svh_seven_benchmarks import worker as w
OLD_RUN=Path("/home/featurize/ate-runs/internvl3-svh-r-deep-exit-textvqa-3090");METHOD="Deep7_L20_26";K=list(range(20,27));CUTOFF=27
def gpu():
 return subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,driver_version,clocks.sm,power.draw,temperature.gpu,memory.used","--format=csv,noheader"],text=True).strip()
def ready(r):
 p=torch.load(OLD_RUN/"prepared"/(r["image_sha256"]+".pt"),weights_only=True,map_location="cpu")
 assert set(p)=={"ids","pixels","roles","request","prompt"} and p["prompt"]==r["prompt"] and p["roles"]==r["roles"]
 d=dict(image_sha256=r["image_sha256"],input_ids_sha256=w.tensor_sha(p["ids"]),pixels_bf16_sha256=w.tensor_sha(p["pixels"]),prompt_sha256=hashlib.sha256(p["prompt"].encode()).hexdigest())
 assert d==BIND[r["question_id"]]["details"] and w.bind(d)==BIND[r["question_id"]]["binding"]
 return {**p,"ids":p["ids"].to("cuda:0"),"pixels":p["pixels"].to("cuda:0")}
def newdb():
 db=sqlite3.connect(RUN/"checkpoints/predictions.sqlite");db.execute("pragma journal_mode=WAL");db.execute("pragma synchronous=FULL")
 db.execute("create table if not exists metadata(key text primary key,value text)")
 db.execute("create table if not exists predictions(phase text,qid text,method text,origin text,checksum text,record text,primary key(phase,qid,method))")
 for k,v in dict(protocol_sha256=PH,source_manifest_sha256=SH,sample_manifest_sha256=sha(OUT/"sample_manifest.json")).items():
  x=db.execute("select value from metadata where key=?",(k,)).fetchone()
  if x:assert x[0]==v
  else:db.execute("insert into metadata values (?,?)",(k,v))
 db.commit();return db
def put(db,phase,r,m,x,origin):
 assert db.execute("select count(*) from predictions where phase=? and qid=? and method=?",(phase,r["question_id"],m)).fetchone()[0]==0
 x=dict(x,phase=phase,method=m,question_id=r["question_id"],sample_sha256=r["sample_sha256"],input_binding=BIND[r["question_id"]]["binding"],protocol_sha256=PH,source_manifest_sha256=SH,generation_origin=origin)
 raw=json.dumps(x,ensure_ascii=False,sort_keys=True);h=hashlib.sha256(raw.encode()).hexdigest();db.execute("insert into predictions values (?,?,?,?,?,?)",(phase,r["question_id"],m,origin,h,raw));db.commit()
def smoke(model,tok,mf):
 checks=[];audits=[];hidden=[]
 for j,r in enumerate(mf["Smoke"]):
  p=ready(r)
  if j==0:
   n,ns,nc=e.base.generate(model,tok,p,"Native",max_tokens=32,audit=True)
   z,zs,zc=e.base.generate(model,tok,p,"No-op",max_tokens=32,audit=True)
   assert n["token_ids"]==z["token_ids"] and len(ns)==len(zs) and all(torch.equal(a,b) for a,b in zip(ns,zs))
   old=read(PARENT/"new_GPU_native_smoke.json")
   write(OUT/"native_smoke.json",dict(strict_Native_NoOp_tokens_and_logits_equal=True,same_GPU_previous_Native_tokens_equal=n["token_ids"]==old["new_token_ids"],new_tokens=n["token_ids"],previous_tokens=old["new_token_ids"],question_id=r["question_id"]))
   del n,z,ns,zs,nc,zc;gc.collect()
  dense,ds,dc=e.generate(model,tok,p,METHOD,K,cutoff=CUTOFF,max_tokens=32,audit=True,physical=False)
  phys,ps,pc=e.generate(model,tok,p,METHOD,K,cutoff=CUTOFF,max_tokens=32,audit=True,physical=True)
  assert phys["audit"]["hidden_exit"]==[dict(before=r["input_tokens"],after=r["input_tokens"]-256,removed=256)] and phys["audit"]["cutoff"]==27
  assert [x["layer"] for x in phys["kv_layout"] if x["visual_kv"]]==K
  diffs=[];aligned=[]
  for step,(a,b) in enumerate(zip(ds,ps)):
   if step and dense["token_ids"][:step]!=phys["token_ids"][:step]:break
   delta=(a.float()-b.float()).abs();diffs.append(dict(step=step,max_abs=float(delta.max()),mean_abs=float(delta.mean()),top1_equal=int(a.argmax())==int(b.argmax())));aligned.append(step)
  for (step,i),(pos,h) in pc.hidden.items():
   if step not in aligned:continue
   dpos,dh=dc.hidden[(step,i)];lookup={int(x):j for j,x in enumerate(dpos)};delta=(h.float()-dh[:,[lookup[int(x)] for x in pos]].float()).abs();hidden.append(dict(question_id=r["question_id"],step=step,layer=i,aligned_rows=len(pos),max_abs=float(delta.max()),mean_abs=float(delta.mean())))
  a=quality(dense,r);b=quality(phys,r)
  checks.append(dict(question_id=r["question_id"],full_tokens_equal=dense["token_ids"]==phys["token_ids"],normalized_answers_equal=a["parsed_answer"]==b["parsed_answer"],dense_tokens=dense["token_ids"],physical_tokens=phys["token_ids"],dense_text=dense["text"],physical_text=phys["text"],aligned_logits=diffs))
  audits.append(dict(question_id=r["question_id"],condition=METHOD,physical=phys["audit"],dense=dense["audit"],physical_KV=phys["kv_layout"],dense_KV=dense["kv_layout"]))
  del dense,phys,ds,ps,dc,pc,p,a,b;gc.collect()
  print("SMOKE_PASS",r["question_id"],flush=True)
 write(OUT/"smoke_validation.json",dict(status="PASS_STRUCTURAL_WITH_NUMERICAL_WAIVER",condition=METHOD,Native_NoOp_strict_equal=True,checks=checks,strict_numerical_equivalence=False,historical_status="NUMERICAL_EQUIVALENCE_FAILED"))
 write(OUT/"physical_cutoff_audit.json",dict(status="PASS",R_to_Image_boundary=20,cutoff=27,keep=K,checks=audits));csvout("smoke_hidden_differences.csv",hidden)
def main():
 global PH,SH,BIND
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 assert GPU in gpu() and "3090" in gpu()
 assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True).strip()
 assert read(OUT/"cpu_tests.json")["status"]=="PASS"
 for rel,h in read(OUT/"source_manifest.json")["runtime_source_sha256"].items():assert sha(RUN/"code"/rel)==h
 PH=sha(OUT/"protocol.json");SH=sha(OUT/"source_manifest.json");BIND=read(OUT/"input_bindings.json");mf=read(OUT/"sample_manifest.json");lookup={r["question_id"]:r for r in mf["Main"]};db=newdb()
 old=sqlite3.connect("file:"+str(PARENT/"server_only/predictions.sqlite")+"?mode=ro&immutable=1",uri=True)
 for phase,qid,m,origin,h,raw in old.execute("select phase,qid,method,origin,checksum,record from predictions where phase='Main' or (phase='Controls' and method='Exit27_NoLateRead')"):
  assert hashlib.sha256(raw.encode()).hexdigest()==h
  if db.execute("select count(*) from predictions where phase=? and qid=? and method=?",(phase,qid,m)).fetchone()[0]:continue
  x=json.loads(raw);x["reused_from_current_parent"]=dict(commit=PARENT_COMMIT,record_sha256=h,original_protocol_sha256=x["protocol_sha256"],original_source_manifest_sha256=x["source_manifest_sha256"],original_origin=origin)
  put(db,phase,lookup[qid],m,x,"REUSED")
 old.close()
 model,tok=w.load_model();env=w.environment(model,tok);env.update(GPU_UUID=GPU,model_loads=1,quality_only=True,timm=md.version("timm"));write(OUT/"environment.json",env)
 try:
  smoke(model,tok,mf)
  for i,r in enumerate(mf["Main"]):
   q=r["question_id"]
   if not db.execute("select count(*) from predictions where phase='Main' and qid=? and method=?",(q,METHOD)).fetchone()[0]:
    p=ready(r);x,ss,c=e.generate(model,tok,p,METHOD,K,cutoff=27,max_tokens=32,audit=False,physical=True,fixed=False)
    assert not ss and x["generated_tokens"]<=32 and x["independent_cache"];put(db,"Main",r,METHOD,quality(x,r),"NEW");del p,x,ss,c
   if (i+1)%16==0:
    write(OUT/"progress.json",dict(phase="QUALITY",completed=i+1,total=128,GPU=gpu()));print("QUALITY",i+1,128,flush=True);gc.collect()
  assert db.execute("select count(*) from predictions where origin='NEW'").fetchone()[0]==128
  assert db.execute("select count(*) from predictions").fetchone()[0]==944
  write(OUT/"run_completion.json",dict(status="COMPLETE",new_formal_quality=128,reused=816,new_speed=0,model_loads=1,GPU=GPU,cutoff=27,keep=K))
  db.close()
 finally:
  del model,tok;gc.collect();torch.cuda.empty_cache()
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  write(OUT/"failure.json",dict(status="FAILED",error=repr(ex),traceback=traceback.format_exc(),GPU=gpu()));print(traceback.format_exc(),flush=True);raise

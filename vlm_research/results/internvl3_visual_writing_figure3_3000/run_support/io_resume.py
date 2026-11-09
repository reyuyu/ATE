"""Storage-only controlled resume. Frozen model/generation files are never edited."""
import pathlib,json,os,signal,time,subprocess,hashlib,shutil,sys
W=pathlib.Path("/home/featurize/work/ATE-internvl3-figure3-3000")
O=W/"vlm_research/results/internvl3_visual_writing_figure3_3000"
BASE=pathlib.Path("/home/featurize/ate-runs/internvl3-figure3-3000")
R=BASE/"runtime"
CP=BASE/"checkpoints"
PY="/home/featurize/.venvs/ate-frozen/bin/python"
def sha(b):return hashlib.sha256(b).hexdigest()
def alive(pid):
 try:return bool((pathlib.Path("/proc")/str(pid)/"cmdline").read_bytes())
 except FileNotFoundError:return False
def atomic(p,b):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+".io-tmp");t.write_bytes(b);t.replace(p)
def jwrite(p,v):atomic(p,(json.dumps(v,ensure_ascii=False,indent=2)+"\n").encode())
def stop_own(pid,expected):
 p=pathlib.Path("/proc")/str(pid)/"cmdline"
 if not p.exists():return
 assert expected.encode() in p.read_bytes(),(pid,"unexpected process")
 os.kill(pid,signal.SIGTERM)
 for i in range(30):
  if not alive(pid):return
  time.sleep(.2)
 os.kill(pid,signal.SIGKILL)
 for i in range(50):
  if not alive(pid):return
  time.sleep(.2)
 raise RuntimeError("Owned process did not stop")
def prepare():
 assert not R.exists(),"Storage resume already prepared; inspect existing state"
 R.mkdir(parents=True)
 names=["protocol.json","source_manifest.json","model_manifest.json","sample_manifest.json",
        "cpu_input_bindings.json","smoke_selection.json","preflight_complete.json"]
 for n in names:atomic(R/n,(O/n).read_bytes())
 print("RUNTIME_BYTES_COPIED",flush=True)
 stop_own(7791,"ate_internvl_finish_pipeline.py")
 stop_own(7056,"vlm_research.internvl3_visual_writing_figure3_3000.run")
 print("OWN_GPU_WORKER_STOPPED_AT_SAVED_CHECKPOINT",flush=True)
 for _ in range(120):
  if not alive(7378):break
  time.sleep(.5)
 else:raise RuntimeError("Prior disk mirror still active; preserve and inspect")
 assert (O/"persistence_complete.json").exists() and not (O/"checkpoints").is_symlink()
 old={p.name:sha(p.read_bytes()) for p in CP.glob("*.json")}
 assert all(sha((O/"checkpoints"/n).read_bytes())==h for n,h in old.items())
 D=O/"engineering_attempts/io_resume";D.mkdir(parents=True,exist_ok=False)
 for n in ["worker_launch.json","runtime_loaded.json","smoke_partial.json","smoke_summary.json",
           "persistence_complete.json","persistence_progress.json","progress.json","gpu_release.json"]:
  p=O/n
  if p.exists():shutil.copy2(p,D/n)
 audit=dict(storage_only=True,reason="NFS metadata/log/progress stalls despite checkpoint symlink; run I/O directly on SSD",
     old_worker_PID=7056,protocol_sha256=sha((R/"protocol.json").read_bytes()),preserved_checkpoint_SHA256=old,
     frozen_model_and_generation_sources_unchanged=True,old_complete=sum(json.loads(p.read_text())["complete"] for p in CP.glob("*.json")),
     old_saved_requests=sum(len(json.loads(p.read_text())["conditions"]) for p in CP.glob("*.json")),
     runtime_path=str(R),persistent_result_path=str(O),wrapper_sha256=sha(pathlib.Path(__file__).read_bytes()))
 jwrite(R/"storage_resume_audit.json",audit);jwrite(D/"storage_resume_audit.json",audit)
 (R/"checkpoints").symlink_to(CP,target_is_directory=True)
 worker=subprocess.Popen([PY,"-u",__file__,"worker"],cwd=W,env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONUNBUFFERED="1",TOKENIZERS_PARALLELISM="false"),
     stdout=(R/"worker_resume.log").open("ab"),stderr=subprocess.STDOUT,start_new_session=True)
 cfg=dict(worker=worker.pid,runtime=str(R),checkpoints=str(CP),mirror=None,finalizer=None)
 jwrite(BASE/"active_runtime.json",cfg)
 mirror=subprocess.Popen([sys.executable,"-u",__file__,"mirror"],stdout=(BASE/"mirror_resume.log").open("ab"),stderr=subprocess.STDOUT,start_new_session=True)
 cfg["mirror"]=mirror.pid
 final=subprocess.Popen([sys.executable,"-u",__file__,"finish"],stdout=(BASE/"finish_resume.log").open("ab"),stderr=subprocess.STDOUT,start_new_session=True)
 cfg["finalizer"]=final.pid;jwrite(BASE/"active_runtime.json",cfg)
 jwrite(O/"storage_resume_launch.json",cfg)
 print("STORAGE_RESUME_STARTED",json.dumps(cfg),flush=True)
def worker():
 sys.path.insert(0,str(W))
 import vlm_research.internvl3_visual_writing_figure3_3000.prepare as prep
 import vlm_research.internvl3_visual_writing_figure3_3000.run as run
 # Only output path globals change. Model, input paths, core source and generation kwargs stay frozen.
 assert prep.OUT==run.OUT==O
 prep.OUT=R;run.OUT=R
 run.verify()
 run.main()
def mirror():
 cfg=json.loads((BASE/"active_runtime.json").read_text());pid=cfg["worker"]
 seen={};hashes={};partial={};complete={};iteration=0
 while True:
  running=alive(pid)
  for p in sorted(CP.glob("*.json")):
   st=p.stat();mark=(st.st_mtime_ns,st.st_size)
   if seen.get(p.name)==mark:continue
   b=p.read_bytes();v=json.loads(b)
   if running and not v["complete"] and time.time()-partial.get(p.name,0)<60:continue
   dest=O/"checkpoints"/p.name;atomic(dest,b);assert sha(dest.read_bytes())==sha(b)
   seen[p.name]=mark;hashes[p.name]=sha(b);partial[p.name]=time.time();complete[p.name]=v["complete"]
  for p in R.glob("*.json"):
   b=p.read_bytes();dest=O/p.name
   if dest.exists() and sha(dest.read_bytes())==sha(b):continue
   atomic(dest,b);assert sha(dest.read_bytes())==sha(b)
  if iteration%6==0 or not running:
   log=R/"worker_resume.log"
   if log.exists():atomic(O/"worker_resume.log",log.read_bytes())
  heartbeat=dict(status="MIRRORING" if running else "FINALIZING",files=len(hashes),
      complete_files=sum(complete.values()),worker_PID=pid,updated=time.time())
  jwrite(BASE/"mirror_heartbeat.json",heartbeat)
  if not running:break
  iteration+=1;time.sleep(10)
 for p in CP.glob("*.json"):
  b=p.read_bytes();assert sha((O/"checkpoints"/p.name).read_bytes())==sha(b);hashes[p.name]=sha(b)
 result=dict(passed=True,files=len(hashes),all_files_SHA256=hashes,
     canonical_checkpoints=str(O/"checkpoints"),source_and_protocol_unchanged=True)
 jwrite(O/"persistence_complete.json",result);jwrite(BASE/"mirror_complete.json",result)
 print("PERSISTENT_MIRROR_SHA_PASS",len(hashes),flush=True)
def finish():
 # Wait until the launch record has all child PIDs.
 for _ in range(100):
  cfg=json.loads((BASE/"active_runtime.json").read_text())
  if cfg["mirror"] and cfg["finalizer"]:break
  time.sleep(.2)
 while alive(cfg["worker"]):time.sleep(10)
 if not (R/"generation_complete.json").exists():raise RuntimeError("Generation incomplete; preserve and stop without automatic retry")
 assert json.loads((R/"generation_complete.json").read_text())["requests"]==93000
 while not (BASE/"mirror_complete.json").exists():
  if not alive(cfg["mirror"]):raise RuntimeError("Mirror stopped without completion")
  time.sleep(5)
 assert json.loads((BASE/"mirror_complete.json").read_text())["files"]==3000
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="4",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
 commands=[[PY,"-u","-m","vlm_research.internvl3_visual_writing_figure3_3000.analyze"],
  [PY,str(O/"plot_figure3.py")],[PY,"-u","-m","vlm_research.internvl3_visual_writing_figure3_3000.finalize"]]
 for cmd in commands:
  print("CPU_PHASE",cmd[-1],flush=True);subprocess.run(cmd,cwd=W,env=env,check=True)
 print("ALL_AUTHORIZED_WORK_COMPLETE_REVIEW_PENDING",flush=True)
if __name__=="__main__":
 {"prepare":prepare,"worker":worker,"mirror":mirror,"finish":finish}[sys.argv[1]]()

"""Persist immutable evidence, then publish only this scope; no GPU execution."""
from .common import *
import subprocess,shutil,sys,re
PERSIST=ROOT/REL
ENV=dict(os.environ,GIT_SSH_COMMAND="ssh -i /home/featurize/.ssh/ate_github_internvl3090_ed25519 -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20")
def git(*args,timeout=180):
 x=subprocess.run(["git","-C",str(ROOT),"-c","pack.threads=1","-c","pack.windowMemory=32m","-c","pack.deltaCacheSize=16m",*args],capture_output=True,text=True,env=ENV,timeout=timeout)
 if x.returncode:raise RuntimeError("GIT "+repr(args)+" "+x.stdout[-2000:]+x.stderr[-3000:])
 return x.stdout.strip()
def copy_once(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert sha(src)==sha(dst),("PERSIST_WOULD_OVERWRITE",str(dst));return
 tmp=dst.with_name(dst.name+"."+uuid.uuid4().hex+".tmp")
 try:
  shutil.copyfile(src,tmp);assert sha(src)==sha(tmp)
  os.link(tmp,dst)
 finally:tmp.unlink(missing_ok=True)
def prepare():
 assert readj(OUT/"validation.json")["evidence_integrity"]=="PASS"
 assert git("branch","--show-current")==BRANCH and git("rev-parse","HEAD")==PARENT
 assert not git("diff","--name-only") and not git("diff","--cached","--name-only")
 writej(OUT/"source_manifest_publication.json",{"phase":"POST_GPU_PUBLICATION_ONLY","code_sha256":{str(p.relative_to(RUN/"code")):sha(p) for p in sorted((RUN/"code"/PKG).glob("*.py"))},"original_protocol_sha256":sha(OUT/"protocol.json"),"numerical_status":"PARTIAL_REPAIR"})
 code=sorted(p for p in (RUN/"code"/PKG).iterdir() if p.is_file() and p.suffix in [".py",".md"])
 light=sorted(p for p in OUT.rglob("*") if p.is_file())
 names=[]
 for src in code:
  rel=PKG/src.name;copy_once(src,ROOT/rel);names.append(str(rel))
 for src in light:
  rel=REL/src.relative_to(OUT);copy_once(src,ROOT/rel);names.append(str(rel))
 for name,h in readj(OUT/"source_manifest.json")["source_sha256"].items():assert sha(ROOT/name)==h,("BRANCH_SOURCE_NOT_EXECUTED_SOURCE",name)
 raw=[p for folder in ["raw","logs","references"] for p in (RUN/folder).rglob("*") if p.is_file() and p.name not in ["persist_prepare_v001.log","publish_v001.log"]]
 size=sum(p.stat().st_size for p in raw);free=shutil.disk_usage(PERSIST).free
 assert free>size+1024*1048576,("PERSIST_SPACE_TOO_LOW",free,size)
 server=[]
 for i,src in enumerate(sorted(raw)):
  dst=PERSIST/"server_only"/src.relative_to(RUN);copy_once(src,dst)
  h=sha(dst);assert h==sha(src)
  server.append({"source":str(src),"persistent_path":str(dst),"bytes":dst.stat().st_size,"sha256":h,"uploaded_to_GitHub":False})
  print("PERSIST_RAW_VERIFIED",i+1,len(raw),dst.name,dst.stat().st_size,flush=True)
 for n in names:
  p=ROOT/n;assert p.stat().st_size<16*1048576 and "/server_only/" not in n and p.suffix not in [".pt",".safetensors",".jpg",".jpeg",".bin",".sqlite"]
  if p.suffix not in [".png",".pdf"]:
   assert not re.search(r"-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE) PRIVATE KEY|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}",p.read_text()),("SECRET_PATTERN",n)
 manifest={"status":"PARTIAL_REPAIR","parent_commit":PARENT,"branch":BRANCH,"GPU_UUID":GPU,"model_revision":REV,"protocol_sha256":sha(OUT/"protocol.json"),"sample_binding_sha256":sha(OUT/"sample_binding.json"),"GitHub_files":[{"path":n,"sha256":sha(ROOT/n),"bytes":(ROOT/n).stat().st_size} for n in sorted(names)],"server_only":server,"raw_unique_phase_directories_and_all_receipts_verified":True,"no_model_images_raw_dataset_or_secrets_uploaded":True,"numerical_gate_not_passed":True,"search_not_run":True,"publication":"PENDING_AT_COMMIT_CREATION; actual Commit and all remote Blob checks in server_only/delivery_receipt.json","self_reference":"This manifest and SHA256SUMS also uploaded; recursive self-hashes and post-push receipt intentionally excluded"}
 writej(OUT/"artifact_manifest.json",manifest);copy_once(OUT/"artifact_manifest.json",PERSIST/"artifact_manifest.json");names.append(str(REL/"artifact_manifest.json"))
 sums="".join(sha(ROOT/n)+"  "+n+"\n" for n in sorted(names));atomic_bytes(OUT/"SHA256SUMS",sums.encode());copy_once(OUT/"SHA256SUMS",PERSIST/"SHA256SUMS");names.append(str(REL/"SHA256SUMS"))
 for line in sums.splitlines():
  h,n=line.split("  ",1);assert sha(ROOT/n)==h
 writej(RUN/"publication_plan.json",{"files":sorted(names),"light_bytes":sum((ROOT/n).stat().st_size for n in names),"server_only_bytes":size,"server_only_files":len(server),"all_file_sha256_verified":True})
 print("PREPARE_COMPLETE",len(names),"light",sum((ROOT/n).stat().st_size for n in names),"raw",size,flush=True)
def publish():
 plan=readj(RUN/"publication_plan.json");names=plan["files"]
 assert git("branch","--show-current")==BRANCH and git("rev-parse","HEAD")==PARENT
 for line in (PERSIST/"SHA256SUMS").read_text().splitlines():
  h,n=line.split("  ",1);assert sha(ROOT/n)==h
 git("add","--sparse","-f","--",*names)
 assert set(git("diff","--cached","--name-only").splitlines())==set(names)
 assert all(n.startswith(str(PKG)+"/") or n.startswith(str(REL)+"/") for n in names)
 git("-c","user.name=Codex","-c","user.email=codex@openai.com","commit","-m","Diagnose InternVL3 L20 shape-dependent BF16 projection errors and record partial repairs")
 head=git("rev-parse","HEAD")
 receipt={"status":"LOCAL_COMMIT_CREATED","local_commit":head,"branch":BRANCH,"numerical_status":"PARTIAL_REPAIR"}
 writej(PERSIST/"server_only/local_commit_receipt.json",receipt)
 try:
  print("PUSH",git("push","origin","HEAD:refs/heads/"+BRANCH,timeout=240),flush=True)
  assert git("ls-remote","--heads","origin","refs/heads/"+BRANCH).split()[0]==head
  ref="refs/ate-verification/svhr-l20-diagnosis"
  git("fetch","--no-tags","origin","refs/heads/"+BRANCH+":"+ref)
  assert git("rev-parse",ref)==head
  paths=git("ls-tree","-r","--name-only",ref,"--",str(PKG),str(REL)).splitlines();assert set(paths)==set(names)
  for n in names:assert git("rev-parse",ref+":"+n)==git("hash-object","--",n),("REMOTE_BLOB_MISMATCH",n)
  assert not git("status","--porcelain","--untracked-files=no")
  gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip()
  apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
  assert gpu.startswith(GPU) and not apps
  receipt.update(status="PUSH_AND_ALL_REMOTE_FILES_VERIFIED",remote_commit=head,remote_files=len(names),all_blob_hashes_match=True,all_persistent_copy_SHA256_match=True,server_only_files=plan["server_only_files"],server_only_bytes=plan["server_only_bytes"],light_bytes=plan["light_bytes"],GPU_released=True,remaining_compute_processes=[],final_gpu_snapshot=gpu,branch_URL="https://github.com/reyuyu/ATE/tree/"+BRANCH,commit_URL="https://github.com/reyuyu/ATE/commit/"+head,report_URL="https://github.com/reyuyu/ATE/blob/"+head+"/"+str(REL/"report_zh.md"),checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
 except Exception as ex:
  receipt.update(status="PUSH_OR_VERIFICATION_FAILED",error=str(ex));raise
 finally:
  writej(PERSIST/"server_only/delivery_receipt.json",receipt);writej(RUN/"delivery_receipt.json",receipt);print("DELIVERY",json.dumps(receipt),flush=True)
if __name__=="__main__":
 if sys.argv[1:]==["prepare"]:prepare()
 elif sys.argv[1:]==["publish"]:publish()
 else:raise SystemExit("prepare or publish only; no GPU experiments")

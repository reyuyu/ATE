"""Persist verified evidence and publish ONLY this independent failed-Smoke scope."""
from .common import *
import subprocess,shutil,sys,re
KEY="/home/featurize/.ssh/ate_github_internvl3090_ed25519"
ENV=dict(os.environ,GIT_SSH_COMMAND="ssh -i "+KEY+" -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20")
def git(*args,timeout=180):
 p=subprocess.run(["git","-C",str(ROOT),"-c","pack.threads=1","-c","pack.windowMemory=32m","-c","pack.deltaCacheSize=16m",*args],capture_output=True,text=True,env=ENV,timeout=timeout)
 if p.returncode:raise RuntimeError("GIT "+repr(args)+" "+p.stdout[-3000:]+p.stderr[-3000:])
 return p.stdout.strip()
def copy_verified(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() and sha(dst)==sha(src):return
 shutil.copyfile(src,dst)
 assert sha(src)==sha(dst),("COPY_SHA_MISMATCH",str(dst))
def prepare():
 assert readj(OUT/"validation.json")["evidence_integrity"]=="PASS"
 assert git("branch","--show-current")==BRANCH and git("rev-parse","HEAD")==PARENT_COMMIT
 assert not git("diff","--name-only") and not git("diff","--cached","--name-only"),"PREEXISTING_TRACKED_CHANGES"
 # Preserve frozen inference manifest exactly; new post-failure code has its own binding.
 a=readj(OUT/"analysis_source_manifest.json")
 a["code_sha256"]={str(x.relative_to(RUN/"code")):sha(x) for x in sorted((RUN/"code"/PKG).glob("*.py"))}
 writej(OUT/"analysis_source_manifest.json",a)
 code=sorted(x for x in (RUN/"code"/PKG).iterdir() if x.is_file() and x.suffix in [".py",".md"])
 light=sorted(x for x in OUT.rglob("*") if x.is_file() and x.name not in ["artifact_manifest.json","SHA256SUMS"])
 names=[]
 for src in code:
  rel=PKG/src.name;copy_verified(src,ROOT/rel);names.append(str(rel))
 for src in light:
  rel=REL/src.relative_to(OUT);copy_verified(src,ROOT/rel);names.append(str(rel))
 for name,want in readj(OUT/"source_manifest.json")["inference_source_SHA256"].items():
  assert sha(ROOT/name)==want,("BRANCH_SOURCE_DIFFERS_FROM_EXECUTION",name)
 server=[]
 for folder in ["diagnostics","logs","references"]:
  for src in sorted((RUN/folder).rglob("*")):
   if not src.is_file() or src.name in ["persist_prepare.log","publish.log"]:continue
   dst=PERSIST/"server_only"/folder/src.relative_to(RUN/folder)
   copy_verified(src,dst);server.append({"original":str(src),"persistent_path":str(dst),"bytes":dst.stat().st_size,"sha256":sha(dst),"uploaded_to_GitHub":False})
 for src in [RUN/"progress.json",RUN/"bootstrap.log"]:
  if src.exists():
   dst=PERSIST/"server_only"/src.name;copy_verified(src,dst);server.append({"original":str(src),"persistent_path":str(dst),"bytes":dst.stat().st_size,"sha256":sha(dst),"uploaded_to_GitHub":False})
 # Git artifacts are enumerated; models/images/full manifests/.pt never staged.
 for name in names:
  p=ROOT/name
  assert p.stat().st_size<=16*1048576 and "server_only" not in p.parts and p.suffix not in [".pt",".sqlite",".safetensors",".jpg",".jpeg",".bin"]
  if p.suffix not in [".png",".pdf"]:
   t=p.read_text()
   assert not re.search(r"-----BEGIN (?:OPENSSH|RSA|EC|PRIVATE) PRIVATE KEY|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}",t),("SENSITIVE_DATA",name)
 manifest={"experiment_status":"FAILED_NUMERICAL_EQUIVALENCE","formal_generations":0,"branch":BRANCH,"parent_commit":PARENT_COMMIT,"model_revision":REV,"GPU_UUID":GPU,"protocol_sha256":sha(OUT/"protocol.json"),"source_manifest_sha256":sha(OUT/"source_manifest.json"),"sample_manifest_sha256":sha(OUT/"sample_manifest.json"),"github_scope":"Only explicitly enumerated files below; manifest and SHA256SUMS are also included, excluding circular self-hashes","github_files":[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(names)],"server_only":server,"raw_evidence_limitations":readj(OUT/"validation.json")["server_raw_retention_limitations"],"publication_state":"PENDING_AT_COMMIT_CREATION; read server_only/delivery_receipt.json for actual verified Commit SHA","commit_identity":"The containing Git commit identifies this manifest. Its own SHA is not recursively embedded.","no_model_images_credentials_or_original_full_dataset_uploaded":True}
 writej(OUT/"artifact_manifest.json",manifest);copy_verified(OUT/"artifact_manifest.json",PERSIST/"artifact_manifest.json");names.append(str(REL/"artifact_manifest.json"))
 sums="".join(sha(ROOT/n)+"  "+n+"\n" for n in sorted(names))
 (OUT/"SHA256SUMS").write_text(sums);copy_verified(OUT/"SHA256SUMS",PERSIST/"SHA256SUMS");names.append(str(REL/"SHA256SUMS"))
 for line in sums.splitlines():
  h,n=line.split("  ",1);assert sha(ROOT/n)==h
 writej(RUN/"publication_plan.json",{"files":sorted(names),"total_bytes":sum((ROOT/n).stat().st_size for n in names),"server_only_bytes":sum(x["bytes"] for x in server),"server_only_files":len(server),"checksums_verified":True})
 print("PREPARED",len(names),"Git files",sum((ROOT/n).stat().st_size for n in names),"bytes; raw",sum(x["bytes"] for x in server),"bytes",flush=True)
 print(git("status","--short"),flush=True)
def publish():
 plan=readj(RUN/"publication_plan.json");names=plan["files"]
 assert git("branch","--show-current")==BRANCH and git("rev-parse","HEAD")==PARENT_COMMIT
 for line in (PERSIST/"SHA256SUMS").read_text().splitlines():
  h,n=line.split("  ",1);assert sha(ROOT/n)==h
 git("add","--sparse","-f","--",*names)
 staged=git("diff","--cached","--name-only").splitlines();assert set(staged)==set(names),("UNEXPECTED_STAGED_PATHS",staged)
 assert all(n.startswith(str(PKG)+"/") or n.startswith(str(REL)+"/") for n in staged)
 git("-c","user.name=Codex","-c","user.email=codex@openai.com","commit","-m","Record InternVL3 SVH-R numerical-gate failure and reproducible smoke evidence")
 head=git("rev-parse","HEAD")
 receipt={"status":"LOCAL_COMMIT_CREATED","local_commit":head,"branch":BRANCH,"experiment_status":"FAILED","formal_generations":0}
 writej(PERSIST/"server_only/delivery_receipt.json",receipt)
 try:
  pushed=git("push","origin","HEAD:refs/heads/"+BRANCH,timeout=240);print("PUSH",pushed,flush=True)
  remote=git("ls-remote","--heads","origin","refs/heads/"+BRANCH)
  assert remote.split()[0]==head,("REMOTE_HEAD_MISMATCH",remote,head)
  verify="refs/ate-verification/svhr-middle-reaccess"
  git("fetch","--no-tags","origin","refs/heads/"+BRANCH+":"+verify,timeout=180)
  assert git("rev-parse",verify)==head
  # Verify every uploaded path via the fetched remote tree, not only local HEAD.
  paths=git("ls-tree","-r","--name-only",verify,"--",str(PKG),str(REL)).splitlines()
  assert set(paths)==set(names),("REMOTE_SCOPE_MISMATCH",paths)
  for name in names:
   assert git("rev-parse",verify+":"+name)==git("hash-object","--",name),("REMOTE_BLOB_MISMATCH",name)
  assert not git("status","--porcelain","--untracked-files=no")
  gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.used,utilization.gpu","--format=csv,noheader"],text=True).strip()
  apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"],text=True).strip()
  assert gpu.startswith(GPU) and not apps
  receipt.update(status="PUSH_AND_REMOTE_ALL_FILES_VERIFIED",remote_commit=head,remote_verified_files=len(names),server_only_files_not_uploaded=plan["server_only_files"],all_uploaded_bytes=plan["total_bytes"],all_remote_blob_hashes_match=True,all_persistent_copy_sha256_match=True,GPU_released=True,final_gpu_snapshot=gpu,remaining_compute_processes=[],branch_URL="https://github.com/reyuyu/ATE/tree/"+BRANCH,commit_URL="https://github.com/reyuyu/ATE/commit/"+head,report_URL="https://github.com/reyuyu/ATE/blob/"+head+"/"+str(REL/"report_zh.md"),checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
  print("PUBLISHED_VERIFIED",json.dumps(receipt),flush=True)
 except Exception as x:
  receipt.update(status="PUSH_OR_VERIFICATION_FAILED",error=str(x));print("PUBLICATION_FAILED",str(x),flush=True);raise
 finally:
  writej(PERSIST/"server_only/delivery_receipt.json",receipt);writej(RUN/"delivery_receipt.json",receipt)
if __name__=="__main__":
 if sys.argv[1:]==["prepare"]:prepare()
 elif sys.argv[1:]==["publish"]:publish()
 else:raise SystemExit("Use prepare or publish; never starts GPU generation")

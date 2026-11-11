"""Explicit allowlist commit/push and independent remote Git object readback."""
from .common import *
import subprocess,shlex
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r-deep-exit-textvqa-3090")
REL=Path("vlm_research/results/internvl3_svh_r_deep_exit_textvqa_3090")
BRANCH="codex/internvl3-svh-r-deep-exit-textvqa-3090"
SSH="ssh -i /home/featurize/.ssh/ate_deep_exit_14890_ed25519 -o IdentitiesOnly=yes -o IdentityAgent=none -o HostName=github.com -o HostKeyAlias=github.com -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=15"
ENV={**os.environ,"GIT_SSH_COMMAND":SSH,"GIT_TERMINAL_PROMPT":"0"}
def git(*args,cwd=ROOT,binary=False):
 cmd=["git","-c","pack.threads=1","-c","pack.windowMemory=32m","-c","pack.deltaCacheSize=16m","-c","core.autocrlf=false","-C",str(cwd),*map(str,args)]
 x=subprocess.run(cmd,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if x.returncode:raise RuntimeError("Git failed: "+shlex.join(cmd)+"\n"+x.stderr.decode(errors="replace"))
 return x.stdout if binary else x.stdout.decode().strip()
def main():
 receipt=ROOT/REL/"server_only/git_publish_receipt.json"
 plan=read(RUN/"publish_plan.json");assert plan["branch"]==BRANCH
 assert git("branch","--show-current")==BRANCH and git("rev-parse","HEAD")==PARENT_COMMIT
 assert not git("diff","--cached","--name-only"),"PREEXISTING_STAGED_CHANGES"
 paths=[x["path"] for x in plan["files"]]
 assert len(paths)==len(set(paths))
 for x in plan["files"]:
  p=ROOT/x["path"];assert p.is_file() and sha(p)==x["sha256"] and p.stat().st_size==x["bytes"]
  assert x["path"].startswith((PKG+"/",str(REL)+"/")) and "server_only" not in p.relative_to(ROOT).parts
  assert p.stat().st_size<10*1024*1024,"OVERSIZED_LIGHT_ARTIFACT"
  if p.suffix not in (".gz",):
   b=p.read_bytes()
   for kind in ("OPENSSH","RSA","EC"):
    assert ("-----BEGIN "+kind+" PRIVATE KEY-----").encode() not in b
   assert not re.search(rb"(?:gh[pousr]_[A-Za-z0-9]{25,}|hf_[A-Za-z0-9]{25,})",b)
 git("add","-f","--",*paths)
 staged=git("diff","--cached","--name-only").splitlines();assert sorted(staged)==sorted(paths)
 assert not any(p.startswith("main/") for p in staged)
 git("-c","user.name=Codex","-c","user.email=codex@openai.com","commit","-m","Add InternVL3 deep visual reaccess TextVQA experiment and audited evidence")
 commit=git("rev-parse","HEAD")
 receipt_data=dict(status="LOCAL_COMMIT_CREATED",branch=BRANCH,commit=commit,parent=PARENT_COMMIT,files=len(paths),allowlist_sha256=sha(RUN/"publish_plan.json"),source_manifest_sha256=sha(ROOT/REL/"source_manifest.json"),protocol_sha256=sha(ROOT/REL/"protocol.json"),server_data_preserved=True)
 write(receipt,receipt_data)
 try:
  git("push","origin","HEAD:refs/heads/"+BRANCH)
  head=git("ls-remote","--heads","origin","refs/heads/"+BRANCH).split()[0];assert head==commit,"REMOTE_HEAD_MISMATCH"
  # Independent new local object DB; download remote trees without full repo/model history.
  verify=RUN/"remote_verification.git";verify.mkdir(exist_ok=False)
  git("init","--bare",cwd=verify)
  git("remote","add","origin",git("remote","get-url","origin"),cwd=verify)
  git("config","remote.origin.promisor","true",cwd=verify);git("config","remote.origin.partialclonefilter","blob:none",cwd=verify)
  git("fetch","--depth=1","--no-tags","--filter=blob:none","origin",BRANCH,cwd=verify)
  assert git("rev-parse","FETCH_HEAD",cwd=verify)==commit
  objects={}
  for x in plan["files"]:
   line=git("ls-tree",commit,"--",x["path"],cwd=verify)
   assert line and line.split()[1]=="blob";objects[x["path"]]=line.split()[2]
  # Batch read exact selected blobs from remote; no alternate object DB or local source is used.
  git("fetch","--no-tags","--no-write-fetch-head","origin",*sorted(set(objects.values())),cwd=verify)
  verified=[]
  for x in plan["files"]:
   b=git("cat-file","blob",objects[x["path"]],cwd=verify,binary=True)
   assert hashlib.sha256(b).hexdigest()==x["sha256"] and len(b)==x["bytes"],("REMOTE_FILE_MISMATCH",x["path"])
   verified.append(dict(path=x["path"],sha256=x["sha256"],git_blob=objects[x["path"]],remote_verified=True))
  assert git("ls-remote","--heads","origin","refs/heads/"+BRANCH).split()[0]==commit
  assert not git("diff","HEAD","--",*paths)
  receipt_data.update(status="PUSHED_REMOTE_ALL_FILES_VERIFIED",remote_head=commit,remote_independent_object_database=str(verify),verified_files=verified,branch_url="https://github.com/reyuyu/ATE/tree/"+BRANCH,commit_url="https://github.com/reyuyu/ATE/commit/"+commit,report_url="https://github.com/reyuyu/ATE/blob/"+commit+"/"+str(REL)+"/report_zh.md",checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
  write(receipt,receipt_data);write(RUN/"git_publish_receipt.json",receipt_data)
  print(json.dumps({k:v for k,v in receipt_data.items() if k!="verified_files"}),flush=True)
 except BaseException as ex:
  receipt_data.update(status="PUSH_OR_REMOTE_VERIFICATION_FAILED",error=repr(ex));write(receipt,receipt_data);write(RUN/"git_publish_receipt.json",receipt_data);raise
if __name__=="__main__":main()

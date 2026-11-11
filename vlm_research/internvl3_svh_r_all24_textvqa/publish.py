"""Explicit allowlist commit/push and immutable-Commit GitHub HTTPS readback."""
from .common import *
import subprocess,shlex,urllib.request,urllib.parse,concurrent.futures
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r-all24-textvqa-3090")
REL=Path("vlm_research/results/internvl3_svh_r_all24_textvqa_3090")
BRANCH="codex/internvl3-svh-r-all24-textvqa-3090"
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
 assert git("branch","--show-current")==BRANCH
 git("merge-base","--is-ancestor",PARENT_COMMIT,"HEAD")
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
 staged=git("diff","--cached","--name-only").splitlines();assert set(staged)<=set(paths)
 assert not any(p.startswith("main/") for p in staged)
 if staged:git("-c","user.name=Codex","-c","user.email=codex@openai.com","commit","-m","Add requested InternVL3 all24-except-L2-L5 TextVQA experiment")
 commit=git("rev-parse","HEAD")
 receipt_data=dict(status="LOCAL_COMMIT_CREATED",branch=BRANCH,commit=commit,parent=PARENT_COMMIT,files=len(paths),allowlist_sha256=sha(RUN/"publish_plan.json"),source_manifest_sha256=sha(ROOT/REL/"source_manifest.json"),protocol_sha256=sha(ROOT/REL/"protocol.json"),server_data_preserved=True)
 write(receipt,receipt_data)
 try:
  before=git("ls-remote","--heads","origin","refs/heads/"+BRANCH).split()
  if not before or before[0]!=commit:git("push","origin","HEAD:refs/heads/"+BRANCH)
  head=git("ls-remote","--heads","origin","refs/heads/"+BRANCH).split()[0];assert head==commit,"REMOTE_HEAD_MISMATCH"
  # Read bytes independently from GitHub at the immutable Commit URL.
  # Avoid Git 2.34 partial-fetch blob negotiation; all source data stay on server.
  def remote_file(x):
   url="https://raw.githubusercontent.com/reyuyu/ATE/"+commit+"/"+urllib.parse.quote(x["path"],safe="/")
   request=urllib.request.Request(url,headers={"User-Agent":"ATE-evidence-integrity-check"})
   for attempt in range(3):
    try:
     with urllib.request.urlopen(request,timeout=40) as response:b=response.read()
     break
    except (urllib.error.URLError,TimeoutError,OSError):
     if attempt==2:raise
     time.sleep(2*(attempt+1))
   assert hashlib.sha256(b).hexdigest()==x["sha256"] and len(b)==x["bytes"],("REMOTE_FILE_MISMATCH",x["path"])
   return dict(path=x["path"],sha256=x["sha256"],git_blob=git("rev-parse",commit+":"+x["path"]),remote_verified=True,transport="GitHub raw HTTPS immutable Commit")
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:verified=list(pool.map(remote_file,plan["files"]))
  assert git("ls-remote","--heads","origin","refs/heads/"+BRANCH).split()[0]==commit
  assert not git("diff","HEAD","--",*paths)
  receipt_data.update(status="PUSHED_REMOTE_ALL_FILES_VERIFIED",remote_head=commit,verification_transport="SSH remote branch head plus independent immutable-Commit GitHub raw HTTPS bytes",verified_files=verified,branch_url="https://github.com/reyuyu/ATE/tree/"+BRANCH,commit_url="https://github.com/reyuyu/ATE/commit/"+commit,report_url="https://github.com/reyuyu/ATE/blob/"+commit+"/"+str(REL)+"/report_zh.md",checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
  write(receipt,receipt_data);write(RUN/"git_publish_receipt.json",receipt_data)
  print(json.dumps({k:v for k,v in receipt_data.items() if k!="verified_files"}),flush=True)
 except BaseException as ex:
  receipt_data.update(status="PUSH_OR_REMOTE_VERIFICATION_FAILED",error=repr(ex));write(receipt,receipt_data);write(RUN/"git_publish_receipt.json",receipt_data);raise
if __name__=="__main__":main()

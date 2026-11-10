"""Scoped non-forced publication and remote blob verification; no PR or main change."""
from pathlib import Path
import subprocess,os,json,hashlib,time
RUN=Path("/home/featurize/ate-runs/internvl3-svh-r-fast-textvqa-imageonly-3090")
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r-fast-textvqa-imageonly-3090")
BRANCH="codex/internvl3-svh-r-fast-textvqa-imageonly-3090"
REL="vlm_research/results/internvl3_svh_r_fast_textvqa_imageonly_3090"
ENV=dict(os.environ,GIT_SSH_COMMAND="ssh -i /home/featurize/.ssh/ate_github_internvl3090_ed25519 -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20")
def git(*args,timeout=120):
 p=subprocess.run(["git","-c","pack.threads=1","-c","pack.windowMemory=32m","-c","pack.deltaCacheSize=16m","-C",str(ROOT),*args],env=ENV,capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError(str(args[:3])+" "+p.stderr.decode(errors="replace")[-2500:])
 return p.stdout
def main():
 files=json.load(open(RUN/"publish_allowlist.json"));assert git("branch","--show-current").decode().strip()==BRANCH
 status=git("diff","--cached","--name-only").decode().splitlines();assert not status or set(status)<=set(files)
 for f in files:
  assert f.startswith("vlm_research/internvl3_svh_r_fast_textvqa_imageonly/") or f.startswith(REL+"/")
  assert "/server_only/" not in f and (ROOT/f).is_file() and (ROOT/f).stat().st_size<5*1024*1024
  if f.endswith((".py",".json",".md",".csv")):
   b=(ROOT/f).read_bytes();assert (b"-----BEGIN "+b"OPENSSH PRIVATE KEY-----") not in b and (b"-----BEGIN "+b"RSA PRIVATE KEY-----") not in b
 git("add","--sparse","-f","--",*files)
 staged=git("diff","--cached","--name-only").decode().splitlines();assert set(staged)==set(files),(len(staged),len(files))
 git("-c","user.name=Codex","-c","user.email=codex@openai.com","commit","-m","Record Image-only TextVQA SVH-R search, paired soft scores and gated speed",timeout=180)
 commit=git("rev-parse","HEAD").decode().strip();print("LOCAL_COMMIT",commit,flush=True)
 git("push","origin","HEAD:refs/heads/"+BRANCH,timeout=180);print("PUSH_OK",flush=True)
 remote=git("ls-remote","origin","refs/heads/"+BRANCH).decode().split()[0];assert remote==commit
 git("fetch","--no-tags","origin","refs/heads/"+BRANCH,timeout=180);assert git("rev-parse","FETCH_HEAD").decode().strip()==commit
 checks=[]
 for f in files:
  data=git("show",commit+":"+f);local=(ROOT/f).read_bytes();assert data==local,("REMOTE_BLOB_MISMATCH",f)
  checks.append({"path":f,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
 receipt={"status":"REMOTE_VERIFIED","branch":BRANCH,"commit":commit,"remote_head":remote,"file_count":len(checks),"files":checks,"branch_url":"https://github.com/reyuyu/ATE/tree/"+BRANCH,"commit_url":"https://github.com/reyuyu/ATE/commit/"+commit,"report_url":"https://github.com/reyuyu/ATE/blob/"+commit+"/"+REL+"/report_zh.md","GPU_released":not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()}
 for p in [RUN/"github_delivery_receipt.json",ROOT/REL/"server_only/github_delivery_receipt.json"]:p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
 print(json.dumps({k:v for k,v in receipt.items() if k!="files"},ensure_ascii=False),flush=True)
if __name__=="__main__":
 try:main()
 except Exception as e:
  (RUN/"publish_failure.json").write_text(json.dumps({"status":"PUBLISH_FAILED","error":repr(e)},indent=2));raise

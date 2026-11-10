"""Explicit light-artifact sync and non-force push on the authorized independent branch."""
from .common import *
import shutil,requests,base64
BRANCH="codex/internvl3-svh-physical-d-seven-benchmarks-3090"
PKG=Path("vlm_research/internvl3_svh_seven_benchmarks")
REL=Path("vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090")
SSH="ssh -i /home/featurize/.ssh/ate_github_internvl3090_ed25519 -o IdentitiesOnly=yes -o IdentityAgent=none -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20"
def git(*args,timeout=120):
    env=dict(os.environ,GIT_SSH_COMMAND=SSH)
    return subprocess.check_output(["git","-c","pack.threads=1","-c","pack.windowMemory=32m","-c","pack.deltaCacheSize=16m","-C",str(ROOT),*args],env=env,text=True,stderr=subprocess.STDOUT,timeout=timeout).strip()
def copy_verified(src,dst):
    src=Path(src);dst=Path(dst);h=sha(src);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and dst.stat().st_size==src.stat().st_size and sha(dst)==h:return h
    temp=dst.with_name(dst.name+".part");shutil.copyfile(src,temp)
    if sha(temp)!=h:raise RuntimeError("PERSISTENT_COPY_SHA_MISMATCH "+str(src))
    os.replace(temp,dst);return h
def main(label):
    assert git("branch","--show-current")==BRANCH
    # No worker/GPU process may be held while NFS/Git operations run.
    gpu=gpu_idle();writej(OUT/"gpu_release.json",{"status":"RELEASED","compute_processes":"","check":gpu,"after_stage":label})
    # Each benchmark environment remains independent; aggregate file maps them explicitly.
    envs={b:readj(OUT/b/"environment.json") for b in BENCHES if (OUT/b/"environment.json").exists()}
    writej(OUT/"environment.json",{"GPU_UUID":GPU,"benchmarks":envs,"one_model_at_a_time":True,"no_cpu_offload":True,"no_other_GPU":True})
    manifests=OUT/"manifests";manifests.mkdir(exist_ok=True)
    for p in (RUN/"manifests").glob("*.jsonl.gz"):copy_verified(p,manifests/p.name)
    refs=OUT/"official_references";refs.mkdir(exist_ok=True)
    for p in (RUN/"references").rglob("*"):
        if p.is_file() and not p.name.startswith("tree_") and p.suffix in (".py",".json",".md",".sh"):copy_verified(p,refs/p.relative_to(RUN/"references"))
    attempts=OUT/"attempt_diagnostics";attempts.mkdir(exist_ok=True)
    for p in (RUN/"attempts").rglob("*"):
        if p.is_file() and p.suffix in (".json",".py",".log"):copy_verified(p,attempts/p.relative_to(RUN/"attempts"))
    raw=[]
    for p in list((RUN/"checkpoints").glob("*.sqlite"))+[p for p in RUN.glob("*.log") if not p.name.startswith("controller")]:
        rel=Path("server_raw")/p.name;h=copy_verified(p,PERSIST/rel)
        raw.append({"ssd_path":str(p),"persistent_path":str(PERSIST/rel),"bytes":p.stat().st_size,"sha256":h,"uploaded":False,"reason":"Raw resumable SQLite/checkpoint or detailed log; light predictions and input/prompt manifests are uploaded separately."})
    files=[]
    for p in (RUN/"code"/PKG).iterdir():
        if p.is_file() and (p.suffix==".py" or p.name in ("README.md",".gitignore")):
            rel=PKG/p.name;copy_verified(p,ROOT/rel);files.append(str(rel))
    uploaded=[]
    # Exclude self-referential checksum/index and all server-only raw artifacts.
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name in ("artifact_manifest.json","SHA256SUMS","publication.json"):continue
        if "__pycache__" in p.parts or p.suffix in (".pyc",".sqlite",".pt",".bin",".safetensors",".jpg",".jpeg"):raise RuntimeError("FORBIDDEN_ARTIFACT "+str(p))
        if p.stat().st_size>45*1024**2:raise RuntimeError("ARTIFACT_TOO_LARGE "+str(p))
        rel=REL/p.relative_to(OUT);uploaded.append({"path":str(rel),"sha256":sha(p),"bytes":p.stat().st_size})
    sourcefiles=[{"path":n,"sha256":sha(ROOT/n),"bytes":(ROOT/n).stat().st_size} for n in files]
    artifact={"status":readj(OUT/"validation.json")["status"],"source_commit":SOURCE,"model_revision":REV,"GPU_UUID":GPU,"protocol_sha256":sha(OUT/"protocol.json"),"dataset_manifest_sha256":sha(OUT/"dataset_manifest.json"),"github_branch":BRANCH,"uploaded_files":uploaded+sourcefiles,"server_only":raw,"commit_binding":"This manifest is bound by its enclosing commit. The non-self-referential publication receipt records that commit separately."}
    writej(OUT/"artifact_manifest.json",artifact)
    sums=[f'{x["sha256"]}  {x["path"]}' for x in uploaded+sourcefiles]+[sha(OUT/"artifact_manifest.json")+"  "+str(REL/"artifact_manifest.json")]
    (OUT/"SHA256SUMS").write_text("\n".join(sums)+"\n")
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name=="publication.json":continue
        rel=REL/p.relative_to(OUT);copy_verified(p,ROOT/rel);files.append(str(rel))
    for x in files:
        p=ROOT/x
        assert p.is_file()
        if p.suffix in (".pyc",".safetensors",".pt",".jpg",".jpeg",".sqlite"):raise RuntimeError("BAD_GIT_FILE")
    # Explicit file allowlist only, including ignored results outside the sparse cone. Never stage a directory.
    for i in range(0,len(files),40):git("add","--sparse","-f","--",*files[i:i+40])
    staged=git("diff","--cached","--name-only").splitlines()
    assert all(x.startswith(str(PKG)+"/") or x.startswith(str(REL)+"/") for x in staged),staged
    if staged:
        git("-c","user.name=Codex","-c","user.email=codex@openai.com","commit","-m","Record InternVL3-SVH seven-benchmark "+label,timeout=180)
    head=git("rev-parse","HEAD")
    try:
        pushed=git("push","origin","HEAD:refs/heads/"+BRANCH,timeout=240)
        remote=git("ls-remote","origin","refs/heads/"+BRANCH).split()[0];assert remote==head
        tree=git("rev-parse",head+":"+str(REL));session=requests.Session();session.trust_env=False
        response=session.get("https://api.github.com/repos/reyuyu/ATE/git/trees/"+tree+"?recursive=1",timeout=30);response.raise_for_status();data=response.json();assert data["sha"]==tree and not data.get("truncated")
        remote_files={x["path"]:x["sha"] for x in data["tree"] if x["type"]=="blob"}
        for name in ["report_zh.md","protocol.json","validation.json","dataset_manifest.json","prompt_manifest.json","accuracy_summary.csv","SHA256SUMS"]:
            assert remote_files[name]==git("rev-parse",head+":"+str(REL/name))
        receipt={"status":"PUSH_VERIFIED","commit":head,"remote_head":remote,"result_tree_sha":tree,"remote_key_files_verified":True,"branch_url":"https://github.com/reyuyu/ATE/tree/"+BRANCH,"commit_url":"https://github.com/reyuyu/ATE/commit/"+head,"report_url":"https://github.com/reyuyu/ATE/blob/"+head+"/"+str(REL/"report_zh.md"),"stage":label}
        writej(RUN/"publication.json",receipt);writej(PERSIST/"publication.json",receipt);print(canonical(receipt),flush=True)
    except Exception as e:
        receipt={"status":"PUSH_FAILED_OR_REMOTE_CONTENT_CHECK_BLOCKED","commit":head,"error":str(e),"stage":label}
        writej(RUN/"publication.json",receipt);writej(PERSIST/"publication.json",receipt);print(canonical(receipt),flush=True)
        # Do not regenerate predictions, force-push or overwrite remote branches.
if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser();a.add_argument("--label",required=True);args=a.parse_args();main(args.label)

"""Create/reuse only this task's independent Git worktree; keep GPU work on SSD."""
from pathlib import Path
import subprocess,json
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r14-full-middle-reaccess-textvqa-3090")
PARENT="/home/featurize/work/ATE-internvl3-svh-r-fast-textvqa-imageonly-3090"
RUN=Path("/home/featurize/ate-runs/internvl3-svh-r14-full-middle-reaccess-textvqa-3090")
BRANCH="codex/internvl3-svh-r14-full-middle-reaccess-textvqa-3090"
COMMIT="3af6596b1ce59d80e203ebf68c6760e242d4df47"
def git(root,*args):return subprocess.check_output(["git","-C",str(root),*args],stderr=subprocess.STDOUT,timeout=180).decode().strip()
def main():
 if not (ROOT/".git").exists():
  exists=subprocess.run(["git","-C",PARENT,"show-ref","--verify","--quiet","refs/heads/"+BRANCH],timeout=180).returncode
  assert exists in (0,1),exists
  args=["worktree","add","--no-checkout"]+([] if exists==0 else ["-b",BRANCH])+[str(ROOT),BRANCH if exists==0 else COMMIT]
  print(git(PARENT,*args),flush=True)
 assert git(ROOT,"branch","--show-current")==BRANCH
 assert git(ROOT,"rev-parse","HEAD")==COMMIT
 git(ROOT,"config","core.sparseCheckout","true");gd=Path(git(ROOT,"rev-parse","--absolute-git-dir"));(gd/"info").mkdir(exist_ok=True);(gd/"info/sparse-checkout").write_text("/vlm_research/internvl3_svh_r14_full_middle_reaccess_textvqa/\n/vlm_research/results/internvl3_svh_r14_full_middle_reaccess_textvqa_3090/\n");git(ROOT,"read-tree","-mu","HEAD")
 (RUN/"git_setup_validation.json").write_text(json.dumps(dict(status="PASS",root=str(ROOT),branch=BRANCH,parent=COMMIT),indent=2)+"\n");print("GIT_WORKTREE_READY",flush=True)
if __name__=="__main__":main()

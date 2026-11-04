import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,subprocess,json,shutil
base=pathlib.Path('/data/lk/ATE-worktrees/textvqa-visual-reaccess-budget')
def run(*args):
 r=subprocess.run(list(args),cwd=base,text=True,capture_output=True);return dict(code=r.returncode,stdout=r.stdout,stderr=r.stderr)
print(json.dumps(dict(remote=run('git','remote','-v'),status=run('git','status','--porcelain'),worktrees=run('git','worktree','list'),head=run('git','rev-parse','HEAD','main'),branches=run('git','branch','--list','codex/textvqa-ocr-reaccess-mechanism'),disk=shutil.disk_usage('/data')._asdict(),processes=[s for s in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines() if 'ocr_reaccess' in s or 'ocr-reaccess' in s],instructions=[str(p) for root in ['/data','/data/lk','/data/lk/ATE-worktrees',str(base)] for p in [pathlib.Path(root)/'AGENTS.md'] if p.exists()],tree_instructions=run('git','ls-tree','-r','--name-only','HEAD','--','**/AGENTS.md'),ocr_files=[s for s in [str(pathlib.Path(d)/f) for d,dirs,files in os.walk('/data/lk/ATE-resources') for f in files] if any(x in s.lower() for x in ['textocr','ocr_val','ocr_train','textvqa','readme']) and not any(x in s for x in ['/images/','/checkpoints/','/input_specs/','/token_manifests/'])][:160]),ensure_ascii=False,indent=2))


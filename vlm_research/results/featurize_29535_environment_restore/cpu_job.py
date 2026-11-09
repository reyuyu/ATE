"""Run one resumable CPU-only recovery job; never initialize a model or CUDA."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,json,os,subprocess,sys,traceback
P=argparse.ArgumentParser();P.add_argument('job',choices=['model','environment']);a=P.parse_args()
O=Path('/home/featurize/work/ATE/vlm_research/results/featurize_29535_environment_restore')
L=Path('/home/featurize/ate-local-envs');L.mkdir(exist_ok=True)
lock=(L/('recovery29535-'+a.job+'.lock')).open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:print('ALREADY_RUNNING',a.job,flush=True);sys.exit(0)
state={'job':a.job,'status':'RUNNING','pid':os.getpid(),'started_at_UTC':datetime.now(timezone.utc).isoformat(),'GPU_inference':False}
def save():
 p=O/(a.job+'_job.json');t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
save()
try:
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',ATE_RECOVERY_OUT=str(O),PYTHONUNBUFFERED='1')
 args=[sys.executable,str(O/'restore_model.py')] if a.job=='model' else ['bash',str(O/'install_environment.sh')]
 p=subprocess.run(args,cwd=O,env=env)
 state.update(status='PASS' if p.returncode==0 else 'FAILED',exit_code=p.returncode,finished_at_UTC=datetime.now(timezone.utc).isoformat());save()
 sys.exit(p.returncode)
except Exception as e:
 state.update(status='FAILED',error_type=type(e).__name__,finished_at_UTC=datetime.now(timezone.utc).isoformat());save();traceback.print_exc();raise

"""Wait for authorized CPU transfers, restore data, then validate without CUDA."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time
O=Path('/home/featurize/work/ATE/vlm_research/results/featurize_29535_environment_restore')
lock=Path('/home/featurize/ate-local-envs/recovery29535-validation.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:print('CPU_VALIDATION_ALREADY_RUNNING');sys.exit(0)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',ATE_RECOVERY_OUT=str(O),PYTHONUNBUFFERED='1')
state={'status':'WAITING_FOR_TRANSFERS','pid':os.getpid(),'started_at_UTC':datetime.now(timezone.utc).isoformat(),'GPU_inference':False}
def save():
 p=O/'validation_job.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
save()
try:
 while True:
  jobs=[json.loads((O/(n+'_job.json')).read_text()) for n in ('model','environment')]
  if any(x['status']=='FAILED' for x in jobs):raise RuntimeError('CPU prerequisite recovery job failed')
  p=O/'transfer_progress.json';transfer=json.loads(p.read_text()) if p.exists() else {}
  if transfer.get('status')=='PARTIAL':raise RuntimeError('Data transfer failed')
  if transfer.get('status')=='ALL_ARCHIVES_SHA256_VERIFIED' and all(x['status']=='PASS' for x in jobs):break
  time.sleep(10)
 for phase,args in [
  ('RESTORING_DATA',[sys.executable,str(O/'restore_datasets.py')]),
  ('CPU_INPUT_VERIFICATION',['/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python',str(O/'verify_restored_environment.py')]),
  ('FINALIZING_REPORT',[sys.executable,str(O/'finish_recovery.py')])]:
  state.update(status=phase);save();subprocess.run(args,cwd=O,env=env,check=True)
 state.update(status='PASS',finished_at_UTC=datetime.now(timezone.utc).isoformat());save()
except Exception as e:
 state.update(status='FAILED',error_type=type(e).__name__,finished_at_UTC=datetime.now(timezone.utc).isoformat());save();raise

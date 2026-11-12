"""Exactly two authorized independent workers; never retries a failed session."""
from pathlib import Path
import os,sys,subprocess,json,sqlite3,statistics,datetime,shutil
B=Path('/home/featurize/work/ATE/vlm_research/results/unified_e2e32_eight_methods_3090_repeatability')
LOCAL=Path('/home/featurize/ate-local-experiments/unified_e2e32_eight_methods_3090_repeatability')
P='/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python'
def save(n,v):(B/n).write_text(json.dumps(v,indent=2)+'\n')
def release():return dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu,temperature.gpu,power.limit','--format=csv,noheader'],text=True).strip(),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip(),server_left_running=True)
assert not (B/'coordinator_started.json').exists(),'No repeat/relaunch permitted'
save('coordinator_started.json',dict(pid=os.getpid(),authorized_sessions=2,automatic_retries=0))
state=dict(status='RUNNING',sessions=[],current_session=None)
try:
 for session in [1,2]:
  state['current_session']=session;save('coordinator_status.json',state)
  D=B/('session'+str(session));L=LOCAL/D.name
  assert not (L/'checkpoint.sqlite').exists()
  assert not release()['processes'],'GPU conflict before independent session'
  env=dict(os.environ,ATE_REPEAT_SESSION=str(session),PYTHONDONTWRITEBYTECODE='1')
  with (L/'runner.log').open('x') as log:
   child=subprocess.Popen([P,'-B',str(L/'code/run_3090.py')],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   state['worker_pid']=child.pid;save('coordinator_status.json',state);rc=child.wait()
  (L/'runner.exit').write_text(str(rc)+'\n');shutil.copyfile(L/'runner.log',D/'runner.log');shutil.copyfile(L/'runner.exit',D/'runner.exit')
  rel=release();(D/'gpu_release.json').write_text(json.dumps(rel,indent=2)+'\n');assert not rel['processes']
  completion=json.loads((D/'completion.json').read_text());state['sessions'].append(dict(session=session,exit_code=rc,status=completion['status'],formal_completed=completion['formal_completed']));save('coordinator_status.json',state)
  if rc!=0 or not completion['status'].startswith('COMPLETE_') or completion['formal_completed']!=240:
   state.update(status='STOPPED_AFTER_HARD_FAILURE',failed_session=session);break
  assert completion['correctness']=='PASS'
  db=sqlite3.connect('file:'+str(D/'checkpoint.sqlite')+'?mode=ro&immutable=1',uri=True)
  rows=[json.loads(p) for (p,) in db.execute('SELECT payload FROM records WHERE category="formal" ORDER BY id')];db.close();assert len(rows)==240
  plan=json.loads((D/'protocol.json').read_text());summary=[]
  for method in plan['requested_methods']:
   rs=[r for r in rows if r['method']==method];assert len(rs)==30 and all(r['history_acceptance']['accepted'] for r in rs)
   vs={m:[r['generation'][m] for r in rs] for m in ['TTFT_ms','TPOT_ms','CUDA_Event_Decode_TPOT_ms','E2E_ms']};vs['KV_MiB']=[r['generation']['cache']['live_bytes']/2**20 for r in rs]
   summary.append(dict(method=method,n=30,mean={k:statistics.mean(v) for k,v in vs.items()}))
  (D/'independent_summary_at_exit.json').write_text(json.dumps(dict(session=session,status=completion['status'],methods=summary),indent=2)+'\n')
  print('SESSION_COMPLETED',session,completion['status'],'formal240',flush=True)
 else:state.update(status='BOTH_SESSIONS_COMPLETE',current_session=None)
except BaseException as ex:
 import traceback
 state.update(status='COORDINATOR_BLOCKED',failure=repr(ex),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
finally:
 save('coordinator_status.json',state);save('gpu_release_final.json',release());print(json.dumps(state),flush=True)

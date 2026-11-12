from pathlib import Path
import json,time,subprocess,traceback
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def run(kind,phase):
 env='/home/featurize/ate-local-envs/'+({'Qwen':'qwen3vl-torch260-tf4576','InternVL':'internvl3-torch260-tf449'}[kind])+'/bin/python'
 log=A/(kind.lower()+'-refine-'+phase+'.log')
 assert not log.exists(),str(log)
 with log.open('x') as f:
  p=subprocess.Popen([env,'-B',str(A/'code/run_refinement_e.py'),'--kind',kind,'--phase',phase],stdout=f,stderr=subprocess.STDOUT)
  save(A/'coordinator_status.json',dict(status='RUNNING',kind=kind,phase='refine_'+phase,pid=p.pid))
  rc=p.wait()
 assert rc==0,(kind,phase,rc)
 print('REFINE_COMPLETE',kind,phase,flush=True)
try:
 save(A/'refinement_coordinator_status.json',dict(status='WAITING_PRIMARY_ALL_PHASES'))
 while True:
  old=json.loads((A/'coordinator_status.json').read_text())
  if old['status']=='COMPLETE_ALL_PHASES':break
  if old['status']=='STOPPED':raise RuntimeError(('PRIMARY_STOPPED',old))
  time.sleep(5)
 save(A/'primary_coordinator_complete.json',old)
 for kind in ['Qwen','InternVL']:
  run(kind,'screen');run(kind,'audit')
  selection=json.loads((A/kind/'refine_selected_stages.json').read_text())
  if selection['SVH']['stage']:
   run(kind,'session1');run(kind,'session2')
  else:print('NO_INCREMENTAL_USEFUL_GAIN_VS_B',kind,flush=True)
  run(kind,'profile')
 save(A/'coordinator_status.json',dict(status='COMPLETE_ALL_PHASES',primary_and_refinements_complete=True))
 save(A/'refinement_coordinator_status.json',dict(status='COMPLETE'))
except BaseException as e:
 err=dict(status='STOPPED',error=repr(e),traceback=traceback.format_exc())
 save(A/'refinement_coordinator_status.json',err)
 # Preserve original stop reason when waiting on a failed primary.
 if (A/'primary_coordinator_complete.json').exists():save(A/'coordinator_status.json',err)
 print(traceback.format_exc(),flush=True);raise

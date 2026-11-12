from pathlib import Path
import subprocess,json,time,traceback
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
S=A.parent/'svh-cross-opt-final-delivery-20261113.json'
def save(x):S.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
try:
 save(dict(status='WAITING_ALL_GPU_PHASES'))
 while True:
  try:
   master=json.loads((A/'coordinator_status.json').read_text())
   refine=json.loads((A/'refinement_coordinator_status.json').read_text())
  except json.JSONDecodeError:
   time.sleep(2);continue
  if master['status']=='STOPPED' or refine['status']=='STOPPED':raise RuntimeError(('GPU_PHASE_STOPPED',master,refine))
  if master['status']=='COMPLETE_ALL_PHASES' and refine['status']=='COMPLETE':break
  time.sleep(10)
 save(dict(status='RUNNING_INDEPENDENT_CPU_VALIDATION_AND_DELIVERY'))
 subprocess.run(['/environment/miniconda3/bin/python','-B',str(A/'code/deliver_validated_artifacts.py')],check=True)
 save(dict(status='COMPLETE_VERIFIED_PERSISTENT_DELIVERY'))
except BaseException as e:
 save(dict(status='STOPPED',error=repr(e),traceback=traceback.format_exc()))
 print(traceback.format_exc(),flush=True);raise

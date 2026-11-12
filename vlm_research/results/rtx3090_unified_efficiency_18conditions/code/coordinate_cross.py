from pathlib import Path
import json,time,subprocess
O=Path(__file__).resolve().parent.parent
# Dependency only: never wait to choose a faster thermal/clock state.
while True:
 p=O/'llava_coordinator_status.json'
 if p.exists():
  s=json.loads(p.read_text())
  if s['status']=='COMPLETE':break
  if s['status']=='STOPPED':
   (O/'cross_coordinator_status.json').write_text(json.dumps(dict(status='WAITING_FOR_LLAVA_CORRECTNESS',llava=s)));raise SystemExit(1)
 time.sleep(10)
completed={}
for kind in ['Qwen','InternVL']:
 env='qwen3vl-torch260-tf4576' if kind=='Qwen' else 'internvl3-torch260-tf449'
 for phase in ['smoke','session1','session2']:
  cmd='/home/featurize/ate-local-envs/'+env+'/bin/python -u -B '+str(O/'code/run_cross.py')+' --kind '+kind+' --phase '+phase
  with (O/(kind+'_'+phase+'_runner.log')).open('x') as log:p=subprocess.Popen(cmd.split(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (O/'cross_coordinator_status.json').write_text(json.dumps(dict(status='RUNNING',backbone=kind,phase=phase,pid=p.pid,completed=completed)))
  rc=p.wait();f=O/kind/phase/'completion.json';s=json.loads(f.read_text()) if f.exists() else dict(status='FAILED_NO_COMPLETION')
  completed[kind+'_'+phase]=s
  if rc or not s['status'].startswith(('COMPLETE','PARTIAL')):break
  if phase=='smoke' and not s.get('usable'):break
(O/'cross_coordinator_status.json').write_text(json.dumps(dict(status='TERMINAL',completed=completed,finished_ns=time.time_ns())))

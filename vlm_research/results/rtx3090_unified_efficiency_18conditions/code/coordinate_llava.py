from pathlib import Path
import subprocess,json,time
O=Path(__file__).resolve().parent.parent
for phase in ['session1','session2']:
 cmd='source /home/featurize/work/ATE/vlm_research/results/featurize_23137-20261113_resource_restore/activate.sh\nexec python -u -B '+str(O/'code/run_llava_v2.py')+' --phase '+phase
 with (O/(phase+'_runner.log')).open('x') as log:p=subprocess.Popen(['bash','-c',cmd],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 (O/'llava_coordinator_status.json').write_text(json.dumps(dict(phase=phase,pid=p.pid,status='RUNNING',started_ns=time.time_ns())))
 rc=p.wait()
 state=json.loads((O/phase/'completion.json').read_text()) if (O/phase/'completion.json').exists() else dict(status='FAILED_NO_COMPLETION')
 if rc or not state['status'].startswith('COMPLETE'):
  (O/'llava_coordinator_status.json').write_text(json.dumps(dict(phase=phase,status='STOPPED',returncode=rc,completion=state)));raise SystemExit(1)
(O/'llava_coordinator_status.json').write_text(json.dumps(dict(status='COMPLETE',formal_completed=720,finished_ns=time.time_ns())))

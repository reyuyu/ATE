from pathlib import Path
import json,time,subprocess,sys
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
status=A/'coordinator_status.json'
def save(x):status.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def wait_initial():
 p=A/'Qwen/screen/completion.json'
 while not p.exists():
  if (A/'STOP_ALL.flag').exists():raise RuntimeError('STOP_ALL')
  time.sleep(5)
 d=json.loads(p.read_text());assert d['status']=='COMPLETE_SCREEN',d
def run(kind,phase):
 if (A/'STOP_ALL.flag').exists():raise RuntimeError('STOP_ALL')
 env='/home/featurize/ate-local-envs/'+({'Qwen':'qwen3vl-torch260-tf4576','InternVL':'internvl3-torch260-tf449','LLaVA':'ate-torch260-cu124-tf449'}[kind])+'/bin/python'
 log=A/(kind.lower()+'-'+phase+'.log')
 assert not log.exists(),str(log)
 save(dict(status='RUNNING',kind=kind,phase=phase,pid=None))
 with log.open('x') as f:
  p=subprocess.Popen([env,'-B',str(A/'code/run_optimization_v4.py'),'--kind',kind,'--phase',phase],stdout=f,stderr=subprocess.STDOUT)
  save(dict(status='RUNNING',kind=kind,phase=phase,pid=p.pid))
  rc=p.wait()
 assert rc==0,(kind,phase,rc)
 print('PHASE_COMPLETE',kind,phase,flush=True)
def follow(kind):
 run(kind,'audit')
 p=A/kind/'selected_stages.json';s=json.loads(p.read_text())
 if any(x['stage'] for x in s.values()):
  run(kind,'session1');run(kind,'session2')
 else:print('NO_BENEFICIAL_CORRECT_CANDIDATE',kind,flush=True)
 run(kind,'profile')
try:
 save(dict(status='RESUMING_LLAVA_CAPTURE_ONLY'))
 run('LLaVA','screen');follow('LLaVA')
 save(dict(status='COMPLETE_ALL_PHASES'));print('COMPLETE_ALL_PHASES',flush=True)
except BaseException as e:
 import traceback
 save(dict(status='STOPPED',error=repr(e),traceback=traceback.format_exc()))
 print(traceback.format_exc(),flush=True);raise

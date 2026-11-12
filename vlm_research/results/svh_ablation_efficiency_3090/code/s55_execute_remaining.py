"""Authorized serial continuation; no retry, no third efficiency session."""
from s55_common import *
import traceback
def main():
 state={'status':'WAITING_FOR_ABLATION','pid':os.getpid(),'started_ns':time.time_ns()}
 def status(**kwargs):state.update(kwargs);save(O/'execution_progress.json',dict(state,updated_ns=time.time_ns()))
 def run(script,arguments,log):
  with Path(log).open('x') as f:
   result=subprocess.run([str(P),'-u','-B',str(O/'code'/script),*arguments],cwd=O,stdout=f,stderr=subprocess.STDOUT)
  assert result.returncode==0,(script,arguments,result.returncode,str(log))
 try:
  status()
  while True:
   state_a=read(O/'ablation/progress.json')
   if state_a['status']!='RUNNING':break
   time.sleep(15)
  assert state_a['status']=='GENERATIONS_COMPLETE_PENDING_CPU_ANALYSIS',state_a
  status(status='QUALITY_CPU_VALIDATION')
  run('s55_analyze.py',['--quality-only'],O/'ablation/cpu_analysis.log')
  assert read(O/'ablation/validation.json')['status']=='PASS'
  for session in (1,2):
   status(status='GENERATION_LENGTH_SESSION'+str(session))
   run('s55_length_run.py',['--session',str(session)],O/'generation_length'/('session'+str(session)+'_run.log'))
   result=read(O/'generation_length'/('session'+str(session))/'progress.json')
   assert result['formal_completed']==360 and result['status'] in ('COMPLETE','COMPLETE_WITH_PERFORMANCE_WARNINGS'),result
  status(status='LENGTH_CPU_VALIDATION')
  run('s55_analyze.py',['--length-only'],O/'generation_length/cpu_analysis.log')
  assert read(O/'generation_length/validation.json')['formal_requests']==720
  status(status='ALL_MEASUREMENTS_AND_CPU_VALIDATION_COMPLETE',finished_ns=time.time_ns())
 except BaseException as ex:
  status(status='STOPPED_NO_AUTOMATIC_RETRY',error=repr(ex),traceback=traceback.format_exc());raise
if __name__=='__main__':main()

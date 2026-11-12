"""Serial two-session controller; no automatic retries or third session."""
from c55_common import *
import traceback
def main():
 state=dict(status='STARTING',pid=os.getpid(),started_ns=time.time_ns())
 try:
  for session in (1,2):
   state.update(status='SESSION'+str(session),updated_ns=time.time_ns());save(N/'execution_progress.json',state)
   with (N/f'session{session}_run.log').open('x') as log:subprocess.run([str(P),'-u','-B',str(N/'code/c55_run.py'),'--session',str(session)],stdout=log,stderr=subprocess.STDOUT,check=True)
   assert read(N/f'session{session}/progress.json')['status'] in ('COMPLETE','COMPLETE_WITH_PERFORMANCE_WARNINGS')
  state.update(status='BOTH_SESSIONS_COMPLETE_PENDING_CPU_ANALYSIS',finished_ns=time.time_ns());save(N/'execution_progress.json',state)
 except BaseException as ex:
  state.update(status='STOPPED_NO_AUTOMATIC_RETRY',error=repr(ex),traceback=traceback.format_exc(),failed_ns=time.time_ns());save(N/'execution_progress.json',state);raise
if __name__=='__main__':main()

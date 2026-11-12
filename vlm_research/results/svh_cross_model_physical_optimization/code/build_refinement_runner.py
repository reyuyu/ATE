from pathlib import Path
import json,hashlib,subprocess
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
s=(A/'code/run_optimization_v3.py').read_text()
s=s.replace("G=A/K/phase;","G=A/K/('refine_'+phase);")
s=s.replace("for ns in ['opt_b','opt_c','opt_d']:","for ns in ['opt_b','opt_e']:")
s=s.replace("{'B':'opt_b','C':'opt_c','D':'opt_d'}","{'B':'opt_b','E':'opt_e'}")
s=s.replace("methods=['SVH','SVH-R']","methods=['SVH']")
s=s.replace("session=phase,","session='refine_'+phase,")
s=s.replace("stages=['B'] if K=='LLaVA' else ['B','C','D']","stages=['E']")
s=s.replace("if stage in ['B','C']:","if stage in ['B','C','E']:")
s=s.replace("exact B/C; frozen BF16 bounded D","exact B/C/E; frozen BF16 bounded D")
s=s.replace("A/K/'selected_stages.json'","A/K/'refine_selected_stages.json'")
s=s.replace("A/K/'screen/candidate_eligibility.json'","A/K/'refine_screen/candidate_eligibility.json'")
s=s.replace("candidates=[('Native','A')]+[(m,'A') for m in methods]+[(m,s)","candidates=[('Native','A')]+[(m,'A') for m in methods]+[(m,'B') for m in methods]+[(m,s)")
s=s.replace("benefit={k:1-mean(s,k)/mean('A',k)","benefit={k:1-mean(s,k)/mean('B',k)")
s=s.replace("assert read(A/'cpu_tests.json')['status']=='PASS_CPU_BEFORE_GPU'","assert read(A/'cpu_tests.json')['status']=='PASS_CPU_BEFORE_GPU'\n assert read(A/'refinement_cpu_tests.json')['status']=='PASS_BEFORE_REFINEMENT_GPU'\n assert read(A/K/'screen'/'candidate_eligibility.json')['eligible']['SVH']['B']")
p=A/'code/run_refinement_e.py';p.write_text(s);compile(s,str(p),'exec')
m=json.loads((A/'optimized_source_manifest.json').read_text());m.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()));(A/'optimized_source_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
coord="""from pathlib import Path
import json,time,subprocess,traceback
A=Path('/home/featurize/ate-experiments/svh-cross-opt-20261113')
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\\n')
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
"""
p=A/'code/coordinate_refinement_e.py';p.write_text(coord);compile(coord,str(p),'exec')
f=(A/'refinement_coordinator.log').open('x');proc=subprocess.Popen(['/environment/miniconda3/bin/python','-B',str(p)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);f.close()
print(json.dumps(dict(refinement_coordinator_pid=proc.pid,waiting_primary=True,runner_sha256=hashlib.sha256((A/'code/run_refinement_e.py').read_bytes()).hexdigest())))

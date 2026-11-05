import pathlib,subprocess,os,json,time,datetime
W=pathlib.Path('/data/lk/ATE-worktrees/textvqa-single-layer-visual-ablation');R=W/'vlm_research';O=R/'results/textvqa_single_layer_visual_ablation';P=pathlib.Path('/data/lk/ATE-resources/experiments/textvqa_single_layer_visual_ablation')
PY='/data/lk/深度学习/.envs/ate-vlm-research/bin/python'
assert not (O/'execution_processes.json').exists() and not (O/'STOP.json').exists()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='f8f6f0550b6b0575ce6eb2f973f37b98c3367280'
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
prot=json.loads((O/'protocol.json').read_text())
for uuid in prot['GPU_UUIDs']:
 line=next(x for x in inventory.splitlines() if uuid in x);a=[s.strip() for s in line.split(',')]
 assert int(a[2].split()[0])<300 and int(a[3].split()[0])>23000 and uuid not in processes
workers=[];logs=[]
for i,uuid in enumerate(prot['GPU_UUIDs']):
 log=(O/f'execution_worker{i}.log').open('x');logs.append(log)
 env=dict(os.environ,PYTHONPATH=str(R),CUDA_VISIBLE_DEVICES=uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 cmd=[PY,'-m','textvqa_single_layer_visual_ablation.run','--worker',str(i)]
 p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);workers.append(p)
record=dict(start_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),launcher_pid=os.getpid(),workers=[dict(worker=i,pid=p.pid,UUID=prot['GPU_UUIDs'][i]) for i,p in enumerate(workers)])
(O/'execution_processes.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
last=None
while any(p.poll() is None for p in workers):
 status=[p.poll() for p in workers]
 if status!=last:print('WORKER_STATUS',status,flush=True);last=status
 time.sleep(5)
record['exit_codes']=[p.returncode for p in workers];record['end_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(O/'execution_exit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
for f in logs:f.close()
assert record['exit_codes']==[0,0]

import pathlib,subprocess,json,hashlib,os,datetime,sys
w=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer');o=w/'vlm_research/results/qwen3vl_c3v2_physical_transfer';p=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_physical_transfer')
phase=sys.argv[1]
if phase=='freeze':
 e=os.environ.copy()
 for k in ['AUTHOR','COMMITTER']:e['GIT_'+k+'_NAME']='Codex';e['GIT_'+k+'_EMAIL']='codex@openai.com'
 subprocess.run(['git','add','vlm_research/qwen3vl_c3v2_physical','vlm_research/results/qwen3vl_c3v2_physical_transfer'],cwd=w,check=True)
 subprocess.run(['git','-c','gc.auto=0','-c','maintenance.auto=false','commit','-m','Freeze corrected observational instrumentation before Qwen physical smoke'],cwd=w,env=e,check=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()
 paths=list((w/'vlm_research/qwen3vl_c3v2_physical').glob('*.py'))+[w/'vlm_research/qwen3vl_early_windows/experiment.py',w/'vlm_research/qwen3vl_smoke/common.py']
 files={str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 model=pathlib.Path('/data/lk/深度学习/.envs/ate-qwen3vl/lib/python3.11/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py');h=hashlib.sha256(model.read_bytes()).hexdigest();assert h=='dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da'
 (o/'source_manifest.json').write_text(json.dumps(dict(commit=commit,files=files,official_modeling=dict(path=str(model),sha256=h)),indent=2,sort_keys=True)+'\n');sys.exit()
assert not (o/'STOP.json').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
rows=[x.strip().split(', ') for x in gpu.strip().splitlines()]
selected=[next(r for r in rows if r[0]==str(i)) for i in [3,2]]
assert all(int(r[2])<256 and int(r[3])>23000 and int(r[4])==0 and r[1] not in procs for r in selected),(gpu,procs)
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
(o/f'resource_preflight/{phase}_{stamp}.json').parent.mkdir(exist_ok=True)
(o/f'resource_preflight/{phase}_{stamp}.json').write_text(json.dumps(dict(at=stamp,gpu=gpu,processes=procs,selected=selected),indent=2)+'\n')
for rank,row in enumerate(selected):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=row[1],PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
 log=open(p/f'{phase}_shard{rank}_{stamp}.log','w')
 proc=subprocess.Popen(['/data/lk/深度学习/.envs/ate-qwen3vl/bin/python','-m','vlm_research.qwen3vl_c3v2_physical.worker',str(rank),phase],cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(phase,rank,row[1],proc.pid,log.name);(p/f'{phase}_shard{rank}.pid').write_text(str(proc.pid))

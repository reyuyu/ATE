"""New Section 5.5 harness; frozen ATE modules are imported, never edited."""
import os,sys,json,gzip,hashlib,time,subprocess
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',USE_TF='0',USE_FLAX='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_MODULE_LOADING='LAZY')
R=Path('/home/featurize/work/ATE'); O=R/'vlm_research/results/svh_ablation_efficiency_3090'
REST=R/'vlm_research/results/featurize_22484-20261113-r1_resource_restore'
M=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
P=Path('/home/featurize/ate-local-envs/ate-torch260-cu124-tf449/bin/python')
U=R/'vlm_research/results/unified_e2e32_4090'
Q=R/'vlm_research/results/featurize_new3090_c3v2_evaluation'
UUID='GPU-4612d0cb-a47a-674d-b647-3f4065f1627f'
REV='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
sys.path[:0]=[str(R),str(REST),str(O/'code'),str(U/'code')]+[str(R/'vlm_research/external_baselines'/x) for x in ['fastv/compatibility','v2drop/compatibility','e2e32_compatibility']]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def lines(p):
 with gzip.open(p,'rt') as f:return [json.loads(x) for x in f]
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def writegz(p,rows):
 with gzip.GzipFile(filename=str(p),mode='wb',mtime=0) as f:
  for r in rows:f.write((json.dumps(r,ensure_ascii=False)+'\n').encode())
def command(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def gpu():
 fields='uuid,name,temperature.gpu,clocks.sm,clocks.mem,pstate,power.draw,utilization.gpu,memory.used'
 values=command(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits']).split(', ')
 return dict(zip(fields.split(','),values))
def verify_sources():
 manifest=read(O/'source_manifest.json')
 for path,h in manifest['files'].items():assert sha(path)==h,path
 assert command(['git','-C',str(R),'status','--porcelain','--untracked-files=no'])==''
 assert sha(M/'.ate_revision.json')==manifest['model_receipt_sha256']
def guard():
 assert not (O/'STOP.flag').exists(),'User stop flag'
 state=gpu();assert state['uuid']==UUID;assert int(state['temperature.gpu'])<85,'GPU >=85C'
 ps=command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
 assert all(int(p)==os.getpid() for p in ps.splitlines() if p.strip()),'Other GPU process'
 return state
def load_model():
 import torch,transformers
 from vlm_research.mme_seed_quality import engine as loader
 from vlm_research.information_horizon_alignment import engine as frozen
 verify_sources();guard()
 assert torch.__version__=='2.6.0+cu124' and transformers.__version__=='4.49.0'
 torch.set_num_interop_threads(6);loader.MODELS=dict(loader.MODELS,llava=M)
 a=frozen.load('B')
 assert {str(p.device) for p in a.model.parameters()}=={'cuda:0'}
 assert {p.dtype for p in a.model.parameters()}=={torch.float16}
 assert a.model.config.text_config._attn_implementation==a.model.config.vision_config._attn_implementation=='eager'
 return a

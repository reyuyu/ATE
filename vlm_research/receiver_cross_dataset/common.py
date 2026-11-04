import os,json,hashlib,pathlib,csv,datetime,shutil,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'results/receiver_cross_dataset_validation'
PRIVATE=pathlib.Path('/data/lk/ATE-resources/experiments/receiver_cross_dataset_validation')
MODEL=pathlib.Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
UUID='GPU-b08489af-0a8a-eb48-55ba-8346733af417'
CONDITIONS=['All-Writer','Question-Only','Threshold-15','Threshold-18','Budget-Matched Random']
SEED=20261104
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,sort_keys=True,ensure_ascii=False)+'\n');os.replace(t,p)
def csvwrite(p,rows,fields=None):
 rows=list(rows);p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def tensors_hash(inputs):
 return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in inputs.items()}
def guard():
 assert not (OUT/'STOP.json').exists(),'STOP file exists'
 assert shutil.disk_usage('/data').free>20*1024**3,'Disk reserve below 20 GiB'
def hardware():
 return dict(UTC=utc(),disk=shutil.disk_usage('/data')._asdict(),gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True),processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True))
def sources():
 files=list(pathlib.Path(__file__).parent.glob('*.py'))
 for rel in ['gqa3000_token_visual_writing/engine.py','gqa3000_token_visual_writing/common.py','semantic_visual_writers/engine.py','semantic_visual_writers/groups.py','vlm_common.py','paper_rq1/engine.py','query_role_attention/engine.py','benchmarks/_vendor/m4c_evaluator.py','benchmarks/scoring.py']:
  files.append(ROOT/rel)
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}

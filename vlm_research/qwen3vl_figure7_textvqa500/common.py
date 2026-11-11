from pathlib import Path
import os,json,hashlib,sqlite3,csv,subprocess,datetime,random
RUN=Path(os.environ.get('ATE_QFIG_RUN','/home/featurize/ate-runs/qwen3vl-figure7-textvqa500'));CODE=RUN/'code';OUT=RUN/'results';S=json.loads((RUN/'state.json').read_text());GPU=S['gpu'];N=36
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(tuple,list,dict)) else v for k,v in r.items()})
def th(t):
 import torch
 return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def configs():
 perm=list(range(N));random.Random(20261108).shuffle(perm)
 cs={'Native':list(range(N)),'NoImage':[]}
 for family in ('Early','Middle','Late','Random'):
  for k in S['budgets'][1:-1]:
   start=(N+4-k)//2
   keep=list(range(k)) if family=='Early' else list(range(start,start+k)) if family=='Middle' else list(range(N-k,N)) if family=='Late' else sorted(perm[:k])
   cs[family+str(k)]=keep
 return cs
def gpu(own=False):
 g=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip();assert GPU in g and '3090' in g
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 assert not apps if not own else all(int(x)==os.getpid() for x in apps.splitlines()),apps
 return g
def verify():
 for name,h in read(OUT/'source_manifest.json')['files'].items():assert sha(CODE/name)==h,name
def connect():
 db=sqlite3.connect(RUN/'checkpoints/predictions.sqlite');db.execute('pragma journal_mode=WAL');db.execute('pragma synchronous=FULL')
 db.execute('create table if not exists metadata(key text primary key,value text)')
 db.execute('create table if not exists predictions(key text,condition text,checksum text,record text,primary key(key,condition))')
 values={'protocol':sha(OUT/'protocol.json'),'source':sha(OUT/'source_manifest.json'),'sample':sha(OUT/'sample_manifest.json'),'GPU':GPU}
 old=dict(db.execute('select key,value from metadata'))
 if old:assert old==values,'Refuse mixed protocol/source/GPU resume'
 else:
  with db:db.executemany('insert into metadata values(?,?)',values.items())
 return db

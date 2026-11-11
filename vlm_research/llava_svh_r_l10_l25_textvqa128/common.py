from pathlib import Path
import os,sys,json,hashlib,csv,sqlite3,datetime
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE/'frozen_sources'))
RUN=Path(os.environ.get('ATE_LLAVA128_RUN','/home/featurize/ate-runs/llava-svh-r-l10-l25-textvqa128-3090'))
OUT=RUN/'results'
MODEL=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
REVISION='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
GPU='GPU-6c96bba5-5cc6-8be2-b277-5533b4cdab27'
KEEP=tuple(range(10,26))
LABELS=('L10_L25',)
def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})
def th(t):
 import torch
 return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def verify_sources():
 m=read(OUT/'source_manifest.json')
 for rel,x in m['files'].items():assert sha(CODE/rel)==x['sha256'],rel
 return sha(OUT/'source_manifest.json')
def connect():
 db=sqlite3.connect(RUN/'predictions.sqlite');db.execute('pragma journal_mode=WAL');db.execute('pragma synchronous=FULL')
 db.execute('create table if not exists metadata(key text primary key,value text not null)')
 db.execute('create table if not exists predictions(qid text,method text,checksum text not null,record text not null,primary key(qid,method))')
 bound=dict(protocol=sha(OUT/'protocol.json'),source=verify_sources(),sample=sha(OUT/'sample_manifest.json'),inputs=sha(OUT/'input_bindings.json'))
 old=dict(db.execute('select key,value from metadata'));expected={k:json.dumps(v) for k,v in bound.items()}
 if old:assert old==expected
 else:
  with db:db.executemany('insert into metadata values(?,?)',expected.items())
 return db

import hashlib,json,os,datetime,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/qwen3vl_c3_quality_recovery'
HIST=ROOT/'vlm_research/results/qwen3vl_c3v2_exploratory_quality_speed'
PRIVATE=Path('/data/lk/ATE-resources/experiments/qwen3vl_c3_quality_recovery')
MODEL=Path('/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
BASE='30d3e7a0c9318e71a5951aca9a05b6b78108db72'
UUIDS=['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
CONDITIONS=['A','B','C','D','E'];SKIP=tuple(range(1,7));SEED=20261106
SPECS={
 'F':dict(p1=False,decode=False,early=False,late_writer=False),
 'A':dict(p1=True,decode=False,early=False,late_writer=False),
 'B':dict(p1=False,decode=True,early=False,late_writer=False),
 'C':dict(p1=True,decode=True,early=False,late_writer=False),
 'D':dict(p1=False,decode=False,early=True,late_writer=False),
 'E':dict(p1=False,decode=False,early=False,late_writer=True)}
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a]).decode().strip()
def check_sources():
 f=read(OUT/'source_manifest.json')
 for p,h in f['files'].items():assert sha(ROOT/p)==h,p
 return digest(f)

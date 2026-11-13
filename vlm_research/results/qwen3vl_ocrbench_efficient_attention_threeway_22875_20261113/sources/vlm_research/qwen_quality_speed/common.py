import hashlib,json,os,subprocess,datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/qwen3vl_c3v2_exploratory_quality_speed'
PRIVATE=Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_exploratory_quality_speed')
MODEL=Path('/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
BASE='fc3f10eabf48a30b05f4e927a00f50bcd20780ad'
UUIDS=['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
CONDITIONS=['Native','Dense-F','Physical-Original','Physical-Repaired']
SEED=20261106
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args]).decode().strip()
def check_sources():
 f=read(OUT/'source_manifest.json')
 for p,h in f['files'].items():assert sha(ROOT/p)==h,p
 assert sha(f['bridge']['path'])==f['bridge']['sha256']
 return digest(f)

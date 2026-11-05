import hashlib,json,os,datetime,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/qwen3vl_keep_l0_21_gqa1000_reproduction'
RECOVERY=ROOT/'vlm_research/results/qwen3vl_c3_quality_recovery'
PRIOR=ROOT/'vlm_research/results/qwen3vl_c3v2_exploratory_quality_speed'
HISTORY=ROOT/'vlm_research/results/qwen3vl_keep_l0_21_gqa3000'
PRIVATE=Path('/data/lk/ATE-resources/experiments/qwen3vl_keep_l0_21_gqa1000_reproduction')
MODEL=Path('/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
BASE='c2ab2892994a39149f7c3203241fa6f6e9ceb341'
UUIDS=['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6','GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
SEED=20261106
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a]).decode().strip()
def check_sources():
 f=read(OUT/'source_manifest.json')
 for p,h in f['files'].items():assert sha(ROOT/p)==h,p
 return digest(f)
def baseline_path(s,c):
 q=s['question_id'];rank=s['rank']
 return RECOVERY/f'checkpoints/rank{rank}/{q}__{c}.json' if c in ['D','E'] else PRIOR/f'checkpoints/rank{rank}/{q}__{("Dense-F" if c=="F" else c)}.json'

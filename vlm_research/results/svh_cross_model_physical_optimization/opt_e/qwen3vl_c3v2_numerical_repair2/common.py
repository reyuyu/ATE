import pathlib,json,hashlib,os,datetime,shutil
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/qwen3vl_c3v2_numerical_repair2'
PRIVATE=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_numerical_repair2')
PREVIOUS=ROOT/'vlm_research/results/qwen3vl_c3v2_numerical_debug'
ORIGINAL='GQA__06905471'
CONDITIONS=['F','B','W22-functional','P','C']
PAIRS=[('F','B'),('W22-functional','P'),('B','C'),('F','C')]
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def save(p,x):
 import torch
 p.parent.mkdir(parents=True,exist_ok=True);torch.save(x,p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),external_to_git=True)
def check():
 assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(read(OUT/'protocol.json')['GPU_cutoff_UTC'])
 assert shutil.disk_usage('/data').free>20*2**30

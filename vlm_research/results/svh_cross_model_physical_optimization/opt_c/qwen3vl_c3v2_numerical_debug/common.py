"""Frozen diagnostic constants; never starts GPU work on import."""
import pathlib,json,hashlib,os
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/qwen3vl_c3v2_numerical_debug'
PRIVATE=pathlib.Path('/data/lk/ATE-resources/experiments/qwen3vl_c3v2_numerical_debug')
OLD=pathlib.Path('/data/lk/ATE-worktrees/qwen3vl-c3v2-physical-transfer/vlm_research/results/qwen3vl_c3v2_physical_transfer')
BASE='1cd2534f7a79ac7b3c746d17ceb19e823cf53ab3'
FROZEN='dae5207d016acea8225f236ce3b81285d3d8bd90'
ORIGINAL='GQA__06905471'
CONDITIONS=['F','B','W22-functional','P','C']
PAIRS=[('F','B'),('W22-functional','P'),('B','C'),('F','C')]
SHADOW_LAYERS=[1,2,3,4,5,6,22,23]
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def write(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def sample(key):
 m=read(OLD/'sample_manifest.json');return next(s for s in m['samples']+m['diagnostic_only'] if s['id']==key)

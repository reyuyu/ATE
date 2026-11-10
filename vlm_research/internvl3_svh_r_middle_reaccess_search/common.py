"""Frozen small-sample SVH-R search constants; no model loading."""
from pathlib import Path
import os,json,hashlib,time,csv
ROOT=Path("/home/featurize/work/ATE-internvl3-svh-r-middle-reaccess-search-3090")
RUN=Path("/home/featurize/ate-runs/internvl3-svh-r-middle-reaccess-search-3090")
OUT=RUN/"results";PKG=Path("vlm_research/internvl3_svh_r_middle_reaccess_search")
REL=Path("vlm_research/results/internvl3_svh_r_middle_reaccess_search_3090")
PERSIST=ROOT/REL
PARENT_RUN=Path("/home/featurize/ate-runs/internvl3-svh-stop-policy-v2-3090")
PARENT_COMMIT="3297559f21d3de55a6630a5491c9ecf66301c4b7"
BRANCH="codex/internvl3-svh-r-middle-reaccess-search-3090"
GPU="GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e"
REV="853e3a797a661694b1b8ece0cb72dc2b23e3dac9"
MODEL=Path("/home/featurize/ate-local-models/InternVL3-8B")
SPACE={"R0":[],"R2":[16,19],"R4":[16,17,18,19],"R8":list(range(12,20)),"R12":list(range(8,20)),"R14":list(range(6,20)),"RS4":[9,14,16,19]}
METHODS=["Native",*SPACE]
SEED=20261110
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":"))
def sha(x):
 if isinstance(x,Path):
  h=hashlib.sha256()
  with x.open("rb") as f:
   for a in iter(lambda:f.read(1048576),b""):h.update(a)
  return h.hexdigest()
 return hashlib.sha256(x.encode() if isinstance(x,str) else x).hexdigest()
def bind(x):return sha(canonical(x))
def readj(p):return json.loads(Path(p).read_text())
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+".tmp");tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");os.replace(tmp,p)
def csvwrite(p,rows,fields=None):
 p=Path(p);fields=fields or list(dict.fromkeys(k for x in rows for k in x));p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for x in rows:w.writerow({k:canonical(v) if isinstance(v,(list,dict)) else v for k,v in x.items()})
def state(status,**kw):writej(RUN/"progress.json",dict(status=status,updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**kw))

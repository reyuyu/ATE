"""Immutable pilot constants and small artifact helpers."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/internvl3_svh_physical_d_pilot_3090'
RUN=Path('/home/featurize/ate-runs/internvl3-svh-physical-d-pilot-3090')
MODEL=Path('/home/featurize/ate-local-models/InternVL3-8B')
REV='853e3a797a661694b1b8ece0cb72dc2b23e3dac9'
OLD=Path('/home/featurize/work/ATE-internvl3-svh-joint-3000/vlm_research/results/internvl3_svh_joint_l20_skip2to5_3000')
FIG=Path('/home/featurize/work/ATE-internvl3-figure3-3000/vlm_research/results/internvl3_visual_writing_figure3_3000')
DATA=Path('/home/featurize/data/ate-recovery/normalized/GQA-testdev-balanced')
TASKS=['ChooseAttr','ChooseCat','ChooseRel','CompareAttr','LogicalObj','QueryAttr']
METHODS=['Native','Dense-D','Physical-D']
GPU='GPU-e392cec6-7ec8-1fee-397b-fde03a79e79e'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def score(text,gold):return int(text.strip().rstrip('.').lower()==gold.lower())
def command(*a):return subprocess.check_output(a,text=True).strip()
def csvwrite(p,rows,fields):
 import csv
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

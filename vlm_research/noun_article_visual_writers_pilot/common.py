import csv,json,hashlib,subprocess,os
from pathlib import Path
import torch
from query_role_attention.common import ROOT,MODEL,REVISION,TASKS,read,sha256,digest
from gqa3000_token_visual_writing.common import atomic_json,atomic_npz
from semantic_visual_writers.groups import flatten,subsets,intact
BASE='01c82b1d39360fc73bca3d10d12886672ffee370'
OUT=ROOT/'results/noun_article_visual_writers_pilot'
HIST=ROOT/'results/gqa3000_token_visual_writing'
RAW=Path('/data/lk/ATE-resources/experiments/noun_article_visual_writers_pilot')
SEEDS=[20261103,20261107]
def tie(*parts):return hashlib.sha256('|'.join(map(str,('noun-article-v1',)+parts)).encode()).hexdigest()
def csvwrite(p,rows):
    rows=list(rows);assert rows;p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def gitsha(p,base=BASE):return hashlib.sha256(subprocess.check_output(['git','show',base+':'+str(Path(p).relative_to(ROOT.parent))],cwd=ROOT.parent)).hexdigest()
def hist(name):
    p=HIST/name;assert sha256(p)==gitsha(p),name
    return read(p)
def plan():return read(OUT/'token_plan.json')
def bind():return sha256(OUT/'token_plan.json')
def active():assert not (OUT/'STOP.json').exists()
def sources():
    p=list((ROOT/'noun_article_visual_writers_pilot').glob('*.py'))+[ROOT/'gqa3000_token_visual_writing/engine.py',ROOT/'semantic_visual_writers/engine.py',ROOT/'vlm_common.py']
    return {str(x.relative_to(ROOT)):sha256(x) for x in sorted(p)}

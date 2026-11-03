import csv,json,hashlib,subprocess,os
from pathlib import Path
import numpy as np
import torch
from query_role_attention.common import ROOT,MODEL,REVISION,read,sha256,digest
from gqa3000_token_visual_writing.common import atomic_json,atomic_npz
from semantic_visual_writers.groups import intact,flatten
BASE='425286a2566cc6b0387460492a0f99566b142a38'
SEM='01c82b1d39360fc73bca3d10d12886672ffee370'
OUT=ROOT/'results/semantic_omitted_token_attention_mass'
HISTORY=ROOT/'results/semantic_writer_error_cases'
RAW=Path('/data/lk/ATE-resources/experiments/semantic_omitted_token_attention_mass')
BANDS={'L0-3':list(range(4)),'L4-7':list(range(4,8)),'L8-11':list(range(8,12)),'L12-15':list(range(12,16)),'L16-19':list(range(16,20)),'L20-30':list(range(20,31)),'L31':[31],'L0-15':list(range(16)),'L0-19':list(range(20)),'L0-31':list(range(32))}
DISPLAY=['L8-11','L0-15','L16-19']
def csvwrite(path,rows):
    rows=list(rows);assert rows;path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def gitsha(p,commit=BASE):return hashlib.sha256(subprocess.check_output(['git','show',commit+':'+str(Path(p).relative_to(ROOT.parent))],cwd=ROOT.parent)).hexdigest()
def historical(name):
    p=HISTORY/name;assert sha256(p)==gitsha(p),name;return read(p)
def cases():return read(OUT/'case_manifest.json')['cases']
def binding():return sha256(OUT/'case_manifest.json')
def sources():
    names=['__init__.py','common.py','prepare.py','engine.py','tests.py','worker.py','driver.py']
    paths=[ROOT/'semantic_omitted_token_attention_mass'/n for n in names]
    paths += [ROOT/'semantic_visual_writers/engine.py',ROOT/'knockout_probe.py',ROOT/'vlm_common.py',ROOT/'paper_rq1_single_layer_gqa6/run.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}
def normalized(mass,legal,image=576):
    mass=np.asarray(mass,dtype=np.float64);legal=np.asarray(legal,dtype=np.float64);p0=image/legal;non=legal-image
    denom=np.full(np.broadcast_shapes(mass.shape,legal.shape),np.nan);np.divide(1-mass,non,out=denom,where=non>0)
    valid=(non>0)&(mass>=0)&(mass<1)&np.isfinite(mass)
    enrich=np.full(np.broadcast_shapes(mass.shape,legal.shape),np.nan);np.divide(mass/image,denom,out=enrich,where=valid)
    return p0,mass-p0,enrich
def rankkey(row):return (-row['mass_mean'],row['first_position'],row['group_id'])

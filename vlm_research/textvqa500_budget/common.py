import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/textvqa500_early_middle_late_budget'
AUDIT=Path(os.environ.get('ATE_BUDGET_AUDIT','/home/featurize/ate-section33-audit-20261108'))
MODEL=Path(os.environ.get('ATE_MODEL_PATH','/home/featurize/data/ate-recovery/models/llava-1.5-7b-hf')).resolve()
DATA=Path('/home/featurize/data/ate-recovery/normalized/TextVQA')
BUNDLE=Path('/home/featurize/data/ate-recovery/bundles-local-20261108/ate3090-textvqa-ocr/ate3090-textvqa-ocr')
REV='b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
BUDGETS=[0,4,8,12,14,16,21,24,32]
MIDDLE={4:(16,20),8:(14,22),12:(12,24),14:(11,25),16:(10,26),21:(7,28),24:(6,30)}
CONDITIONS={'NoImage':[], 'Native':list(range(32))}
for region in ['Early','Middle','Late']:
 for k in BUDGETS[1:-1]:
  lo,hi=(0,k) if region=='Early' else (32-k,32) if region=='Late' else MIDDLE[k]
  CONDITIONS[f'{region}{k}']=list(range(lo,hi))
REUSE={'Native':('middle','Native'),'Early21':('middle','Early'),'Late21':('middle','Late'),'Middle14':('l11','Decode-L11-L24')}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2)+'\n');tmp.replace(p)
def load_samples():
 import gzip
 return json.loads(gzip.decompress((AUDIT/'middle/sample_manifest.json.gz').read_bytes()))

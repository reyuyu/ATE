from pathlib import Path
import json,gzip,hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'vlm_research/results/textvqa500_early_middle_late_random_budget'
LOCAL=Path('/home/featurize/ate-random-audit-20261108')
MODEL=Path('/home/featurize/ate-local-models/llava-1.5-7b-hf')
DATA=Path('/home/featurize/data/ate-recovery/normalized/TextVQA')
BUNDLE=Path('/home/featurize/data/ate-recovery/restored_bundles/ate3090-textvqa-ocr')
REF='d27f7641c158e1393e58982fd1022359d55d8019'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def load_samples():return json.loads(gzip.decompress((OUT/'sample_manifest.json.gz').read_bytes()))
def load_history():
 with gzip.open(OUT/'historical_per_question.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
 return {c:{r['key']:r for r in rows if r['condition']==c} for c in {r['condition'] for r in rows}}

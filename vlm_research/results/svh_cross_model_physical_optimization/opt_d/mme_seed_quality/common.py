"""Quality-only experiment utilities; no model timing functions."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/mme_seed_physical_c3_quality'
PRIVATE = Path('/data/lk/ATE-resources/experiments/mme_seed_physical_c3_quality')
DATA = Path('/data/lk/ATE-benchmarks-20261105/datasets')
MODELS = {
    'llava': '/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
    'qwen': '/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
}
UUIDS = ['GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6', 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
TASKS = {'Perception': ['existence','count','position','color','posters','celebrity','scene','landmark','artwork','OCR'],
         'Cognition': ['commonsense_reasoning','numerical_calculation','text_translation','code_reasoning']}

def now():
    return datetime.now(timezone.utc).isoformat()

def read(p):
    p=Path(p)
    if p.suffix=='.gz':
        import gzip
        return json.loads(gzip.decompress(p.read_bytes()))
    return json.loads(p.read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def digest(o):
    return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def atomic(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(o,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def tensor_sha(v):
    import torch
    return hashlib.sha256(v.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

def hook_counts(model):
    return [(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]

def mme_parse(text):
    # Exact parse_pred_ans from the pinned official calculation.py after lower().
    text=text.lower()
    if text in ('yes','no'): return text
    if 'yes' in text[:4]: return 'yes'
    if 'no' in text[:4]: return 'no'
    return 'other'

def source_hashes():
    paths=list((ROOT/'vlm_research/mme_seed_quality').glob('*.py'))
    paths += [ROOT/'vlm_research'/p for p in ['qwen3vl_c3v2_physical/engine.py',
        'ate_fulljoint_v2/engine.py','physical_handoff_fastpath/engine.py']]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths if p.name not in ('analyze.py','report.py')}

def git_head():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

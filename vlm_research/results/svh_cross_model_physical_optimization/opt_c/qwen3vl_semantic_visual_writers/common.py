"""Immutable bindings; native Qwen environment and unchanged LLaVA selector."""
import csv
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from ..qwen3vl_early_windows.experiment import MODEL, REVISION, digest, tensor_digest
from ..qwen3vl_smoke.common import sha256
from ..qwen3vl_query_role.common import read, seal as write
from ..semantic_visual_writers.groups import RANDOM_SEEDS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'vlm_research/results/qwen3vl_semantic_visual_writers_240'
RAW = Path('/data/lk/ATE-resources/experiments/qwen3vl_semantic_visual_writers_240')
BASE = '4d3667870132c51cce3c5ab0282ac368077e0217'
LLAVA_COMMIT = '9687b298efcdca651cc5446b5cb07da4a2b4e37c'
LLAVA = Path('/data/lk/ATE-worktrees/semantic-visual-writers-240/vlm_research/results/semantic_visual_writers_240')
HISTORY = ROOT/'vlm_research/results/qwen3vl_query_role_l1_l6_240'
MANIFEST_SHA = '258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c'
SELECTOR_SHA = 'bdfeea43befd9c59cf373e6972d550ee6340416c7550b8ce3a395989a9492e22'
CONDITIONS = ['All-Writer','No-Writer','Semantic-k','Question-Last-k','R-Last-k','Scaffold-k'] + [f'Random-k-{s}' for s in RANDOM_SEEDS] + ['Drop-Semantic']
SWAP_CONDITIONS = ['All-Writer','No-Writer','Semantic-k','R-Last-k',f'Random-k-{RANDOM_SEEDS[0]}']
SEED = 20261103
PRIMARY = [('Semantic-k','Random-mean(3)'),('Semantic-k','Question-Last-k'),
    ('Semantic-k','R-Last-k'),('Semantic-k','Scaffold-k'),('Semantic-k','All-Writer'),
    ('Semantic-k','No-Writer'),('Drop-Semantic','All-Writer')]


def command(*args):
    return subprocess.check_output(args,text=True).strip()


def key(s):
    return s['task']+'__'+s['question_id']


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'vlm_research'/p for p in ['semantic_visual_writers/groups.py',
        'qwen3vl_early_windows/experiment.py','qwen3vl_smoke/common.py',
        'qwen3vl_query_role/common.py','qwen3vl_query_role/engine.py',
        'qwen3vl_query_role/analyze.py']]
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}


def hardware():
    return dict(table=command('nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'), compute=command('nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'))


def select_gpu():
    snapshot = hardware()
    busy = {line.split(',')[0].strip() for line in snapshot['compute'].splitlines()}
    candidates = []
    for line in snapshot['table'].splitlines():
        index,uuid,name,used,free,util = [x.strip() for x in line.split(',')]
        if uuid not in busy and int(used)<200 and int(free)>22000 and int(util)==0:
            candidates.append(dict(index=int(index),uuid=uuid,name=name,logical='cuda:0'))
    write(OUT/'gpu_selection.json',dict(**snapshot,candidates=candidates))
    assert candidates, 'No idle GPU; stop without waiting or interfering'
    return candidates[0]


def preflight(gpu):
    snapshot = hardware()
    line = next(x for x in snapshot['table'].splitlines() if x.split(',')[1].strip()==gpu['uuid'])
    fields = [x.strip() for x in line.split(',')]
    busy = [x for x in snapshot['compute'].splitlines() if x.split(',')[0].strip()==gpu['uuid']]
    passed = not busy and int(fields[3])<200 and int(fields[4])>22000 and int(fields[5])==0
    write(OUT/f'gpu_preflight_{os.getpid()}.json',dict(**snapshot,selected=gpu,passed=passed))
    assert passed and os.environ.get('CUDA_VISIBLE_DEVICES')==gpu['uuid'], 'GPU occupied or wrong visibility; stop'


def verify():
    p = read(OUT/'protocol.json')
    assert p['sources']==sources(), 'Frozen source changed'
    for name,h in p['artifacts'].items():
        assert sha256(OUT/name)==h,name
    for path,h in p['official_sources'].items():
        assert sha256(path)==h,path
    for f in read(OUT/'model_manifest.json')['files']:
        assert (MODEL/f['path']).stat().st_size==f['size']
    assert sha256(OUT/'sample_manifest.json')==MANIFEST_SHA
    assert sha256(ROOT/'vlm_research/semantic_visual_writers/groups.py')==SELECTOR_SHA
    return p


def checkpoint(stem,condition,swap=False):
    return OUT/('swap_checkpoints' if swap else 'checkpoints')/stem/(condition+'.json')


def save_raw(stem,condition,first,trace=None,swap=False):
    path = RAW/('swap' if swap else 'main')/stem/(condition+'.npz')
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(), 'Orphan raw output; retain for diagnosis, never overwrite'
    arrays = dict(first_logits=first.numpy())
    if trace is not None:
        arrays['valid_logits'] = np.stack([v.numpy() for v in trace])
    temp = path.with_suffix('.tmp')
    with temp.open('xb') as handle:
        np.savez_compressed(handle,**arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp,path)
    return dict(path=str(path),sha256=sha256(path))


def raw(record,trace=False):
    artifact = record['raw']
    assert sha256(artifact['path'])==artifact['sha256']
    with np.load(artifact['path'],allow_pickle=False) as data:
        v = data['valid_logits' if trace else 'first_logits'].copy()
    assert np.isfinite(v).all()
    return v


def table(path,rows):
    assert rows
    path = Path(path)
    assert not path.exists(), f'Never overwrite completed table {path}'
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

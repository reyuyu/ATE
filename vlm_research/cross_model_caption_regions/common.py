import json
import os
from pathlib import Path
import random
import subprocess

from ..visual_pathways.common import sha, digest, read, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'vlm_research/results/cross_model_coco_decode_regions_60'
HISTORY = Path('/data/lk/ATE-worktrees/coco-decode-region-60/vlm_research')
MANIFEST = HISTORY/'results/rq2_sparse_visual_caption_60/sample_manifest.json'
LLAVA = HISTORY/'results/coco_decode_visual_regions_60'
UUID = 'GPU-b08489af-0a8a-eb48-55ba-8346733af417'
PROMPT = 'Describe the image in one sentence.'
QMODEL = Path('/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
IMODEL = Path('/data/lk/ATE-resources/models/internvl3-8b/853e3a797a661694b1b8ece0cb72dc2b23e3dac9')
MODELS = {'qwen': QMODEL, 'internvl': IMODEL}


def conditions(kind):
    n = 36 if kind == 'qwen' else 28
    bands = ([range(1,12), range(12,24), range(24,36)] if kind == 'qwen'
             else [range(1,10), range(10,19), range(19,28)])
    c = {'Full': list(range(n)), 'Decode None': [],
         **{name: [0]+list(r) for name, r in zip(('Early','Middle','Late'), bands)}}
    for size, seed in zip(([11,12,12] if kind == 'qwen' else [9,9,9]),
                          [20261102,20261103,20261104]):
        c[f'Random{size}-{seed}'] = [0]+sorted(random.Random(seed).sample(range(1,n), size))
    return c


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def preflight():
    gpu = command('nvidia-smi','-i','1','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits')
    apps = command('nvidia-smi','-i','1','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits')
    row = [v.strip() for v in gpu.split(',')]
    assert row[0] == '1' and row[1] == UUID and '3090' in row[2]
    assert not apps, f'GPU1 occupied; stop without touching other GPUs: {apps}'
    assert int(row[4]) >= 22000 and int(row[5]) == 0, gpu
    return dict(gpu=gpu, processes=apps, pid=os.getpid(), physical_gpu=1, uuid=UUID)


def sources():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT/'vlm_research'/n for n in (
        'visual_pathways/common.py','visual_pathways/engine.py',
        'visual_pathways/sparse_accuracy.py','visual_pathways/scan.py',
        'internvl3_smoke/engine.py','internvl3_smoke/run.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def verify():
    p = read(OUT/'protocol.json')
    assert p['source_sha256'] == sources()
    assert p['conditions'] == {m: conditions(m) for m in MODELS}
    for name, h in p['bindings'].items():
        assert sha(name) == h, name
    for kind, entries in p['model_files'].items():
        for e in entries:
            st = (MODELS[kind]/e['path']).stat()
            assert st.st_size == e['bytes'] and st.st_mtime_ns == e['mtime_ns']
    return p


def divergent(a, b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x != y),
                min(len(a),len(b)) if len(a) != len(b) else -1)

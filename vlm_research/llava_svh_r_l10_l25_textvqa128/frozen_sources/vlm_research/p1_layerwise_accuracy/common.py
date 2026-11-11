import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/gqa3000_p1_layerwise_accuracy'
OLD = Path('/data/lk/ATE-worktrees/single-layer-gqa6/vlm_research')
ORIGINAL = OLD / 'results/paper_rq1_single_layer_gqa6'
FIG3 = Path('/data/lk/ATE-worktrees/figure3-decode-window-comparison/vlm_research/results/figure3_decode_window_comparison')
SPARSE = Path('/data/lk/ATE-worktrees/qwen-internvl-sparse-visual-3000/vlm_research/results/qwen_internvl_sparse_visual_3000')
SCAN = Path('/data/lk/ATE-worktrees/qwen3vl-gqa3000-visual-writing-necessity/vlm_research/results/three_vlm_layer_sensitivity_3000')
SEED = 20261104
BRANCH = 'codex/gqa3000-p1-layerwise-accuracy-llava-qwen'
MODELS = {
    'llava': dict(layers=32, revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9', directory='llava-1.5-7b-hf',
        transformers='4.49.0', dtype='float16', logits_to_keep=0, python='/data/lk/深度学习/.envs/ate-vlm-research/bin/python'),
    'qwen': dict(layers=36, revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', directory='qwen3-vl-8b-instruct',
        transformers='4.57.6', dtype='bfloat16', logits_to_keep=1, python='/data/lk/深度学习/.envs/ate-qwen3vl/bin/python')}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def seal(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Immutable artifact conflict: {path}'
    else:
        atomic(path, value)


def key(sample):
    return sample['task']+'__'+sample['question_id']


def conditions(kind):
    return ['Native Full']+[f'Drop-P1-L{i}' for i in range(MODELS[kind]['layers'])]


def layer_of(kind, condition):
    assert condition in conditions(kind)
    return None if condition == 'Native Full' else int(condition.removeprefix('Drop-P1-L'))


def model_path(kind):
    m = MODELS[kind]
    return Path('/data/lk/ATE-resources/models') / m['directory'] / m['revision']


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def hardware():
    rows = command('nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits')
    procs = command('nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits')
    return dict(gpus=[dict(zip(('index','uuid','used','free','util'),[x.strip() for x in row.split(',')])) for row in rows.splitlines()], compute=procs)


def idle(gpu, snapshot):
    return gpu['uuid'] not in snapshot['compute'] and int(gpu['used']) < 200 and int(gpu['free']) > 22000 and int(gpu['util']) == 0


def select_gpus():
    h = hardware()
    candidates = [next(g for g in h['gpus'] if int(g['index']) == i) for i in (0,3,2,1)]
    return h, [g for g in candidates if idle(g,h)][:2]


def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [OLD/n for n in ('vlm_common.py','knockout_probe.py','paper_rq1/engine.py', 'paper_rq1/scorers.py',
        'paper_rq1_single_layer_gqa6/engine.py','paper_rq1_single_layer_gqa6/analyze.py')]
    paths += [ROOT/'vlm_research'/n for n in ('qwen3vl_early_windows/experiment.py', 'qwen3vl_smoke/common.py')]
    return {str(p): sha(p) for p in sorted(paths)}


def verify(kind):
    p = read(OUT/kind/'protocol.json')
    assert source_hashes() == p['sources'], 'Frozen source changed'
    for name, value in p['artifacts'].items():
        assert sha(OUT/kind/name) == value, name
    return p


def checkpoint(kind, shard, identifier, condition):
    return OUT/kind/f'shard{shard}/checkpoints'/identifier/(condition.replace(' ','_')+'.json')


def validate_checkpoint(record, protocol, sample, condition):
    assert record['binding'] == digest(protocol) and record['sample_sha256'] == digest(sample)
    assert record['condition'] == condition and record['image_sha256'] == sample['image_sha256']
    assert record['generation_sha256'] == digest(record['generation'])
    assert record['generation']['token_ids'] and record['generation']['condition_layer'] == layer_of(protocol['kind'],condition)

from pathlib import Path
import subprocess

from ..three_vlm_sensitivity.common import (
    ROOT, MODELS, TASKS, model_path, read, write, sha256, digest, csv_write,
)

OUT = ROOT / 'vlm_research/results/early_attention_boundary_120'
HISTORY = ROOT / 'vlm_research/results/three_vlm_layer_sensitivity_3000'
RAW = Path('/data/lk/ATE-resources/experiments/early_attention_boundary_120')
SEED = 20261102
UUID = 'GPU-b08489af-0a8a-eb48-55ba-8346733af417'
CONDITIONS = {
    'qwen': {'Full': [], 'L1-L4': list(range(1, 5)), 'L1-L6': list(range(1, 7)),
             'L1-L8': list(range(1, 9)), 'L2-L8': list(range(2, 9)), 'L1-L9': list(range(1, 10))},
    'internvl': {'Full': [], 'L1-L4': list(range(1, 5)), 'L2-L4': list(range(2, 5)),
                'L2-L5': list(range(2, 6)), 'L1-L5': list(range(1, 6))},
}


def key(row):
    return row['task'] + '__' + row['question_id']


def select(rows):
    pools = {task: sorted((r for r in rows if r['task'] == task),
             key=lambda r: digest([SEED, task, r['question_id'], r['image_sha256']])) for task in TASKS}
    selected, used, ids = [], set(), set()
    for _ in range(20):
        for task in TASKS:
            candidates = [r for r in pools[task] if key(r) not in ids]
            row = next((r for r in candidates if r['image_sha256'] not in used), candidates[0])
            selected.append(row)
            used.add(row['image_sha256'])
            ids.add(key(row))
    return selected


def source_hashes():
    folders = ['early_attention_boundary', 'three_vlm_sensitivity', 'internvl3_smoke',
               'internvl3_stage', 'qwen3vl_smoke', 'qwen3vl_early_windows']
    return {str(p.relative_to(ROOT)): sha256(p) for folder in folders
            for p in sorted((ROOT / 'vlm_research' / folder).glob('*.py'))}


def verify():
    protocol = read(OUT / 'protocol.json')
    assert protocol['code_sha256'] == source_hashes(), 'Frozen source changed'
    for name, value in protocol['file_sha256'].items():
        assert sha256(OUT / name) == value, name
    return protocol


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).strip()


def preflight():
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1', 'Physical GPU1 only'
    snapshot = gpu_snapshot()
    row = next(r for r in snapshot.splitlines() if r.split(',')[0].strip() == '1')
    values = [v.strip() for v in row.split(',')]
    assert values[1] == UUID and '3090' in values[2]
    assert int(values[3]) < 200 and int(values[4]) >= 23000, ('GPU1 is not free', row)
    apps = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    assert not apps, ('GPU1 has an active compute process', apps)
    return snapshot

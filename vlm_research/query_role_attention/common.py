import csv
import json
import os
from pathlib import Path
import subprocess

from benchmarks.dataset_registry import digest, sha256
from paper_rq1_single_layer_gqa6.run import atomic_json as write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/query_role_attention_boundary_240'
RAW = Path('/data/lk/ATE-resources/experiments/query_role_attention_boundary_240')
BASE = 'bd171312fd0489e8bc027744b7a976367da11668'
REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
MODEL = Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf')/REVISION
UUID = 'GPU-b08489af-0a8a-eb48-55ba-8346733af417'
SEED = 20261102
TASKS = ['ChooseAttr', 'ChooseCat', 'ChooseRel', 'CompareAttr', 'LogicalObj', 'QueryAttr']
CONDITIONS = ['Full', 'I-Bypass', 'R-Bypass', 'A-Bypass']
COMPARISONS = [('I-Bypass', 'Full'), ('R-Bypass', 'Full'), ('A-Bypass', 'Full'),
               ('R-Bypass', 'I-Bypass'), ('R-Bypass', 'A-Bypass')]
LAYERS = [2, 3, 4, 5]
POOL = ROOT/'results/paper_rq1_single_layer_gqa6/sample_manifest.json'


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    assert rows
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def sources():
    paths = list((ROOT/'query_role_attention').glob('*.py'))
    names = ['paper_rq1/engine.py', 'paper_rq1/scorers.py', 'knockout_probe.py',
             'vlm_common.py', 'benchmarks/dataset_registry.py', 'benchmarks/smoke.py',
             'paper_rq1_single_layer_gqa6/run.py', 'paper_rq1_single_layer_gqa6/engine.py',
             'paper_rq1_early_joint_attention/engine.py', 'first_answer_attention/engine.py',
             'paper_rq1_decode_layerwise_quality/engine.py']
    paths += [ROOT/n for n in names]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def verify():
    p = read(OUT/'protocol.json')
    assert p['sources'] == sources(), 'Frozen source changed'
    for name, value in p['bindings'].items():
        assert sha256(OUT/name) == value, name
    for name, value in p['model_files'].items():
        stat = Path(name).stat()
        assert stat.st_size == value['bytes'] and stat.st_mtime_ns == value['mtime_ns'], name
    for name, value in {**p['transformers_sources'], **p['scorer_sources']}.items():
        assert sha256(name) == value, name
    return p


def preflight():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUID
    raw = subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    fields = [v.strip() for v in raw.strip().split(',')]
    apps = subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True).strip()
    assert fields[0] == '1' and fields[1] == UUID and '3090' in fields[2]
    assert int(fields[3]) < 200 and int(fields[4]) > 22000 and not apps, (raw, apps)
    return dict(physical_gpu=1, uuid=UUID, logical_device='cuda:0', table=raw, compute_processes=apps)


def selected(pool):
    if len(pool) != 3000 or len({s['question_id'] for s in pool}) != 3000:
        raise ValueError('Require exact original 3000-question pool')
    if any(sum(s['task'] == t for s in pool) != 500 for t in TASKS):
        raise ValueError('Wrong task counts')
    used, result = set(), []
    for task in TASKS:
        ordered = sorted([s for s in pool if s['task'] == task], key=lambda s: digest([
            SEED, 'query-role-input-v1', task, s['question_id'], s['image_sha256'],
            s['input_mapping']['prompt_ids']]))
        chosen = []
        for sample in ordered:
            if sample['image_sha256'] in used:
                continue
            chosen.append(sample)
            used.add(sample['image_sha256'])
            if len(chosen) == 40:
                break
        if len(chosen) != 40:
            raise ValueError('Insufficient unique image clusters; never substitute')
        result.extend(chosen)
    return result

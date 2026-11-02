import csv
import json
import os
from pathlib import Path
import subprocess

from benchmarks.dataset_registry import digest, sha256
from paper_rq1_single_layer_gqa6.run import atomic_json as write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/attention_relay_joint_240'
RAW = Path('/data/lk/ATE-resources/experiments/attention_relay_joint_240')
BASE = 'bd171312fd0489e8bc027744b7a976367da11668'
REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
MODEL = Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf') / REVISION
GPU_INDEX = 0
GPU_UUID = 'GPU-a63c6362-dcfd-4a26-c44b-014b40127c89'
SEED = 20261102
TASKS = ['ChooseAttr', 'ChooseCat', 'ChooseRel', 'CompareAttr', 'LogicalObj', 'QueryAttr']
LAYERS = [2, 3, 4, 5]
CONDITIONS = {'Full': (False, False), 'A-Skip': (True, False),
              'Text-Only': (False, True), 'Joint': (True, True)}


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    if not rows:
        raise ValueError('Cannot silently emit an empty table')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sources():
    names = ['paper_rq1/engine.py', 'paper_rq1/scorers.py', 'knockout_probe.py',
             'vlm_common.py', 'benchmarks/dataset_registry.py', 'benchmarks/smoke.py',
             'paper_rq1_single_layer_gqa6/run.py', 'paper_rq1_single_layer_gqa6/engine.py',
             'paper_rq1_early_joint_attention/engine.py', 'paper_rq1_early_joint_attention/statistics.py',
             'paper_rq1_decode_layerwise_quality/engine.py', 'paper_rq1_decode_layerwise_quality/statistics.py',
             'rq2_decode_visual/engine.py', 'rq2_source_knockout/engine.py',
             'rq2_visual_sufficiency/engine.py', 'first_answer_attention/engine.py']
    files = [ROOT / n for n in names] + sorted((ROOT / 'attention_relay_joint').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def snapshot():
    table = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True)
    return dict(table=table, compute_apps=apps)


def preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise ValueError('Expose only the frozen physical GPU UUID')
    snap = snapshot()
    row = next(r for r in csv.reader(snap['table'].splitlines()) if int(r[0]) == GPU_INDEX)
    row = [v.strip() for v in row]
    okay = row[1] == GPU_UUID and int(row[3]) < 200 and int(row[4]) > 22000
    okay = okay and not any(r.startswith(GPU_UUID) for r in snap['compute_apps'].splitlines())
    if not okay:
        write(OUT / f'gpu_preflight_blocked_{os.getpid()}.json', snap)
        raise RuntimeError('Selected physical GPU is not idle; stop, do not wait or preempt')
    return dict(physical_gpu=GPU_INDEX, uuid=GPU_UUID, logical_device='cuda:0', **snap)


def device_binding():
    import torch
    if torch.cuda.device_count() != 1:
        raise AssertionError('Require exactly one logical GPU')
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.lower().removeprefix('gpu-') != GPU_UUID.lower().removeprefix('gpu-'):
        raise AssertionError('Logical CUDA device does not match frozen physical UUID')
    return dict(logical_device='cuda:0', actual_uuid=actual, physical_index=GPU_INDEX)


def verify():
    p = read(OUT / 'protocol.json')
    if p['source'] != sources():
        raise AssertionError('Frozen source changed')
    for n, h in p['bindings'].items():
        assert sha256(OUT / n) == h, n
    for n, b in p['model_files'].items():
        st = Path(n).stat()
        assert st.st_size == b['bytes'] and st.st_mtime_ns == b['mtime_ns'], n
    for n, h in {**p['transformers_sources'], **p['scorer_sources']}.items():
        assert sha256(n) == h, n
    return p

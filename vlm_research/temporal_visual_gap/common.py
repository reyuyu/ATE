import csv
import json
import os
from pathlib import Path
import subprocess

from benchmarks.dataset_registry import digest, sha256
from paper_rq1_single_layer_gqa6.run import atomic_json as write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/gqa_coco_temporal_visual_gap'
RAW = Path('/data/lk/ATE-resources/experiments/gqa_coco_temporal_visual_gap')
HROOT = Path('/data/lk/ATE-worktrees/coco-decode-region-60/vlm_research')
HCOCO = HROOT/'results/rq2_sparse_visual_caption_60'
UUID = 'GPU-b08489af-0a8a-eb48-55ba-8346733af417'
SEED = 20261102
CONDITIONS = ['Full', 'None', 'Periodic25', 'Periodic50',
              'Random25-20261102', 'Random25-20261103',
              'Random50-20261102', 'Random50-20261103']
TASKS = ['ChooseAttr', 'ChooseCat', 'ChooseRel', 'CompareAttr', 'LogicalObj', 'QueryAttr']


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert rows
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def gate(condition, image, step):
    if condition not in CONDITIONS or step < 1:
        raise ValueError('Only frozen conditions and real Decode steps')
    if condition == 'Full':
        return True
    if condition == 'None':
        return False
    if condition.startswith('Periodic'):
        return (step-1) % (4 if condition == 'Periodic25' else 2) == 0
    name, seed = condition.split('-')
    number = int(digest(['temporal-image-gate-v1', image, step, int(seed)]), 16)
    return number < (2**256 // (4 if name == 'Random25' else 2))


def sources():
    paths = list((ROOT/'temporal_visual_gap').glob('*.py'))
    names = ['paper_rq1/engine.py', 'paper_rq1/scorers.py', 'rq2_decode_visual/engine.py',
             'rq2_visual_sufficiency/engine.py', 'rq2_source_knockout/engine.py',
             'knockout_probe.py', 'vlm_common.py', 'benchmarks/dataset_registry.py',
             'benchmarks/smoke.py', 'paper_rq1_single_layer_gqa6/run.py',
             'paper_rq1_token_role_decode/labels.py', 'paper_rq1_token_role_decode/protocol.py']
    paths += [ROOT/name for name in names]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def verify():
    p = read(OUT/'protocol.json')
    assert p['sources'] == sources(), 'Frozen source changed'
    for name, value in p['bindings'].items():
        assert sha256(OUT/name) == value
    for name, value in p['model_files'].items():
        s = Path(name).stat()
        assert s.st_size == value['bytes'] and s.st_mtime_ns == value['mtime_ns']
    for name, value in {**p['transformers_source'], **p['scorer_source']}.items():
        assert sha256(name) == value
    return p


def preflight():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUID
    table = subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    fields = [v.strip() for v in table.strip().split(',')]
    apps = subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True).strip()
    assert fields[0] == '1' and fields[1] == UUID and '3090' in fields[2]
    assert int(fields[3]) < 200 and int(fields[4]) > 22000 and not apps, (table, apps)
    return dict(physical_gpu=1, uuid=UUID, table=table, processes=apps)


def history(sample, condition):
    assert condition in ['Full', 'None']
    q = sample['question_id']
    if condition == 'Full':
        b = read(HCOCO/'history_bindings.json')[q]['A']
        path, raw = HROOT/b['path'], HROOT/b['raw_file']
        assert sha256(path) == b['sha256'] and sha256(raw) == b['raw_sha256']
        r = read(path)
        assert r['sample_sha256'] == digest(sample) == b['sample_sha256']
        assert digest(r['result']) == r['result_sha256']
        g = r['result']['generation']
    else:
        path = HCOCO/f"checkpoints/{q.replace(':','_')}/DN.json"
        r = read(path)
        assert r['sample_sha256'] == digest(sample)
        assert r['binding'] == digest(read(HCOCO/'protocol.json'))
        assert digest(r['generation']) == r['generation_sha256']
        assert r['first_logits_exact_historical_Full']
        g, raw = r['generation'], HCOCO/r['raw_file']
        assert g['prefill_blocked'] == [] and g['decode_blocked'] == list(range(32))
        assert sha256(raw) == r['raw_sha256']
    assert g['own_cache'] and g['own_prefix'] and g['eos'] and not g['truncated']
    return g, dict(path=str(path), sha256=sha256(path), raw=str(raw), raw_sha256=sha256(raw))

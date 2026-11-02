import csv
import json
import os
from pathlib import Path
import subprocess

from benchmarks.dataset_registry import digest, sha256
from paper_rq1_single_layer_gqa6.run import atomic_json as write
from paper_rq1_token_role_decode.protocol import MODEL_PATH, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/coco_wordwise_attention_sources_200'
OLD = ROOT / 'results/paper_rq1_late_attention_sources'
RAW = Path('/data/lk/ATE-resources/experiments/coco_wordwise_attention_sources_200')
UUID = 'GPU-b08489af-0a8a-eb48-55ba-8346733af417'
SOURCES = ['BOS', 'IMAGE', 'PROMPT_TEXT', 'GENERATED_OLDER', 'SELF']
MAIN = ['BOS', 'IMAGE', 'PROMPT_TEXT', 'GENERATED_HISTORY']
ROLES = ['Function', 'Visual', 'Relation', 'Unknown', 'Punctuation', 'EOS']
SELECTED = [14, 26, 31]
SEED = 20261102


def read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sources():
    folders = ['coco_wordwise_sources', 'paper_rq1_late_attention_sources', 'paper_rq1_token_role_decode',
               'paper_rq1', 'paper_rq1_single_layer_gqa6']
    paths = [p for folder in folders for p in (ROOT / folder).glob('*.py')]
    paths += [ROOT / name for name in ['vlm_common.py', 'knockout_probe.py', 'benchmarks/smoke.py',
                                      'benchmarks/dataset_registry.py']]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def verify():
    config = read(OUT / 'protocol.json')
    assert config['source_sha256'] == sources(), 'Frozen source changed'
    assert config['sample_sha256'] == sha256(OUT / 'sample_manifest.json')
    assert config['cpu_sha256'] == sha256(OUT / 'cpu_validation.json')
    return config


def preflight():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == UUID, 'Only physical GPU1 UUID may be exposed'
    snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
    row = next(r for r in snapshot.splitlines() if r.split(',')[0].strip() == '1')
    values = [v.strip() for v in row.split(',')]
    assert values[1] == UUID and '3090' in values[2]
    assert int(values[3]) < 200 and int(values[4]) >= 20000, ('GPU1 occupied/insufficient memory', row)
    processes = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-compute-apps=gpu_uuid,pid,process_name',
                                         '--format=csv,noheader'], text=True).strip()
    assert not processes, ('GPU1 is occupied; stopping', processes)
    return dict(snapshot=snapshot, physical_gpu=1, uuid=UUID, processes=processes)

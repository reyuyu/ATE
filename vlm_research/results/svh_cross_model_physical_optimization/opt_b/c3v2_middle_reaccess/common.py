import collections
import csv
import datetime
import json
from pathlib import Path

from ..physical_visual_handoff.common import atomic, digest, read, sha, TOLERANCES
from ..accuracy_autoresearch.common import config
from ..benchmarks.scoring import textvqa_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/c3v2_middle_visual_reaccess'
HISTORY = Path('/data/lk/ATE-worktrees/c3v2-reverse-writer-pruning/vlm_research/results/c3v2_reverse_writer_pruning')
BASE_COMMIT = 'e936ff3dcf3ae02e4395dd1f2e9dd05cd8da25cb'
LABELS = ('Native', 'C3-V2', 'C3+Middle9', 'C3+Middle10')
KEEP = {'Native': tuple(range(32)), 'C3-V2': (),
        'C3+Middle9': tuple(range(11, 20)), 'C3+Middle10': tuple(range(11, 21))}
CUTOFF = {'Native': 32, 'C3-V2': 20, 'C3+Middle9': 20, 'C3+Middle10': 21}
DEVICES = [dict(index=0, uuid='GPU-a63c6362-dcfd-4a26-c44b-014b40127c89'),
           dict(index=1, uuid='GPU-b08489af-0a8a-eb48-55ba-8346733af417')]
SEED = 20261105
MAX_TOKENS = 32
CFG = config('all', 'both', head=1)


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def sources():
    files = [p for p in Path(__file__).parent.glob('*.py') if p.name != 'analyze.py']
    files += [ROOT / 'vlm_research' / p for p in (
        'accuracy_autoresearch/engine.py', 'accuracy_autoresearch/common.py',
        'ate_fulljoint_v2/engine.py', 'physical_handoff_fastpath/engine.py',
        'reverse_writer/engine.py', 'reverse_writer/common.py',
        'p1_layerwise_accuracy/engine.py', 'p1_layerwise_accuracy/common.py',
        'benchmarks/scoring.py', 'benchmarks/protocols.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def save(path, payload):
    assert not path.exists(), f'Refuse overwrite: {path}'
    atomic(path, dict(payload=payload, sha256=digest(payload)))


def load(path):
    value = read(path)
    assert digest(value['payload']) == value['sha256']
    return value['payload']


def quality_path(key, label):
    return OUT / 'checkpoints' / key / (label + '.json')


def performance_path(key, mode, repeat, label):
    return OUT / 'performance' / key / mode / str(repeat) / (label + '.json')


def score(sample, text):
    assert len(sample['answer']) == 10
    return float(textvqa_score(text, sample['answer']))

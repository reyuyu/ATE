import datetime
import gzip
import json
import sqlite3
from pathlib import Path

from ..physical_visual_handoff.common import atomic, digest, read, sha, TOLERANCES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/c3v2_gqa_testdev_balanced_12578'
REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
MODEL = Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf') / REVISION
LABELS = ('C3-V2', 'C3-V2+L11-L24', 'Native')
KEEP = {'C3-V2': (), 'C3-V2+L11-L24': tuple(range(11, 25))}
CUTOFF = {'C3-V2': 20, 'C3-V2+L11-L24': 25}
SEED = 20261106
DEVICES = [dict(index=0, uuid='GPU-a63c6362-dcfd-4a26-c44b-014b40127c89'),
           dict(index=1, uuid='GPU-b08489af-0a8a-eb48-55ba-8346733af417')]


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def prediction(text):
    # Identical to the pinned LLaVA convert_gqa_for_eval.py, not VQA soft scoring.
    return text.rstrip('.').lower()


def score(sample, text):
    return int(prediction(text) == sample['answer'])


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT / 'vlm_research' / p for p in (
        'c3v2_middle_reaccess/engine.py', 'c3v2_middle_reaccess/common.py',
        'ate_fulljoint_v2/engine.py', 'accuracy_autoresearch/engine.py',
        'accuracy_autoresearch/common.py', 'physical_handoff_fastpath/engine.py',
        'physical_visual_handoff/engine.py', 'physical_visual_handoff/common.py',
        'p1_layerwise_accuracy/engine.py', 'p1_layerwise_accuracy/common.py',
        'reverse_writer/engine.py', 'reverse_writer/common.py', 'benchmarks/protocols.py')]
    old = Path('/data/lk/ATE-worktrees/single-layer-gqa6/vlm_research')
    paths += [old / p for p in ('vlm_common.py', 'knockout_probe.py')]
    return {str(p): sha(p) for p in sorted(paths)}


def samples():
    with gzip.open(OUT / 'sample_manifest.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def expand(sample):
    compact = sample['compact_prompt_ids']
    i = compact.index(32000)
    assert compact.count(32000) == 1
    return compact[:i] + [32000] * 576 + compact[i+1:]


def freeze_check():
    f = read(OUT / 'execution_freeze.json')
    assert f['sources'] == sources(), 'Frozen inference source changed'
    for name, value in f['artifacts'].items():
        assert sha(OUT / name) == value, name
    return f


def database(rank, binding):
    path = OUT / f'worker{rank}/results.sqlite'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS results '
               '(qid TEXT, condition TEXT, payload TEXT NOT NULL, sha256 TEXT NOT NULL, '
               'PRIMARY KEY(qid,condition))')
    old = db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
    if old:
        assert old[0] == binding, 'Checkpoint belongs to a different frozen experiment'
    else:
        db.execute('INSERT INTO metadata VALUES (?,?)', ('binding', binding))
    db.commit()
    return db


def put(db, record):
    # One durable transaction per condition. Duplicate results are never overwritten.
    with db:
        db.execute('INSERT INTO results VALUES (?,?,?,?)',
                   (record['question_id'], record['condition'],
                    json.dumps(record, ensure_ascii=False, separators=(',', ':')), digest(record)))


def unpack(row):
    record = json.loads(row[0])
    assert digest(record) == row[1], 'Checkpoint digest mismatch'
    return record

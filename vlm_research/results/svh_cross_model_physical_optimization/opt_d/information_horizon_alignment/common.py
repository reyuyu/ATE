"""Small, frozen quality-only runner shared by the two distinct protocols."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/cvpr2026_information_horizon_alignment'
RES = Path('/data/lk/ATE-resources/information-horizon-alignment')
AUTHOR = Path('/data/lk/ATE-resources/Information-Horizon')
DATA = Path('/data/lk/ATE-benchmarks-20261105/datasets')
AUTHOR_SHA = '75909b2936a13d7214e83514f0bc0cb9cc91139e'
NATIVE_REV = '4481d270cc22fd5c4d1bb5df129622006ccd9234'
NATIVE_MODEL = RES / 'hf-hub/models--liuhaotian--llava-v1.5-7b/snapshots' / NATIVE_REV
HF_MODEL = Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
UUIDS = ['GPU-a63c6362-dcfd-4a26-c44b-014b40127c89',
         'GPU-b08489af-0a8a-eb48-55ba-8346733af417',
         'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6',
         'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
ORDER = ['TextVQA', 'MME', 'ScienceQA-IMG', 'MMBench', 'POPE']
CONDITIONS = {'A': ['Author-Native'], 'B': ['HF-Native', 'Physical-C3', 'Physical-L11-L24']}
SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")
MC = "Answer with the option's letter from the given choices directly."


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def prompt(text):
    return SYSTEM + ' USER: <image>\n' + text + ' ASSISTANT:'


def manifest():
    with gzip.open(OUT / 'sample_manifest.jsonl.gz', 'rt') as stream:
        return [json.loads(line) for line in stream]


def source_hashes():
    paths = list((ROOT / 'vlm_research/information_horizon_alignment').glob('*.py'))
    paths += [ROOT / 'vlm_research' / name for name in (
        'mme_seed_quality/engine.py', 'c3v2_middle_reaccess/engine.py',
        'ate_fulljoint_v2/engine.py', 'accuracy_autoresearch/engine.py',
        'physical_handoff_fastpath/engine.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths
            if p.name not in ('analyze.py', 'report.py', 'supervise.py', 'resources.py')}


def verify(stage):
    p = read(OUT / ('protocol_paper_native.json' if stage == 'A' else 'protocol_ate_paper.json'))
    assert p['sources'] == source_hashes(), 'Executing source changed after freeze'
    for name, expected in p['artifacts'].items():
        assert sha(OUT / name) == expected, name
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=AUTHOR, text=True).strip() == AUTHOR_SHA
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=AUTHOR)
    return p


def database(stage, rank, protocol):
    path = OUT / f'workers/{stage}/rank{rank}/results.sqlite'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS results(key TEXT,condition TEXT,benchmark TEXT,payload TEXT,sha256 TEXT,PRIMARY KEY(key,condition))')
    binding = digest(protocol)
    old = db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
    if old:
        assert old[0] == binding, 'Wrong checkpoint protocol'
    else:
        with db:
            db.execute('INSERT INTO metadata VALUES (?,?)', ('protocol', binding))
    return db


def prior(db, sample, condition):
    row = db.execute('SELECT payload,sha256 FROM results WHERE key=? AND condition=?',
                     (sample['key'], condition)).fetchone()
    if row:
        value = json.loads(row[0])
        assert digest(value) == row[1]
        assert value['sample_sha256'] == digest(sample)
        return value


def put(db, sample, condition, value):
    with db:
        db.execute('INSERT INTO results VALUES (?,?,?,?,?)',
                   (sample['key'], condition, sample['benchmark'],
                    json.dumps(value, ensure_ascii=False, separators=(',', ':')), digest(value)))


def hook_counts(model):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]

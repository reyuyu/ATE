import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import ast
import base64
import copy
import hashlib
import logging
import os
import re
import string
from functools import lru_cache

from ..mme_seed_quality.common import atomic, digest, sha, now, read, MODELS
from ..benchmarks.scoring import pope_answer, scienceqa_index, textvqa_score
from ..benchmarks.dataset_registry import official_registry

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/seven_benchmark_native_physical_quality'
DATA = Path('/data/lk/ATE-benchmarks-20261105/datasets')
AUDIT = Path('/data/lk/ATE-benchmark-audit-20261105-2230')
REGISTRY = official_registry()
ORDER = tuple(REGISTRY['paper_queue'])
COUNTS = {name: REGISTRY['benchmarks'][name]['expected_questions'] for name in ORDER}
CONDITIONS = ('Native', 'Physical-C3')
DEVICES = ('GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6',
           'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185')
SHORT = 'Answer the question using a single word or phrase.'
MC = "Answer with the option's letter from the given choices directly."


@lru_cache(maxsize=1)
def official_mmbench_matcher():
    path = OUT / 'official/matching_util_api.json'
    entry = read(path)
    content = base64.b64decode(entry['content'])
    blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
    assert blob == entry['sha'] == '84ee053d7d267ffe7ad397d3a99c2f5e80716c56'
    module = ast.parse(content)
    names = {'can_infer_option', 'can_infer_text', 'can_infer'}
    selected = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names
                or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id == '_VERBOSE_ANSWER_RE' for t in n.targets)]
    assert {n.name for n in selected if isinstance(n, ast.FunctionDef)} == names
    scope = dict(cp=copy, os=os, re=re, string=string, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope['can_infer']


def sources():
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in sorted(
        (ROOT / 'vlm_research').rglob('*.py')) if 'results' not in p.parts}
    registry_path = ROOT / 'vlm_research/benchmarks/official_registry.json'
    bindings[str(registry_path.relative_to(ROOT))] = sha(registry_path)
    return bindings


def prompt(row, benchmark):
    if benchmark in ('MMBench', 'ScienceQA-IMG'):
        choices = row['choices']
        if isinstance(choices, list):
            choices = dict(zip('ABCDE', choices))
        context = ('Context: ' + row['hint'] + '\n') if row.get('hint') else ''
        return context + row['question'] + '\n' + '\n'.join(
            f'{k}. {v}' for k, v in choices.items()) + '\n' + MC
    if benchmark == 'MM-Vet':
        return row['question']
    return row['question'] + '\n' + SHORT


def score(sample, text):
    if sample.get('gold_available') is False or sample['benchmark'] == 'vqav2_testdev':
        if 'answer' in sample:
            raise ValueError('Gold must not be attached to VQAv2 test-dev')
        return dict(score=None, status='GENERATED_AWAITING_OFFICIAL_SCORING')
    bench, gold = sample['benchmark'], sample['answer']
    if bench == 'POPE':
        parsed = pope_answer(text)
        return dict(score=float(parsed == gold), parsed=parsed)
    if bench == 'ScienceQA-IMG':
        parsed = scienceqa_index(text, len(sample['raw']['choices']))
        return dict(score=float(parsed == gold), parsed=parsed, parse_failed=parsed < 0)
    if bench == 'TextVQA':
        return dict(score=float(textvqa_score(text, gold)))
    if bench == 'VQAv2':
        from ..ate_fulljoint_vqav2.common import score as vqa_score
        return dict(score=float(vqa_score(text, [a['answer'] for a in gold['answers']])))
    if bench == 'GQA':
        # Same prediction conversion as the completed full-val historical run.
        parsed = text.rstrip('.').lower()
        return dict(score=float(parsed == gold), parsed=parsed)
    if bench == 'MMBench':
        parsed = official_mmbench_matcher()(text, dict(sample['raw']['choices'])) or 'Z'
        return dict(score=float(parsed == gold), parsed=parsed, parse_failed=parsed == 'Z',
                    policy='Official VLMEvalKit exact_matching; main summary is circular all-rolls hit')
    return dict(score=None, status='GENERATED_AWAITING_OFFICIAL_SCORING')


def safety():
    free = shutil.disk_usage('/data').free
    if free < 20 * 1024**3:
        raise RuntimeError(f'/data below mandatory 20 GiB free: {free}')
    if (OUT / 'STOP').exists():
        raise RuntimeError('Explicit experiment STOP file')


def samples(benchmark):
    with gzip.open(OUT / f'manifests/{benchmark}.jsonl.gz', 'rt') as handle:
        return [json.loads(line) for line in handle]


def verify():
    protocol = read(OUT / 'protocol.json')
    assert tuple(protocol['order']) == ORDER and protocol['counts'] == COUNTS, 'Old validation queue is not the official test-dev protocol'
    assert protocol['sources'] == sources(), 'Frozen source changed'
    for name, value in protocol['artifacts'].items():
        assert sha(OUT / name) == value, name
    return protocol


def database(rank, protocol):
    path = OUT / f'worker{rank}/results.sqlite'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS results '
               '(key TEXT, condition TEXT, benchmark TEXT, payload TEXT, sha256 TEXT, '
               'PRIMARY KEY(key, condition))')
    binding = digest(protocol)
    previous = db.execute("SELECT value FROM metadata WHERE key='protocol'").fetchone()
    if previous:
        assert previous[0] == binding, 'Checkpoint protocol mismatch'
    else:
        with db:
            db.execute('INSERT INTO metadata VALUES (?,?)', ('protocol', binding))
    return db


def put(db, sample, condition, record):
    with db:
        db.execute('INSERT INTO results VALUES (?,?,?,?,?)',
                   (sample['key'], condition, sample['benchmark'],
                    json.dumps(record, ensure_ascii=False, separators=(',', ':')), digest(record)))

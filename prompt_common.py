"""Independent prompt experiment; frozen model/algorithm/scorers are read-only."""
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import zlib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path('/data/lk/ATE-worktrees/qwen3vl-svh-quality-then-svhr-search-2x3090')
FIRST = BASE / 'vlm_research/results/qwen3vl_ate_hf_native_svh_sixbench'
OUT = ROOT / 'vlm_research/results/qwen3vl_short_answer_prompt_ablation'
PARSER = Path('/data/lk/ATE-worktrees/qwen3vl-sqa-textvqa-loss-audit/vlm_research/qwen3vl_sqa_textvqa_loss_audit/parsers.py')
UUIDS = ['GPU-b08489af-0a8a-eb48-55ba-8346733af417', 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
TASKS = ['GQA', 'ScienceQA-IMG', 'MMBench']
SEED = 20261111
P1_GQA = 'Answer using only a single word or short phrase. No explanation.'
P1_CHOICE = 'Answer with exactly one option letter from the listed choices. No explanation. Do not repeat the option text.'
P2_CHOICE = 'Your entire response must be exactly one uppercase option letter from the listed choices, with no other characters.'
sys.path.insert(0, str(BASE))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


@lru_cache(None)
def load_parser():
    spec = importlib.util.spec_from_file_location('frozen_diagnostic_parser', PARSER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prompt_body(row, condition):
    if condition == 'P0':
        return row['body']
    if condition == 'P1':
        return row['body'] + '\n' + (P1_GQA if row['benchmark'] == 'GQA' else P1_CHOICE)
    assert condition == 'P2' and row['benchmark'] != 'GQA'
    return row['body'] + '\n' + P2_CHOICE


def conditions(task):
    return ['P0', 'P1'] if task == 'GQA' else ['P0', 'P1', 'P2']


def render(processor, row, prompt):
    new = dict(row, body=prompt_body(row, prompt))
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': row['image_path']}, {'type': 'text', 'text': new['body']}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    visual = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ids = processor.tokenizer(text, add_special_tokens=False)['input_ids']
    assert ids.count(visual) == 1
    pos = ids.index(visual)
    ids = ids[:pos] + [visual] * row['image_tokens'] + ids[pos+1:]
    new.update(input_length=len(ids), prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
    return new, text, ids


def readonly(path):
    conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    conn.execute('PRAGMA query_only=ON')
    return conn


def historical(keys):
    found = {}
    for shard in [0, 1]:
        with readonly(FIRST / f'shard{shard}.sqlite') as conn:
            for key in sorted(keys):
                for condition, state, blob, expected in conn.execute(
                        'SELECT condition,state,payload,payload_sha256 FROM predictions WHERE key=?', (key,)):
                    assert state == 'COMPLETE', (key, condition, state)
                    raw = zlib.decompress(blob)
                    assert hashlib.sha256(raw).hexdigest() == expected
                    rec = json.loads(raw)
                    assert digest(rec['input_binding']) == rec['input_binding_sha256']
                    found[key, condition] = rec
    assert len(found) == len(keys) * 2
    return found


class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, payload BLOB, sha256 TEXT)')
        self.conn.commit()

    def get(self, key):
        row = self.conn.execute('SELECT payload,sha256 FROM records WHERE id=?', (key,)).fetchone()
        if row is None:
            return None
        raw = zlib.decompress(row[0])
        assert hashlib.sha256(raw).hexdigest() == row[1]
        return json.loads(raw)

    def put(self, key, value):
        raw = json.dumps(value, ensure_ascii=True, sort_keys=True).encode()
        if self.get(key) is not None:
            assert self.get(key) == value, ('Refuse overwrite', key)
            return
        with self.conn:
            self.conn.execute('INSERT INTO records VALUES (?,?,?)', (key, zlib.compress(raw), hashlib.sha256(raw).hexdigest()))


def disk():
    free = shutil.disk_usage('/data').free
    # Independent bounded text-only experiment. The original 2-GiB policy is untouched.
    assert free > 512 * 2**20, ('OUTPUT_SPACE_BLOCKED', free)
    total = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert total < 180 * 2**20, ('OUTPUT_BUDGET_REACHED', total)
    return free


def check_sources():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, expected in protocol['frozen_source_sha256'].items():
        assert sha(path) == expected, ('Frozen source changed', path)
    assert sha(OUT / 'sample_manifest.json') == protocol['manifest_sha256']
    return protocol


def scored(row, text, truncated=False):
    from vlm_research.qwen3vl_overnight.common import scorers
    score = scorers().score(row, text)
    if row['benchmark'] == 'ScienceQA-IMG':
        p = load_parser().option_choice(text, row['options'], truncated=truncated)
        gold = chr(65 + row['gold']) if isinstance(row['gold'], int) else row['gold']
        score.update(diagnostic_score=float(p.letter == gold), diagnostic_choice=p.letter, diagnostic_status=p.status,
                     format_only_failure=bool(p.letter == gold and not score['score']))
    return score

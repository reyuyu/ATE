"""Local, read-only reuse of the frozen Qwen runtime and scorers."""
import hashlib
import json
import os
import sqlite3
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path('/data/lk/ATE-worktrees/qwen3vl-svh-quality-then-svhr-search-2x3090')
P2ROOT = Path('/data/lk/ATE-worktrees/qwen3vl-p2-full-gqa-type-audit')
PROMPT = Path('/data/lk/ATE-worktrees/qwen3vl-short-answer-prompt-ablation')
sys.path.insert(0, str(PROMPT))
from prompt_common import Store, digest, sha, write, table, readonly, scored, UUIDS
from vlm_research.qwen3vl_overnight.common import MODEL, inventory, prepare

OUT = ROOT / 'vlm_research/results/qwen3vl_svhr_l11_l24_budget_search'
FIRST = BASE / 'vlm_research/results/qwen3vl_ate_hf_native_svh_sixbench'
P2 = P2ROOT / 'vlm_research/results/qwen3vl_p2_full_and_gqa_type_audit'
SEED = 20261111
TASKS = ['TextVQA', 'GQA-Query', 'GQA-NonQuery', 'ScienceQA-IMG', 'MMBench']
PROTECTION = ['POPE','MME']


def read_store(path):
    with readonly(path) as db:
        for key, blob, expected in db.execute('SELECT id,payload,sha256 FROM records'):
            raw = zlib.decompress(blob)
            assert hashlib.sha256(raw).hexdigest() == expected
            yield key, json.loads(raw)


def historical(keys):
    found = {}
    for shard in [0, 1]:
        with readonly(FIRST / f'shard{shard}.sqlite') as db:
            for key, cond, state, blob, expected in db.execute('SELECT key,condition,state,payload,payload_sha256 FROM predictions'):
                if key not in keys or key.startswith(('ScienceQA-IMG:', 'MMBench:')):
                    continue
                raw = zlib.decompress(blob)
                assert hashlib.sha256(raw).hexdigest() == expected
                r = json.loads(raw)
                found[key, cond] = r
    for name in ['worker0', 'worker1', 'reused_validation', 'aligned0', 'aligned1']:
        for _, r in read_store(P2 / (name + '.sqlite')):
            if r['key'] in keys and r.get('state', 'COMPLETE') == 'COMPLETE':
                found[r['key'], r['condition']] = r
    return found


def disk():
    import shutil
    free = shutil.disk_usage('/data').free
    assert free >= 2 * 2**30, ('STOP_DISK_BELOW_2GIB', free)
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert used < 1500 * 2**20, ('STOP_OUTPUT_BUDGET_1500MIB', used)
    return free


def check():
    p = json.loads((OUT / 'protocol.json').read_text())
    for path, expected in p['sources'].items():
        assert sha(path) == expected, ('FROZEN_SOURCE_CHANGED', path)
    assert sha(OUT / 'sample_manifest.json') == p['manifest_sha256']
    return p


def records():
    out = {}
    for shard in [0, 1]:
        path = OUT / f'worker{shard}.sqlite'
        if path.exists():
            for key, row in read_store(path):
                assert key not in out
                out[key] = row
    return out

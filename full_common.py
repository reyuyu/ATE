"""Independent full P2 evaluation; historical code and results are read-only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = Path('/data/lk/ATE-worktrees/qwen3vl-short-answer-prompt-ablation')
sys.path.insert(0, str(PRIOR))
from prompt_common import (BASE, FIRST, PARSER, UUIDS, digest, sha, write, table,
                           readonly, Store, scored, load_parser)
import hashlib
import json
import os
import shutil
import zlib

OLD = PRIOR / 'vlm_research/results/qwen3vl_short_answer_prompt_ablation'
OUT = ROOT / 'vlm_research/results/qwen3vl_p2_full_and_gqa_type_audit'
TASKS = ['ScienceQA-IMG', 'MMBench']
SEED = 20261111
GQA_PATH = Path('/data/lk/ATE-resources/data/gqa-testdev-balanced-official/testdev_balanced_questions.json')
SQA_PATH = Path('/data/lk/ATE-resources/benchmarks/phase4_0/sources/scienceqa/data/scienceqa/problems.json')
MMB_PATH = Path('/data/lk/ATE-benchmarks-20261105/datasets/MMBench/mmbench_dev_20230712.tsv')


def read_store(path):
    with readonly(path) as db:
        for key, blob, expected in db.execute('SELECT id,payload,sha256 FROM records'):
            raw = zlib.decompress(blob)
            assert hashlib.sha256(raw).hexdigest() == expected, (path, key)
            yield key, json.loads(raw)


def original_records(task=None):
    for shard in [0, 1]:
        with readonly(FIRST / f'shard{shard}.sqlite') as db:
            for key, cond, state, blob, expected in db.execute('SELECT key,condition,state,payload,payload_sha256 FROM predictions'):
                raw = zlib.decompress(blob)
                assert hashlib.sha256(raw).hexdigest() == expected
                r = json.loads(raw)
                if task is not None and r['benchmark'] not in task:
                    continue
                assert r['key'] == key and r['condition'] == cond
                yield state, r


def render(processor, row, suffix):
    new = dict(row, body=row['body'] + '\n' + suffix)
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': row['image_path']}, {'type': 'text', 'text': new['body']}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = processor.tokenizer(text, add_special_tokens=False)['input_ids']
    visual = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    assert ids.count(visual) == 1
    pos = ids.index(visual)
    ids = ids[:pos] + [visual] * row['image_tokens'] + ids[pos+1:]
    new.update(input_length=len(ids), prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
    return new, text, ids


def disk():
    free = shutil.disk_usage('/data').free
    assert free > 512 * 2**20, ('STOP_DISK', free)
    size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert size < 300 * 2**20, ('STOP_OUTPUT_BUDGET', size)
    return free


def check():
    p = json.loads((OUT/'p2_full_protocol.json').read_text())
    for path, expected in p['sources'].items():
        assert sha(path) == expected, ('SOURCE_CHANGED', path)
    assert sha(OUT/'sample_manifest.json') == p['manifest_sha256']
    for path, state in p['model_files'].items():
        stat = Path(path).stat()
        assert [stat.st_size, stat.st_mtime_ns] == state['stat']
    return p


def body(question, options, hint):
    return ((f'Hint: {hint}\n' if hint else '') + f'Question: {question}\nOptions:\n'
            + '\n'.join(f'{k}. {v}' for k, v in options.items())
            + '\nPlease select the correct answer from the options above. \n')

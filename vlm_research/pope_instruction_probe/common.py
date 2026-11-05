import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from ..physical_visual_handoff.common import atomic, digest, read, sha, write_csv
from ..reverse_writer.common import CFG, writer_set
from ..benchmarks.protocols import MODEL_PATH

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/pope3_instruction_attention_addback'
HISTORY = Path('/data/lk/ATE-worktrees/c3v2-reverse-writer-pruning/vlm_research/results/c3v2_reverse_writer_pruning')
RESOURCES = Path('/data/lk/ATE-resources/experiments/pope3_instruction_attention_addback')
KEYS = ('pope_random__1351', 'pope_random__2459', 'pope_random__301')
TARGETS = {'1351': 'truck', '2459': 'boat', '301': 'vase'}
# Chosen from official boxes and original-image inspection, before any new GPU outputs.
TARGET_ANNOTATIONS = {'1351': [1799923], '2459': [1803349], '301': [1154230, 1670746]}
GPU = dict(index=0, uuid='GPU-a63c6362-dcfd-4a26-c44b-014b40127c89')
SOURCE_COMMIT = 'e936ff3dcf3ae02e4395dd1f2e9dd05cd8da25cb'
FORMS = (' Yes', ' No', ' yes', ' no', 'Yes', 'No', 'yes', 'no')


def conditions(n):
    items = [('E1', []), ('C3', list(range(n)))]
    for size in (1, 2, 4):
        items += [(f'add{size}_{i:02d}', list(range(i, i+size))) for i in range(n-size+1)]
    return items


def writers(mapping, restored):
    inst = mapping['parts']['Instruction']
    assert len(restored) == len(set(restored)) and all(0 <= i < len(inst) for i in restored)
    return sorted(set(writer_set(mapping, 'E1')) | {inst[i] for i in restored})


def sources():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT/'vlm_research'/p for p in ('accuracy_autoresearch/engine.py',
        'ate_fulljoint_v2/engine.py', 'physical_handoff_fastpath/engine.py',
        'reverse_writer/common.py', 'reverse_writer/engine.py', 'p1_layerwise_accuracy/engine.py',
        'roi_sink_common.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in files if p.name != 'analyze.py'}


def edge_mask():
    y, x = np.indices((24, 24))
    return ((x < 2) | (x >= 22) | (y < 2) | (y >= 22)).reshape(-1)


def candidate_ids(tokenizer, mapping, form):
    text = mapping['rendered_prompt'].replace('<image>', '<image>' * 576)
    ids = tokenizer(text + form, add_special_tokens=True)['input_ids']
    prefix = mapping['prompt_ids']
    assert ids[:len(prefix)] == prefix, 'Answer boundary changes frozen Prompt tokens'
    result = ids[len(prefix):]
    assert result and 1 not in result and 2 not in result, 'Extra BOS/EOS in scored answer'
    return result


def checkpoint(qid, condition):
    return OUT / 'checkpoints' / qid / (condition + '.json')


def save_record(path, payload):
    assert not path.exists(), f'Refuse overwrite: {path}'
    atomic(path, dict(payload=payload, sha256=digest(payload)))


def load_record(path):
    b = read(path)
    assert b['sha256'] == digest(b['payload'])
    return b['payload']

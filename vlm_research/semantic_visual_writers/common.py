"""Frozen inputs, audited historical All-Writer, and immutable experiment records."""
import subprocess
from pathlib import Path

import numpy as np
import torch

from query_role_attention.common import ROOT, MODEL, REVISION, TASKS, read, write, sha256, digest, csv_write
from visual_write_window.common import sources as historical_sources, hardware
from .groups import RANDOM_SEEDS

BASE = '6b9467d6fb4e0f8535040809bee60397ca798ea8'
OUT = ROOT/'results/semantic_visual_writers_240'
HISTORY = ROOT/'results/visual_write_window_240'
FORMAL = ROOT/'results/query_role_attention_boundary_240_formal'
SWAP = ROOT/'results/rq2_visual_text_handoff_240/image_swap_manifest.json'
RAW = Path('/data/lk/ATE-resources/experiments/semantic_visual_writers_240')
CONDITIONS = ['All-Writer', 'No-Writer', 'Semantic-k', 'Question-Last-k', 'R-Last-k',
              'Scaffold-k'] + [f'Random-k-{seed}' for seed in RANDOM_SEEDS] + ['Drop-Semantic']
SWAP_CONDITIONS = ['All-Writer', 'No-Writer', 'Semantic-k', 'R-Last-k', f'Random-k-{RANDOM_SEEDS[0]}']
SEED = 20261103
SMOKE_IDS = ['00150746', '11697551', '11127723']
MANIFEST_SHA = '258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c'


def sources():
    return {**historical_sources(), **{str(p.relative_to(ROOT)): sha256(p)
            for p in sorted(Path(__file__).parent.glob('*.py'))}}


def key(s):
    return s['task']+'__'+s['question_id']


def first_logits(record):
    assert sha256(record['raw']['path']) == record['raw']['sha256']
    with np.load(record['raw']['path'], allow_pickle=False) as arrays:
        value = torch.from_numpy(arrays['first_logits'].copy())
    assert value.ndim == 1 and torch.isfinite(value).all()
    return value


def audit_all_writer():
    p = read(HISTORY/'protocol.json')
    assert read(HISTORY/'completion.json')['status'] == 'COMPLETE'
    assert p['sources'] == historical_sources()
    assert p['model_revision'] == REVISION and p['generation']['logits_to_keep'] == 1
    assert p['dtype'] == 'float16' and p['attention'] == 'eager' and p['transformers'] == '4.49.0'
    assert p['conditions']['All-Writer'] == list(range(32))
    assert read(HISTORY/'gpu_smoke_validation.json')['passed']
    assert read(HISTORY/'image_independence_validation.json')['passed']
    assert sha256(HISTORY/'sample_manifest.json') == MANIFEST_SHA == sha256(FORMAL/'sample_manifest.json')
    mappings = read(HISTORY/'input_mappings.json')
    samples = read(HISTORY/'sample_manifest.json')['samples']
    entries = []
    for s, mapping in zip(samples, mappings):
        path = HISTORY/'checkpoints'/key(s)/'All-Writer.json'
        r = read(path)
        g = r['generation']
        assert r['binding'] == digest(p) and r['condition'] == 'All-Writer'
        assert r['sample_sha256'] == digest(s) and r['input_hashes'] == mapping['input_hashes']
        assert g['query_layout'] == mapping['role_layout'] and r['result_sha256'] == digest(g)
        assert g['answer_image_blocked'] and g['writer_layers'] == list(range(32))
        assert len(g['source_events']) == 32*len(g['token_ids'])
        for e in g['source_events']:
            assert e['answer_image_masked'] and e['blocked_probability_max'] == 0
            assert not e['R_image_masked'] and e['non_target_mask_unchanged']
        v = first_logits(r)
        assert int(v.argmax()) == g['token_ids'][0] and digest(v.tolist()) == g['first_logits_sha256']
        assert sha256(s['image_path']) == s['image_sha256']
        entries.append(dict(key=key(s), checkpoint=str(path.relative_to(ROOT)), sha256=sha256(path),
            raw=r['raw'], correct=g['correct'], input_hashes=mapping['input_hashes']))
    assert len(entries) == 240 and sum(e['correct'] for e in entries) == 202
    return dict(passed=True, reused_main_generations=240, new_main_generations=2160,
        source_commit=BASE, source_protocol_sha256=sha256(HISTORY/'protocol.json'), entries=entries,
        reason='Same 240 images, prompt/pixel hashes, native FP16/eager, keep=1; all 32 layers block answer image keys and allow all R.')


def select_gpu():
    state = hardware()
    busy = {line.split(',')[0].strip() for line in state['compute_processes'].splitlines()}
    candidates = []
    for line in state['table'].splitlines():
        index, uuid, name, used, free, util = [v.strip() for v in line.split(',')]
        if uuid not in busy and int(used) < 200 and int(free) > 22000 and int(util) == 0:
            candidates.append(dict(index=int(index), uuid=uuid, name=name, logical='cuda:0'))
    state['selected'] = candidates[0] if candidates else None
    write(OUT/'gpu_selection.json', state)
    assert candidates, 'No idle GPU: stop without automatic waiting'
    return candidates[0]


def preflight(gpu):
    import os
    state = hardware()
    row = next(line for line in state['table'].splitlines() if line.split(',')[1].strip() == gpu['uuid'])
    fields = [s.strip() for s in row.split(',')]
    busy = [line for line in state['compute_processes'].splitlines() if line.split(',')[0].strip() == gpu['uuid']]
    state.update(selected=gpu, passed=not busy and int(fields[3]) < 200 and int(fields[4]) > 22000 and int(fields[5]) == 0)
    write(OUT/f'gpu_preflight_{os.getpid()}.json', state)
    assert state['passed'] and os.environ.get('CUDA_VISIBLE_DEVICES') == gpu['uuid'], 'GPU occupied; stop'


def verify():
    p = read(OUT/'protocol.json')
    assert p['sources'] == sources(), 'Frozen code changed'
    for name, expected in p['bindings'].items():
        assert sha256(OUT/name) == expected, name
    for path, expected in {**p['transformers_sources'], **p['scorer_sources']}.items():
        assert sha256(path) == expected, path
    for path, info in p['model_files'].items():
        st = Path(path).stat()
        assert st.st_size == info['bytes'] and st.st_mtime_ns == info['mtime_ns'], path
    for e in read(OUT/'all_writer_reuse_audit.json')['entries']:
        assert sha256(ROOT/e['checkpoint']) == e['sha256']
        assert sha256(e['raw']['path']) == e['raw']['sha256']
    return p

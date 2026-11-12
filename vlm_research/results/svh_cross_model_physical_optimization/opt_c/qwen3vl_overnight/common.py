import gzip
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from ..qwen3vl_svh_sqa7908_repair.run import BASE, MODEL, sources as inherited_sources
from ..qwen3vl_svh_fasttrack.freeze import GPUS, scorers
from ..qwen3vl_svh_preflight48.freeze import COUNTS, TASKS, digest, prepare, sha, table, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/qwen3vl_ate_hf_native_svh_sixbench'
OLD = BASE / 'vlm_research/results/qwen3vl_svh_fasttrack_qk'
REPAIR = Path('/data/lk/ATE-worktrees/qwen3vl-svh-sqa7908-projection-repair')
INVENTORY = BASE / 'vlm_research/results/qwen3vl_svh_expanded_preflight48/inventory.jsonl.gz'
MIN_DISK = 2 * 2**30
CORE = ['__init__.py', 'common.py', 'store.py', 'engine.py', 'run.py']


def sources():
    return {**inherited_sources(), **{str((Path(__file__).parent / n).relative_to(ROOT)): sha(Path(__file__).parent / n) for n in CORE},
        'vlm_research/qwen3vl_svh_preflight48/freeze.py': sha(ROOT / 'vlm_research/qwen3vl_svh_preflight48/freeze.py'),
        'vlm_research/qwen3vl_svh_preflight48/score.py': sha(ROOT / 'vlm_research/qwen3vl_svh_preflight48/score.py')}


def inventory():
    with gzip.open(INVENTORY, 'rt') as stream:
        return [json.loads(line) for line in stream]


def check():
    p = json.loads((OUT / 'protocol.json').read_text())
    assert p['mode'] == 'RELAXED_NUMERICAL_QUALITY_ONLY' and p['sources'] == sources()
    assert sha(INVENTORY) == p['inventory_sha256']
    assert sha(OUT / 'shards.json') == p['shards_sha256']
    for path, expected in p['scorer_sources'].items():
        assert sha(path) == expected
    for path, state in p['model_files'].items():
        stat = Path(path).stat()
        assert [stat.st_size, stat.st_mtime_ns] == state['stat'], 'Model/cache changed after freeze'
    return p


def disk():
    free = shutil.disk_usage('/data').free
    if free < MIN_DISK:
        raise RuntimeError(f'PAUSED_DISK: {free} free; minimum {MIN_DISK}')
    return free


def freeze():
    import torch
    assert not torch.cuda.is_initialized() and not (OUT / 'protocol.json').exists()
    disk()
    old = json.loads((OLD / 'protocol.json').read_text())
    assert sha(INVENTORY) == old['inventory_sha256']
    rows = inventory()
    assert len(rows) == len({r['key'] for r in rows}) == 35346
    assert Counter(r['benchmark'] for r in rows) == COUNTS
    mapping = json.loads((OLD / 'shards.json').read_text())
    assert sha(OLD / 'shards.json') == old['frozen_files']['shards.json']
    assert set(mapping) == {r['key'] for r in rows}
    clusters = {}
    images = {}
    for r in rows:
        tags = [('image', r['image_sha256'])]
        if r['benchmark'] == 'MME':
            tags.append(('mme', r['pair_id']))
        if r['benchmark'] == 'MMBench':
            tags.append(('mmbench', r['circular_group']))
        for tag in tags:
            assert clusters.setdefault(tag, mapping[r['key']]['shard']) == mapping[r['key']]['shard']
        path = r['image_path']
        if path not in images:
            assert sha(path) == r['image_sha256'], ('Image SHA mismatch', path)
            images[path] = r['image_sha256']
    model_files = {}
    for path in sorted(MODEL.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.safetensors', '.txt', '.model', '.jinja'):
            stat = path.stat()
            model_files[str(path)] = dict(sha256=sha(path), stat=[stat.st_size, stat.st_mtime_ns])
    scorer = scorers()
    scorer.adapters()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OLD / 'shards.json', OUT / 'shards.json')
    shutil.copyfile(OLD / 'numerical_cases.json', OUT / 'smoke_candidates.json')
    shutil.copyfile(OLD / 'input_bindings.json', OUT / 'known_input_bindings.json')
    p = dict(mode='RELAXED_NUMERICAL_QUALITY_ONLY', strict_gate=old['gate'],
        strict_status='FAIL_RETAINED', full_59_numerical_validated=False, fast_track_validated=False,
        plan='B: ATE unified HF, not author official reproduction', plan_A='BLOCKED',
        runtime='ORIGINAL_VISION_EAGER_TEXT_EAGER', low_memory_runtime_not_used=True,
        model=str(MODEL), revision=MODEL.name, model_files=model_files,
        inherited_algorithm_commit='fa3c9a7747', original_base_commit='d5ca88d7de604728eb1f0bcea3185f01d266e125',
        sources=sources(), inventory_path=str(INVENTORY), inventory_sha256=sha(INVENTORY),
        shards_sha256=sha(OUT / 'shards.json'), counts=COUNTS, questions=35346, generations=70692,
        generation=old['generation'], caps=old['caps'], visual_budget=[65536, 2097152],
        scorer_sources={path: sha(path) for path in scorer.scorer_sources().values()}, gpu_uuids=GPUS,
        semantics=dict(exit_before_layer=22, writer_layers=list(range(22)), answer_skip_layers=list(range(1, 7)),
            answer_image_access=[], persistent_original_visual_KV=0),
        oom_policy='SKIPPED_OOM; no shrinking, precision/backend change or selected answer retry',
        disk_min_free_bytes=MIN_DISK, canary_pairs=128, bootstrap_replicates=20000, bootstrap_seed=20261111,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(OUT / 'protocol.json', p)
    write(OUT / 'image_sha256.json', images)
    write(OUT / 'cpu_inventory.json', dict(status='PASS', rows=len(rows), images=len(images),
        tasks=COUNTS, grouping='Image SHA plus MME pair and MMBench Circular union', cuda_initialized=False))
    print('FROZEN quality-only', len(rows), 'images', len(images), flush=True)

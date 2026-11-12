import hashlib
from pathlib import Path
import subprocess

from ..qwen3vl_visual_writing_necessity.common import (
    read, seal, atomic, key, digest, sha256, MODEL, REVISION, hardware, assert_idle, command)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/qwen3vl_keep_l0_21_gqa3000'
HISTORY = Path('/data/lk/ATE-worktrees/qwen3vl-gqa3000-visual-writing-necessity/vlm_research/results/qwen3vl_visual_writing_necessity_3000')
BASE = '96a66a30bc81db9ffe5c3ed77716673a81aaaa01'
REL = 'vlm_research/results/qwen3vl_visual_writing_necessity_3000'
CONDITION = 'Keep-L0-L21'
OPEN_LAYERS = list(range(22))
SEED = 20261104


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    inherited = read(HISTORY / 'protocol.json')['sources']
    for name, expected in inherited.items():
        assert sha256(ROOT / name) == expected, 'Historical implementation changed: '+name
        paths.append(ROOT / name)
    paths += [ROOT / 'vlm_research/qwen3vl_visual_writing_necessity/analyze.py']
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


def committed(name):
    blob = subprocess.check_output(['git', 'show', f'{BASE}:{REL}/{name}'], cwd=ROOT)
    expected = hashlib.sha256(blob).hexdigest()
    assert sha256(HISTORY / name) == expected, 'Historical artifact differs from source Commit: '+name
    return expected


def verify():
    p = read(OUT / 'protocol.json')
    assert sources() == p['sources'], 'Frozen source changed'
    for name, expected in p['artifacts'].items():
        assert sha256(OUT / name) == expected, name
    for name, expected in p['official_sources'].items():
        assert sha256(name) == expected, name
    return p


def checkpoint(shard, identifier):
    return OUT / f'shard{shard}/checkpoints' / (identifier+'.json')

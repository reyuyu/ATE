import os
from pathlib import Path

from ..textvqa_deep_add_drop.common import (
    PREVIOUS, BASELINES, SEED, historical, historical_path, previous,
    atomic, read, load, save, digest, sha, score, write_csv, now, TOLERANCES,
    sources as inherited_sources, stable,
)

ROOT = Path(__file__).resolve().parents[2]
FIRST_LAYER = int(os.environ.get('ATE_TEXTVQA_DECODE_FIRST_LAYER', '11'))
assert FIRST_LAYER in (10, 11, 12), 'Only the user-requested fixed regions are supported'
OUT = ROOT / f'vlm_research/results/textvqa_decode_l{FIRST_LAYER}_l24_500'
SUBSET = Path('/data/lk/ATE-worktrees/textvqa-decode-deep-add-drop-200/vlm_research/results/textvqa_decode_deep_add_drop_200')
CONDITION = f'Decode-L{FIRST_LAYER}-L24'
KEEP = tuple(range(FIRST_LAYER, 25))


def sources():
    files = inherited_sources()
    files.update({str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')
                  if p.name != 'analyze.py'})
    return files


def checkpoint(key):
    return OUT / 'checkpoints' / (key+'.json')

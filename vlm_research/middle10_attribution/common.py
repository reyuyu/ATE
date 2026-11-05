from pathlib import Path

from ..middle10_recovery.common import (
    HISTORY, EARLY7, now, write_csv, load, save, score, atomic, digest, read, sha,
    TOLERANCES, sources as inherited_sources,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/middle10_remaining_quality_attribution'
PREVIOUS = Path('/data/lk/ATE-worktrees/c3v2-middle10-quality-recovery/vlm_research/results/c3v2_middle10_quality_recovery')
BASE_COMMIT = '1828dc28dcb2923b72741a5fac9e5cb4c7baf6ba'
LABELS = ('A', 'B', 'C', 'D', 'E')
DECODE_IMAGE = tuple(range(11, 21))
BASELINES = ('Native', 'C3+Middle10', 'E3')
SEED = 20261106
THRESHOLD = 47.68


def policy(label):
    if label not in LABELS:
        raise ValueError(label)
    return dict(
        receiver_image_layers=tuple(range(32 if label in ('B', 'D', 'E') else 21)),
        p1_image_layers=tuple(range(32)) if label in ('C', 'D', 'E') else tuple(sorted(EARLY7+DECODE_IMAGE)),
        prefill_p1_attention_skip=(2, 3, 4, 5) if label in ('A', 'B') else (),
        decode_attention_skip=() if label == 'E' else (2, 3, 4, 5),
        decode_image_layers=DECODE_IMAGE,
        physical_hidden_exit=False,
    )


def sources():
    files = inherited_sources()
    files.update({str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')
                  if p.name != 'analyze.py'})
    return files


def checkpoint(key, label):
    return OUT / 'checkpoints' / key / (label+'.json')


def historical(key, label):
    root = PREVIOUS if label == 'E3' else HISTORY
    return load(root / 'checkpoints' / key / (label+'.json'))

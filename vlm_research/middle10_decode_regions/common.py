from pathlib import Path

from ..middle10_attribution.common import (
    HISTORY, now, write_csv, load, save, score, atomic, digest, read, sha,
    TOLERANCES, sources as inherited_sources,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/middle10_decode_early_late_recovery'
PREVIOUS = Path('/data/lk/ATE-worktrees/middle10-remaining-quality-attribution/vlm_research/results/middle10_remaining_quality_attribution')
BASE_COMMIT = '0a27cef983b5ae1f20930f2660afd59c11bbac7e'
BASELINES = ('Native', 'E')
LABELS = ('Early', 'Late')
KEEP = {'E': tuple(range(11, 21)), 'Early': tuple(range(21)),
        'Late': tuple(range(11, 32))}
SEED = 20261106
THRESHOLD = 47.68


def sources():
    files = inherited_sources()
    files.update({str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')
                  if p.name != 'analyze.py'})
    return files


def checkpoint(key, label):
    return OUT / 'checkpoints' / key / (label+'.json')


def historical_path(key, label):
    assert label in BASELINES
    return (HISTORY if label == 'Native' else PREVIOUS) / 'checkpoints' / key / (label+'.json')


def historical(key, label):
    return load(historical_path(key, label))

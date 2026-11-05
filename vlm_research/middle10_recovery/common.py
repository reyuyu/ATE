from pathlib import Path

from ..c3v2_middle_reaccess.common import (
    KEEP as OLD_KEEP, CUTOFF as OLD_CUTOFF, LABELS as BASELINES,
    now, write_csv, load, save, score,
)
from ..physical_visual_handoff.common import atomic, digest, read, sha, TOLERANCES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/c3v2_middle10_quality_recovery'
HISTORY = Path('/data/lk/ATE-worktrees/c3v2-middle-visual-reaccess/vlm_research/results/c3v2_middle_visual_reaccess')
BASE_COMMIT = 'a3727ba0808b183bc8bede4396a86e82855c7d5a'
NEW = ('E1', 'E2', 'E3', 'E4', 'E5')
LABELS = BASELINES + NEW
EARLY7 = (0, 1, 6, 7, 8, 9, 10)
KEEP = {**OLD_KEEP, **{label: tuple(range(11, 21)) for label in NEW[:-1]},
        'E5': tuple(range(11, 20))}
CUTOFF = {**OLD_CUTOFF, **{label: 21 for label in NEW}}
PREFILL_KEEP = {label: tuple(sorted(set(KEEP[label]) | (
    set(EARLY7) if label in ('E2', 'E3', 'E4') else set()))) for label in LABELS}
WRITER_LAST = {label: 31 if label == 'Native' else
               20 if label in ('E1', 'E3', 'E4') else 19 for label in LABELS}
SEED = 20261105
COMPARISONS = (
    ('E1', 'C3+Middle10'), ('E2', 'C3+Middle10'), ('E3', 'C3+Middle10'),
    ('E3', 'E1'), ('E3', 'E2'), ('E4', 'E3'),
    ('E5', 'C3+Middle9'), ('C3+Middle10', 'E5'),
)


def sources():
    from ..c3v2_middle_reaccess.common import sources as inherited
    files = inherited()
    files.update({str(p.relative_to(ROOT)): sha(p) for p in
                  (ROOT / 'vlm_research/c3v2_middle_reaccess').glob('*.py')
                  if p.name != 'analyze.py'})
    files.update({str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')
                  if p.name not in ('analyze.py',)})
    return files


def quality_path(key, label):
    return OUT / 'checkpoints' / key / (label + '.json')


def performance_path(key, mode, repeat, label):
    return OUT / 'performance' / key / mode / str(repeat) / (label + '.json')

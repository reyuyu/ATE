import hashlib
from pathlib import Path

from ..middle10_decode_regions import common as previous
from ..middle10_decode_regions.common import (
    now, write_csv, load, save, score, atomic, digest, read, sha, TOLERANCES,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/textvqa_decode_deep_add_drop_200'
PREVIOUS = Path('/data/lk/ATE-worktrees/middle10-decode-early-late-recovery/vlm_research/results/middle10_decode_early_late_recovery')
BASE_COMMIT = '5627bc822a7827ae11edfaec84c31f6b73398e25'
SEED = 20261106
LAYERS = tuple(range(21, 32))
BASELINES = ('Native', 'E', 'Late')
LABELS = tuple(f'{mode}-L{i}' for mode in ('Add', 'Drop') for i in LAYERS)
KEEP = {f'Add-L{i}': tuple(range(11, 21))+(i,) for i in LAYERS}
KEEP.update({f'Drop-L{i}': tuple(j for j in range(11, 32) if j != i) for i in LAYERS})


def stable(value):
    return hashlib.sha256(f'{SEED}|{value}'.encode('ascii')).hexdigest()


def select_samples(samples, target=200):
    groups = {}
    for key, sample in samples.items():
        groups.setdefault(sample['image_sha256'], []).append(key)
    ordered = sorted(groups, key=stable)
    sizes = [0]
    for image in ordered:
        sizes.append(sizes[-1]+len(groups[image]))
    end = min(range(1, len(sizes)), key=lambda i: (abs(sizes[i]-target), i))
    selected = set(ordered[:end])
    return {k: samples[k] for k in sorted(samples) if samples[k]['image_sha256'] in selected}, ordered, end


def sources():
    files = previous.sources()
    files.update({str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')
                  if p.name != 'analyze.py'})
    return files


def historical_path(key, label):
    assert label in BASELINES
    return PREVIOUS / 'checkpoints' / key / 'Late.json' if label == 'Late' else previous.historical_path(key, label)


def historical(key, label):
    return load(historical_path(key, label))


def checkpoint(key, label):
    assert label in LABELS
    return OUT / 'checkpoints' / key / (label+'.json')

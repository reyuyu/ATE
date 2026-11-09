"""Frozen seven-benchmark Middle16 protocol and isolated output paths."""
from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/svh_r_middle16_sevenbench_3090'
AUDIT = Path('/data/lk/ATE-audits/svh-cross-model-readiness-20261110')
PAPER = Path('/data/lk/ATE-worktrees/cvpr2026-information-horizon-alignment/vlm_research/results/cvpr2026_information_horizon_alignment')
MODEL = Path('/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
ORIGINAL_TOKENIZER = Path('/data/lk/ATE-resources/information-horizon-alignment/runtime-staging/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234')
GQA_REF = '71a5762cd8ccb8cc99efe9b6b0267d8abef70c46'
PAPER_REF = '0414e553ba6fd1e21626ecf741489d225b44279c'
BASE_REF = 'e281754ef17b3accc09d6938b5b9f020048b2aec'
GQA_PROTOCOL_SHA = '1a7db2cacd807105fe6ccd1bf9c91652032050c242c2a49ddd5d7f490dcdeec4'
COUNTS = {'GQA': 12578, 'ScienceQA-IMG': 2017, 'TextVQA': 5000,
          'POPE': 8910, 'MME': 2374, 'MMBench': 4377, 'OCRBench-v1': 1000}
SYSTEM = ('A chat between a curious user and an artificial intelligence assistant. '
          "The assistant gives helpful, detailed, and polite answers to the user's questions.")


@dataclass(frozen=True)
class Algorithm:
    name: str
    reaccess: tuple
    exit_before: int
    writer_layers: tuple = tuple(range(20))
    answer_skip: tuple = (2, 3, 4, 5)

    def __post_init__(self):
        assert self.reaccess and tuple(sorted(set(self.reaccess))) == self.reaccess
        assert set(self.reaccess).isdisjoint(self.answer_skip)
        assert max(self.reaccess) < self.exit_before <= 32
        assert self.writer_layers == tuple(range(20))


MIDDLE16 = Algorithm('llava-SVH-R-Middle16', tuple(range(8, 24)), 24)
OLD_REACCESS = Algorithm('SVH-R-L11-L24', tuple(range(11, 25)), 25)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def sources():
    paths = list((ROOT / 'vlm_research/svh_middle16').glob('*.py'))
    paths += [ROOT / 'vlm_research' / p for p in (
        'c3v2_middle_reaccess/engine.py', 'c3v2_middle_reaccess/common.py',
        'physical_handoff_fastpath/engine.py', 'ate_fulljoint_v2/engine.py',
        'accuracy_autoresearch/engine.py', 'accuracy_autoresearch/common.py',
        'information_horizon_alignment/engine.py', 'information_horizon_alignment/score.py',
        'information_horizon_alignment/common.py', 'mme_seed_quality/engine.py',
        'mme_seed_quality/common.py', 'physical_visual_handoff/engine.py',
        'physical_visual_handoff/common.py', 'p1_layerwise_accuracy/engine.py',
        'p1_layerwise_accuracy/common.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def verify():
    p = read(OUT / 'protocol.json')
    assert p['sources'] == sources(), 'Frozen source changed; do not resume'
    assert p['manifest_sha256'] == sha(OUT / 'sample_manifest.jsonl.gz')
    assert p['partition_sha256'] == sha(OUT / 'partition.json')
    return p

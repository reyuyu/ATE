import csv
import hashlib
import json
from pathlib import Path

from vlm_research.internvl3_smoke.common import read, sha256, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'vlm_research/results/internvl3_layer_stage_pilot_120'
MODEL = Path('/data/lk/ATE-resources/models/internvl3-8b/853e3a797a661694b1b8ece0cb72dc2b23e3dac9')
REVISION = MODEL.name
RAW = Path('/data/lk/ATE-resources/experiments/internvl3_layer_stage_pilot_120')
SOURCE = Path('/data/lk/ATE-worktrees/single-layer-gqa6/vlm_research/results/paper_rq1_single_layer_gqa6/sample_manifest.json')
TASKS = ('ChooseAttr','ChooseCat','ChooseRel','CompareAttr','LogicalObj','QueryAttr')
SEED = 20261101
CONDITIONS = {'Full': [], 'W0': list(range(4)), 'W1': list(range(1,5)),
              'W2': list(range(2,6)), 'W3': list(range(3,7)),
              'WM': list(range(12,16)), 'WL': list(range(24,28))}
HISTORY = {
    'llava_mass': Path('/data/lk/ATE-worktrees/first-answer-attention-mass/vlm_research/results/first_answer_attention_mass/layer_summary.csv'),
    'llava_sensitivity': Path('/data/lk/ATE-worktrees/first-answer-attention-mass/vlm_research/results/first_answer_attention_mass/aligned_layer_sensitivity.csv'),
    'llava_tasks': SOURCE.parent/'first_token_layer_summary.csv',
    'llava_windows': Path('/data/lk/ATE-worktrees/four-layer-p1-decode-scan/vlm_research/results/four_layer_p1_decode_scan/accuracy_summary.csv'),
    'scorer': Path('/data/lk/ATE-worktrees/first-answer-early-attention/vlm_research/paper_rq1/scorers.py'),
    'gqa_evaluator': Path('/data/lk/ATE-resources/paper_rq1/annotations/gqa_eval.py'),
    'gqa_converter': Path('/data/lk/ATE-resources/benchmarks/phase4_0/sources/llava/scripts/convert_gqa_for_eval.py'),
    'llava_metric_definition': SOURCE.parents[2]/'paper_rq1_single_layer_gqa6/engine.py',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select(rows):
    chosen, used, qids = [], set(), set()
    key = lambda r: digest([SEED, r['task'], r['question_id'], r['image_sha256']])
    for _ in range(20):
        for task in TASKS:
            candidates = [r for r in rows if r['task'] == task and (task,r['question_id']) not in qids]
            unique = [r for r in candidates if r['image_sha256'] not in used]
            row = min(unique or candidates, key=key)
            chosen.append(row)
            qids.add((task,row['question_id']))
            used.add(row['image_sha256'])
    return chosen


def prediction(text):
    return text.rstrip('.').lower()


def csv_write(name, rows):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT/'vlm_research/internvl3_smoke').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def verify():
    protocol = read(OUT/'protocol.json')
    assert protocol['code_sha256'] == sources(), 'Frozen runtime source changed'
    assert sha256(OUT/'sample_manifest.json') == protocol['sample_manifest_sha256']
    for path, expected in protocol['history_sha256'].items():
        assert sha256(path) == expected
    return protocol

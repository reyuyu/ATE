import csv
import hashlib
import json
from pathlib import Path

from ..p1_layerwise_accuracy.common import MODELS, atomic, digest, key, read, sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/physical_visual_handoff_pilot'
HISTORY = ROOT / 'vlm_research/results/gqa3000_p1_layerwise_accuracy'
SEED = 20261104
BRANCH = 'codex/dual-gpu-physical-visual-handoff-pilot'
CONDITIONS = ['N0', 'D0', 'D1', 'P1', 'P2', 'D2', 'P3', 'D3', 'P4']
PAIRS = [('D1', 'P1'), ('D1', 'P2'), ('D2', 'P3'), ('D3', 'P4')]
# Frozen before any pilot GPU measurements. Shape-changing paths are not bit exact.
TOLERANCES = {
    'llava': dict(logit_atol=0.10, logit_rtol=0.003, hidden_relative_rmse=0.01, hidden_cosine=1e-4),
    'qwen': dict(logit_atol=0.50, logit_rtol=0.01, hidden_relative_rmse=0.04, hidden_cosine=0.002),
}


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def prepare():
    pool = read(HISTORY / 'llava/sample_manifest.json')['samples']
    used = set()
    selected = []
    for task in sorted({s['task'] for s in pool}):
        ranked = sorted((s for s in pool if s['task'] == task),
                        key=lambda s: digest([SEED, 'physical-handoff', task, s['question_id']]))
        chosen = []
        for s in ranked:
            if s['image_sha256'] not in used:
                used.add(s['image_sha256'])
                chosen.append(s)
            if len(chosen) == 40:
                break
        assert len(chosen) == 40
        selected += chosen
    assert len(selected) == len(used) == 240
    # Six independent smoke images, excluded from the 240-image quality cohort.
    remain = [s for s in pool if s['image_sha256'] not in used]
    smoke = [min(remain, key=lambda s: len(s['question'])), max(remain, key=lambda s: len(s['question']))]
    smoke += [next(s for s in remain if s['task'] == 'ChooseRel'),
              next(s for s in remain if s['task'] == 'CompareAttr'),
              next(s for s in remain if len(s['answer'].split()) > 1)]
    qbindings = read(HISTORY / 'qwen/input_bindings.json')
    smoke += [max(remain, key=lambda s: qbindings[key(s)]['layout']['image_tokens'])]
    smoke = list({key(s): s for s in smoke}.values())
    pilot60 = [key(s) for t in sorted({s['task'] for s in selected})
               for s in [v for v in selected if v['task'] == t][:10]]
    for kind in MODELS:
        binding = read(HISTORY / kind / 'input_bindings.json')
        for s in selected + smoke:
            assert sha(s['image_path']) == s['image_sha256']
        atomic(OUT / kind / 'sample_manifest.json', dict(seed=SEED, samples=selected, smoke=smoke, pilot60=pilot60,
               selection='Stable hash; unique image SHA across six tasks; no correctness filtering; smoke images disjoint'))
        atomic(OUT / kind / 'input_bindings.json', {key(s): binding[key(s)] for s in selected + smoke})
    atomic(OUT / 'numerical_acceptance.json', dict(tolerances=TOLERANCES,
           noop='torch.equal all effective returned logits, same LM-head shape',
           physical='Allclose plus retained-text hidden RMSE/cosine; no NaN/Inf; exact greedy sequence in equivalence gate',
           interpretation='Relative error is unstable near zero; report it, gate using combined absolute/relative allclose.'))

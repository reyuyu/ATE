import hashlib
import json
import math
from pathlib import Path
import random
import time

from ..visual_pathways.common import ROOT, TASKS, read, write, sha, digest, command

OUT = ROOT/'vlm_research/results/gqa_random_visual_budget_3000'
OTHER = Path('/data/lk/ATE-worktrees/qwen-internvl-sparse-visual-3000/vlm_research')
LLAVA = Path('/data/lk/ATE-worktrees/rq2-gqa-text-vs-sparse-3000/vlm_research')
MODELS = {'qwen': dict(gpu=0, layers=36, revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
          'internvl': dict(gpu=2, layers=28, revision='853e3a797a661694b1b8ece0cb72dc2b23e3dac9'),
          'llava': dict(gpu=3, layers=32, revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9')}
SEEDS = (20261102, 20261103, 20261104)
BUDGETS = (0., .2, .4, .6, .8, 1.)


def compact_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def conditions(model):
    n = MODELS[model]['layers']
    result = {'Only Text': dict(p=0., seed=None, keep=[], off=list(range(n))),
              'Full': dict(p=1., seed=None, keep=list(range(n)), off=[])}
    permutations = {}
    for seed in SEEDS:
        order = list(range(n)); random.Random(seed).shuffle(order)
        permutations[str(seed)] = order
        for p in BUDGETS[1:-1]:
            keep = order[:math.floor(p*n+.5)]
            result[f'p{p:.1f}_s{seed}'] = dict(p=p, seed=seed, keep=keep,
                off=sorted(set(range(n))-set(keep)))
    return result, permutations


def sources():
    paths = [Path(__file__).with_name(n) for n in ['__init__.py', 'common.py', 'engine.py', 'run.py', 'tests.py']]
    paths += [ROOT/'vlm_research/visual_pathways'/n for n in ['sparse_accuracy.py', 'engine.py', 'scan.py', 'common.py']]
    paths += list((ROOT/'vlm_research/internvl3_smoke').glob('*.py'))
    paths += [ROOT/'vlm_research/qwen3vl_keep_l26/run.py', ROOT/'vlm_research/qwen3vl_early_windows/experiment.py',
              ROOT/'vlm_research/vlm_common.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def freeze():
    from .history import bind, model_files
    assert not (OUT/'protocol.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    samples = read(LLAVA/'results/rq2_gqa_text_vs_sparse_3000/sample_manifest.json')['samples']
    assert len(samples) == 3000 and len({s['image_sha256'] for s in samples}) == 2409
    assert all(sum(s['task']==t for s in samples) == 500 for t in TASKS)
    for s in samples:
        assert sha(s['image_path']) == s['image_sha256']
    write(OUT/'sample_manifest.json', dict(samples=samples, count=3000, image_clusters=2409,
        original_commit='221cdde56dde53908aaa953fe26672fd33334f84', selection='Exact historical samples; no resampling'))
    c = {}; orders = {}
    for model in MODELS:
        c[model], orders[model] = conditions(model)
    write(OUT/'layer_sets.json', dict(rng='Python random.Random(seed).shuffle(range(num_layers))',
        rounding='floor(p*num_layers+0.5)', permutations=orders, conditions=c))
    bindings = bind(samples)
    for model, records in bindings.items():
        write(OUT/f'endpoint_bindings/{model}.json', records)
    files = model_files()
    for path, value in files.items():
        assert sha(path) == value
    write(OUT/'model_files.json', files)
    write(OUT/'protocol.json', dict(started_epoch=start, base_commit=command('git','rev-parse','HEAD'),
        models=MODELS, seeds=SEEDS, budgets=BUDGETS, sources=sources(),
        history_source_sha256=sha(Path(__file__).with_name('history.py')),
        samples_sha256=sha(OUT/'sample_manifest.json'), layer_sets_sha256=sha(OUT/'layer_sets.json'),
        bindings_sha256={m:sha(OUT/f'endpoint_bindings/{m}.json') for m in MODELS},
        model_files_sha256=sha(OUT/'model_files.json'),
        intervention='Only original P-1 and every actual incremental Decode Query -> original Image Keys: additive -inf before native Softmax; all other Prefill Queries native',
        generation=dict(greedy=True, max_new_tokens=32, use_cache=True, independent_cache=True, independent_actual_prefix=True),
        qwen_deepstack='Native', internvl=dict(max_num=1, image_tokens=256, logits_to_keep=1),
        scope='3000 questions x 14 conditions x 3 models; only 12 random conditions are new complete generations per model',
        endpoints='Reuse strictly bound historical complete generation; LLaVA Only Text numeric first-token metrics replay one prefill if not stored',
        bootstrap=dict(seed=20261102, repetitions=20000, cluster='Image SHA256',
            estimator='Question weighted; average the 3 seeds within the same question BEFORE image-cluster resampling',
            intervals='Ordinary paired 95%; all comparisons exploratory'),
        resources=dict(physical_gpu0='qwen', physical_gpu2='internvl', physical_gpu3='llava', forbidden_gpu=1),
        no_acceleration_or_noninferiority_claim=True))
    print('Frozen 3000 questions / 2409 image clusters, 3 x 14 conditions; 108000 new generations.', flush=True)


def verify():
    p = read(OUT/'protocol.json')
    assert p['sources'] == sources()
    assert p['history_source_sha256'] == sha(Path(__file__).with_name('history.py'))
    assert p['samples_sha256'] == sha(OUT/'sample_manifest.json')
    assert p['layer_sets_sha256'] == sha(OUT/'layer_sets.json')
    assert p['model_files_sha256'] == sha(OUT/'model_files.json')
    for m,h in p['bindings_sha256'].items(): assert sha(OUT/f'endpoint_bindings/{m}.json') == h
    return p

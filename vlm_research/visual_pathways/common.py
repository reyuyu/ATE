import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/visual_pathways_seven_hour'
SOURCE = Path('/data/lk/ATE-worktrees/single-layer-gqa6/vlm_research/results/paper_rq1_single_layer_gqa6/sample_manifest.json')
QH = ROOT / 'vlm_research/results/qwen3vl_image_source_240'
IMODEL = Path('/data/lk/ATE-resources/models/internvl3-8b/853e3a797a661694b1b8ece0cb72dc2b23e3dac9')
START = '2026-11-02T04:18:13+08:00'
TASK1_END = '2026-11-02T08:48:13+08:00'
EXPERIMENT_END = '2026-11-02T10:58:13+08:00'
DEADLINE = '2026-11-02T11:18:13+08:00'
TASKS = ('ChooseAttr', 'ChooseCat', 'ChooseRel', 'CompareAttr', 'LogicalObj', 'QueryAttr')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def sources():
    paths = list((ROOT/'vlm_research/visual_pathways').glob('*.py'))
    paths += list((ROOT/'vlm_research/internvl3_smoke').glob('*.py'))
    paths += [ROOT/'vlm_research/qwen3vl_image_source'/n for n in ('engine.py', 'run.py')]
    paths += [ROOT/'vlm_research/qwen3vl_early_windows/experiment.py', ROOT/'vlm_research/vlm_common.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def assign(samples, count):
    groups = {}
    for s in samples:
        groups.setdefault(s['image_sha256'], []).append(s)
    bins = [[] for _ in range(count)]
    for image, rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        i = min(range(count), key=lambda k: (len(bins[k]), k))
        bins[i].extend(rows)
    return bins


def freeze():
    assert not (OUT/'protocol.json').exists()
    raw = read(SOURCE)['samples']
    old = read(QH/'sample_manifest.json')['samples']
    lookup = {(s['task'], s['question_id']): s for s in raw}
    assert len(lookup) == 3000 and all(sum(s['task'] == t for s in raw) == 500 for t in TASKS)
    keys = set()
    for s in old:
        key = (s['task'], s['question_id'])
        original = lookup[key]
        for field in ('image_id', 'image_sha256', 'question', 'answer'):
            assert original[field] == s[field], (key, field)
        assert sha(s['image_path']) == s['image_sha256']
        keys.add(key)
    assert len(keys) == 240
    im = Path('/data/lk/ATE-worktrees/internvl3-layer-stage-pilot-120/vlm_research/results/internvl3_8b_single_gpu_smoke/model_files_sha256.json')
    model_manifest = read(im)
    for entry in model_manifest['files']:
        assert sha(IMODEL/entry['path']) == entry['sha256'], entry['path']
    write(OUT/'internvl_model_manifest.json',model_manifest)
    samples = []
    for s in sorted(raw, key=lambda s: (s['task'], s['question_id'])):
        samples.append(dict(s, pilot=(s['task'], s['question_id']) in keys))
    write(OUT/'sample_manifest.json', dict(samples=samples, count=3000, pilot=240,
          original_commit='221cdde56dde53908aaa953fe26672fd33334f84', source_sha256=sha(SOURCE),
          selection='Exact frozen six tasks x 500; pilot exact Qwen Image-source/RQ2.1 cohort, no new sampling'))
    write(OUT/'protocol.json', dict(start=START, task1_end=TASK1_END, experiment_end=EXPERIMENT_END, deadline=DEADLINE,
        budget_hours=7, final_reserve_minutes=20, gpu_ids=[0,2,3], prohibited_gpu=1,
        manifest_sha256=sha(OUT/'sample_manifest.json'), source_sha256=sources(),
        qwen_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', internvl_revision=IMODEL.name,
        internvl_model_manifest_sha256=sha(OUT/'internvl_model_manifest.json'),
        llava_revision='b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
        qwen_history={n:sha(QH/n) for n in ('protocol.json','sample_manifest.json','environment.json','per_question.csv','model_manifest.json')},
        task1=dict(conditions='One layer only, original P-1 -> actual Image keys -inf before native softmax',
            target='Same-sample native Full top1, FP32 full-vocabulary log_softmax',
            primary='absolute_logp_change', secondary=['signed_logp_loss','top1_flip'],
            qwen='36 layers BF16/eager, native DeepStack, exact history240 reuse',
            internvl='28 layers BF16/eager, official template/max_num=1/256 visual tokens',
            cache=False, smoke_exact=True),
        task2=dict(conditions={'A':[True,False],'B':[True,True],'C':[False,False],'D':[False,True]},
            relay='All nonvisual original Prefill queries strictly before P-1 -> Image OFF at every layer; causal legal edges only. Decode native.',
            skip='Only original P-1 o_proj output zero in L2-L5; MLP/residual and decode unchanged.',
            interaction='(loss_D-loss_C) - (loss_B-loss_A); positive means greater fixed-Full-token skip harm with relay OFF',
            extension_gate='Image-cluster paired two-sided 95% CI for the predeclared LogP interaction lies strictly above zero. Accuracy interaction reported separately; no NI claim.',
            no_extra_conditions=True),
        task3=dict(optional=True, scope='Relay OFF; P-1 Full/All Image OFF/Keep sparse, decode native',
            llava_keep=[14,19], qwen_keep=[26], internvl_keep='Top two mean absolute sensitivity layers on fixed pilot240, frozen before rescue'),
        bootstrap=dict(seed=20261102, repetitions=10000, cluster='Image SHA256', estimator='Question-weighted; whole-image paired resampling'),
        stop_policy='Task1 at 4.5h; experiments at 6h40; no silent skips/tolerance relaxation, all incomplete coverage explicit'))
    print('Frozen 3000, exact pilot240, deadline', DEADLINE, flush=True)


def verify():
    p = read(OUT/'protocol.json')
    assert sha(OUT/'sample_manifest.json') == p['manifest_sha256']
    # Analysis files are not part of the immutable execution surface.
    current = sources()
    assert all(current.get(k) == v for k,v in p['source_sha256'].items())
    for name, expected in p['qwen_history'].items():
        assert sha(QH/name) == expected
    return p


def preflight(gpu, folder):
    assert gpu in (0,2,3) and os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    apps = command('nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader')
    assert not apps, f'Physical GPU{gpu} occupied: {apps}'
    inventory = command('nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader')
    write(folder/'preflight.json', dict(pid=os.getpid(),gpu=gpu,inventory=inventory,time=time.time()))


if __name__ == '__main__':
    freeze()

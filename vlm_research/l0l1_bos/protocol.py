"""Fixed scope, immutable historical checkpoint bindings and outcome-blind cases."""
import json
import os
from pathlib import Path
import random
import subprocess

from benchmarks.dataset_registry import digest, freeze_json, sha256
from first_answer_attention import run as old
from paper_rq1_single_layer_gqa6.run import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gpu2_l0l1_bos_mechanism'
BASE = 'e495adf4fd93337a3638981bbb66c2bbac81242d'
GPU = 'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6'
CONDITION = dict(layers=[0, 1], decode=True)
SEED = 20261031
KEYS = ('Full', 'L0L1_OFF', 'L2L5_OFF', 'L0L5_OFF')


def read(path):
    return json.loads(Path(path).read_text())


def identity(sample):
    return f'{sample["task"]}__{sample["question_id"]}'


def old_paths(sample):
    directory = old.OUT / f'shard{old.rank_for(sample)}/checkpoints' / identity(sample)
    return {'Full': old.baseline_path(sample), 'L2L5_OFF': directory / 'E.json', 'L0L5_OFF': directory / 'D.json'}


def prior(sample, entry=None):
    result = {}
    for name, path in old_paths(sample).items():
        if entry is not None:
            assert sha256(path) == entry['historical_sha256'][name]
        record = read(path)
        assert record['sample_sha256'] == digest(sample)
        g = record['generation']
        assert record['result_sha256'] == digest(g)
        result[name] = g
    d, e, full = result['L0L5_OFF'], result['L2L5_OFF'], result['Full']
    assert d['full_first_logits_sha256'] == e['full_first_logits_sha256']
    assert d['full_first_token'] == e['full_first_token'] == full['token_ids'][0]
    assert d['full_first_token_logp'] == e['full_first_token_logp']
    return result


def cases(samples):
    selected = []
    for task in old.TASKS:
        pool = sorted([s for s in samples if s['task'] == task], key=lambda s: s['question_id'])
        selected += sorted(random.Random(SEED).sample(pool, 2), key=lambda s: s['question_id'])
    return selected


def sources():
    paths = list((ROOT / 'l0l1_bos').glob('*.py'))
    paths += [ROOT / n for n in ('test_l0l1_bos.py', 'paper_rq1/engine.py',
        'paper_rq1_early_joint_attention/engine.py', 'first_answer_attention/engine.py',
        'paper_rq1_single_layer_gqa6/run.py', 'vlm_common.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def prepare():
    old_manifest = old.verify()
    samples = old.cohort()
    assert len(samples) == 3000
    reused, totals = [], {k: 0 for k in ('Full', 'L2L5_OFF', 'L0L5_OFF')}
    for sample in samples:
        previous = prior(sample)
        for name in totals:
            totals[name] += previous[name]['score']
        reused.append(dict(task=sample['task'], question_id=sample['question_id'],
            sample_sha256=digest(sample), historical_sha256={k: sha256(p) for k, p in old_paths(sample).items()}))
    assert totals == {'Full': 2627, 'L2L5_OFF': 2630, 'L0L5_OFF': 2558}
    freeze_json(OUT / 'reuse_bindings.json', reused)
    protected = [old.OUT / n for n in ('execution_manifest.json', 'per_question.csv', 'result.json', 'accuracy_summary.csv')]
    protected += [old.previous.ORIGINAL / 'sample_manifest.json']
    frozen = dict(base_commit=BASE, questions=3000, images=2409, per_task={t: 500 for t in old.TASKS},
        condition=CONDITION, gpu_uuid=GPU, only_new_condition='L0L1_OFF',
        reuse_bindings_sha256=sha256(OUT / 'reuse_bindings.json'), old_binding=digest(old_manifest),
        protected_sha256={str(p.relative_to(ROOT)): sha256(p) for p in protected},
        cases=[dict(task=s['task'], question_id=s['question_id'], sample_sha256=digest(s)) for s in cases(samples)],
        selection='Independent Random(20261031).sample(sorted original task by question_id,2), no outcome filtering',
        temporal_scope='Only P-1 in Prefill and every subsequent Decode query; o_proj output zero; MLP and KV normal',
        generation=dict(greedy=True, max_new_tokens=32, eos_token_id=2, independent_cache=True),
        native_reuse='Full/D/E complete answers and Full reference LogP reused, no 3000 Full reference forwards',
        mass_partition='BOS token positions; Image576; Prompt = all remaining accessible original input tokens, excluding BOS/Image',
        mechanism='Only original P-1, one independent cached Prefill per condition; observe layers0..5, all heads',
        statistics=dict(seed=SEED, bootstrap_repetitions=200000, cluster='Whole image SHA256, question-weighted paired ratio',
            harm='H01=Full-F; H01_given25=E-D; Full=Native, F=L0L1_OFF, E=L2L5_OFF, D=L0L5_OFF',
            interaction='I=(E-D)-(Full-F)=F+E-D-Full. Positive means super-additive harm on accuracy probability scale.',
            family='Five distinct predeclared contrasts: F-Full,E-Full,D-Full,E-D,interaction; ordinary paired95 and Bonferroni simultaneous95; taskwise descriptive only'),
        illustrative='After accuracy: first <=3 Full-correct/new-wrong cases in task/question order, distinct tasks and images, clearly posthoc')
    freeze_json(OUT / 'protocol.json', frozen)
    print('Prepared3000; Full/D/E reused;12 outcome-blind cases frozen', flush=True)


def verify():
    config = read(OUT / 'protocol.json')
    assert config['reuse_bindings_sha256'] == sha256(OUT / 'reuse_bindings.json')
    for path, expected in config['protected_sha256'].items():
        assert sha256(ROOT / path) == expected, path
    return config


def freeze():
    config = verify()
    tested = read(OUT / 'cpu_validation.json')
    assert tested['passed'] and tested['source_sha256'] == sources()
    paths = list(sources())
    assert not subprocess.check_output(['git', 'diff', 'HEAD', '--', *paths], cwd=ROOT, text=True).strip()
    freeze_json(OUT / 'execution_manifest.json', dict(protocol_sha256=sha256(OUT / 'protocol.json'),
        cpu_sha256=sha256(OUT / 'cpu_validation.json'), source_sha256=sources(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))


def execution():
    verify()
    binding = read(OUT / 'execution_manifest.json')
    assert binding['protocol_sha256'] == sha256(OUT / 'protocol.json')
    assert binding['cpu_sha256'] == sha256(OUT / 'cpu_validation.json')
    assert binding['source_sha256'] == sources()
    return binding


def preflight():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'Only physical GPU2 UUID may be exposed'
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
    assert not any(line.startswith(GPU) for line in apps.splitlines()), 'GPU2 occupied; do not terminate processes'
    free = int(subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True))
    assert free >= 19000
    return dict(gpu_uuid=GPU, gpu2_free_mib=free, all_compute_processes_before=apps)

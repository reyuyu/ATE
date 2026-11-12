"""Bindings and immutable artifacts for the five-condition experiment."""
import json
from pathlib import Path

from ..qwen3vl_early_windows.experiment import MODEL, REVISION, digest, sha256, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/qwen3vl_query_role_l1_l6_240'
RAW = Path('/data/lk/ATE-resources/experiments/qwen3vl_query_role_l1_l6_240')
SOURCE_ROOT = Path('/data/lk/ATE-worktrees/query-role-attention-boundary-240-formal')
SOURCE = SOURCE_ROOT / 'vlm_research/results/query_role_attention_boundary_240_formal/sample_manifest.json'
SOURCE_SHA = '258c32d470270829e126ae9abf3a030dbc1485b2f329fdfc326f4d6e9b3e686c'
LLAVA_IA = Path('/data/lk/ATE-worktrees/query-role-ia-joint-240/vlm_research/results/query_role_ia_joint_240/per_question.csv')
LLAVA_FORMAL = SOURCE_ROOT / 'vlm_research/results/query_role_attention_boundary_240_formal/per_question.csv'
MODEL_MANIFEST = Path('/data/lk/ATE-worktrees/qwen3vl-early-attention-windows-240/vlm_research/results/qwen3vl_early_attention_windows_240/model_manifest.json')
LAYERS = list(range(1, 7))
CONDITIONS = ['Full', 'I-Bypass', 'R-Bypass', 'A-Bypass', 'I+A-Bypass']
GPU_UUID = 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185'
SEED = 20261103
COMPARISONS = [('I-Bypass', 'Full'), ('R-Bypass', 'Full'), ('A-Bypass', 'Full'),
               ('I+A-Bypass', 'Full'), ('R-Bypass', 'I-Bypass'), ('R-Bypass', 'A-Bypass')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seal(path, value):
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Immutable artifact differs: {path}'
    else:
        write(path, value)


def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT / 'vlm_research/qwen3vl_early_windows/experiment.py',
              ROOT / 'vlm_research/qwen3vl_smoke/common.py']
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def roles(layout, vision_end_id):
    ids = layout['input_ids']
    image = layout['image_positions']
    p1 = layout['p_minus_1']
    assert image and image == list(range(image[0], image[-1] + 1))
    assert p1 == len(ids) - 1 and ids[image[-1] + 1] == vision_end_id
    relay = list(range(image[-1] + 2, p1))
    assert relay and set(layout['question_positions']).issubset(relay)
    assert not set(image) & set(relay) and p1 not in image + relay
    return {'I': image, 'R': relay, 'A': [p1]}


def target_rows(condition, mapping, prefill):
    assert condition in CONDITIONS
    if not prefill:
        return [0] if condition in ('A-Bypass', 'I+A-Bypass') else []
    names = {'Full': [], 'I-Bypass': ['I'], 'R-Bypass': ['R'],
             'A-Bypass': ['A'], 'I+A-Bypass': ['I', 'A']}[condition]
    return sorted(i for name in names for i in mapping[name])


def key(sample):
    return sample['task'] + '__' + sample['question_id']


def checkpoint(sample, condition):
    return OUT / 'checkpoints' / key(sample) / (condition + '.json')


def verify():
    p = read(OUT / 'protocol.json')
    assert p['sources'] == sources(), 'Frozen source changed'
    for name in ('sample_manifest.json', 'qwen_input_manifest.json', 'model_manifest.json',
                 'environment.json', 'analysis_environment.json', 'llava_reference.csv', 'llava_formal_reference.csv'):
        assert sha256(OUT / name) == p['artifacts'][name], name
    assert sha256(OUT / 'sample_manifest.json') == SOURCE_SHA
    return p

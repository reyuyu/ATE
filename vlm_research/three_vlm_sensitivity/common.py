import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/three_vlm_layer_sensitivity_3000'
RAW = Path('/data/lk/ATE-resources/experiments/three_vlm_layer_sensitivity_3000')
HISTORY = Path('/data/lk/ATE-worktrees')
LLAVA = HISTORY / 'single-layer-gqa6/vlm_research/results/paper_rq1_single_layer_gqa6'
QWEN = HISTORY / 'qwen3vl-four-stage-pilot-120/vlm_research/results/qwen3vl_four_stage_pilot_120'
INTERN = HISTORY / 'internvl3-layer-stage-pilot-120/vlm_research/results/internvl3_layer_stage_pilot_120'
TASKS = ['ChooseAttr', 'ChooseCat', 'ChooseRel', 'CompareAttr', 'LogicalObj', 'QueryAttr']
MODELS = {
    'qwen': dict(repo='Qwen/Qwen3-VL-8B-Instruct', revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
        directory='qwen3-vl-8b-instruct', layers=36, transformers='4.57.6'),
    'internvl': dict(repo='OpenGVLab/InternVL3-8B', revision='853e3a797a661694b1b8ece0cb72dc2b23e3dac9',
        directory='internvl3-8b', layers=28, transformers='4.49.0'),
}


def model_path(name):
    m = MODELS[name]
    return Path('/data/lk/ATE-resources/models') / m['directory'] / m['revision']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    folders = ['three_vlm_sensitivity', 'internvl3_smoke', 'internvl3_stage',
               'qwen3vl_smoke', 'qwen3vl_early_windows', 'qwen3vl_four_stage_pilot']
    return {str(p.relative_to(ROOT)): sha256(p) for folder in folders
            for p in sorted((ROOT / 'vlm_research' / folder).glob('*.py'))}


def verify_sources():
    protocol = read(OUT / 'protocol.json')
    for path, expected in protocol['code_sha256'].items():
        assert sha256(ROOT / path) == expected, f'Frozen source changed: {path}'
    assert sha256(OUT / 'shared_3000_manifest.json') == protocol['shared_manifest_sha256']
    return protocol

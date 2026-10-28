"""Protocol, content binding and checkpoint validation, without GPU imports."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'ed666b68eef6380a8bd57852636c18bef1673b02'
HISTORY = ROOT / 'results/phase3_2_fixed_k_exit'
CACHE = ROOT / 'results/phase3_4_roi_sink_diagnostic/attention_rows'
DISCOVERY = '15133706'
PINS = {'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124',
        'transformers': '4.49.0', 'accelerate': '1.4.0', 'huggingface_hub': '0.29.3',
        'safetensors': '0.5.3', 'pillow': '11.1.0', 'numpy': '1.26.4',
        'matplotlib': '3.10.1', 'sentencepiece': '0.2.0'}
CONDITIONS = [dict(name='full', source=None, start=None, strength=None)]
CONDITIONS += [dict(name=f'noop_L{s}', source=s, start=24, strength=0.) for s in (23, 24)]
CONDITIONS += [dict(name=f'L{s}_from24_lambda{v:g}', source=s, start=24, strength=v)
               for s in (23, 24) for v in (.5, 1.)]
CONDITIONS += [dict(name=f'L{s}_from25_lambda1', source=s, start=25, strength=1.) for s in (23, 24)]
RESEARCH = [c['name'] for c in CONDITIONS if c['strength'] in (.5, 1.)]
STATISTICS = dict(bootstrap_seed=360240, bootstrap_repetitions=5000,
                  weak_full_margin_abs_at_most=.15, severe_gain_at_most=-.5,
                  interval='paired image-level percentile 95%; exploratory, not multiplicity adjusted',
                  flips='exact two-sided discordant-pair binomial (McNemar)',
                  direction='0<=gain(.5)<=gain(1), or 0>=gain(.5)>=gain(1); zeros separate',
                  cohorts=['primary_239', 'uncached_187', 'all_240', 'discovery_1'],
                  historical_policy='Record exactness and numerical differences; no tolerance fitted. '
                  'All templates freshly captured from this image on this machine. '
                  'Same-machine no-op and causal invariants remain bitwise mandatory.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def samples():
    result = sorted(read(HISTORY / 'manifest.json')['samples'], key=lambda s: s['question_id'])
    if len(result) != 240 or len({s['question_id'] for s in result}) != 240 or len({s['img_id'] for s in result}) != 240:
        raise ValueError('Frozen cohort must be exactly 240 distinct questions/images')
    if [sum(s['answer'] == label for s in result) for label in ('left', 'right')] != [120, 120]:
        raise ValueError('Frozen cohort label balance changed')
    return result


def split(rows):
    if len(rows) != 240 or len({s['question_id'] for s in rows}) != 240:
        raise ValueError('Cannot shard an incomplete or duplicated cohort')
    ordered = sorted(rows, key=lambda s: s['question_id'])
    return [[s['question_id'] for s in ordered[i*60:(i+1)*60]] for i in range(4)]


def frozen_sources():
    paths = list(ROOT.glob('*.py')) + list((ROOT / 'phase3_6').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def history_k24():
    rows = list(csv.DictReader((HISTORY / 'per_sample_results.csv').open()))
    chosen = [r for r in rows if r['cutoff'] == '24']
    if len(rows) != 1920 or len(chosen) != 240 or len({r['question_id'] for r in chosen}) != 240:
        raise ValueError('Invalid Physical K24 join/denominator')
    return {r['question_id']: r for r in chosen}


def prepare(output, image_root, model_path):
    if output.exists():
        raise FileExistsError('Use a new protocol directory; never replace a frozen protocol')
    cohort = samples()
    cached = {p.stem.removesuffix('_task') for p in CACHE.glob('*_task.npz')}
    ids = {s['question_id'] for s in cohort}
    if len(cached) != 53 or not cached <= ids or DISCOVERY not in cached:
        raise ValueError('Expected exactly 53 cached cohort images including discovery')
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASELINE, 'HEAD'], cwd=ROOT, check=True)
    sources = frozen_sources()
    if git('diff', 'HEAD', '--', '*.py'):
        raise ValueError('Commit and freeze implementation before preparing a run')
    model_config = read(ROOT / 'configs/model.json')
    model_files = {}
    for item in model_config['files']:
        p = model_path / item['name']
        value = sha(p)
        if p.stat().st_size != item['bytes'] or (item.get('lfs_sha256') and value != item['lfs_sha256']):
            raise ValueError('Model resource differs from fixed HF revision')
        model_files[item['name']] = value
    for sample in cohort:
        if sha(image_root / f"image_{sample['img_id']}.jpg") != sample['image_sha256']:
            raise ValueError('Frozen input image changed')
    files = [HISTORY / 'manifest.json', HISTORY / 'per_sample_results.csv', ROOT / 'configs/model.json',
             ROOT / 'requirements.txt']
    files += [HISTORY / 'checkpoints/run' / f"{s['question_id']}.json" for s in cohort]
    files += sorted(CACHE.glob('*_task.npz'))
    config = dict(phase='3.6', baseline=BASELINE, code_commit=git('rev-parse', 'HEAD'),
                  sources=sources, inputs={str(p.relative_to(ROOT)): sha(p) for p in files},
                  conditions=CONDITIONS, statistics=STATISTICS, packages=PINS,
                  model_revision=model_config['revision'], model_files_sha256=model_files,
                  image_root=str(image_root), model_path=str(model_path), cached_question_ids=sorted(cached),
                  shards=split(cohort), samples=cohort,
                  smoke_ids=[DISCOVERY, next(s['question_id'] for s in cohort if s['answer'] == 'left')])
    output.mkdir(parents=True)
    atomic(output / 'protocol.json', dict(binding=digest(config), config=config))
    return config


def protocol(output, require_commit=True):
    envelope = read(output / 'protocol.json')
    config = envelope['config']
    if digest(config) != envelope['binding'] or config['conditions'] != CONDITIONS or config['statistics'] != STATISTICS:
        raise ValueError('Frozen protocol/binding changed')
    if config['sources'] != frozen_sources():
        raise ValueError('Worker source hashes differ from frozen implementation')
    if require_commit and git('rev-parse', 'HEAD') != config['code_commit']:
        raise ValueError('Worker must use the exact frozen code commit')
    for name, expected in config['inputs'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen input/cache/history changed: ' + name)
    if config['shards'] != split(config['samples']):
        raise ValueError('Shard assignment changed')
    return config, envelope['binding']


def checkpoint(path, binding, expected_sample, artifact_root):
    envelope = read(path)
    body = envelope['body']
    if digest(body) != envelope['sha256'] or body['binding'] != binding or body['sample_sha256'] != digest(expected_sample):
        raise ValueError('Checkpoint binding/sample/hash mismatch')
    if body['question_id'] != expected_sample['question_id'] or not body['passed']:
        raise ValueError('Invalid sample checkpoint')
    if [r['condition'] for r in body['rows']] != [c['name'] for c in CONDITIONS]:
        raise ValueError('Missing or duplicated sample condition')
    for name, expected in body['artifacts'].items():
        path = (artifact_root / name).resolve()
        if not path.is_relative_to(artifact_root.resolve()) or sha(path) != expected:
            raise ValueError('Checkpoint artifact changed/escaped output root')
    return body

"""Package completed results without model forwards, refitting, or new gates."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from benchmarks.dataset_registry import digest, freeze_json, sha256
from bos_cheap.protocol import OUT, UPDATES
from bos_cheap.run import gate, verify


def prepare():
    completion = json.loads((OUT / 'completion.json').read_text())
    if completion['status'] != 'complete_exploratory':
        raise ValueError('Require exploratory completion')
    for name, expected in completion['files_sha256'].items():
        if sha256(OUT / name) != expected:
            raise ValueError('Original completion changed: ' + name)
    archive = OUT / 'audit/pre_review'
    archive.mkdir(parents=True, exist_ok=True)
    for name in ['completion.json', 'report.md', 'research_summary.md']:
        target = archive / name
        if target.exists():
            raise ValueError('Refuse to overwrite original review snapshot')
        shutil.copy2(OUT / name, target)
    config = verify()
    manifest = json.loads((OUT / 'sample_manifest.json').read_text())
    failures = list(OUT.glob('**/failures/*.json'))
    if failures:
        raise ValueError('Unexpected failed attempts: ' + str(failures))
    summary = {}
    max_historical = 0.
    for stage in ['smoke', 'old', 'new']:
        workers = gate(stage)
        expected = manifest['old_samples'][:1] * 4 if stage == 'smoke' else manifest[stage + '_samples']
        paths = list(OUT.glob(f'{stage}/shard*/checkpoints/*.json'))
        if len(paths) != len(expected):
            raise ValueError('Wrong checkpoint count: ' + stage)
        seen = set()
        for path in paths:
            ck = json.loads(path.read_text())
            key = (ck['task'], ck['question_id'])
            if stage != 'smoke' and key in seen:
                raise ValueError('Duplicate checkpoint')
            seen.add(key)
            if ck['binding'] != digest(config) or not all(ck['checks'][k] for k in ['inputs_unchanged', 'hooks_clean']):
                raise ValueError('Invalid checkpoint binding or cleanup')
            if sha256(ck['raw_path']) != ck['raw_sha256']:
                raise ValueError('Changed raw vectors')
            if [r['layer'] for r in ck['features']] != list(range(32)):
                raise ValueError('Missing feature layers')
            with np.load(ck['raw_path']) as raw:
                for name in raw.files:
                    if not np.isfinite(raw[name]).all():
                        raise ValueError('Nonfinite raw array')
                if raw['qkv'].shape != (32, 3, 32, 128):
                    raise ValueError('Wrong native QKV shape')
                if stage == 'new' and (raw['ablated_logits'].shape[0] != 32 or len(ck['metrics']) != 32):
                    raise ValueError('Missing new labels')
            if ck['historical_full_max_difference'] is not None:
                max_historical = max(max_historical, ck['historical_full_max_difference'])
        if stage != 'smoke' and seen != {(s['task'], s['question_id']) for s in expected}:
            raise ValueError('Sample membership differs')
        summary[stage] = dict(checkpoints=len(paths), workers=workers, finite=True, raw_sha256_verified=True)
    with (UPDATES / 'per_question_updates.csv').open() as f:
        prior = {(r['task'], r['question_id'], int(r['layer'])): r for r in csv.DictReader(f)}
    differences = []
    with (OUT / 'matched_update_mechanisms.csv').open() as f:
        for row in csv.DictReader(f):
            key = row['task'], row['question_id'], int(row['layer'])
            differences.append(abs(float(row['residual_norm']) - float(prior[key]['norm_residual'])))
    if len(differences) != 19200:
        raise ValueError('Invalid decomposition join')
    freeze_json(OUT / 'release_validation.json', dict(
        passed=True, source_code_commit=config['code_commit'], stages=summary,
        failures=0, historical_full_max_abs_difference=max_historical,
        matched_decomposition_rows=len(differences),
        matched_residual_norm_max_abs_difference=max(differences),
        model_forwards=0, predictor_fits=0, threshold_changes=0,
        script_sha256=sha256(Path(__file__))))
    source = OUT / 'cheap_features_per_question.csv'
    target = OUT / 'cheap_features_per_question.csv.gz'
    with source.open('rb') as src, target.open('xb') as dst:
        with gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as zipped:
            shutil.copyfileobj(src, zipped)
    restored = hashlib.sha256()
    with gzip.open(target, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            restored.update(block)
    if restored.hexdigest() != sha256(source):
        raise ValueError('Compression is not lossless')
    freeze_json(OUT / 'question_artifact_manifest.json', dict(
        rows=113728, local_uncompressed=str(source), uncompressed_sha256=sha256(source),
        uncompressed_bytes=source.stat().st_size,
        git_lossless_gzip=target.name, gzip_sha256=sha256(target), gzip_bytes=target.stat().st_size,
        decompressed_sha256=restored.hexdigest(),
        note='Plain CSV exists locally; Git carries identical bytes in gzip to respect the100MiB file limit.'))
    print('Original manifest verified; raw arrays finite; lossless question CSV ready.', flush=True)


def finalize():
    original_path = OUT / 'audit/pre_review/completion.json'
    original = json.loads(original_path.read_text())
    changes = []
    for name, expected in original['files_sha256'].items():
        if sha256(OUT / name) != expected:
            changes.append(name)
    if set(changes) != {'report.md', 'research_summary.md'}:
        raise ValueError('Unexpected frozen-result changes: ' + str(changes))
    figure_rows = []
    for p in sorted((OUT / 'figures').glob('*.png')):
        with Image.open(p) as image:
            pixels = np.asarray(image.convert('RGB'))
            if min(image.size) < 400 or pixels.std() < 3:
                raise ValueError('Blank or invalid figure: ' + str(p))
            figure_rows.append(dict(path=str(p.relative_to(OUT)), width=image.width, height=image.height,
                                    pixel_std=float(pixels.std()), pdf_sha256=sha256(p.with_suffix('.pdf'))))
    freeze_json(OUT / 'figure_validation.json', dict(passed=True, figures=figure_rows,
        note='Pixel/nonblank checks plus manual visual inspection; no synthetic or interpolated results.'))
    freeze_json(OUT / 'audit/release_revision.json', dict(
        original_completion_sha256=sha256(original_path),
        original_hash_inventory_verified_before_review=True,
        original_inventory_files=len(original['files_sha256']), revised_original_files=changes,
        reason='Add Chinese nine-question interpretation, readable threshold definitions and review figures; preserve original auto-generated artifacts.',
        frozen_worker_sources_unchanged=True, predictor_sha256=sha256(OUT / 'frozen_predictors.json'),
        script_sha256=sha256(Path(__file__)), predictor_fits=0, threshold_changes=0, model_forwards=0))
    inventory = {}
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p == OUT / 'completion.json' or p.suffix in ['.lock', '.tmp']:
            continue
        if p.name in ['cheap_features_per_question.csv', 'cheap_features_per_head.csv']:
            continue
        if p.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError('Git artifact too large: ' + str(p))
        inventory[str(p.relative_to(OUT))] = sha256(p)
    result = {k: v for k, v in original.items() if k != 'files_sha256'}
    result.update(files_sha256=inventory, release_revision='reviewed-v2',
                  original_completion_sha256=sha256(original_path),
                  source_code_commit=verify()['code_commit'],
                  storage='Lossless compressed question/head CSV in Git; original plain CSV and raw NPZ retained locally.',
                  no_under_one_percent_risk_certification=True, no_speedup_claim=True)
    with tempfile.NamedTemporaryFile('w', dir=OUT, suffix='.tmp', delete=False) as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno()); temp = f.name
    os.replace(temp, OUT / 'completion.json')
    print('Reviewed completion inventory:', len(inventory), 'files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'finalize'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else finalize()

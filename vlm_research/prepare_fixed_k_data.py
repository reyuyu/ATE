"""Freeze Phase 3.2 data without inspecting any model predictions."""
import csv
import hashlib
import json
from pathlib import Path
import random

from PIL import Image
from prepare_physical_exit_data import sha256
from vlm_common import ROOT

OUTPUT = ROOT / 'results/phase3_2_fixed_k_exit'
SEED = 20261029


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def select_rows(rows, excluded, per_label=120, seed=SEED):
    rng, selected, used = random.Random(seed), [], set(excluded)
    for label in ('left', 'right'):
        pool = sorted((r for r in rows if r['answer'] == label and r['true option'] == label
                       and {r['true option'], r['false option']} == {'left', 'right'}
                       and r['question'].strip()), key=lambda r: (r['question_id'], r['img_id']))
        rng.shuffle(pool)
        count = 0
        for row in pool:
            if row['img_id'] in used:
                continue
            selected.append(row)
            used.add(row['img_id'])
            count += 1
            if count == per_label:
                break
        if count != per_label:
            raise ValueError(f'Insufficient unique images for {label}: {count}/{per_label}')
    return selected


def audit_image(sample):
    path = Path(sample['image_path'])
    result = {'image_available': path.is_file()}
    if not path.is_file():
        return dict(result, error='Missing frozen image; do not replace')
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            result['image_size'] = list(im.size)
        result['image_sha256'] = sha256(path)
        if result['image_size'] != sample['source_image_size']:
            result['error'] = 'Image dimensions differ from source CSV'
    except Exception as exc:
        result['error'] = repr(exc)
    return result


def main():
    if OUTPUT.exists():
        raise ValueError('Frozen experiment directory exists; use runner resume, never resample')
    old_path = ROOT / 'configs/phase3_1_data/manifest.json'
    old = json.loads(old_path.read_text())
    source = ROOT / 'configs/phase3_1_data/GQA_val_correct_question_with_relChooser_ChooseRel.csv'
    if sha256(source) != old['source_csv_sha256']:
        raise ValueError('Source CSV changed since pilot')
    rows = list(csv.DictReader(source.open()))
    selected = select_rows(rows, {s['img_id'] for s in old['samples']})
    samples = []
    for r in selected:
        s = {k: r[k] for k in ('question_id', 'img_id', 'question', 'answer', 'true option', 'false option')}
        s.update(candidates=['left', 'right'], correct_candidate=r['true option'], wrong_candidate=r['false option'],
                 image_path=str(Path(old['local_image_root']) / f"image_{r['img_id']}.jpg"),
                 source_image_size=[int(r['image_w']), int(r['image_h'])], source_row_sha256=digest(r))
        s.update(audit_image(s))
        samples.append(s)
    # Prespecified balanced prefix; no predictions or availability used to choose it.
    smoke_ids = [s['question_id'] for label in ('left', 'right')
                 for s in [r for r in samples if r['answer'] == label][:12]]
    manifest = {k: old[k] for k in ('source_repository', 'source_revision', 'source_file',
                                  'source_csv_sha256', 'source_rows', 'local_image_root', 'caveat')}
    manifest.update(seed=SEED, phase='3.2', pilot_manifest_sha256=sha256(old_path),
                    excluded_image_ids=[s['img_id'] for s in old['samples']],
                    selection_rule='Sort by string question_id,img_id; Python Random(20261029), shuffle left pool then right pool; first 120 unused images each, shared exclusion/used set. No model outputs or availability used. Missing/corrupt images are frozen and block execution, never replaced.',
                    smoke_rule='First 12 left and first 12 right in frozen selection order', smoke_question_ids=smoke_ids,
                    validation_rule='All 12 original Phase 3.1 images, all eight K; historical compatibility at shared K',
                    validation_samples=old['samples'], samples=samples)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'manifest_sha256': sha256(OUTPUT/'manifest.json'), 'count': len(samples),
                      'invalid': [s['question_id'] for s in samples if not s['image_available'] or 'error' in s]}))


if __name__ == '__main__':
    main()

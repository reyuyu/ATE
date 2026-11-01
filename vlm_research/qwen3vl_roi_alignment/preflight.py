"""Fail-closed annotation gate; no GPU experiment is launched by this module."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/qwen3vl_roi_sensitivity_alignment_240'
HISTORY = ROOT / 'vlm_research/results/qwen3vl_image_source_240'
CSV_COMMIT = 'c9a1111dac4baa175f67716fab473789320187c0'
ANNOTATION = Path('/data/lk/ATE-resources/paper_rq1/annotations/gqa_val_balanced.parquet')
SOURCES = {
    'ChooseAttr': ('GQA_val_correct_question_with_choose_ChooseAttr.csv', 'fc02b888c567442ed9fb5dacddba75f3ce81e889'),
    'ChooseCat': ('GQA_val_correct_question_with_categoryThatThisChoose_objThisChoose_ChooseCat.csv', '18cf6b85fb702720b55d2f78bf2075e196aea8f0'),
    'ChooseRel': ('GQA_val_correct_question_with_relChooser_ChooseRel.csv', '0b57b6a2fa5de8ec3202bc94b7afbf84f2ebd1f9'),
    'CompareAttr': ('GQA_val_correct_question_with_twoCommon_CompareAttr.csv', '1443b40cb1176016e53bda37f9841233ce74bdbf'),
    'LogicalObj': ('GQA_val_correct_question_with_existThatOr_LogicalObj.csv', 'd4466383593c580722732be6571320f3376c14f0'),
    'QueryAttr': ('GQA_val_correct_question_with_positionQuery_QueryAttr.csv', '79df684ad216270cffb2755474901c7c9733e760'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def canonical(value):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError(f'Invalid ID: {value}')
    return str(int(number))


def bbox(row, prefix):
    values = [float(row[f'{prefix} {v}']) for v in ('x', 'y', 'w', 'h')]
    width, height = float(row['image_w']), float(row['image_h'])
    x, y, w, h = values
    if not all(math.isfinite(v) for v in values + [width, height]):
        raise ValueError('Nonfinite bounding box')
    if min(x, y) < 0 or min(w, h, width, height) <= 0 or x+w > width or y+h > height:
        raise ValueError('Bounding box outside image or empty; no silent clipping')
    return dict(object_id=canonical(row[f'{prefix} id']), name=row[f'{prefix} name'],
                xywh=values, image_size=[width, height])


def binding(row, sample):
    if canonical(row['img_id']) != canonical(sample['image_id']):
        raise ValueError('Image ID mismatch')
    if row['question'] != sample['question'] or row['answer'] != sample['answer']:
        raise ValueError('Question/answer mismatch')


def source_path(task, filename):
    cached = OUT / 'sources' / filename
    if cached.exists():
        return cached
    if task in ('ChooseCat', 'CompareAttr'):
        return OUT / 'sources' / f'GQA_{task}.csv'
    if task == 'ChooseRel':
        return ROOT / 'vlm_research/configs/phase3_1_data' / filename
    return Path('/data/lk/ATE-resources/benchmarks/phase4_0/sources/information_flow/datasets') / filename


def audit():
    import pyarrow.parquet as pq
    from PIL import Image
    manifest = json.loads((HISTORY / 'sample_manifest.json').read_text())
    history_protocol = json.loads((HISTORY / 'protocol.json').read_text())
    assert history_protocol['model_revision'] == '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
    assert history_protocol['dtype'] == 'bfloat16' and history_protocol['attention'] == 'eager'
    assert history_protocol['manifest_sha256'] == sha(HISTORY / 'sample_manifest.json')
    samples = manifest['samples']
    assert len(samples) == 240 and set(Counter(s['task'] for s in samples).values()) == {40}
    originals = {canonical(r['id']): r for r in pq.read_table(ANNOTATION).to_pylist()}
    lookup, objects, source_audit = {}, defaultdict(set), {}
    for task, (filename, git_blob) in SOURCES.items():
        path = source_path(task, filename)
        raw = path.read_bytes()
        actual_blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        assert actual_blob == git_blob, (task, actual_blob, git_blob)
        copied = OUT / 'sources' / filename
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(raw)
        rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
        source_audit[task] = dict(filename=filename, git_blob_sha1=git_blob,
            sha256=sha(path), rows=len(rows), official_commit=CSV_COMMIT)
        for r in rows:
            key = (task, canonical(r['question_id']))
            assert key not in lookup, key
            lookup[key] = r
            for prefix in ('central object', 'object1', 'object2'):
                if f'{prefix} id' not in r:
                    continue
                try:
                    b = bbox(r, prefix)
                except (ValueError, ArithmeticError):
                    continue
                objects[(canonical(r['img_id']), b['object_id'])].add(tuple(b['xywh']))
    assert all(len(v) == 1 for v in objects.values()), 'Conflicting official object BBoxes'
    sensitivity = list(csv.DictReader((HISTORY / 'per_question.csv').open()))
    previous = defaultdict(list)
    for r in sensitivity:
        previous[(r['task'], r['question_id'])].append(r)
    assert len(sensitivity) == 8640 and len(previous) == 240
    records, targets = [], {}
    for s in samples:
        key = (s['task'], s['question_id'])
        orig = originals[canonical(s['question_id'])]
        assert orig['question'] == s['question'] and orig['answer'] == s['answer']
        assert canonical(orig['imageId']) == canonical(s['image_id'])
        assert sha(s['image_path']) == s['image_sha256']
        with Image.open(s['image_path']) as image:
            image_size = list(image.size)
        old = previous[key]
        assert sorted(int(r['layer']) for r in old) == list(range(36))
        assert all(r['image_sha256'] == s['image_sha256'] and r['image_id'] == s['image_id'] for r in old)
        encoded = json.dumps(s['layout'], sort_keys=True, ensure_ascii=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == s['input_binding']
        row = lookup.get((s['task'], canonical(s['question_id'])))
        boxes, error = [], ''
        if row is not None:
            binding(row, s)
            try:
                prefixes = ['central object'] if 'central object id' in row else ['object1', 'object2']
                # '-' is an explicitly nonexistent logical alternative, not a box to invent.
                boxes = [bbox(row, prefix) for prefix in prefixes if row[f'{prefix} id'] != '-']
                if not boxes:
                    raise ValueError('No existing target object has a BBox')
                assert all(b['image_size'] == image_size for b in boxes), 'Image dimensions mismatch'
            except (ValueError, ArithmeticError, AssertionError) as exc:
                error = str(exc)
                boxes = []
        # Exact referenced IDs only. This is a coverage diagnostic, NOT a target-ROI definition.
        mentioned = sorted({a['value'] for group in orig['annotations'].values() for a in group
                            if a['value'].isdigit()})
        missing_mentions = [o for o in mentioned if (canonical(s['image_id']), canonical(o)) not in objects]
        status = 'direct_csv_bbox_available' if boxes else 'invalid_csv_bbox' if row else 'question_absent_from_csv'
        records.append(dict(task=s['task'], question_id=s['question_id'], image_id=s['image_id'],
            image_sha256=s['image_sha256'], input_binding=s['input_binding'], status=status,
            bbox_error=error, direct_csv_question=int(row is not None), valid_target_boxes=len(boxes),
            explicitly_absent_csv_objects=sum(row.get(f'{p} id') == '-' for p in ('object1','object2')) if row else 0,
            official_mentioned_ids=json.dumps(mentioned), mentioned_ids_missing_csv_boxes=json.dumps(missing_mentions),
            all_mentioned_objects_in_csv=int(bool(mentioned) and not missing_mentions)))
        if boxes:
            targets[f'{s["task"]}__{s["question_id"]}'] = boxes
    with (OUT / 'bbox_coverage.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    counts = {t: dict(Counter(r['status'] for r in records if r['task'] == t)) for t in SOURCES}
    complete = len(targets) == 240
    assert not complete, 'Coverage is now complete: review protocol before proceeding'
    dump(OUT / 'sample_manifest.json', manifest)
    dump(OUT / 'available_csv_boxes.json', targets)
    dump(OUT / 'source_audit.json', source_audit)
    dump(OUT / 'validation.json', dict(status='partial_annotation_gate_failed',
        samples=240, valid_direct_bbox_questions=len(targets), coverage_by_task=counts,
        question_image_answer_bindings_checked=240, actual_image_sha256_checked=240,
        historical_sensitivity_rows_bound=8640, historical_input_binding_hashes_checked=240,
        all_mentioned_objects_available=sum(r['all_mentioned_objects_in_csv'] for r in records),
        all_mentioned_objects_unavailable=sum(not r['all_mentioned_objects_in_csv'] for r in records),
        indirect_lookup_is_not_target_roi_validation=True,
        gpu_smoke='NOT RUN: incomplete reliable ROI annotation', gpu_collection='NOT RUN',
        statistical_results='NOT RUN', model_forwards=0))
    dump(OUT / 'protocol.json', dict(status='partial_annotation_gate_failed',
        sample_manifest_sha256=sha(HISTORY / 'sample_manifest.json'),
        sensitivity_csv_sha256=sha(HISTORY / 'per_question.csv'),
        history_protocol_sha256=sha(HISTORY / 'protocol.json'),
        original_annotation_sha256=sha(ANNOTATION), original_annotation_path=str(ANNOTATION),
        source_repository='FightingFighting/cross-modal-information-flow-in-MLLM',
        source_commit=CSV_COMMIT, model='Qwen/Qwen3-VL-8B-Instruct',
        revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b', dtype='bfloat16', attention='eager',
        requested_samples=240, requested_layers=36, allowed_physical_gpus=[0,2,3],
        stop_reason='Official per-task correct_question CSVs do not cover the frozen cohort; no replacement or fabricated BBoxes.',
        no_gpu_experiment_started=True, no_sensitivity_or_correlation_conclusions=True))
    print(json.dumps(counts, indent=2), flush=True)
    print(f'STOP: only {len(targets)}/240 direct target BBoxes validated; no GPU run.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['audit'])
    parser.parse_args()
    audit()

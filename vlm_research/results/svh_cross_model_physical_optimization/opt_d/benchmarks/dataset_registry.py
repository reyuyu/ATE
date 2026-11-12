"""CPU-only schema, immutable manifests, and relocatable read-only resources."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from PIL import Image

SEED = 20261029
DEFAULT_RESOURCES = Path('/data/lk/ATE-resources/benchmarks/phase4_0')
OUTPUT = Path(__file__).resolve().parents[1] / 'results/phase4_0_dataset_suite'
REQUIRED = {'dataset', 'subset', 'split', 'question_id', 'image_id', 'image_path',
            'question', 'answer', 'candidates', 'answer_type', 'source_revision',
            'source_row_sha256', 'image_sha256', 'metadata'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def freeze_json(path, value):
    """Idempotent exact replay only: changes require a new suite version."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen artifact differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two preparers overwriting each other.
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def validate_sample(row):
    if REQUIRED - row.keys():
        raise ValueError(f'Missing fields: {REQUIRED - row.keys()}')
    for key in ('dataset', 'question_id', 'image_id', 'question', 'source_revision'):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f'Invalid {key}')
    candidates = row['candidates']
    if row['answer_type'] == 'closed':
        if not candidates or len(candidates) < 2 or len(set(candidates)) != len(candidates):
            raise ValueError('Closed task requires distinct real candidates')
        if row['answer'] not in candidates:
            raise ValueError('Gold absent from candidates')
    elif row['answer_type'] in ('open', 'vqa_soft'):
        if candidates is not None:
            raise ValueError('Never fabricate candidates for open answers')
        if row['answer_type'] == 'vqa_soft' and (not isinstance(row['answer'], list) or len(row['answer']) != 10):
            raise ValueError('TextVQA requires all 10 answers')
    else:
        raise ValueError('Unknown answer type')
    return row


def audit_image(path, expected_size=None):
    path = Path(path)
    result = {'image_path': str(path), 'image_sha256': None, 'image_size': None, 'image_status': 'missing'}
    if not path.is_file():
        return result
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            result.update(image_size=list(im.size), image_format=im.format)
        result['image_sha256'] = sha256(path)
        if expected_size is not None and list(expected_size) != result['image_size']:
            raise ValueError(f'Image dimensions: {result["image_size"]} != {expected_size}')
        result['image_status'] = 'verified'
    except Exception as exc:
        result.update(image_status='corrupt', error=f'{type(exc).__name__}: {exc}')
    return result


def audit_rows(rows):
    for row in rows:
        validate_sample(row)
    ids = Counter(r['question_id'] for r in rows)
    images = Counter(r['image_id'] for r in rows)
    duplicate_ids = {k: v for k, v in ids.items() if v > 1}
    if duplicate_ids:
        raise ValueError(f'Duplicate Question IDs: {duplicate_ids}')
    return {'questions': len(rows), 'unique_images': len(images), 'duplicate_question_ids': duplicate_ids,
            'repeated_image_ids': {k: v for k, v in images.items() if v > 1},
            'answers': dict(sorted(Counter(str(r['answer']) for r in rows).items())),
            'answer_types': dict(Counter(r['answer_type'] for r in rows)),
            'choice_counts': dict(Counter(str(len(r['candidates'])) if r['candidates'] else 'open' for r in rows))}


def select_unique(rows, count, excluded=(), balanced=False):
    """Sort -> shuffle, prefer previously unused images; never inspect predictions/assets."""
    audit_rows(rows)
    rng = random.Random(SEED)
    groups = [None] if not balanced else ['no', 'yes']
    if balanced and count % 2:
        raise ValueError('Balanced target must be even')
    used, selected = set(), []
    for label in groups:
        pool = sorted([r for r in rows if label is None or r['answer'] == label],
                      key=lambda r: (r['question_id'], r['image_id']))
        rng.shuffle(pool)
        # Stable partition; exclusion is a preference, not a hidden availability filter.
        pool = [r for r in pool if r['image_id'] not in excluded] + [r for r in pool if r['image_id'] in excluded]
        quota = count if label is None else count // 2
        n = 0
        for row in pool:
            if row['image_id'] in used:
                continue
            selected.append(row)
            used.add(row['image_id'])
            n += 1
            if n == quota:
                break
        if n != quota:
            raise ValueError(f'Insufficient distinct images for frozen quota {label}: {n}/{quota}')
    return sorted(selected, key=lambda r: r['question_id'])


def select_groups(rows, target):
    audit_rows(rows)
    by_image = {}
    for row in rows:
        by_image.setdefault(row['image_id'], []).append(row)
    ids = sorted(by_image)
    random.Random(SEED).shuffle(ids)
    out = []
    for image_id in ids:
        out.extend(sorted(by_image[image_id], key=lambda r: r['question_id']))
        if len(out) >= target:
            break
    return sorted(out, key=lambda r: (r['image_id'], r['question_id']))


def load_manifest(path, image_root=None, verify_images=True):
    """Never mutates the frozen file; root mapping is explicit and hash checked."""
    manifest = json.loads(Path(path).read_text())
    body = {k: v for k, v in manifest.items() if k != 'binding_sha256'}
    if digest(body) != manifest['binding_sha256']:
        raise ValueError('Manifest binding mismatch')
    for source in manifest['sources']:
        if sha256(source['path']) != source['sha256']:
            raise ValueError(f'Source revision/file changed: {source["path"]}')
    audit_rows(manifest['samples'])
    for row in manifest['samples']:
        if image_root is not None:
            row['image_path'] = str(Path(image_root) / row['metadata']['resource_relative_path'])
        if verify_images:
            if not row['image_sha256']:
                raise ValueError('blocked_assets: missing frozen image hash')
            if sha256(row['image_path']) != row['image_sha256']:
                raise ValueError(f'Image hash mismatch: {row["image_id"]}')
    return manifest

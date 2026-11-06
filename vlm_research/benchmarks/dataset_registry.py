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
    hidden = row.get('answer_type') == 'hidden'
    required = REQUIRED - {'answer'} if hidden else REQUIRED
    if required - row.keys():
        raise ValueError(f'Missing fields: {required - row.keys()}')
    for key in ('dataset', 'question_id', 'image_id', 'question', 'source_revision'):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f'Invalid {key}')
    candidates = row['candidates']
    if hidden:
        if row.get('gold_available') is not False or row.get('official_server_scoring_required') is not True:
            raise ValueError('Hidden-answer rows must explicitly require official scoring')
        if hidden_gold_present(row):
            raise ValueError('Never attach fabricated or leaked gold to hidden-answer rows')
        if candidates is not None:
            raise ValueError('Hidden open VQA must not fabricate candidates')
    elif row['answer_type'] == 'closed':
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


def hidden_gold_present(value):
    if isinstance(value, dict):
        return any(key in ('answer', 'answers', 'gold', 'lecture', 'solution') or
                   hidden_gold_present(item) for key, item in value.items())
    if isinstance(value, list):
        return any(hidden_gold_present(item) for item in value)
    return False


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
            'answers': dict(sorted(Counter(str(r['answer']) for r in rows if 'answer' in r).items())),
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


def official_registry():
    registry = json.loads(Path(__file__).with_name('official_registry.json').read_text())
    queue = registry['paper_queue']
    if len(set(queue)) != len(queue) or 'vqav2_val2014_diagnostic' in queue or 'VQAv2' in queue:
        raise ValueError('Paper queue must not include the old ambiguous/validation VQAv2 entry')
    entry = registry['benchmarks']['vqav2_testdev']
    if (queue.count('vqav2_testdev') != 1 or entry['split'] != 'test-dev2015'
            or entry['expected_questions'] != 107394 or entry['gold_available'] is not False
            or entry['official_server_scoring_required'] is not True):
        raise ValueError('Invalid official VQAv2 test-dev protocol')
    return registry


def official_benchmark_path(identifier, data_root=None):
    registry = official_registry()
    if identifier not in registry['paper_queue']:
        raise ValueError('Not an enabled paper benchmark: ' + identifier)
    root = Path(data_root or registry['data_root'])
    return root / registry['benchmarks'][identifier]['directory']


def load_prepared_benchmark(identifier, data_root=None, verify_images=True):
    """Load the unified inference-only JSONL; hidden tests intentionally lack gold."""
    registry = official_registry()
    entry = registry['benchmarks'][identifier]
    root = official_benchmark_path(identifier, data_root)
    hidden = entry['gold_available'] is False
    if not hidden:
        raise ValueError('This adapter is for hidden-answer paper benchmarks only')
    if (root / 'answers.json').exists():
        raise ValueError('Hidden test-dev directory must not contain answers.json')
    audit = json.loads((root / entry['audit_filename']).read_text())
    if (audit['status'] != 'COMPLETE' or audit['gold_available'] is not False
            or audit['official_server_scoring_required'] is not True
            or audit['actual_questions'] != entry['expected_questions']
            or audit['split'] != entry['split']):
        raise ValueError('Hidden benchmark has not passed complete data validation')
    inputs = root / 'inputs.jsonl'
    if sha256(inputs) != audit['inputs_sha256']:
        raise ValueError('Inference manifest SHA changed')
    if sha256(root / 'source_manifest.json') != audit['source_manifest_sha256']:
        raise ValueError('Frozen source manifest SHA changed')
    with inputs.open() as f:
        rows = [json.loads(line) for line in f]
    audit_rows(rows)
    if len(rows) != entry['expected_questions']:
        raise ValueError('Unexpected official question count')
    manifest_path = root / 'image_manifest.json'
    if sha256(manifest_path) != audit['image_manifest_sha256']:
        raise ValueError('Image manifest SHA changed')
    image_rows = json.loads(manifest_path.read_text())
    infos = {i['image_id']: i for i in image_rows}
    if len(infos) != len(image_rows):
        raise ValueError('Duplicate image ID in hidden test manifest')
    used = set()
    for row in rows:
        if row['dataset'] != entry['dataset'] or row['split'] != entry['split']:
            raise ValueError('Mixed dataset/split in paper manifest')
        if (row['answer_type'] != 'hidden' or row.get('gold_available') is not False
                or row.get('official_server_scoring_required') is not True):
            raise ValueError('Public/fabricated gold row in hidden benchmark')
        info = infos[row['image_id']]
        if row['image_path'] != info['image_path'] or row['image_sha256'] != info['image_sha256']:
            raise ValueError('Broken frozen image reference')
        if verify_images and row['image_id'] not in used:
            if sha256(row['image_path']) != row['image_sha256']:
                raise ValueError('Image hash mismatch')
        used.add(row['image_id'])
    if used != set(infos):
        raise ValueError('Extra or missing images in hidden test manifest')
    return rows

#!/usr/bin/env python3
"""CPU-only, revision-pinned download, lossless export and validation.

Requires huggingface_hub, pyarrow and Pillow. No torch/model imports.
Only test (OCRBench) and validation (DocVQA/InfoVQA) shards are selected.
"""
import argparse
import collections
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '20'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '10'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

GIB = 2**30
SPECS = [
    ('OCRBench-v1', 'echo840/OCRBench', 'test', 1000),
    ('DocVQA-val', 'vikhyatk/docvqa-val', 'validation', 5349),
    ('InfoVQA-val', 'mm-eval/InfographicVQA', 'validation', 2801),
]
PINS = {
    'echo840/OCRBench': '92a54bd1384387c178d5a07140a2d85e0a3d12e1',
    'vikhyatk/docvqa-val': '94ab3ed9c10895c28f77a17847e8202e858628e6',
    'mm-eval/InfographicVQA': 'c8e35ef221f0836de9c467e6da181b38d44b552a',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def save(path, value):
    """Idempotent without silently overwriting a conflicting existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encoded(value)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError('Conflicting existing file: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(data)


def event(audit, name, value):
    with (audit / name).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


class SpaceError(RuntimeError):
    pass


class Guard:
    def __init__(self, root, audit, floor):
        self.root, self.audit, self.floor = root, audit, floor

    def check(self, name, need=0):
        v = os.statvfs(self.root)
        free = v.f_bavail * v.f_frsize
        permitted = free - need >= self.floor + 64 * 2**20
        event(self.audit, 'disk_events.jsonl', dict(
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            event=name, free_bytes=free, need_bytes=need,
            floor_bytes=self.floor, permitted=permitted))
        if not permitted:
            raise SpaceError(f'{name}: available={free}, need={need}, floor={self.floor}')
        return free


def api_json(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last = None
    for _ in range(2):
        try:
            with opener.open(url, timeout=15) as response:
                return json.load(response)
        except Exception as exc:
            last = exc
    raise RuntimeError(f'Metadata unavailable: {url}: {last}')


def metadata(repo, split, audit):
    path = audit / (repo.replace('/', '__') + '.json')
    if path.exists():
        result = json.loads(path.read_text())
    else:
        errors = []
        for endpoint in ('https://huggingface.co', 'https://hf-mirror.com'):
            try:
                result = api_json(f'{endpoint}/api/datasets/{repo}/revision/{PINS[repo]}?blobs=true')
                result['_metadata_endpoint'] = endpoint
                result['_metadata_errors'] = errors
                save(path, result)
                break
            except Exception as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError('; '.join(errors))
    revision = result['sha']
    if revision != PINS[repo] or not re.fullmatch('[0-9a-f]{40}', revision) or result['id'] != repo:
        raise ValueError('Invalid repository/revision metadata')
    files = [x for x in result['siblings']
             if re.fullmatch(r'data/' + split + r'-\d+-of-\d+\.parquet', x['rfilename'])]
    if not files or any(not x.get('lfs', {}).get('sha256') for x in files):
        raise ValueError('Missing selected-split LFS checksums')
    return result, sorted(files, key=lambda x: x['rfilename'])


def hf_worker(repo, revision, filename, target, endpoint):
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo, filename, repo_type='dataset', revision=revision,
                    local_dir=target, endpoint=endpoint, token=False,
                    etag_timeout=10)


def download(repo, revision, item, root, audit, guard):
    raw = root / 'raw'
    target = raw / item['rfilename']
    n = item['size']
    expected = item['lfs']['sha256']
    if target.exists():
        if target.stat().st_size != n or sha(target) != expected:
            raise RuntimeError('Existing raw file fails hash: ' + str(target))
        reused = True
    else:
        reused = False
        guard.check('download:' + item['rfilename'], n + 128 * 2**20)
        raw.mkdir(exist_ok=True, parents=True)
        errors = []
        for attempt in range(2):
            endpoint = 'https://hf-mirror.com'
            command = [sys.executable, __file__, '--hf-worker', repo, revision,
                       item['rfilename'], str(raw), endpoint]
            with (audit / 'transfer.log').open('ab') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
                deadline = time.monotonic() + 600
                stop = None
                try:
                    while proc.poll() is None:
                        try:
                            guard.check('monitor:' + item['rfilename'])
                        except SpaceError as exc:
                            stop = exc
                            break
                        if time.monotonic() > deadline:
                            stop = RuntimeError('Download deadline exceeded')
                            break
                        time.sleep(2)
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                if isinstance(stop, SpaceError):
                    raise stop
            if not stop and proc.returncode == 0 and target.exists():
                break
            errors.append(dict(attempt=attempt + 1, exit=proc.returncode, error=str(stop)))
        if not target.exists() or target.stat().st_size != n or sha(target) != expected:
            raise RuntimeError('Download/hash failure: ' + str(target) + ': ' + str(errors))
    result = dict(repository=repo, revision=revision, filename=item['rfilename'],
                  original_url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{item["rfilename"]}',
                  transport_url=f'https://hf-mirror.com/datasets/{repo}/resolve/{revision}/{item["rfilename"]}',
                  local_path=str(target), bytes=n, sha256=expected,
                  verification='HF LFS SHA256 + exact length', reused=reused)
    event(audit, 'downloads.jsonl', result)
    result.pop('reused')
    return target, result


def image_bytes(value):
    if not isinstance(value, dict) or not isinstance(value.get('bytes'), bytes):
        raise ValueError('Expected embedded image bytes; no implicit external image fetching')
    if not value['bytes']:
        raise ValueError('Empty embedded image')
    return value['bytes']


def valid_answers(value):
    if not isinstance(value, list) or not value:
        raise ValueError('Expected non-empty answer list')
    if any(not isinstance(a, str) or not a.strip() for a in value):
        raise ValueError('Invalid reference answer; no silent filtering')
    return value


def question_from_messages(value):
    messages = json.loads(value) if isinstance(value, str) else value
    if not isinstance(messages, list):
        raise ValueError('Unknown messages schema')
    user = [m for m in messages if m.get('role') in ('user', 'human')]
    if len(user) != 1:
        raise ValueError('Expected exactly one user question; preserve source before adapting multi-turn')
    message = user[0]
    if 'question' in message:
        if 'content' in message and message['content'] != message['question']:
            raise ValueError('Conflicting question/content message fields')
        content = message['question']
    elif 'content' in message:
        content = message['content']
    else:
        raise ValueError('No recognized user question field')
    if isinstance(content, str):
        question = content
    elif isinstance(content, list):
        text = []
        for part in content:
            if part.get('type') in ('text', 'input_text'):
                text.append(part['text'])
            elif part.get('type') not in ('image', 'image_url', 'input_image'):
                raise ValueError('Unknown non-text message part: ' + repr(part))
        question = '\n'.join(text)
    else:
        raise ValueError('Unknown user content schema')
    if not question.strip():
        raise ValueError('Empty question')
    return question


def normalized_questions(name, split, row_index, row, media):
    prefix = f'{name}:{split}:row{row_index:06d}'
    if name == 'DocVQA-val':
        if not row['qa']:
            raise ValueError('Empty DocVQA QA list')
        values = [(f'{prefix}:qa{j:03d}', qa['question'], valid_answers(qa['answers']),
                   dict(source_row_index=row_index, source_qa_index=j,
                        original_id=qa.get('id'), image_original_path=row['image'].get('path')))
                  for j, qa in enumerate(row['qa'])]
    elif name == 'OCRBench-v1':
        values = [(prefix, row['question'], valid_answers(row['answer']),
                   dict(source_row_index=row_index, original_id=row.get('id'),
                        question_type=row['question_type'], original_dataset=row['dataset']))]
    else:
        values = [(prefix, question_from_messages(row['messages']), valid_answers(row['answer']),
                   dict(source_row_index=row_index, original_id=row.get('id'),
                        question_type=row.get('question_type'), media_count=len(media),
                        multi_image=len(media) > 1))]
    for qid, question, answers, meta in values:
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Empty question; refusing to drop record')
        first = media[0]
        record = dict(dataset=name, split=split, question_id=qid,
                      image_id=first['image_id'], image_path=first['image_path'],
                      image_sha256=first['sha256'], question=question,
                      metadata=meta, images=[dict(image_id=m['image_id'], image_path=m['image_path'],
                                                image_sha256=m['sha256']) for m in media])
        yield record, answers


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p / 100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def image_stats(images):
    return {field: {f'P{p}': percentile([m[field] for m in images], p) for p in (50, 90, 95, 99, 100)}
            for field in ('bytes', 'width', 'height', 'pixels')}


def allocated(root):
    return sum(p.lstat().st_blocks * 512 for p in [root, *root.rglob('*')] if not p.is_symlink())


def export_dataset(name, split, expected, files, root, guard, result):
    import pyarrow.parquet as pq
    from PIL import Image
    guard.check('export_start:' + name, 64 * 2**20)
    images = {}
    inputs, answers, full, source_records = [], {}, [], []
    question_types, categories = collections.Counter(), collections.Counter()
    row_index = 0
    schemas = []
    original_image_refs = 0
    source_path_keys = set()
    for path in files:
        parquet = pq.ParquetFile(path)
        schemas.append(dict(path=str(path), schema=str(parquet.schema_arrow), rows=parquet.metadata.num_rows))
        for batch in parquet.iter_batches(batch_size=4):
            for row in batch.to_pylist():
                values = row['media'] if name == 'InfoVQA-val' else [row['image']]
                if not isinstance(values, list) or not values:
                    raise ValueError('No media; refusing to skip record')
                media = []
                for index, value in enumerate(values):
                    data = image_bytes(value)
                    h = digest(data)
                    if h not in images:
                        with Image.open(io.BytesIO(data)) as im:
                            im.load()
                            width, height, fmt = im.width, im.height, im.format
                        if width <= 0 or height <= 0:
                            raise ValueError('Invalid image dimensions')
                        ext = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp', 'TIFF': 'tiff', 'GIF': 'gif', 'BMP': 'bmp'}.get(fmt)
                        if not ext:
                            raise ValueError('Unrecognized original encoding: ' + str(fmt))
                        destination = root / 'images' / (h + '.' + ext)
                        if destination.exists():
                            if sha(destination) != h:
                                raise RuntimeError('Conflicting exported image')
                        else:
                            guard.check('export_image:' + name, len(data) + 8192)
                            destination.parent.mkdir(exist_ok=True)
                            with destination.open('xb') as stream:
                                stream.write(data)
                        if sha(destination) != h:
                            raise RuntimeError('Written image hash mismatch')
                        with Image.open(destination) as im:
                            im.load()
                            if im.size != (width, height) or im.format != fmt:
                                raise RuntimeError('Exported image validation mismatch')
                        images[h] = dict(image_id=h, image_path=str(destination), sha256=h,
                                         bytes=len(data), width=width, height=height, pixels=width*height,
                                         format=fmt, conversion=False, source_refs=[])
                    info = images[h]
                    info['source_refs'].append(dict(row_index=row_index, media_index=index, original_path=value.get('path')))
                    source_path_keys.add((value.get('path'), h))
                    media.append(info)
                    original_image_refs += 1
                original = dict(row)
                key = 'media' if name == 'InfoVQA-val' else 'image'
                replacements = []
                for value, info in zip(values, media):
                    replacement = dict(value)
                    replacement['bytes'] = dict(sha256=info['sha256'], exported_path=info['image_path'],
                                                original_encoding_preserved=True)
                    replacements.append(replacement)
                original[key] = replacements if key == 'media' else replacements[0]
                source_records.append(dict(source_row_index=row_index, source_file=str(path), original=original))
                for record, gold in normalized_questions(name, split, row_index, row, media):
                    qid = record['question_id']
                    if qid in answers:
                        raise ValueError('Duplicate normalized question ID')
                    inputs.append(record)
                    answers[qid] = dict(answers=gold, original_id=record['metadata'].get('original_id'))
                    full.append(dict(record, answers=gold))
                    question_types[str(record['metadata'].get('question_type', 'NOT_PROVIDED'))] += 1
                    if 'original_dataset' in record['metadata']:
                        categories[record['metadata']['original_dataset']] += 1
                row_index += 1
    if len(inputs) != expected:
        raise ValueError(f'Unexpected QA count: {len(inputs)} != {expected}; no samples dropped')
    if name == 'DocVQA-val' and row_index != 1286:
        raise ValueError('Unexpected DocVQA source image-row count')
    if name == 'OCRBench-v1' and len(question_types) != 10:
        raise ValueError('OCRBench question types differ from expected original 10')
    guard.check('write_annotations:' + name, sum(len(json.dumps(v, ensure_ascii=False).encode()) for v in full) * 5 + 8*2**20)
    for filename, rows in [('inputs.jsonl', inputs), ('records_with_answers.jsonl', full),
                           ('source_records_with_answers.jsonl', source_records)]:
        path = root / filename
        body = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
        if path.exists():
            if path.read_bytes() != body:
                raise FileExistsError('Conflicting existing normalized data: ' + str(path))
        else:
            with path.open('xb') as stream:
                stream.write(body)
    save(root / 'answers.json', answers)
    save(root / 'image_manifest.json', list(images.values()))
    stats = image_stats(list(images.values()))
    result.update(status='COMPLETE', actual_qa=len(inputs), source_rows=row_index,
                  image_references=sum(len(r['images']) for r in inputs),
                  source_image_references=original_image_refs,
                  unique_images=len(images), unique_image_sha256=len(images),
                  unique_source_path_and_sha_pairs=len(source_path_keys),
                  image_stats=stats, question_type_distribution=dict(question_types),
                  original_dataset_distribution=dict(categories), parquet_schemas=schemas,
                  byte_preserving_export=True, answers_separated=True,
                  validations=dict(all_images_decoded=True, hashes_verified=True, unique_question_ids=True,
                                   nonempty_questions=True, nonempty_answers=True, no_missing_images=True,
                                   expected_qa_count=True, all_qa_expanded=True, selected_split_only=True,
                                   original_metadata_and_media_preserved=True),
                  image_disk_bytes=sum(v['bytes'] for v in images.values()))
    save(root / 'data_audit.json', result)
    result['allocated_bytes'] = allocated(root)
    return result


def run(args):
    root, audit = Path(args.root).resolve(), Path(args.audit).resolve()
    root.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    guard = Guard(root, audit, int(args.floor_gib * GIB))
    guard.check('task_start')
    results = []
    for name, repo, split, expected in SPECS:
        directory = root / name
        directory.mkdir(exist_ok=True)
        result = dict(dataset=name, repository=repo, split=split, expected_qa=expected,
                      local_path=str(directory), actual_qa=0, unique_images=0, status='MISSING')
        print('START', name, flush=True)
        try:
            meta, items = metadata(repo, split, audit)
            result.update(revision=meta['sha'], metadata_endpoint=meta['_metadata_endpoint'],
                          source_download_bytes=sum(x['size'] for x in items))
            existing_audit = directory / 'data_audit.json'
            if existing_audit.exists():
                old = json.loads(existing_audit.read_text())
                if old['revision'] != meta['sha']:
                    raise RuntimeError('Existing exported revision conflict')
                from finalize import verify
                verify(old)
                old['allocated_bytes'] = allocated(directory)
                results.append(old)
                event(audit, 'dataset_runs.jsonl', old)
                (audit / 'run_results.json').write_bytes(encoded(results))
                print('REVALIDATED', name, old['status'], old['actual_qa'], flush=True)
                continue
            need = sum(x['size'] for x in items if not (directory / 'raw' / x['rfilename']).exists())
            guard.check('raw_split_reservation:' + name, need + 128 * 2**20)
            files, downloads = [], []
            for item in items:
                path, entry = download(repo, meta['sha'], item, directory, audit, guard)
                files.append(path)
                downloads.append(entry)
            result['source_files'] = downloads
            # Read embedded lengths and hashes before export; reserve exact deduplicated bytes.
            import pyarrow.parquet as pq
            unique = {}
            for path in files:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=4, columns=['media' if name == 'InfoVQA-val' else 'image']):
                    for row in batch.to_pylist():
                        for value in (row['media'] if name == 'InfoVQA-val' else [row['image']]):
                            data = image_bytes(value)
                            unique[digest(data)] = len(data)
            need = sum(n + 8192 for h, n in unique.items() if not list((directory / 'images').glob(h + '.*')))
            result['export_peak_additional_bytes'] = need + 64 * 2**20
            guard.check('exact_export_reservation:' + name, need + 64 * 2**20)
            result = export_dataset(name, split, expected, files, directory, guard, result)
        except Exception as exc:
            result.update(status='BLOCKED' if isinstance(exc, SpaceError) else 'FAILED',
                          error=repr(exc), allocated_bytes=allocated(directory))
            print('BLOCKED_OR_FAILED', name, repr(exc), flush=True)
        results.append(result)
        event(audit, 'dataset_runs.jsonl', result)
        # The latest task summary is separate from immutable dataset evidence.
        latest = audit / 'run_results.json'
        latest.write_bytes(encoded(results))
        print('RESULT', name, result['status'], result.get('actual_qa'), result.get('allocated_bytes'), flush=True)
    guard.check('task_end')
    return 0 if all(r['status'] == 'COMPLETE' for r in results) else 2


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--hf-worker':
        hf_worker(*sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--root', default='/data/lk/ATE-benchmarks-20261105/datasets')
        parser.add_argument('--audit', required=True)
        parser.add_argument('--floor-gib', type=float, default=15)
        sys.exit(run(parser.parse_args()))

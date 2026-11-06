#!/usr/bin/env python3
"""Prepare only official VQA v2 test-dev2015, without torch or GPU imports."""
import argparse
import ast
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import zipfile
import zlib

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import requests
from PIL import Image

GIB = 2**30
REVISION = 'c121f0432da27facab705978f83c4ada465e46fd'
QUESTION_URL = 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip'
IMAGE_ZIP = 'https://s3.amazonaws.com/images.cocodataset.org/zips/test2015.zip'
IMAGE_BASE = 'https://s3.amazonaws.com/images.cocodataset.org/test2015/'
EXPECTED = 107394
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = Path('/data/lk/ATE-benchmarks-20261105/datasets/VQAv2-testdev2015')
DEFAULT_SOURCE = Path('/data1/likai/ATE-vqav2-testdev2015-sources-20261107')
TLS = threading.local()
STOP = threading.Event()
FLOOR = 8 * GIB


def session():
    if not hasattr(TLS, 'session'):
        TLS.session = requests.Session()
        TLS.session.trust_env = False
    return TLS.session


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError('Conflicting existing artifact: ' + str(path))
    else:
        with path.open('xb') as f:
            f.write(data)


def event(path, obj):
    with path.open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def disk(path):
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def guard(root, audit, name, need=0, floor=FLOOR):
    free = disk(root)
    event(audit / 'disk_events.jsonl', dict(time=time.time(), event=name, free_bytes=free,
                                          need_bytes=need, floor_bytes=floor))
    if free - need < floor + 64 * 2**20:
        raise RuntimeError(f'DISK_BLOCKED {name}: free={free}, need={need}, floor={floor}')
    return free


def fetch(url, destination, limit, headers=None):
    """Bounded HTTP streaming with range resume and exclusive partial ownership."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    partial = destination.with_suffix(destination.suffix + '.part')
    last = None
    for attempt in range(2):
        if url.startswith(IMAGE_BASE) and (STOP.is_set() or disk(destination.parent) < FLOOR + 64*2**20):
            STOP.set()
            raise RuntimeError('Image download stopped at disk safety floor')
        offset = partial.stat().st_size if partial.exists() else 0
        request_headers = dict(headers or {})
        if offset:
            request_headers['Range'] = f'bytes={offset}-'
        try:
            with session().get(url, headers=request_headers, stream=True, timeout=(8, 20)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise RuntimeError('Server did not honor exact resume range')
                size = offset
                with partial.open('ab' if offset else 'xb') as f:
                    start = time.monotonic()
                    for chunk in response.iter_content(256 * 1024):
                        if url.startswith(IMAGE_BASE) and (STOP.is_set() or disk(destination.parent) < FLOOR + 64*2**20):
                            STOP.set()
                            raise RuntimeError('Image transfer stopped at disk safety floor')
                        size += len(chunk)
                        if size > limit or time.monotonic() - start > 300:
                            raise RuntimeError('Download size/deadline limit')
                        f.write(chunk)
            partial.rename(destination)
            return destination
        except Exception as exc:
            last = exc
    raise RuntimeError(f'Download failed after two attempts: {url}: {last}')


def remote_headers(url):
    last = None
    for _ in range(2):
        try:
            with session().head(url, timeout=(8, 20), allow_redirects=True) as response:
                response.raise_for_status()
                return dict(response.headers)
        except Exception as exc:
            last = exc
    raise RuntimeError(f'HEAD failed: {url}: {last}')


class RemoteZip(io.RawIOBase):
    def __init__(self, url, length, etag):
        self.url, self.length, self.etag, self.position = url, length, etag, 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        return self.position

    def read(self, count=-1):
        count = self.length - self.position if count < 0 else min(count, self.length - self.position)
        if not count:
            return b''
        if count < 0 or count > 32 * 2**20:
            raise RuntimeError('Unexpected ZIP metadata range size')
        start, end = self.position, self.position + count - 1
        last = None
        for _ in range(2):
            try:
                with session().get(self.url, headers={'Range': f'bytes={start}-{end}', 'If-Match': self.etag},
                                   timeout=(8, 30)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.length}':
                        raise RuntimeError('Incorrect Content-Range')
                    data = response.content
                    if len(data) != count:
                        raise RuntimeError('Truncated ZIP metadata range')
                    self.position += count
                    return data
            except Exception as exc:
                last = exc
        raise RuntimeError('Remote ZIP range failed: ' + repr(last))


def read_official_questions(archive):
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Official question ZIP CRC failure')
        dev_name = 'v2_OpenEnded_mscoco_test-dev2015_questions.json'
        full_name = 'v2_OpenEnded_mscoco_test2015_questions.json'
        dev = json.loads(z.read(dev_name))
        full = json.loads(z.read(full_name))
        manifest = [dict(name=i.filename, bytes=i.file_size, crc32=i.CRC) for i in z.infolist()]
    if dev.get('data_subtype') != 'test-dev2015' or full.get('data_subtype') != 'test2015':
        raise ValueError('Wrong official split metadata')
    questions, complete = dev['questions'], full['questions']
    if len(questions) != EXPECTED or len(complete) != 447793:
        raise ValueError('Wrong official split count')
    lookup = {q['question_id']: q for q in complete}
    if len(lookup) != len(complete) or len({q['question_id'] for q in questions}) != EXPECTED:
        raise ValueError('Duplicate official question IDs')
    for q in questions:
        if lookup.get(q['question_id']) != q or not q['question'].strip():
            raise ValueError('Test-dev question differs from official Test2015')
        if q['question_id'] // 1000 != q['image_id']:
            raise ValueError('Official Question ID / Image ID mismatch')
    return questions, complete, manifest


def inspect_image(path, expected):
    data = path.read_bytes()
    if len(data) != expected['bytes'] or zlib.crc32(data) & 0xffffffff != expected['crc32']:
        raise ValueError('Image size/CRC differs from official COCO ZIP: ' + str(path))
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        width, height, fmt = im.width, im.height, im.format
    if fmt != 'JPEG' or min(width, height) <= 0:
        raise ValueError('Invalid original COCO image')
    return dict(image_id=str(expected['image_id']), image_path=str(path),
                image_sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                width=width, height=height, format=fmt, crc32=expected['crc32'],
                url=IMAGE_BASE + path.name, original_encoding_preserved=True)


def download_image(expected, directory):
    path = directory / expected['filename']
    if path.exists():
        return inspect_image(path, expected)
    fetch(IMAGE_BASE + expected['filename'], path, expected['bytes'])
    try:
        return inspect_image(path, expected)
    except Exception:
        # Never erase a conflicting completed image; preserve it for audit.
        raise


def old_snapshot():
    root = Path('/data/lk/ATE-benchmarks-20261105/datasets/VQAv2')
    return {str(root / name): sha(root / name) for name in ('inputs.jsonl', 'answers.json')}


def prepare(args):
    root, sources, audit = Path(args.root).resolve(), Path(args.sources).resolve(), Path(args.audit).resolve()
    audit.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    owner = sources / 'ownership.json'
    ownership = dict(task='vqav2-testdev2015-alignment-20261107', root=str(root), sources=str(sources))
    if root.exists() and not owner.exists():
        raise FileExistsError('Existing dataset without task ownership')
    save(owner, ownership)
    root.mkdir(parents=True, exist_ok=True)
    before = guard(root, audit, 'task_start')
    baseline = sources / 'baseline.json'
    if not baseline.exists():
        save(baseline, dict(free_bytes=before, old_val2014_files=old_snapshot()))
    zip_path = sources / 'v2_Questions_Test_mscoco.zip'
    headers = remote_headers(QUESTION_URL)
    save(sources / 'question_zip_headers.json', headers) if not (sources / 'question_zip_headers.json').exists() else None
    fetch(QUESTION_URL, zip_path, 16 * 2**20)
    zip_hash = sha(zip_path)
    if zip_path.stat().st_size != int(headers['Content-Length']):
        raise ValueError('Question ZIP length mismatch')
    etag = headers['ETag'].strip('"')
    if re.fullmatch('[0-9a-f]{32}', etag) and hashlib.md5(zip_path.read_bytes()).hexdigest() != etag:
        raise ValueError('Question ZIP ETag/MD5 mismatch')
    questions, full, zip_manifest = read_official_questions(zip_path)
    print('OFFICIAL_QUESTIONS', len(questions), 'FULL_SUBMISSION_IDS', len(full), flush=True)
    official_files = []
    for rel in ('scripts/v1_5/eval/vqav2.sh', 'scripts/convert_vqav2_for_submission.py',
                'docs/Evaluation.md', 'llava/eval/m4c_evaluator.py'):
        target = sources / 'LLaVA' / rel
        url = f'https://api.github.com/repos/haotian-liu/LLaVA/contents/{rel}?ref={REVISION}'
        fetch(url, target, 256*1024, {'Accept': 'application/vnd.github.raw+json'})
        official_files.append(dict(repository='haotian-liu/LLaVA', revision=REVISION,
                                   path=rel, url=url, local_path=str(target), sha256=sha(target)))
    official_module = ast.parse((sources / 'LLaVA/llava/eval/m4c_evaluator.py').read_text())
    local_module = ast.parse((ROOT / 'vlm_research/benchmarks/_vendor/m4c_evaluator.py').read_text())
    get_class = lambda m: next(n for n in m.body if isinstance(n, ast.ClassDef) and n.name == 'EvalAIAnswerProcessor')
    normalizer_matches = ast.dump(get_class(official_module)) == ast.dump(get_class(local_module))
    if not normalizer_matches:
        raise ValueError('Existing EvalAI normalizer differs from pinned LLaVA class')
    wanted = {q['image_id'] for q in questions}
    image_headers_path = sources / 'image_zip_headers.json'
    if image_headers_path.exists():
        image_headers = json.loads(image_headers_path.read_text())
    else:
        image_headers = remote_headers(IMAGE_ZIP)
        save(image_headers_path, image_headers)
    remote = RemoteZip(IMAGE_ZIP, int(image_headers['Content-Length']), image_headers['ETag'])
    with zipfile.ZipFile(remote) as z:
        infos = {i.filename: i for i in z.infolist()}
        selected = []
        for image_id in sorted(wanted):
            name = f'COCO_test2015_{image_id:012d}.jpg'
            i = infos['test2015/' + name]
            if i.is_dir() or i.file_size <= 0:
                raise ValueError('Invalid official ZIP member')
            selected.append(dict(image_id=image_id, filename=name, bytes=i.file_size, crc32=i.CRC))
    save(sources / 'selected_coco_zip_members.json', selected)
    image_dir = root / 'images' / 'test2015'
    image_dir.mkdir(parents=True, exist_ok=True)
    needed = sum(x['bytes'] + 4096 for x in selected if not (image_dir / x['filename']).exists())
    print('SELECTED_IMAGES', len(selected), 'BYTES', sum(x['bytes'] for x in selected),
          'FREE_GIB', disk(root)/GIB, flush=True)
    guard(root, audit, 'all_images_and_manifests_peak', needed + 256 * 2**20)
    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_image, item, image_dir): item for item in selected}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(dict(image_id=item['image_id'], error=repr(exc)))
            if (len(results) + len(failures)) % 500 == 0:
                guard(root, audit, 'images:' + str(len(results)))
                print('IMAGES', len(results), 'FAILED', len(failures), 'FREE_GIB', round(disk(root)/GIB, 3), flush=True)
    if failures:
        save(audit / 'image_download_failures.json', failures)
        raise RuntimeError(f'Images incomplete: {len(failures)}; rerun resumes exact missing files')
    images = {int(i['image_id']): i for i in results}
    save(root / 'image_manifest.json', [images[k] for k in sorted(images)])
    input_path = root / 'inputs.jsonl'
    output_temp = sources / 'canonical_testdev_inputs.jsonl.tmp'
    with output_temp.open('w') as f:
        for index, q in enumerate(questions):
            info = images[q['image_id']]
            record = dict(dataset='VQAv2', subset='testdev', split='test-dev2015',
                          question_id=str(q['question_id']), image_id=str(q['image_id']),
                          image_path=info['image_path'], image_sha256=info['image_sha256'],
                          question=q['question'], candidates=None, answer_type='hidden',
                          gold_available=False, official_server_scoring_required=True,
                          source_revision=zip_hash, source_row_sha256=content_sha(q),
                          metadata=dict(source_row_index=index, official_question_id=q['question_id'],
                                        coco_split='test2015', official_source_sha256=zip_hash))
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    temp_hash = sha(output_temp)
    if input_path.exists():
        if sha(input_path) != temp_hash:
            raise FileExistsError('Conflicting existing inputs manifest')
    else:
        guard(root, audit, 'copy_canonical_inputs', output_temp.stat().st_size + 2**20)
        with output_temp.open('rb') as src, input_path.open('xb') as dst:
            import shutil
            shutil.copyfileobj(src, dst, 8*2**20)
    if sha(input_path) != temp_hash:
        raise ValueError('Canonical input copy SHA mismatch')
    cleanup = dict(task_owned_only=True, protected_old_val2014=True, planned_deletions=[
        dict(path=str(output_temp), bytes=output_temp.stat().st_size, sha256=temp_hash,
             reason='Exact verified duplicate of final inputs.jsonl', replacement=str(input_path))],
        deleted=[], preserved=[dict(path=str(zip_path), reason='Authoritative official question archive used by submission exporter')])
    save(audit / 'cleanup_manifest_planned.json', cleanup)
    if output_temp.parent != sources or sha(output_temp) != temp_hash:
        raise ValueError('Cleanup ownership/hash mismatch')
    output_temp.unlink()
    cleanup['deleted'] = cleanup['planned_deletions']
    save(audit / 'prepare_cleanup_manifest.json', cleanup)
    combined_cleanup = audit / 'cleanup_manifest.json'
    if not combined_cleanup.exists():
        save(combined_cleanup, cleanup)
    else:
        existing_cleanup = json.loads(combined_cleanup.read_text())
        if (existing_cleanup['planned_deletions'] != cleanup['planned_deletions'] or
                not all(item in existing_cleanup['deleted'] for item in cleanup['deleted'])):
            raise ValueError('Conflicting previous cleanup record')
    source_manifest = dict(benchmark='VQAv2', version='2.0', split='test-dev2015',
        question_source=dict(url=QUESTION_URL, local_path=str(zip_path), sha256=zip_hash,
                             etag=headers['ETag'], bytes=zip_path.stat().st_size, zip_members=zip_manifest,
                             official_testdev_subset_verified_against_full_test2015=True),
        images=dict(url=IMAGE_ZIP, etag=image_headers['ETag'], bytes=int(image_headers['Content-Length']),
                    download_method='Only selected official individual JPEGs; exact ZIP member size and CRC32 checked',
                    selected_members_sha256=sha(sources / 'selected_coco_zip_members.json')),
        llava_sources=official_files, llava_answer_processor_ast_matches_existing_vendor=normalizer_matches,
        llava_eval_zip=dict(url='https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view',
                            status='UNAVAILABLE_CONNECTION_TIMEOUT', fallback='VQA official test question archive, not a third-party question mirror'),
        gold_available=False, official_server_scoring_required=True,
        official_submission_ids=447793, inference_questions=EXPECTED)
    save(root / 'source_manifest.json', source_manifest)
    if old_snapshot() != json.loads(baseline.read_text())['old_val2014_files']:
        raise ValueError('Protected val2014 annotation snapshot changed')
    audit_value = dict(status='COMPLETE', dataset='VQAv2', version='2.0', split='test-dev2015',
        actual_questions=EXPECTED, unique_images=len(images), missing_images=0,
        duplicate_question_ids=0, gold_available=False, official_server_scoring_required=True,
        inputs_sha256=sha(input_path), image_manifest_sha256=sha(root / 'image_manifest.json'),
        source_manifest_sha256=sha(root / 'source_manifest.json'), local_path=str(root),
        old_val2014_preserved=True, model_predictions_generated=0, gpu_used=False)
    save(root / 'dataset_audit.json', audit_value)
    guard(root, audit, 'prepare_complete')
    print('COMPLETE', json.dumps(audit_value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(DEFAULT_DATA))
    parser.add_argument('--sources', default=str(DEFAULT_SOURCE))
    parser.add_argument('--audit', default=str(ROOT / 'vlm_research/results/vqav2_testdev2015_alignment'))
    parser.add_argument('--workers', type=int, default=16)
    prepare(parser.parse_args())

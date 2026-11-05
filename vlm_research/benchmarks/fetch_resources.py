"""Selected official images only. ZIP parsing/CRC is delegated to stdlib zipfile."""
import argparse
import io
import json
from pathlib import Path
import shutil
import zipfile

import requests

from .acquire import download
from .dataset_registry import DEFAULT_RESOURCES, OUTPUT, digest, freeze_json, select_groups, select_unique, sha256
from .pope_adapter import read_pope
from .scienceqa_adapter import scienceqa_rows
from .scienceqa_adapter import MATCH_FIELDS, verify_publisher_order
from .textvqa_adapter import textvqa_rows


class RangeReader(io.RawIOBase):
    """Seekable bounded HTTP object with pinned ETag, used by ZipFile."""
    def __init__(self, session, url):
        self.session, self.url, self.position = session, url, 0
        r = session.head(url, timeout=(5, 30))
        r.raise_for_status()
        self.size = int(r.headers['Content-Length'])
        self.etag = r.headers['ETag']
        self.version = r.headers.get('x-amz-version-id')
        self.records = []
        self.transferred_bytes = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        new = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if new < 0:
            raise ValueError('Negative seek')
        self.position = new
        return new

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if size <= 0:
            return b''
        if size > 16 * 1024**2 or self.transferred_bytes + size > 512 * 1024**2:
            raise ValueError('Explicit ZIP range budget exceeded')
        start, end = self.position, self.position+size-1
        r = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}', 'If-Match': self.etag},
                             timeout=(5, 45), stream=True)
        with r:
            if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Server ignored range; refuse full archive transfer')
            if r.headers.get('ETag') != self.etag:
                raise ValueError('ZIP ETag changed')
            content = r.raw.read(size+1)
        if len(content) != size:
            raise ValueError('Truncated/oversized ZIP range')
        self.records.append({'start': start, 'end': end, 'sha256': __import__('hashlib').sha256(content).hexdigest()})
        self.transferred_bytes += size
        self.position += size
        return content


def selected_inputs(root):
    sources = root/'sources'
    pope_rev = '08d957b917e5a378a2f99d35b6293c536a66298b'
    pope = {strategy: read_pope(sources/f'pope/output/coco/coco_pope_{strategy}.json', strategy,
                              pope_rev, root/'images/coco/val2014')
            for strategy in ('random', 'popular', 'adversarial')}
    image_sets = [{r['image_id'] for r in rows} for rows in pope.values()]
    if not all(ids == image_sets[0] for ids in image_sets):
        raise ValueError('POPE image sets differ: require explicit new sampling protocol')
    common = {r['image_id'] for r in select_groups(pope['random'], 300)}
    pope_selected = {k: [r for r in rows if r['image_id'] in common] for k, rows in pope.items()}
    science = scienceqa_rows(json.loads((sources/'scienceqa/data/scienceqa/problems.json').read_text()),
                            json.loads((sources/'scienceqa/data/scienceqa/pid_splits.json').read_text()),
                            '2cbf8318e07b9ece895bb2ae605e71e38d623264', root/'images/scienceqa')
    science_selected = select_unique(science, 500)
    tv_path = sources/'textvqa/TextVQA_0.5.1_val.json'
    tv = textvqa_rows(json.loads(tv_path.read_text()), 'official-v0.5.1', root/'images/textvqa') if tv_path.exists() else []
    tv_selected = select_groups(tv, 500) if tv else []
    selections = dict(pope=pope_selected, scienceqa_img=science_selected, textvqa=tv_selected)
    lock = {'seed': 20261029, 'rule': 'POPE: shared complete random-image groups until >=300 questions; '
            'ScienceQA: sorted shuffled official image-test IDs, first500; '
            'TextVQA: sorted shuffled complete image groups until >=500 questions. No asset/output filter.',
            'question_ids': {**{f'pope_{k}': [r['question_id'] for r in v] for k, v in pope_selected.items()},
                             'scienceqa_img': [r['question_id'] for r in science_selected],
                             'textvqa': [r['question_id'] for r in tv_selected]},
            'rows_binding': digest(selections)}
    freeze_json(root/'selection_lock.json', lock)
    return pope, science, tv, selections


def fetch_textvqa(s, root, rows, records):
    url = 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip'
    reader = RangeReader(s, url)
    archive_record = dict(url=url, bytes=reader.size, etag=reader.etag, s3_version_id=reader.version,
                          archive_sha256=None, caveat='Only selected ZIP ranges downloaded; no full archive SHA256 claim')
    freeze_json(root/'textvqa_archive_identity.json', {k: archive_record[k] for k in ('url', 'bytes', 'etag', 's3_version_id')})
    try:
        with zipfile.ZipFile(reader) as z:
            members = {Path(info.filename).name: info for info in z.infolist() if not info.is_dir()}
            ids = sorted({r['image_id'] for r in rows})
            for n, iid in enumerate(ids):
                name = iid + '.jpg'
                info = members.get(name)
                path = root/'images/textvqa'/name
                if info is None:
                    records.append(dict(image_id=iid, status='failed', error='Missing official ZIP member'))
                    continue
                if path.exists():
                    records.append(dict(image_id=iid, status='present', path=str(path), sha256=sha256(path)))
                    continue
                if info.file_size > 20*1024**2 or shutil.disk_usage(root).free < 50*1024**3:
                    raise ValueError('Image size / disk safety limit')
                data = z.read(info)  # Includes zipfile CRC verification.
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as f:
                    f.write(data)
                records.append(dict(image_id=iid, status='downloaded', path=str(path), sha256=sha256(path),
                                    archive_member=info.filename, crc32=info.CRC, bytes=info.file_size))
                if n % 25 == 0:
                    print('TextVQA images', n+1, '/', len(ids), flush=True)
    finally:
        archive_record.update(range_requests=reader.records, transferred_bytes=reader.transferred_bytes)
        records.append(archive_record)


def fetch_scienceqa(s, root, selected, records):
    """HF mirror test split; match ALL public content fields to official test PID."""
    meta = json.loads((root/'sources/preflight/scienceqa_hf.json').read_text())
    rev, repo = meta['sha'], meta['id']
    if rev != 'f18b0a70359ebfb41f658fd564208d0355b013f4' or repo != 'derek-thomas/ScienceQA':
        raise ValueError('ScienceQA mirror revision changed; do not mix versions in this resource root')
    tree_url = f'https://hf-mirror.com/api/datasets/{repo}/tree/{rev}/data'
    r = s.get(tree_url, timeout=(5, 30))
    r.raise_for_status()
    files = [x for x in r.json() if x['path'].startswith('data/test-')]
    if len(files) != 1:
        raise ValueError('ScienceQA test file count changed')
    item = files[0]
    record = download(s, f'https://hf-mirror.com/datasets/{repo}/resolve/{rev}/{item["path"]}',
                      root/'sources/scienceqa_hf'/Path(item['path']).name, cap=256*1024**2)
    records.append(record)
    if record['status'] == 'failed':
        return
    if record['sha256'] != item['lfs']['oid'] or record['bytes'] != item['size']:
        raise ValueError('ScienceQA parquet LFS SHA256 mismatch')
    record.update(repository=repo, revision=rev, lfs_sha256=item['lfs']['oid'])
    import pyarrow.parquet as pq
    mirror_rows = []
    for batch in pq.ParquetFile(record['path']).iter_batches(batch_size=32):
        mirror_rows.extend(batch.to_pylist())
    problems = json.loads((root/'sources/scienceqa/data/scienceqa/problems.json').read_text())
    test_ids = json.loads((root/'sources/scienceqa/data/scienceqa/pid_splits.json').read_text())['test']
    builder = root/'sources/scienceqa_hf/tutorial/ScienceQA.py'
    if sha256(builder) != 'a0806fb01130872aecb6ede47cf4fd392b373c993b303ccc6830a1ae9a258d1e':
        raise ValueError('Publisher builder source hash differs; review mapping first')
    by_pid = verify_publisher_order(problems, test_ids, mirror_rows)
    records.append({'status': 'publisher_order_verified', 'official_test_rows': len(test_ids),
                    'fields_exact': MATCH_FIELDS, 'builder_sha256': sha256(builder),
                    'mapping_rule': 'Exact publisher problems.items() insertion order; agrees with official pid_splits.test; '
                                    'all 4241 rows and 12 content fields + image presence checked, not question-only matching'})
    for row in selected:
        data = by_pid[row['question_id']]['image']['bytes']
        path = Path(row['image_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('Existing ScienceQA image differs')
        else:
            with path.open('xb') as f:
                f.write(data)
        records.append(dict(question_id=row['question_id'], path=str(path), status='verified_extracted', sha256=sha256(path),
                            source_revision=rev, publisher_row_index=test_ids.index(row['question_id']), content_fields_matched=MATCH_FIELDS))
    print('ScienceQA extracted', sum(x.get('status') == 'verified_extracted' for x in records), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['annotations', 'pope', 'scienceqa', 'textvqa'])
    p.add_argument('--root', type=Path, default=DEFAULT_RESOURCES)
    a = p.parse_args()
    s = requests.Session()
    s.trust_env = False
    records = []
    try:
        if a.stage == 'annotations':
            records.append(download(s, 'https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json',
                                    a.root/'sources/textvqa/TextVQA_0.5.1_val.json'))
        else:
            _, _, _, selected = selected_inputs(a.root)
            if a.stage == 'textvqa':
                fetch_textvqa(s, a.root, selected['textvqa'], records)
            elif a.stage == 'scienceqa':
                fetch_scienceqa(s, a.root, selected['scienceqa_img'], records)
            else:
                for i, row in enumerate({r['image_id']: r for r in selected['pope']['random']}.values()):
                    url = f'https://s3.amazonaws.com/images.cocodataset.org/val2014/{row["image_id"]}'
                    records.append(download(s, url, row['image_path'], cap=20*1024**2))
                    print('POPE image', i+1, records[-1]['status'], flush=True)
    except Exception as e:
        records.append(dict(status='failed', error=f'{type(e).__name__}: {e}'))
        raise
    finally:
        path = a.root/f'fetch_{a.stage}_audit.json'
        history = json.loads(path.read_text()) if path.exists() else []
        history.append(records)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()

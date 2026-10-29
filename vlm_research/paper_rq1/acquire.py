"""Acquire revision-pinned annotations and selected official ZIP members only."""
from concurrent.futures import ThreadPoolExecutor
import json
import zipfile
import time

import requests

from benchmarks.acquire import download
from benchmarks.fetch_resources import RangeReader
from .inventory import RESOURCES, OUTPUT


class StrictRanges(RangeReader):
    """Stanford ignores Range when If-Match is present; validate returned ETag instead."""
    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if size <= 0:
            return b''
        if size > 16*1024**2 or self.transferred_bytes+size > 512*1024**2:
            raise ValueError('Explicit range budget exceeded')
        start, end = self.position, self.position+size-1
        for attempt in range(4):
            response = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}'}, stream=True,
                                        timeout=(5, 45))
            if response.status_code not in (429, 503) or attempt == 3:
                break
            retry = response.headers.get('Retry-After', '')
            seconds = int(retry) if retry.isdigit() else 15*2**attempt
            response.close()
            time.sleep(seconds)
        with response:
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError(f'Range not honored: status={response.status_code}, requested={start}-{end}, '
                                 f'returned={response.headers.get("Content-Range")}')
            if response.headers.get('ETag') != self.etag:
                raise ValueError('Archive changed')
            data = response.raw.read(size+1)
        if len(data) != size:
            raise ValueError('Invalid range length')
        from hashlib import sha256
        self.records.append(dict(start=start, end=end, sha256=sha256(data).hexdigest()))
        self.transferred_bytes += size
        self.position += size
        return data


def gqa_evaluator():
    session = requests.Session()
    session.trust_env = False
    reader = StrictRanges(session, 'https://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip')
    with zipfile.ZipFile(reader) as archive:
        candidates = [i for i in archive.infolist() if i.filename.endswith('eval.py')]
        if len(candidates) != 1:
            raise ValueError('Ambiguous official GQA evaluator')
        data = archive.read(candidates[0])
    path = RESOURCES/'annotations/gqa_eval.py'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Official evaluator changed')
    if not path.exists():
        path.write_bytes(data)
    from benchmarks.dataset_registry import sha256
    audit = dict(url=reader.url, etag=reader.etag, member=candidates[0].filename,
                 sha256=sha256(path), bytes_downloaded=reader.transferred_bytes, ranges=reader.records)
    (OUTPUT/'gqa_evaluator_source.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('Official GQA evaluator acquired', audit['sha256'], flush=True)

GQA_REV = 'a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8'
FILES = {
    'gqa_val_balanced.parquet': f'https://hf-mirror.com/datasets/lmms-lab/GQA/resolve/{GQA_REV}/val_balanced_instructions/val-00000-of-00001.parquet',
    'gqa_README.md': f'https://hf-mirror.com/datasets/lmms-lab/GQA/resolve/{GQA_REV}/README.md',
    'textcaps_val.json': 'https://dl.fbaipublicfiles.com/textvqa/data/textcaps/TextCaps_0.1_val.json',
    'gqa_eval.zip': 'https://downloads.cs.stanford.edu/nlp/data/gqa/eval.zip',
}


def fetch(item):
    name, url = item
    session = requests.Session()
    session.trust_env = False
    return dict(name=name, **download(session, url, RESOURCES/'annotations'/name, cap=100*1024**2))


def coco():
    session = requests.Session()
    session.trust_env = False
    url = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'
    path = RESOURCES/'annotations/captions_val2014.json'
    if path.exists():
        from benchmarks.dataset_registry import sha256
        return dict(source=url, status='present', sha256=sha256(path))
    reader = RangeReader(session, url)
    with zipfile.ZipFile(reader) as archive:
        data = archive.read('annotations/captions_val2014.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)
    from benchmarks.dataset_registry import sha256
    return dict(source=url, status='downloaded', sha256=sha256(path), etag=reader.etag,
                size=reader.size, bytes_downloaded=reader.transferred_bytes, ranges=reader.records)


def main():
    records = list(ThreadPoolExecutor(4).map(fetch, FILES.items()))
    try:
        records.append(dict(name='coco_captions', **coco()))
    except Exception as exc:
        records.append(dict(name='coco_captions', status='failed', error=repr(exc)))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT/'annotation_acquisition.json').write_text(json.dumps(records, indent=2)+'\n')
    for record in records:
        print(record.get('name'), record['status'], record.get('bytes', record.get('error')), flush=True)


if __name__ == '__main__':
    main()

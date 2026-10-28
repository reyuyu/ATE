"""Restore only the fixed handoff resources; never load a GPU or edit history."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import traceback

import huggingface_hub
from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'llava-hf/llava-1.5-7b-hf'
MODEL_REVISION = 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
DATA_ID = 'TIGER-Lab/MMEB-V2'
DATA_REVISION = 'e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1'
ARCHIVE = 'image-tasks/mmeb_v1.tar.gz'
ARCHIVE_BYTES = 7031608592
ARCHIVE_SHA256 = 'feeaf4a8319cba587b982ec976fba225d13e134a3217e1cad5c251e64299346a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def retrieve(repo, filename, revision, folder, endpoint, expected_bytes, expected_sha256,
             repo_type=None, transport='hub'):
    if transport == 'hub' or expected_sha256 is None:
        return Path(hf_hub_download(repo, filename, revision=revision, repo_type=repo_type,
                                    local_dir=folder, endpoint=endpoint))
    path = folder / filename
    if path.exists():
        if path.stat().st_size == expected_bytes and sha(path) == expected_sha256:
            return path
        raise ValueError(f'Preserve mismatched existing file: {path}')
    url = hf_hub_url(repo, filename, revision=revision, repo_type=repo_type, endpoint=endpoint)
    metadata = get_hf_file_metadata(url)
    if (metadata.commit_hash != revision or metadata.size != expected_bytes
            or metadata.etag != expected_sha256):
        raise ValueError('Pinned HF metadata disagrees with expected content; no transfer started')
    path.parent.mkdir(parents=True, exist_ok=True)
    transfer = path.with_suffix(path.suffix + '.transfer')
    if not transfer.exists():
        partials = list((folder / '.cache/huggingface/download').rglob(f'*.{expected_sha256}.incomplete'))
        if len(partials) == 1:
            partials[0].rename(transfer)
    subprocess.run(['aria2c', '--no-conf=true', '--all-proxy=', '--continue=true',
                    '--auto-file-renaming=false', '--allow-overwrite=false', '--file-allocation=none',
                    '--split=4', '--max-connection-per-server=4', '--min-split-size=10M',
                    '--max-tries=5', '--retry-wait=15', '--connect-timeout=20', '--timeout=90',
                    '--summary-interval=60', '--console-log-level=warn', '--download-result=hide',
                    '--checksum=sha-256=' + expected_sha256, '--dir=' + str(path.parent),
                    '--out=' + transfer.name, url], check=True)
    if transfer.stat().st_size != expected_bytes or sha(transfer) != expected_sha256:
        raise ValueError('Completed transfer failed final size/SHA256 check')
    transfer.rename(path)
    return path


def restore_data(root, endpoint, transport):
    folder = root / 'data'
    record = folder / 'mmeb_download_source.json'
    if record.exists():
        spec = json.loads(record.read_text())
        assert (spec['repo_id'], spec['revision'], spec['filename']) == (DATA_ID, DATA_REVISION, ARCHIVE)
    else:
        # The historical download revision is unknown; this is the new recovery pin.
        info = HfApi(endpoint=endpoint).dataset_info(DATA_ID, revision=DATA_REVISION)
        assert info.sha == DATA_REVISION
        spec = dict(repo_id=DATA_ID, repo_type='dataset', revision=info.sha,
                    filename=ARCHIVE, endpoint=endpoint,
                    historical_revision_known=False,
                    pinned_at=datetime.now(timezone.utc).isoformat(),
                    expected_bytes=ARCHIVE_BYTES, expected_sha256=ARCHIVE_SHA256)
        save(record, spec)
    path = retrieve(DATA_ID, ARCHIVE, spec['revision'], folder, endpoint,
                    ARCHIVE_BYTES, ARCHIVE_SHA256, repo_type='dataset', transport=transport)
    digest = sha(path)
    if path.stat().st_size != ARCHIVE_BYTES or digest != ARCHIVE_SHA256:
        raise ValueError('Downloaded archive size/SHA256 mismatch; preserved for investigation')
    spec.update(downloaded_path=str(path), archive_sha256=digest, bytes=path.stat().st_size)
    save(record, spec)
    destination = folder / 'image-tasks'
    final = destination / 'MMEB/GQA'
    extraction_audit = folder / 'gqa_extraction.json'
    if final.exists():
        if not extraction_audit.is_file():
            raise FileExistsError('Existing GQA directory has no completed extraction audit')
        audit = json.loads(extraction_audit.read_text())
        if audit.get('archive_sha256') != digest:
            raise ValueError('Existing extraction belongs to a different archive')
        return dict(image_root=str(final), extraction=audit)
    staging = destination / ('extracting_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    staging.mkdir(parents=True, exist_ok=False)
    with tarfile.open(path, 'r:gz') as archive:
        selected = []
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.parts[:2] != ('MMEB', 'GQA'):
                continue
            if name.is_absolute() or '..' in name.parts or not (member.isdir() or member.isfile()):
                raise ValueError(f'Unsafe GQA archive member: {member.name}')
            selected.append(member)
        if not any(m.isfile() and m.name.endswith('image_2324597.jpg') for m in selected):
            raise ValueError('Required GQA case not found in original archive')
        members = folder / 'gqa_archive_members.txt'
        members.write_text(''.join(m.name + '\n' for m in selected))
        required = sum(m.size for m in selected if m.isfile())
        if shutil.disk_usage(destination).free < required + 20 * 1024**3:
            raise OSError('Insufficient extraction space plus 20GiB reserve')
        archive.extractall(staging, members=selected, filter='data')
    (destination / 'MMEB').mkdir(exist_ok=True)
    (staging / 'MMEB/GQA').rename(final)
    audit = dict(archive_sha256=digest, member_count=len(selected), extracted_bytes=required,
                 image_root=str(final), completed_at=datetime.now(timezone.utc).isoformat())
    save(extraction_audit, audit)
    return dict(image_root=str(final), extraction=audit)


def restore_model(root, endpoint, transport):
    config = json.loads((ROOT / 'configs/model.json').read_text())
    assert config['model_id'] == MODEL_ID and config['revision'] == MODEL_REVISION
    folder = root / 'models/llava-1.5-7b-hf' / MODEL_REVISION
    save(root / 'audit/model_download_source.json',
         dict(repo_id=MODEL_ID, revision=MODEL_REVISION, endpoint=endpoint,
              model_dir=str(folder), historical_config_sha256=sha(ROOT / 'configs/model.json')))

    def download(item):
        path = retrieve(MODEL_ID, item['name'], MODEL_REVISION, folder, endpoint,
                        item['bytes'], item.get('lfs_sha256'), transport=transport)
        if path.stat().st_size != item['bytes']:
            raise ValueError(f"Model size mismatch: {item['name']}")
        digest = sha(path)
        if item.get('lfs_sha256') and item['lfs_sha256'] != digest:
            raise ValueError(f"Model SHA256 mismatch: {item['name']}")
        result = dict(file=item['name'], bytes=path.stat().st_size, sha256=digest)
        print(json.dumps(dict(model_file_verified=result)), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(download, sorted(config['files'], key=lambda item: bool(item.get('lfs_sha256')))))
    save(root / 'audit/model_files_verified.json', dict(revision=MODEL_REVISION, files=verified))
    return dict(model_dir=str(folder), verified_files=len(verified))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    parser.add_argument('--resource', required=True, choices=['data', 'model'])
    parser.add_argument('--transport', choices=['hub', 'aria2'], default='hub')
    args = parser.parse_args()
    if huggingface_hub.__version__ != '0.29.3':
        raise RuntimeError('Use pinned huggingface_hub==0.29.3, not a different downloader')
    args.root = args.root.resolve()
    output = args.root / 'audit' / f'restore_{args.resource}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    state = dict(resource=args.resource, started=datetime.now(timezone.utc).isoformat(),
                 status='running', downloader_version=huggingface_hub.__version__, transport=args.transport)
    save(output, state)
    try:
        operation = restore_data if args.resource == 'data' else restore_model
        result = operation(args.root, args.endpoint, args.transport)
        state.update(status='completed_resource_download', result=result)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        state['finished'] = datetime.now(timezone.utc).isoformat()
        save(output, state)


if __name__ == '__main__':
    main()

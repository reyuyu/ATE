"""Explicit archive-only cleanup; keep testdev, benchmark inputs and results."""
import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zlib
import zipfile

from vlm_research.information_horizon_alignment.common import OUT, atomic, manifest, now, read, sha


def check_image(entry):
    path, expected = entry
    assert path.is_file(), str(path)
    actual = sha(path)
    assert actual == expected, str(path)
    return str(path), actual


def zip_copy_audit(archive, root):
    checked = 0
    with zipfile.ZipFile(archive) as stream:
        for item in stream.infolist():
            if item.is_dir() or '/__MACOSX/' in '/' + item.filename or Path(item.filename).name.startswith('._'):
                continue
            path = root / item.filename
            assert path.resolve().is_relative_to(root.resolve()), item.filename
            assert path.is_file() and path.stat().st_size == item.file_size, str(path)
            crc = 0
            with path.open('rb') as source:
                for block in iter(lambda: source.read(4 * 1024**2), b''):
                    crc = zlib.crc32(block, crc)
            assert crc & 0xffffffff == item.CRC, str(path)
            checked += 1
    return checked


def main():
    destination = OUT / 'storage_cleanup_audit.json'
    assert not destination.exists(), 'Cleanup is a one-shot audited operation'
    testdev = Path('/data/lk/ATE-resources/data/gqa-testdev-balanced-official')
    audit = read(testdev / 'download_audit.json')
    assert len(audit['images']) == 398 and audit['questions'] == 12578
    assert sha(testdev / 'testdev_balanced_questions.json') == audit['question_file_sha256']
    protected = {Path(s['image_path']): s['image_sha256'] for s in manifest()}
    protected.update({Path(s['path']): s['sha256'] for s in audit['images']})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        verified = list(pool.map(check_image, protected.items()))
    atomic(OUT / 'protected_image_hashes.json', dict(verified))
    staging = Path('/data/lk/ATE-benchmarks-20261105/_staging')
    candidates = [
        (Path('/data/lk/ATE-resources/data/image-tasks/mmeb_v1.tar.gz'), None,
         'Unused re-downloadable source archive; extracted GQA images and all current inputs retained'),
        (staging / 'MME_Benchmark_release_version.zip',
         Path('/data/lk/ATE-benchmarks-20261105/datasets/MME'), 'All extracted files checked by ZIP size and CRC32'),
        (staging / 'SEED-Bench-image.zip',
         Path('/data/lk/ATE-benchmarks-20261105/datasets/SEED-Bench'), 'All extracted files checked by ZIP size and CRC32'),
    ]
    records = []
    for path, extracted, reason in candidates:
        assert path.is_file() and not path.is_symlink(), str(path)
        assert path.resolve().is_relative_to(Path('/data/lk'))
        checked = zip_copy_audit(path, extracted) if extracted else None
        record = dict(path=str(path), bytes=path.stat().st_size,
                      allocated_bytes=path.stat().st_blocks * 512, sha256=sha(path),
                      reason=reason, extracted_files_crc_verified=checked)
        records.append(record)
        print('VERIFIED', record, flush=True)
    result = dict(at=now(), status='VERIFIED_BEFORE_DELETE',
                  authorization='User authorized removing unused disk data, preserving GQA testdev_balanced',
                  protected_current_and_testdev_images=len(verified),
                  testdev_questions=12578, testdev_images=398,
                  old_gqa_images_retained=True, annotations_and_results_retained=True,
                  free_bytes_before=shutil.disk_usage('/data').free, archives=records)
    atomic(destination, result)
    for record in records:
        path = Path(record['path'])
        assert path.stat().st_size == record['bytes'] and path.stat().st_blocks * 512 == record['allocated_bytes']
        path.unlink()
        record['deleted_at'] = now()
        atomic(destination, result)
    assert all(path.is_file() for path in protected)
    result.update(status='COMPLETE', at=now(),
                  reclaimed_allocated_bytes=sum(r['allocated_bytes'] for r in records),
                  free_bytes_after=shutil.disk_usage('/data').free)
    atomic(destination, result)
    print('RECLAIMED_GIB', result['reclaimed_allocated_bytes'] / 1024**3, flush=True)


if __name__ == '__main__':
    main()

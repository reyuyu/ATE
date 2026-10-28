"""Copy hash-verified fixed CUDA wheel assets; never change the source Conda env."""
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil


PINNED = {
    'nvidia-cublas-cu12': '12.4.5.8',
    'nvidia-cuda-cupti-cu12': '12.4.127',
    'nvidia-cuda-nvrtc-cu12': '12.4.127',
    'nvidia-cuda-runtime-cu12': '12.4.127',
    'nvidia-cudnn-cu12': '9.1.0.70',
    'nvidia-cufft-cu12': '11.2.1.3',
    'nvidia-curand-cu12': '10.3.5.147',
    'nvidia-cusolver-cu12': '11.6.1.9',
    'nvidia-cusparse-cu12': '12.3.1.170',
    'nvidia-cusparselt-cu12': '0.6.2',
    'nvidia-nccl-cu12': '2.21.5',
    'nvidia-nvjitlink-cu12': '12.4.127',
    'nvidia-nvtx-cu12': '12.4.127',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return base64.urlsafe_b64encode(h.digest()).decode().rstrip('=')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-site', type=Path, required=True)
    p.add_argument('--target-site', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    args = p.parse_args()
    if args.audit.exists() or args.source_site.resolve() == args.target_site.resolve():
        p.error('Use distinct environments and a new audit file')
    installed = {d.metadata['Name']: d for d in importlib.metadata.distributions(path=[str(args.source_site)])}
    results = []
    for name, version in PINNED.items():
        dist = installed.get(name)
        if dist is None or dist.version != version:
            results.append(dict(name=name, expected=version, status='download_required'))
            continue
        entries = []
        invalid = []
        for relative, checksum, size in csv.reader(dist.read_text('RECORD').splitlines()):
            # Bytecode has no source integrity claim; regenerate it in the new env.
            if relative.endswith('.pyc'):
                continue
            source = (args.source_site / relative).resolve()
            if not source.is_relative_to(args.source_site.resolve()) or not source.is_file():
                invalid.append(relative)
                continue
            if checksum:
                algo, expected = checksum.split('=', 1)
                if algo != 'sha256' or digest(source) != expected or source.stat().st_size != int(size):
                    invalid.append(relative)
                    continue
            elif not relative.endswith('.dist-info/RECORD'):
                invalid.append(relative)
                continue
            entries.append((relative, source))
        if invalid:
            results.append(dict(name=name, version=version, status='source_record_mismatch_download_required', files=invalid))
            continue
        for relative, source in entries:
            target = args.target_site / relative
            if target.exists() and digest(target) != digest(source):
                raise ValueError(f'Existing target differs; no overwrite: {target}')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        results.append(dict(name=name, version=version, status='copied_after_record_verification',
                            files=len(entries), bytes=sum(s.stat().st_size for _, s in entries)))
        print(name, version, 'verified and copied', flush=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(dict(source=str(args.source_site), target=str(args.target_site),
                                          source_unchanged=True, packages=results), indent=2) + '\n')


if __name__ == '__main__':
    main()

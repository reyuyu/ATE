"""Fetch missing official weights once, through a pinned domestic HF endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import requests

ROOT = Path('/data/lk/ATE-resources/information-horizon-alignment')
HUB = ROOT / 'hf-hub'
ENDPOINT = 'https://hf-mirror.com'
NATIVE_REVISION = '4481d270cc22fd5c4d1bb5df129622006ccd9234'


def session():
    value = requests.Session()
    value.trust_env = False
    return value


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def download(repo, revision, entry):
    name = entry['rfilename']
    target = HUB / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision / name
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get('lfs', {}).get('sha256')
    size = entry['size']
    if target.exists() and target.stat().st_size == size:
        if not expected or checksum(target) == expected:
            return dict(path=str(target), bytes=size, sha256=checksum(target), reused=True)
        raise RuntimeError(f'Existing resource hash mismatch: {target}')
    temp = target.with_name(target.name + '.partial')
    url = f'{ENDPOINT}/{repo}/resolve/{revision}/{name}'
    error = None
    for attempt in range(6):
        try:
            start = temp.stat().st_size if temp.exists() else 0
            response = session().get(url, headers={'Range': f'bytes={start}-'} if start else {},
                                     stream=True, timeout=(15, 60))
            response.raise_for_status()
            if start and response.status_code != 206:
                start = 0
            with temp.open('ab' if start else 'wb') as handle:
                for block in response.iter_content(8 * 1024**2):
                    if shutil.disk_usage('/data').free < 3 * 1024**3:
                        raise RuntimeError('Mandatory /data 3 GiB resource-install safety floor')
                    handle.write(block)
            assert temp.stat().st_size == size, (name, temp.stat().st_size, size)
            actual = checksum(temp)
            assert not expected or actual == expected, f'LFS SHA256 failed: {name}'
            os.replace(temp, target)
            print('VERIFIED', repo, name, size, flush=True)
            return dict(path=str(target), bytes=size, sha256=actual, reused=False)
        except (requests.RequestException, AssertionError) as exc:
            error = repr(exc)
            print('RETRY', name, attempt + 1, error, flush=True)
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f'Resource failed after bounded retries: {name}: {error}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for repo, revision in (('liuhaotian/llava-v1.5-7b', NATIVE_REVISION),
                           ('openai/clip-vit-large-patch14-336', None)):
        response = session().get(f'{ENDPOINT}/api/models/{repo}' +
            (f'/revision/{revision}' if revision else '') + '?blobs=true', timeout=30)
        response.raise_for_status()
        metadata = response.json()
        revision = metadata['sha']
        if repo.startswith('liuhaotian'):
            wanted = {'config.json', 'generation_config.json', 'tokenizer.model',
                      'tokenizer_config.json', 'special_tokens_map.json',
                      'pytorch_model.bin.index.json', 'pytorch_model-00001-of-00002.bin',
                      'pytorch_model-00002-of-00002.bin'}
        else:
            available = {e['rfilename'] for e in metadata['siblings']}
            wanted = {'config.json', 'preprocessor_config.json',
                      'model.safetensors' if 'model.safetensors' in available else 'pytorch_model.bin'}
        entries = [e for e in metadata['siblings'] if e['rfilename'] in wanted]
        assert len(entries) == len(wanted)
        manifest[repo] = dict(revision=revision, entries=entries)
    (ROOT / 'official_resources_requested.json').write_text(json.dumps(manifest, indent=2))
    if args.metadata_only:
        print(json.dumps(manifest, indent=2))
        return
    needed = 0
    for repo, m in manifest.items():
        snapshot = HUB / ('models--' + repo.replace('/', '--')) / 'snapshots' / m['revision']
        for e in m['entries']:
            target = snapshot / e['rfilename']
            partial = target.with_name(target.name + '.partial')
            present = target.stat().st_size if target.exists() else partial.stat().st_size if partial.exists() else 0
            if not snapshot.resolve().is_relative_to(Path('/dev/shm')):
                needed += max(0, e['size'] - present)
    free = shutil.disk_usage('/data').free
    assert needed + 3 * 1024**3 < free, (needed, free)
    for repo, config in manifest.items():
        with ThreadPoolExecutor(max_workers=2) as pool:
            config['downloaded'] = list(pool.map(
                lambda entry: download(repo, config['revision'], entry), config['entries']))
        refs = HUB / ('models--' + repo.replace('/', '--')) / 'refs'
        refs.mkdir(parents=True, exist_ok=True)
        (refs / 'main').write_text(config['revision'])
        (ROOT / 'official_resources_verified.json').write_text(json.dumps(manifest, indent=2))
    print('ALL_OFFICIAL_WEIGHTS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()

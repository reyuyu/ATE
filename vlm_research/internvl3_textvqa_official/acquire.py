"""Download small, immutable upstream sources; never model weights or image datasets."""
import base64
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/internvl3_textvqa_official_alignment'
COMMIT = 'cb1b907680808ffb3c314b8a34e0c90a7a3db1de'
PATHS = [
    'LICENSE', 'vlmeval/config.py', 'vlmeval/vlm/internvl/internvl_chat.py',
    'vlmeval/vlm/internvl/utils.py', 'vlmeval/dataset/image_vqa.py',
    'vlmeval/dataset/utils/vqa_eval.py', 'vlmeval/dataset/image_base.py',
    'vlmeval/vlm/base.py', 'vlmeval/inference.py', 'vlmeval/smp/file.py',
    'vlmeval/smp/vlm.py', 'vlmeval/smp/misc.py',
]


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'ATE-protocol-audit'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main():
    global COMMIT
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', action='store_true')
    args = parser.parse_args()
    folder, manifest = 'sources', 'source_manifest.json'
    if args.release:
        COMMIT = '1cc6d16d34b8f164523388edb0a9642ef8ab7314'
        folder, manifest = 'release_sources', 'release_source_manifest.json'
    old = json.loads((OUT/manifest).read_text()) if (OUT/manifest).exists() else {}
    known = {e['path']: e for e in old.get('entries', []) if e.get('commit') == COMMIT}
    entries = []
    for name in PATHS:
        target = OUT / folder / name
        if name in known and target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest() == known[name]['sha256']
            entries.append(known[name])
            continue
        url = f'https://api.github.com/repos/open-compass/VLMEvalKit/contents/{name}?ref={COMMIT}'
        payload = json.loads(fetch(url))
        data = base64.b64decode(payload['content'])
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == payload['sha']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == data, ('EXISTING_SOURCE_MISMATCH', name)
        else:
            target.write_bytes(data)
        entries.append(dict(path=name, commit=COMMIT, url=url, github_blob=payload['sha'],
                            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
        print('FROZEN_SOURCE', name, flush=True)
    for name, url in [
        ('internvl3_evaluation.html', 'https://internvl.readthedocs.io/en/latest/internvl3.0/evaluation.html'),
        ('internvl3_blog.html', 'https://internvl.github.io/blog/2025-04-11-InternVL-3.0/'),
    ]:
        target = OUT / 'sources' / name
        if not target.exists():
            target.write_bytes(fetch(url))
        data = target.read_bytes()
        entries.append(dict(path=name, url=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    record = dict(framework='open-compass/VLMEvalKit', commit=COMMIT,
        selection=('First upstream InternVL3 registration, 2025-04-11; official HF route without the later LMDeploy default, selected before predictions.' if args.release else
                   'Latest upstream commit on 2025-05-29, the documentation last-update date; chosen before any new predictions.'),
        limitation='Official documentation does not publish the exact leaderboard-run commit; this is a dated, immutable framework snapshot, not proof of that hidden run.',
        entries=entries)
    (OUT / manifest).write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()

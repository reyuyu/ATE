#!/usr/bin/env python3
"""Check a portable ATE dataset on CPU; optionally emit absolute-path inputs."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe relative path: ' + relative)
    result = (root / p).resolve()
    result.relative_to(root)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bundle', type=Path, help='Directory containing bundle_manifest.json')
    ap.add_argument('--output', type=Path, help='New directory for resolved inputs and separate Gold')
    args = ap.parse_args()
    root = args.bundle.resolve()
    manifest = json.loads((root / 'bundle_manifest.json').read_text())
    for rel, expected in manifest['metadata_files'].items():
        assert sha(inside(root, rel)) == expected, rel
    inventory = json.loads((root / 'image_sha256.json').read_text())
    for rel, info in inventory.items():
        p = inside(root, rel)
        assert p.is_file() and not (root / rel).is_symlink(), rel
        assert p.stat().st_size == info['bytes'], rel
    # Reuse the previous full image SHA/decode audit. Only decode three samples here.
    samples = sorted(inventory)
    sampled = []
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image and samples:
        for rel in sorted({samples[0], samples[len(samples)//2], samples[-1]}):
            with Image.open(inside(root, rel)) as im:
                im.load()
                assert im.size[0] > 0 and im.size[1] > 0
            sampled.append(rel)
    output = args.output.resolve() if args.output else None
    if output:
        if output == root or root in output.parents:
            raise ValueError('--output must be outside the immutable bundle')
        output.mkdir(parents=True, exist_ok=False)
    counts = {}
    forbidden = {'answer', 'answers', 'gold', 'label', 'lecture', 'solution', 'correct_answer', 'answer_idx'}
    for name, info in manifest['views'].items():
        source = inside(root, info['directory'])
        answers = json.loads((source / 'answers.json').read_text())
        rows = [json.loads(line) for line in (source / 'inputs.jsonl').read_text().splitlines()]
        ids = [str(row['question_id']) for row in rows]
        assert len(rows) == info['questions'] and len(ids) == len(set(ids)), name
        assert set(ids) == set(answers), name
        for row in rows:
            assert not forbidden.intersection(row), name
            assert row['image_path'] in inventory, row['image_path']
            p = inside(root, row['image_path'])
            if output:
                row['image_path'] = str(p)
        counts[name] = len(rows)
        if output:
            dst = output / name
            dst.mkdir()
            (dst / 'inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n' for r in rows))
            shutil.copyfile(source / 'answers.json', dst / 'answers.json')
            resolved = dict(info, dataset=name, inputs_sha256=sha(dst/'inputs.jsonl'), answers_sha256=sha(dst/'answers.json'), bundle_manifest_sha256=sha(root/'bundle_manifest.json'))
            (dst/'manifest.json').write_text(json.dumps(resolved, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'success': True, 'views': counts, 'images': len(inventory), 'sample_decodes': sampled,
                      'pillow_available': Image is not None, 'image_hashes_reused': True,
                      'gpu_used': False, 'output': str(output) if output else None}, ensure_ascii=False))


if __name__ == '__main__':
    main()

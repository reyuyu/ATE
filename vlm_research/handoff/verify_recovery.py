"""CPU-only recovery audit: committed artifacts, frozen images and optional weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-root', type=Path, help='Directory containing image_<img_id>.jpg')
    p.add_argument('--verify-source-images', action='store_true', help='Also audit all785 images referenced by source CSV; only252 have frozen hashes')
    p.add_argument('--model-dir', type=Path, help='Optional fixed-revision HF snapshot directory')
    p.add_argument('--output', type=Path, help='NEW JSON path, outside historical result directories')
    args = p.parse_args()
    if args.verify_source_images and not args.image_root:
        p.error('--verify-source-images requires --image-root')
    if args.output and args.output.exists():
        p.error('Refuse to overwrite an existing audit')
    bundle = json.loads((ROOT/'handoff/artifact_manifest.json').read_text())
    failures = []
    for relative, expected in bundle['files_sha256'].items():
        path = ROOT/relative
        if not path.is_file():
            failures.append(dict(kind='missing_repository_file', path=relative))
        elif sha(path) != expected:
            failures.append(dict(kind='repository_hash_mismatch', path=relative))
    manifest = json.loads((ROOT/'results/phase3_2_fixed_k_exit/manifest.json').read_text())
    samples = manifest['samples']
    if len(samples) != 240 or len({s['img_id'] for s in samples}) != 240 or any(sum(s['answer']==a for s in samples)!=120 for a in ['left','right']):
        failures.append(dict(kind='cohort_shape_or_balance'))
    unique = {s['img_id']:s for s in samples+manifest['validation_samples']}
    image_results = []
    if args.image_root:
        from PIL import Image
        for img_id, s in sorted(unique.items()):
            path = args.image_root/f'image_{img_id}.jpg'
            record = dict(img_id=img_id, path=str(path), exists=path.is_file())
            if path.is_file():
                record['sha256'] = sha(path)
                record['hash_matches'] = record['sha256']==s['image_sha256']
                try:
                    with Image.open(path) as im: im.verify()
                    with Image.open(path) as im: record['size'] = list(im.size)
                    record['decode_and_size_match'] = record['size']==s['source_image_size']
                except Exception as exc:
                    record.update(decode_and_size_match=False, error=repr(exc))
            if not record['exists'] or not record.get('hash_matches') or not record.get('decode_and_size_match'):
                failures.append(dict(kind='frozen_image_mismatch', **record))
            image_results.append(record)
    source_results = []
    if args.verify_source_images:
        source = ROOT/'configs/phase3_1_data/GQA_val_correct_question_with_relChooser_ChooseRel.csv'
        source_rows = {r['img_id']:r for r in csv.DictReader(source.open())}
        for img_id, r in sorted(source_rows.items()):
            path = args.image_root/f'image_{img_id}.jpg'
            record = dict(img_id=img_id, exists=path.is_file(), frozen_hash_available=img_id in unique)
            if path.is_file():
                record['sha256'] = sha(path)
                try:
                    with Image.open(path) as im: im.verify()
                    with Image.open(path) as im: record['size'] = list(im.size)
                    record['decode_and_size_match'] = record['size']==[int(r['image_w']),int(r['image_h'])]
                except Exception as exc:
                    record.update(decode_and_size_match=False, error=repr(exc))
            if not record['exists'] or not record.get('decode_and_size_match'):
                failures.append(dict(kind='source_image_missing_or_invalid', **record))
            source_results.append(record)
    weights = []
    if args.model_dir:
        config = json.loads((ROOT/'configs/model.json').read_text())
        for item in config['files']:
            path = args.model_dir/item['name']
            record = dict(file=item['name'], exists=path.is_file())
            if path.is_file():
                record['size_matches'] = path.stat().st_size==item['bytes']
                if item.get('lfs_sha256'):
                    record['sha256'] = sha(path)
                    record['hash_matches'] = record['sha256']==item['lfs_sha256']
            if not record['exists'] or not record.get('size_matches') or record.get('hash_matches') is False:
                failures.append(dict(kind='model_file_mismatch', **record))
            weights.append(record)
    result = dict(passed=not failures, original_code_commit=bundle['source_commit'],
                  repository_files_checked=len(bundle['files_sha256']), phase32_n=240, pilot_n=12,
                  image_check_requested=bool(args.image_root), images_checked=len(image_results),
                  images_required_for_full_recovery=len(unique), model_check_requested=bool(args.model_dir),
                  images=image_results, source_images=source_results, source_images_checked=len(source_results),
                  model_files=weights, failures=failures,
                  note='No GPU or model forward. Repository-only success does not certify missing images/environment. Raw CSV has 964 questions/785 images; frozen continuation needs 240+12 distinct images. No resampling or manifest rewriting.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['images','source_images','model_files','failures']},ensure_ascii=False,indent=2))
    if failures:
        print(json.dumps(failures[:10],ensure_ascii=False,indent=2))
        raise SystemExit(1)

if __name__=='__main__':main()

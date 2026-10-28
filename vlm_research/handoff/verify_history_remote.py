"""Verify original Phase3.5 inventories with audited, runtime-only image paths."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_entry(original, image_root, frozen):
    path = Path(original)
    if path.is_absolute():
        if path.name not in frozen:
            raise ValueError('Unknown absolute path, refuse unvalidated remapping: ' + original)
        return image_root.resolve() / path.name, True
    local = (ROOT / path).resolve()
    if not local.is_relative_to(ROOT):
        raise ValueError('Relative checksum path escapes project')
    return local, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file')
    manifest = json.loads((ROOT / 'results/phase3_2_fixed_k_exit/manifest.json').read_text())
    frozen = {f"image_{s['img_id']}.jpg": s['image_sha256']
              for s in manifest['samples'] + manifest['validation_samples']}
    bundle = json.loads((ROOT / 'handoff/artifact_manifest.json').read_text())['files_sha256']
    definitions = [('results/phase3_5_exit_boundary_viz/checksums.json', ['inputs', 'outputs']),
                   ('results/phase3_5_two_case_visualization/validation.json', ['input_sha256', 'output_sha256'])]
    checks = []
    for relative, sections in definitions:
        if sha(ROOT / relative) != bundle[relative]:
            raise ValueError('Historical checksum inventory changed')
        inventory = json.loads((ROOT / relative).read_text())
        for section in sections:
            for original, expected in inventory[section].items():
                actual, image = resolve_entry(original, args.image_root, frozen)
                if image and frozen[actual.name] != expected:
                    raise ValueError('Inventory image hash differs from frozen 252-image cohort')
                record = dict(inventory=relative, section=section, original_path=original,
                              actual_path=str(actual), is_external_image=image,
                              expected_sha256=expected, exists=actual.is_file())
                if record['exists']:
                    record['actual_sha256'] = sha(actual)
                record['passed'] = record.get('actual_sha256') == expected
                checks.append(record)
    repository = [r for r in checks if not r['is_external_image']]
    images = [r for r in checks if r['is_external_image']]
    result = dict(passed=all(r['passed'] for r in checks),
                  repository_checks=len(repository), repository_checks_passed=all(r['passed'] for r in repository),
                  image_checks=len(images), image_checks_passed=all(r['passed'] for r in images),
                  pending_images=sorted({r['original_path'] for r in images if not r['passed']}),
                  mapping='Only known frozen image basenames are relocated; original inventories/SHA256 unchanged',
                  checks=checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

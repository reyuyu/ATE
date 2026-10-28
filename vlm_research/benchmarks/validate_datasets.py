"""Read-only acceptance gate for all manifests and their original resources."""
import argparse
import json
from pathlib import Path

from .dataset_registry import OUTPUT, audit_image, digest, freeze_json, load_manifest, sha256


def validate(output):
    inventory = json.loads((output/'dataset_inventory.json').read_text())
    protocols = json.loads((output/'evaluation_protocols.json').read_text())
    locks = json.loads((output/'sampling_lock.json').read_text())
    results = {}
    for path in sorted((output/'manifests').glob('*.json')):
        item = inventory[path.stem]
        if sha256(path) != item['manifest_sha256']:
            raise ValueError(f'Changed frozen manifest {path}')
        manifest = load_manifest(path, verify_images=False)
        if manifest['protocol_sha256'] != digest(protocols):
            raise ValueError('Protocol changed after sample freeze')
        if [r['question_id'] for r in manifest['samples']] != locks['units'][path.stem]['question_ids']:
            raise ValueError('Frozen IDs changed')
        checked, blocked = [], []
        for sample in manifest['samples']:
            observed = audit_image(sample['image_path'], sample['metadata'].get('expected_image_size'))
            if observed != sample['metadata']['image_audit']:
                raise ValueError(f'Asset differs from freeze: {sample["question_id"]}')
            if observed['image_status'] == 'verified':
                checked.append(sample['question_id'])
            else:
                blocked.append(sample['question_id'])
            mapping = sample['metadata']['input_mapping']
            if mapping['prompt_ids_sha256'] != digest(mapping['prompt_ids']) or len(mapping['visual_positions']) != 576:
                raise ValueError('Prompt/image token binding failed')
            if mapping['p_minus_1'] != mapping['prompt_length']-1 or max(mapping['question_positions']) >= mapping['p_minus_1']:
                raise ValueError('Invalid dynamic P-1/Question')
            if sample['candidates'] and set(mapping['answer_mapping']) != set(sample['candidates']):
                raise ValueError('Candidate mapping incomplete')
            for label, records in mapping['answer_mapping'].items():
                if not records or [r['query_position'] for r in records] != list(range(mapping['p_minus_1'], mapping['p_minus_1']+len(records))):
                    raise ValueError('Multi-token teacher forcing mapping error')
        if manifest['status'] == 'ready_native_full' and blocked:
            raise ValueError('Ready status with blocked assets')
        results[path.stem] = {'manifest_verified': True, 'samples': len(manifest['samples']),
                             'verified_images_per_question': len(checked), 'blocked_question_ids': blocked,
                             'status': manifest['status']}
    expected = {'gqa_chooseattr', 'gqa_queryattr', 'gqa_logicalobj', 'pope_random', 'pope_popular',
                'pope_adversarial', 'scienceqa_img', 'textvqa'}
    if set(results) != expected:
        raise ValueError('Missing dataset registration')
    return {'passed': True, 'meaning': 'Frozen data integrity, not a model-quality result or intervention acceptance',
            'all_assets_ready': all(not x['blocked_question_ids'] for x in results.values()), 'datasets': results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--save', type=Path)
    args = p.parse_args()
    value = validate(args.output)
    if args.save:
        freeze_json(args.save, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

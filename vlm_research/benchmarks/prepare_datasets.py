"""Freeze annotations, sampling, exact prompts and hashes before any model output."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from .dataset_registry import (DEFAULT_RESOURCES, OUTPUT, SEED, audit_image, audit_rows,
                               digest, freeze_json, select_unique, sha256)
from .fetch_resources import selected_inputs
from .gqa_adapter import GQA_FILES, read_gqa
from .protocols import MODEL_PATH, prompt_text, protocols


def source(path, revision, url):
    return {'path': str(path), 'revision': revision, 'url': url, 'sha256': sha256(path), 'bytes': path.stat().st_size}


def token_mapping(processor, sample):
    text = prompt_text(sample)
    message = lambda t: [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}]
    render = lambda t: processor.apply_chat_template(message(t), tokenize=False, add_generation_prompt=True)
    marker = '__ATE_PHASE40_PAYLOAD__'
    if marker in text:
        raise ValueError('Reserved marker')
    before, after = render(marker).split(marker)
    prompt = render(text)
    if prompt != before + text + after:
        raise ValueError('Chat template unexpectedly changes payload')
    raw = processor.tokenizer(prompt, return_offsets_mapping=True)
    image_token = processor.tokenizer.convert_tokens_to_ids('<image>')
    if raw['input_ids'].count(image_token) != 1:
        raise ValueError('Exactly one image placeholder required')
    ids, offsets = [], []
    for tid, offset in zip(raw['input_ids'], raw['offset_mapping']):
        copies = 576 if tid == image_token else 1
        ids.extend([tid]*copies)
        offsets.extend([offset]*copies)
    qoffset = 0
    if sample['dataset'] == 'scienceqa_img':
        hint = sample['metadata']['raw_row'].get('hint')
        qoffset = len('Context: ' + hint + '\n') if hint else 0
    start = len(before) + qoffset
    end = start + len(sample['question'])
    if prompt[start:end] != sample['question']:
        raise ValueError('Question mapping failure')
    question_positions = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and ids[i] != image_token]
    if not question_positions or max(question_positions) >= len(ids)-1:
        raise ValueError('Invalid actual Question/P-1 mapping')
    from attention_position_probe import encode_answer
    answer_mapping = {label: encode_answer(processor.tokenizer, label, len(ids)) for label in sample['candidates'] or []}
    return {'text_payload': text, 'rendered_prompt': prompt, 'rendered_prompt_sha256': digest(prompt),
            'prompt_ids': ids, 'prompt_ids_sha256': digest(ids), 'prompt_length': len(ids), 'p_minus_1': len(ids)-1,
            'question_positions': question_positions, 'question_char_span': [start, end],
            'visual_positions': [i for i, tid in enumerate(ids) if tid == image_token],
            'answer_mapping': answer_mapping,
            'candidate_lengths': {label: len(records) for label, records in answer_mapping.items()},
            'reference_answer_token_ids': {answer: processor.tokenizer(answer, add_special_tokens=False)['input_ids']
                                           for answer in (sample['answer'] if isinstance(sample['answer'], list) else [sample['answer']])},
            'question_span_policy': 'Only original Question, excluding context, choices and output instruction; '
                                    'boundary-overlap tokens retained and spans recorded for future intervention review'}


def prepare(root, output, gqa_root):
    repo_root = Path(__file__).resolve().parents[1]
    old_path = repo_root/'results/phase3_2_fixed_k_exit/manifest.json'
    old = json.loads(old_path.read_text())
    old_images = {r['img_id'] for r in old['samples']}
    gqa_rev = 'c9a1111dac4baa175f67716fab473789320187c0'
    units = {}
    for name, filename in GQA_FILES.items():
        path = root/'sources/information_flow/datasets'/filename
        rows = read_gqa(path, name, gqa_rev, gqa_root)
        selected = select_unique(rows, 240, old_images, balanced=name == 'gqa_logicalobj')
        units[name] = (rows, selected, [source(path, gqa_rev,
            f'https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM/blob/{gqa_rev}/datasets/{filename}')])
    pope, science, textvqa, selected = selected_inputs(root)
    for strategy, rows in pope.items():
        path = root/f'sources/pope/output/coco/coco_pope_{strategy}.json'
        units[f'pope_{strategy}'] = (rows, selected['pope'][strategy], [source(path,
            '08d957b917e5a378a2f99d35b6293c536a66298b', 'https://github.com/AoiDragon/POPE/tree/08d957b917e5a378a2f99d35b6293c536a66298b/output/coco')])
    sqa_sources = [source(root/f'sources/scienceqa/data/scienceqa/{name}',
                         '2cbf8318e07b9ece895bb2ae605e71e38d623264', 'https://github.com/lupantech/ScienceQA')
                   for name in ('problems.json', 'pid_splits.json')]
    units['scienceqa_img'] = (science, selected['scienceqa_img'], sqa_sources)
    tv_path = root/'sources/textvqa/TextVQA_0.5.1_val.json'
    units['textvqa'] = (textvqa, selected['textvqa'], [source(tv_path, 'official-v0.5.1',
        'https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json')] if tv_path.exists() else [])
    # Freeze ID/row/source selection before image checks and any model predictions.
    selection = {'seed': SEED, 'units': {name: {'question_ids': [r['question_id'] for r in chosen],
                 'row_sha256': [r['source_row_sha256'] for r in chosen], 'sources': sources}
                 for name, (_, chosen, sources) in units.items()},
                 'gqa_rule': 'Per task independent sorted/shuffled pool; distinct images; prefer non-ChooseRel images; '
                             'LogicalObj 120no+120yes; others240. Do not filter by availability or predictions.',
                 'group_rule': json.loads((root/'selection_lock.json').read_text())['rule'],
                 'historical_manifest_sha256': sha256(old_path)}
    freeze_json(output/'sampling_lock.json', selection)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    protocol = protocols()
    protocol['common']['chat_template'] = processor.chat_template
    protocol['common']['chat_template_sha256'] = digest(processor.chat_template)
    protocol['common']['image_processor_config'] = processor.image_processor.to_dict()
    protocol['common']['tokenizer_special_ids'] = processor.tokenizer.special_tokens_map
    protocol['common']['tokenizer_class'] = type(processor.tokenizer).__name__
    protocol['common']['upstream_scorer_sha256'] = {str(p.relative_to(root/'sources/llava')): sha256(p)
        for p in (root/'sources/llava/llava/eval').glob('eval_*.py')}
    protocol['common']['upstream_scorer_sha256']['llava/eval/m4c_evaluator.py'] = sha256(root/'sources/llava/llava/eval/m4c_evaluator.py')
    protocol['common']['tokenizer_processor_files'] = {p.name: sha256(p) for p in Path(MODEL_PATH).iterdir()
        if p.is_file() and p.suffix in ('.json', '.jinja', '.model')}
    freeze_json(output/'evaluation_protocols.json', protocol)
    inventory, cached_audit = {}, {}
    for name, (rows, chosen, sources) in units.items():
        full_audit = audit_rows(rows)
        full_image_counts = Counter()
        full_missing = []
        for row in rows:
            path = row['image_path']
            if path not in cached_audit:
                cached_audit[path] = audit_image(path, row['metadata'].get('expected_image_size'))
            a = cached_audit[path]
            full_image_counts[a['image_status']] += 1
            if a['image_status'] != 'verified':
                full_missing.append({'question_id': row['question_id'], 'image_id': row['image_id'], **a})
        freeze_json(output/f'full_inventory/{name}.json', {'sources': sources, 'audit': full_audit,
            'image_audit_counts_per_question': dict(full_image_counts),
            'rows': rows, 'missing_or_corrupt': full_missing})
        samples = []
        for original in chosen:
            row = copy.deepcopy(original)
            image_audit = cached_audit[row['image_path']]
            row['image_sha256'] = image_audit['image_sha256']
            row['metadata']['image_audit'] = image_audit
            row['metadata']['cluster_id'] = image_audit['image_sha256'] or row['image_id']
            row['metadata']['input_mapping'] = token_mapping(processor, row)
            samples.append(row)
        audit = audit_rows(samples)
        missing = [r['question_id'] for r in samples if r['metadata']['image_audit']['image_status'] != 'verified']
        status = 'ready_native_full' if samples and not missing else 'blocked_assets'
        if name == 'scienceqa_img':
            sqa = json.loads(Path(sqa_sources[0]['path']).read_text())
            audit['raw_all_split_questions'] = len(sqa)
            audit['official_test_questions'] = sum(r['split'] == 'test' for r in sqa.values())
            audit['eligible_image_test_questions'] = len(rows)
        item = {'name': name, 'seed': SEED, 'status': status, 'sources': sources,
                'selection_sha256': digest(selection['units'][name]), 'samples': samples,
                'audit': audit, 'full_audit': full_audit, 'missing_or_corrupt_selected_ids': missing,
                'unique_verified_image_sha256': len({r['image_sha256'] for r in samples if r['image_sha256']}),
                'historical_choose_rel_image_overlap': sorted({r['image_id'] for r in samples} & old_images) if name.startswith('gqa_') else [],
                'protocol_sha256': digest(protocol),
                'smoke_question_ids': [r['question_id'] for r in samples[:2]],
                'intervention_ready': False,
                'future_gate': protocol['common']['future_intervention_gate']}
        item['binding_sha256'] = digest(item)
        freeze_json(output/f'manifests/{name}.json', item)
        inventory[name] = {k: v for k, v in item.items() if k not in ('samples', 'binding_sha256')}
        inventory[name]['manifest_sha256'] = sha256(output/f'manifests/{name}.json')
        inventory[name]['image_roots'] = sorted({str(Path(r['image_path']).parent) for r in samples})
        print(name, status, len(samples), audit['unique_images'], 'missing', len(missing), flush=True)
    overlaps = {}
    for a, (_, ar, _) in units.items():
        for b, (_, br, _) in units.items():
            if a < b:
                overlaps[f'{a}__{b}'] = {'image_id_overlap': sorted({r['image_id'] for r in ar} & {r['image_id'] for r in br})}
    freeze_json(output/'overlap_audit.json', overlaps)
    preflight = json.loads((root/'sources/acquisition.json').read_text())[-1]['records']
    inventory['gqa_official_balanced'] = {'status': 'blocked_assets_and_official_scorer', 'frozen_questions': 0,
        'split': 'balanced testdev (separate registration, not author CSVs)',
        'source_url': 'https://cs.stanford.edu/people/dorarad/gqa/download.html',
        'scorer_url': 'https://cs.stanford.edu/people/dorarad/gqa/evaluate.html',
        'preflight': [r for r in preflight if 'dorarad/gqa' in r['url']],
        'recovery': 'Recover official questions1.2 annotations and official eval bundle; inspect testdev answer availability, '
                    'matching original image IDs and local scorer assets. No huge archive downloaded; do not reuse ChooseRel as balanced.'}
    freeze_json(output/'dataset_inventory.json', inventory)
    freeze_json(output/'resource_image_audit.json', cached_audit)
    freeze_json(output/'resource_provenance.json', {
        'gqa': json.loads(Path('/data/lk/ATE-resources/data/mmeb_download_source.json').read_text()),
        'sources': json.loads((root/'sources/acquisition.json').read_text()),
        'asset_downloads': {p.name: json.loads(p.read_text()) for p in root.glob('fetch_*_audit.json')},
        'resource_root': str(root), 'originals_policy': 'Read-only historical assets; new selected images outside Git',
        'source_file_hashes': {str(p.relative_to(root)): sha256(p) for p in (root/'sources').rglob('*')
                               if p.is_file() and not p.name.endswith('.partial')}})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=DEFAULT_RESOURCES)
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--gqa-root', type=Path, default=Path('/data/lk/ATE-resources/data/image-tasks/MMEB/GQA'))
    a = p.parse_args()
    prepare(a.root, a.output, a.gqa_root)


if __name__ == '__main__':
    main()

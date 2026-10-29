"""Freeze unfiltered task cohorts and true processor mappings before GPU scoring."""
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

from attention_position_probe import locate_question_tokens
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.protocols import prompt_text
from paper_rq1.scorers import GQA_EVALUATOR_SHA, GQA_CONVERTER_SHA
from .protocol import ANNOTATION, IMAGES, MODEL_PATH, OUTPUT, PROTOCOL, SEED, TASKS


def order(*parts):
    return digest([SEED, *parts])


def select(rows):
    chosen = []
    counts = {}
    seen = set()
    for task, detailed in TASKS.items():
        pool = [r for r in rows if r['types']['detailed'] in detailed]
        counts[task] = len(pool)
        if len(pool) < 500:
            raise ValueError(f'{task}: only {len(pool)} original questions; refuse silent supplementation')
        current = sorted(pool, key=lambda r: order(task, r['id']))[:500]
        subset = {r['id'] for r in sorted(current, key=lambda r: order('accuracy', task, r['id']))[:80]}
        for r in current:
            if r['id'] in seen or not r['isBalanced'] or not r['answer']:
                raise ValueError('Duplicate question or invalid original answered validation row')
            seen.add(r['id'])
            chosen.append(dict(task=task, question_id=r['id'], image_id=r['imageId'],
                question=r['question'], answer=r['answer'], type_detailed=r['types']['detailed'],
                source_row_sha256=digest(r), source='lmms-lab/GQA@a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8:val_balanced_instructions',
                selection_bias='No model-correctness prefilter; conditional on original task category.',
                accuracy_subset=r['id'] in subset,
                rank=int(order('worker', r['imageId']), 16) % 4))
    return chosen, counts


def freeze():
    import pyarrow.parquet as pq
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT/'execution_manifest.json').exists():
        raise ValueError('Do not refreeze data after execution')
    rows = pq.read_table(ANNOTATION).to_pylist()
    selected, counts = select(rows)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    processor.patch_size, processor.num_additional_image_tokens = 14, 1
    processor.vision_feature_select_strategy = 'default'
    hashes = {}
    failures = []
    for index, sample in enumerate(selected):
        image_path = IMAGES/f"image_{sample['image_id']}.jpg"
        try:
            with Image.open(image_path) as original:
                original.load()
                image = original.convert('RGB')
                size = list(image.size)
            if sample['image_id'] not in hashes:
                hashes[sample['image_id']] = sha256(image_path)
            sample.update(image_path=str(image_path), image_sha256=hashes[sample['image_id']], image_size=size)
            sample['prompt_text'] = prompt_text(dict(dataset='gqa_balanced', question=sample['question']))
            messages = [dict(role='user', content=[dict(type='image'), dict(type='text', text=sample['prompt_text'])])]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            native = processor(images=image, text=chat, return_tensors='pt')['input_ids'][0].tolist()
            gold = processor(images=image, text=chat+' '+sample['answer'], return_tensors='pt')['input_ids'][0].tolist()
            if gold[:len(native)] != native or len(gold) <= len(native):
                raise ValueError('Gold continuation tokenization does not preserve original prompt')
            bos = [i for i, token in enumerate(native) if token == processor.tokenizer.bos_token_id]
            visual = [i for i, token in enumerate(native) if token == 32000]
            if len(bos) != 1 or len(visual) != 576:
                raise ValueError('Unexpected true processor BOS/image layout')
            question_records, _ = locate_question_tokens(processor, sample['prompt_text'], native, 32000)
            # The legacy locator locates its exact text argument; retain the actual instruction-inclusive payload.
            sample['input_mapping'] = dict(prompt_ids=native, p_minus_one=len(native)-1,
                bos_positions=bos, bos_token_id=processor.tokenizer.bos_token_id,
                visual_positions=visual, question_and_instruction_positions=[r['sequence_index'] for r in question_records],
                gold_continuation_ids=gold[len(native):], chat_prompt_sha256=digest(chat))
        except Exception as exc:
            failures.append(dict(task=sample['task'], question_id=sample['question_id'], image_id=sample['image_id'], error=repr(exc)))
        if (index+1) % 500 == 0:
            print('Frozen mappings', index+1, '/', len(selected), flush=True)
    freeze_json(OUTPUT/'selection_audit.json', dict(source_sha256=sha256(ANNOTATION), source_rows=len(rows),
        task_pool_counts=counts, selected_before_image_validation=True, replacements=0, failures=failures))
    if failures:
        raise ValueError(f'{len(failures)} selected images/mappings invalid; no replacements permitted')
    official = ANNOTATION.parent/'gqa_eval.py'
    converter = Path('/data/lk/ATE-resources/benchmarks/phase4_0/sources/llava/scripts/convert_gqa_for_eval.py')
    if sha256(official) != GQA_EVALUATOR_SHA or sha256(converter) != GQA_CONVERTER_SHA:
        raise ValueError('Official scoring source hashes differ')
    import ast
    tree = ast.parse(official.read_text())
    expression = ast.dump(ast.parse('predicted == gold', mode='eval').body)
    if not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'correct' for t in n.targets)
               and ast.dump(n.value) == expression for n in ast.walk(tree)):
        raise ValueError('Official primary equality expression differs')
    scorer = dict(passed=True, official_primary_accuracy_equality_verified=True,
        llava_converter="text.rstrip('.').lower() after native decoded output.strip()",
        source_sha256={str(official):sha256(official), str(converter):sha256(converter)})
    freeze_json(OUTPUT/'scorer_validation.json', scorer)
    freeze_json(OUTPUT/'sample_manifest.json', dict(protocol=PROTOCOL, samples=selected))
    for rank in range(4):
        freeze_json(OUTPUT/f'manifests/shard{rank}.json', dict(samples=[s for s in selected if s['rank'] == rank]))
    image_sets = {t:{s['image_id'] for s in selected if s['task'] == t} for t in TASKS}
    audit = dict(status='ready', total_questions=3000, total_images=len(hashes),
        per_task={t:dict(questions=500, images=len(image_sets[t]), accuracy_subset=80,
            shard_questions={str(i):sum(s['task']==t and s['rank']==i for s in selected) for i in range(4)},
            shard_accuracy={str(i):sum(s['task']==t and s['rank']==i and s['accuracy_subset'] for s in selected) for i in range(4)}) for t in TASKS},
        image_overlap={a:{b:len(image_sets[a]&image_sets[b]) for b in TASKS} for a in TASKS},
        sample_manifest_sha256=sha256(OUTPUT/'sample_manifest.json'),
        shard_sha256={str(i):sha256(OUTPUT/f'manifests/shard{i}.json') for i in range(4)},
        protocol_sha256=digest(PROTOCOL), scorer_validation_sha256=sha256(OUTPUT/'scorer_validation.json'))
    freeze_json(OUTPUT/'data_manifest.json', audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    freeze()

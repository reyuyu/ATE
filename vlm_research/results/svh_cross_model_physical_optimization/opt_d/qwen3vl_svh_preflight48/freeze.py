"""CPU-only binding of Plan-B task bodies to the pinned Qwen processor."""
import csv
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from ..qwen3vl_early_windows.experiment import tensor_digest
from ..qwen3vl_svh_numerical_repair.run import GATE, MODEL, sha, source_files

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'vlm_research/results/qwen3vl_svh_expanded_preflight48'
AUDIT = ROOT / 'vlm_research/results/qwen3vl_svh_six_official_benchmarks_2x3090'
PHASE = Path('/data/lk/ATE-resources/benchmarks/phase4_0/sources')
DATA = Path('/data/lk/ATE-benchmarks-20261105/datasets')
TASKS = ['GQA', 'ScienceQA-IMG', 'TextVQA', 'POPE', 'MME', 'MMBench']
COUNTS = dict(zip(TASKS, [12578, 2017, 5000, 9000, 2374, 4377]))
SEED = 20261110


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + '\n')
    os.replace(tmp, path)


def table(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        w = csv.DictWriter(stream, fields, lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()})


def ranked(rows, label):
    return sorted(rows, key=lambda r: digest([SEED, label, r['key']]))


def options_body(question, options, hint):
    return ((f'Hint: {hint}\n' if hint else '') + f'Question: {question}\nOptions:\n'
            + '\n'.join(f'{k}. {v}' for k, v in options.items())
            + '\nPlease select the correct answer from the options above. \n')


def load_inventory():
    registry = json.loads((AUDIT / 'dataset_manifest.json').read_text())
    manifest = Path(registry['manifest_path'])
    assert sha(manifest) == registry['manifest_sha256']
    with gzip.open(manifest, 'rt') as stream:
        old = [json.loads(line) for line in stream]
    gqa_path = Path('/data/lk/ATE-resources/data/gqa-testdev-balanced-official/testdev_balanced_questions.json')
    sqa_path = PHASE / 'scienceqa/data/scienceqa/problems.json'
    splits_path = PHASE / 'scienceqa/data/scienceqa/pid_splits.json'
    tv_path = PHASE / 'textvqa/TextVQA_0.5.1_val.json'
    tsv_path = DATA / 'MMBench/mmbench_dev_20230712.tsv'
    sources = [manifest, gqa_path, sqa_path, splits_path, tv_path, tsv_path]
    gqa = json.loads(gqa_path.read_text())
    sqa = json.loads(sqa_path.read_text())
    test = json.loads(splits_path.read_text())['test']
    tv = {str(r['question_id']): r for r in json.loads(tv_path.read_text())['data']}
    csv.field_size_limit(2**30)
    with tsv_path.open() as stream:
        mm = {str(r['index']): {k: v for k, v in r.items() if k != 'image'}
              for r in csv.DictReader(stream, delimiter='\t')}
    assert {r['question_id'] for r in old if r['benchmark'] == 'ScienceQA-IMG'} == {
        str(i) for i in test if sqa[str(i)]['image']}
    rows = []
    for r in old:
        task, qid = r['benchmark'], r['question_id']
        if task == 'POPE':
            continue
        row = dict(key=f'{task}:{qid}', benchmark=task, question_id=qid,
                   image_id=r['image_id'], image_path=r['image_path'], image_sha256=r['image_sha256'],
                   question=r['question'], gold=r['answer'], max_new_tokens=32 if task == 'GQA' else 128)
        if task == 'GQA':
            s = gqa[qid]
            assert (s['question'], s['answer'], str(s['imageId'])) == (r['question'], r['answer'], str(r['image_id']))
            row['body'] = r['question'] + '\nAnswer the question using a single word or phrase.'
            row['auxiliary_information'] = []
        elif task == 'ScienceQA-IMG':
            s = sqa[qid]
            assert s['question'] == r['question'] and s['answer'] == r['answer']
            assert list(r['options'].values()) == s['choices'] and r['raw']['hint'] == s['hint']
            row.update(options=r['options'], hint=s['hint'])
            row['body'] = options_body(s['question'], r['options'], s['hint'])
        elif task == 'TextVQA':
            s = tv[qid]
            assert (s['question'], s['image_id'], s['answers']) == (r['question'], r['image_id'], r['answer'])
            assert len(s['answers']) == 10 and r['paper_ocr'] and r['ocr_text'] == r['user_text']
            assert 'Reference OCR token:' in r['ocr_text']
            row.update(body=r['ocr_text'], ocr_policy='FROZEN_PAPER_OCR_INPUT_VARIANT_NOT_ORIGINAL_NO_OCR_PLAN_B')
        elif task == 'MME':
            task_name, filename = r['image_id'].split('/', 1)
            p = DATA / 'MME/MME_Benchmark_release_version' / task_name / Path(filename).with_suffix('.txt')
            if not p.exists():
                p = p.parent / 'questions_answers_YN' / p.name
            lines = [line.strip().split('\t') for line in p.read_text().splitlines() if line.strip()]
            assert [r['question'], r['answer']] in [[q, a.lower()] for q, a in lines]
            sources.append(p)
            row.update(body=r['question'], task=task_name, pair_id=r['pair_id'])
        elif task == 'MMBench':
            s = mm[qid]
            assert s['question'] == r['question'] and s['answer'] == r['answer']
            assert all(s[k] == v for k, v in r['options'].items())
            hint = s.get('hint', '')
            hint = '' if hint.lower() in ('nan', 'none') else hint
            row.update(body=options_body(r['question'], r['options'], hint), hint=hint,
                       options=r['options'], circular_group=int(qid) % 1000000,
                       dataset_version='LEGACY_EN_DEV_20230712_NOT_V1_1')
        rows.append(row)
    for strategy in ['random', 'popular', 'adversarial']:
        p = PHASE / f'pope/output/coco/coco_pope_{strategy}.json'
        sources.append(p)
        src = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        assert len(src) == 3000
        for r in src:
            path = DATA / 'COCO2014/val2014' / r['image']
            qid = f"{strategy}:{r['question_id']}"
            rows.append(dict(key=f'POPE:{qid}', benchmark='POPE', question_id=qid,
                             image_id=r['image'], image_path=str(path), question=r['text'],
                             body=r['text'] + '\nPlease answer yes or no.', gold=r['label'],
                             strategy=strategy, dataset_version='RUCAIBOX_COCO_ORIGINAL_9000', max_new_tokens=128))
    assert Counter(r['benchmark'] for r in rows) == COUNTS
    assert len({r['key'] for r in rows}) == len(rows)
    return rows, {str(p): sha(p) for p in sorted(set(sources))}


def prepare(processor, row):
    assert sha(row['image_path']) == row['image_sha256']
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': row['image_path']},
                                           {'type': 'text', 'text': row['body']}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(row['image_path']) as im:
        inputs = processor(text=[prompt], images=[im.convert('RGB')], return_tensors='pt',
                           min_pixels=65536, max_pixels=2097152)
    ids = inputs['input_ids'][0].tolist()
    visual = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    positions = [i for i, tid in enumerate(ids) if tid == visual]
    grid = inputs['image_grid_thw'][0].tolist()
    assert positions and positions == list(range(positions[0], positions[-1] + 1))
    assert len(positions) == math.prod(grid) // 4 == row['image_tokens']
    assert len(ids) == row['input_length']
    return inputs, dict(prompt=prompt, input_ids=ids, input_length=len(ids), p_minus_1=len(ids) - 1,
                        image_positions=positions, image_tokens=len(positions), image_grid_thw=grid,
                        input_tensor_hashes={k: tensor_digest(v) for k, v in inputs.items()})


def grouped_exact(rows, field, count, label):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    # Exact-size subset selection preserves complete pairs/circular groups.
    dp = {0: []}
    for key in sorted(groups, key=lambda k: digest([SEED, label, str(k)])):
        n = len(groups[key])
        for total in sorted(list(dp), reverse=True):
            if total + n <= count and total + n not in dp:
                dp[total + n] = dp[total] + [key]
        if count in dp:
            return [r for k in dp[count] for r in sorted(groups[k], key=lambda r: r['key'])]
    raise ValueError(('Cannot preserve complete groups at requested size', field, count))


def choose_pretest(rows):
    selected = []
    for task in TASKS:
        pool = [r for r in rows if r['benchmark'] == task]
        if task == 'MME':
            selected += grouped_exact(pool, 'pair_id', 48, task)
        elif task == 'MMBench':
            selected += grouped_exact(pool, 'circular_group', 48, task)
        elif task == 'POPE':
            for strategy in ['random', 'popular', 'adversarial']:
                selected += ranked([r for r in pool if r['strategy'] == strategy], task + strategy)[:16]
        else:
            selected += ranked(pool, task)[:48]
    assert Counter(r['benchmark'] for r in selected) == {t: 48 for t in TASKS}
    return selected


def choose_edges(rows):
    reasons = defaultdict(list)
    index = {r['key']: r for r in rows}
    for task in TASKS:
        pool = [r for r in rows if r['benchmark'] == task]
        for axis in ['image_tokens', 'input_length']:
            ordered = sorted(pool, key=lambda r: (r[axis], digest([SEED, r['key']])))
            for name, q in [('min', 0), ('median', .5), ('p95', .95), ('p99', .99), ('max', 1)]:
                row = ordered[math.ceil(q * (len(ordered) - 1))]
                reasons[row['key']].append(f'{task}/{axis}/{name}')
    return [dict(sample=index[key], reasons=why) for key, why in reasons.items()]


def freeze():
    import torch
    from .score import adapters, scorer_sources
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    assert not (OUT / 'protocol.json').exists(), 'Frozen run exists; do not overwrite'
    rows, sources = load_inventory()
    adapters()
    sources.update({path: sha(path) for path in scorer_sources().values()})
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    images = {}
    for row in rows:
        path = row['image_path']
        if path not in images:
            with Image.open(path) as im:
                width, height = im.size
            h, w = smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=2097152)
            images[path] = dict(sha256=sha(path), width=width, height=height,
                                resized_height=h, resized_width=w, image_tokens=h*w//1024)
        image = images[path]
        assert 'image_sha256' not in row or image['sha256'] == row['image_sha256'], path
        row['image_sha256'] = image['sha256']
        row['image_tokens'] = image['image_tokens']
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': path},
                                               {'type': 'text', 'text': row['body']}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = processor.tokenizer(prompt, add_special_tokens=False)['input_ids']
        assert ids.count(processor.tokenizer.convert_tokens_to_ids(processor.image_token)) == 1
        row['input_length'] = len(ids) + row['image_tokens'] - 1
        row['prompt_sha256'] = hashlib.sha256(prompt.encode()).hexdigest()
    edges = choose_edges(rows)
    pretest = choose_pretest(rows)
    selected = {r['key']: r for r in pretest}
    selected.update({r['sample']['key']: r['sample'] for r in edges})
    layouts = {}
    for key, row in selected.items():
        inputs, layout = prepare(processor, row)
        layouts[key] = layout
        del inputs
    write(OUT / 'sample_manifest.json', pretest)
    write(OUT / 'numerical_cases.json', edges)
    write(OUT / 'input_bindings.json', layouts)
    write(OUT / 'image_inventory.json', images)
    with gzip.open(OUT / 'inventory.jsonl.gz', 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    table(OUT / 'length_boundaries.csv', [dict(key=r['sample']['key'], image_tokens=r['sample']['image_tokens'],
          input_length=r['sample']['input_length'], reasons=r['reasons']) for r in edges])
    local_sources = {str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')}
    write(OUT / 'protocol.json', dict(base_commit='c95e688e3683c3bee68d983dddb3adb0bc40c5f8',
          plan='B_ATE_UNIFIED_HF_NON_AUTHOR_OFFICIAL', plan_A='BLOCKED', gate=GATE,
          native_noop='strict torch.equal on all returned valid logits', seed=SEED,
          model=str(MODEL), revision=MODEL.name, dtype='bfloat16', attention='eager',
          torch=torch.__version__, transformers=__import__('transformers').__version__,
          logits_to_keep=1, min_pixels=65536, max_pixels=2097152,
          generation=dict(do_sample=False, num_beams=1, repetition_penalty=1., eos=[151645,151643], pad=151643),
          tasks=COUNTS, caps={t: 32 if t == 'GQA' else 128 for t in TASKS},
          textvqa_amendment='User-requested frozen Paper-OCR body; NOT the old no-OCR Plan-B input',
          mmbench='Legacy20230712 exact_matching diagnostic; NOT author V1.1 or API Judge',
          edges='Per task min/median/P95/P99/max for visual count and total prompt length; union, no output selection',
          long_decode_minimum_incremental_steps=16, numerical_cases=len(edges), pretest_per_task=48,
          full_evaluation_authorized=False, stop_on_any_failure=True, source_files={**source_files(), **local_sources},
          source_annotations=sources, frozen_files={name:sha(OUT/name) for name in [
              'sample_manifest.json', 'numerical_cases.json', 'input_bindings.json',
              'image_inventory.json', 'inventory.jsonl.gz', 'length_boundaries.csv']}))
    write(OUT / 'cpu_preparation.json', dict(status='PASS', inventory_rows=len(rows),
          unique_image_paths=len(images), actual_processor_bindings=len(layouts),
          numerical_cases=len(edges), cuda_initialized=torch.cuda.is_initialized(),
          counts=dict(Counter(r['benchmark'] for r in rows))))
    print('CPU FREEZE', len(rows), len(images), len(edges), len(layouts), flush=True)


if __name__ == '__main__':
    freeze()

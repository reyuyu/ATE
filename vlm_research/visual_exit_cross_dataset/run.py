import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import torch

from attention_position_probe import append_answer_inputs
from benchmarks.dataset_registry import OUTPUT as SUITE, digest, freeze_json, load_manifest, sha256
from benchmarks.protocols import MODEL_PATH, prompt_text
from benchmarks.scoring import generated_score
from benchmarks.smoke import hook_footprint, preflight, state_hash
from knockout_probe import tensor_hash
from physical_exit import TOLERANCES, aligned, equivalence, measured_forward
from vlm_common import load_model, prepare_inputs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/visual_exit_cross_dataset64'
UNITS = ('scienceqa_img', 'textvqa')
CONDITIONS = ('Full', 'K16', 'K24')
SEED = 20261029


def select_samples(rows, count=32):
    pool = sorted(rows, key=lambda r: (r['question_id'], r['image_sha256']))
    random.Random(SEED).shuffle(pool)
    result, used = [], set()
    for row in pool:
        if row['image_sha256'] in used:
            continue
        result.append(row)
        used.add(row['image_sha256'])
        if len(result) == count:
            return sorted(result, key=lambda r: r['question_id'])
    raise ValueError('Insufficient distinct actual images; never resample by model output')


def source_hashes():
    names = ['physical_exit.py', 'knockout_probe.py', 'vlm_common.py', 'attention_position_probe.py',
             'benchmarks/dataset_registry.py', 'benchmarks/protocols.py', 'benchmarks/scoring.py',
             'benchmarks/smoke.py', 'benchmarks/_vendor/m4c_evaluator.py', 'test_visual_exit_cross_dataset.py']
    return {name: sha256(ROOT/name) for name in names} | {
        str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'visual_exit_cross_dataset').glob('*.py'))}


def freeze(output):
    samples, manifests = {}, {}
    for unit in UNITS:
        path = SUITE/f'manifests/{unit}.json'
        manifest = load_manifest(path)
        samples[unit] = select_samples(manifest['samples'])
        manifests[unit] = sha256(path)
    config = {'seed': SEED, 'samples': samples, 'parent_manifests_sha256': manifests,
        'source_sha256': source_hashes(), 'tolerances': TOLERANCES,
        'generation': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 32,
                       'use_cache': False, 'eos_token_id': 2},
        'conditions': list(CONDITIONS), 'model_path': MODEL_PATH,
        'suite_protocol_sha256': sha256(SUITE/'evaluation_protocols.json'),
        'selection': 'For each frozen 500-question unit: sort IDs, Random(20261029).shuffle; '
                     'first32 distinct actual image SHA256; then sort selected IDs. No predictions used.',
        'smoke_ids': {unit: rows[0]['question_id'] for unit, rows in samples.items()},
        'protocol': 'Every step recomputes complete original prefix, no KV cache, all32 layers. '
                    'Physical K16/K24 delete all576 images immediately before L16/L24. '
                    'Keep original text positions and RoPE. Full uses native uncached forward. '
                    'Unchanged Phase4 prompts/scorers; ScienceQA letter accuracy; TextVQA ten-vote soft score.'}
    freeze_json(output/'manifest.json', config)
    return config


def verify_config(output):
    config = json.loads((output/'manifest.json').read_text())
    if config['source_sha256'] != source_hashes() or config['tolerances'] != TOLERANCES:
        raise ValueError('Frozen source/tolerances changed')
    for unit in UNITS:
        path = SUITE/f'manifests/{unit}.json'
        if sha256(path) != config['parent_manifests_sha256'][unit]:
            raise ValueError('Parent manifest changed')
        manifest = load_manifest(path)
        if select_samples(manifest['samples']) != config['samples'][unit]:
            raise ValueError('Frozen selection inconsistent')
    if sha256(SUITE/'evaluation_protocols.json') != config['suite_protocol_sha256']:
        raise ValueError('Scoring/prompt protocol changed')
    return config


def actual_inputs(model, processor, sample):
    inputs, _ = prepare_inputs(model, processor, sample['image_path'], prompt_text(sample))
    mapping = sample['metadata']['input_mapping']
    if inputs['input_ids'][0].tolist() != mapping['prompt_ids']:
        raise AssertionError('Actual prompt differs from frozen input')
    if int((inputs['input_ids'] == model.config.image_token_index).sum()) != 576:
        raise AssertionError('Not576 image tokens')
    return inputs


def finite_output(output):
    if not torch.isfinite(output['hidden']).all() or not torch.isfinite(output['logits']).all():
        raise AssertionError('Nonfinite hidden/logits')


def step_layout(output, inputs, condition, image_token_id):
    n = inputs['input_ids'].shape[1]
    cutoff = int(condition[1:]) if condition.startswith('K') else 32
    expected = torch.where(inputs['input_ids'][0] != image_token_id)[0] if cutoff < 32 else torch.arange(n, device=inputs['input_ids'].device)
    if not torch.equal(output['original_positions'], expected):
        raise AssertionError('Original retained text positions changed')
    if output['physical_deletion'] != (cutoff < 32):
        raise AssertionError('Physical deletion flag incorrect')
    for record in output['layers']:
        size = n-576 if record['layer'] >= cutoff and cutoff < 32 else n
        if record['sequence_length'] != size or record['mask_shape'] != [1, 1, size, size]:
            raise AssertionError('Actual physical layer sequence/mask length incorrect')
        # The local physical implementation selects original position IDs and
        # rotary cos/sin at the boundary; its returned text positions audit this.
    if cutoff < 32 and output['original_text_position_ids'] != expected.cpu().tolist():
        raise AssertionError('Physical text position IDs were renumbered')
    return {'input_length': n, 'retained_length': len(expected),
            'retained_original_positions': expected.cpu().tolist(), 'layers': output['layers']}


def compare_mask(model, inputs, physical, cutoff):
    mask = measured_forward(model, inputs, 'mask', cutoff)
    text = torch.where((inputs['input_ids'][0] != model.config.image_token_index) & inputs['attention_mask'][0].bool())[0]
    result = equivalence(aligned(mask, text, 'hidden'), aligned(physical, text, 'hidden'),
                         aligned(mask, text, 'logits'), aligned(physical, text, 'logits'))
    if not result['passed']:
        raise AssertionError(f'Mask/Physical exceeds unchanged tolerances: {result}')
    return result


def generate(model, processor, prompt, condition, smoke=False):
    ids, steps = [], []
    start = time.perf_counter()
    for index in range(32):
        inputs = append_answer_inputs(prompt, ids) if ids else prompt
        cutoff = int(condition[1:]) if condition != 'Full' else 32
        output = measured_forward(model, inputs, 'full' if condition == 'Full' else 'physical', cutoff)
        finite_output(output)
        layout = step_layout(output, inputs, condition, model.config.image_token_index)
        logits = aligned(output, [inputs['input_ids'].shape[1]-1], 'logits')[0, 0]
        token = int(logits.argmax())
        record = {'step': index, 'token_id': token,
                  'full_vocabulary_logp': float(logits.double().log_softmax(-1)[token]),
                  'positions_and_lengths': layout, 'measurement': output['measurement']}
        if smoke and condition != 'Full':
            record['mask_equivalence'] = compare_mask(model, inputs, output, cutoff)
        ids.append(token)
        steps.append(record)
        del logits, output
        if token == processor.tokenizer.eos_token_id:
            break
    return {'token_ids': ids, 'text': processor.tokenizer.decode(ids, skip_special_tokens=True),
        'eos_observed': bool(ids and ids[-1] == processor.tokenizer.eos_token_id),
        'length_limit_reached': len(ids) == 32 and ids[-1] != processor.tokenizer.eos_token_id,
        'steps': steps, 'elapsed_seconds': time.perf_counter()-start, 'use_cache': False}


def smoke(model, processor, config, output):
    path = output/'smoke.json'
    if path.exists():
        gate = json.loads(path.read_text())
        if not gate['passed'] or gate['manifest_sha256'] != digest(config):
            raise ValueError('Smoke gate mismatch')
        return gate
    results = []
    for unit in UNITS:
        sample = config['samples'][unit][0]
        prompt = actual_inputs(model, processor, sample)
        full = measured_forward(model, prompt, 'full', 32)
        k32 = measured_forward(model, prompt, 'physical', 32)
        if not torch.equal(full['logits'], k32['logits']) or not torch.equal(full['hidden'], k32['hidden']):
            raise AssertionError('K32 is not bitwise Native Full')
        step_layout(k32, prompt, 'K32', model.config.image_token_index)
        del full, k32
        own = generate(model, processor, prompt, 'Full')
        native = {}
        with torch.inference_mode():
            for cached in (False, True):
                sequence = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=32,
                    use_cache=cached, eos_token_id=2, pad_token_id=processor.tokenizer.eos_token_id,
                    temperature=1., top_p=1.)
                tokens = sequence[0, prompt['input_ids'].shape[1]:].cpu().tolist()
                if tokens != own['token_ids']:
                    raise AssertionError(f'Full uncached vs native generation mismatch, cache={cached}')
                native[str(cached)] = tokens
        exited = {condition: generate(model, processor, prompt, condition, smoke=True) for condition in ('K16', 'K24')}
        results.append({'unit': unit, 'question_id': sample['question_id'], 'passed': True,
            'full_uncached_matches_native_cached_and_uncached_tokens': True, 'native_generation_tokens': native,
            'k32_logits_hidden_bitwise_exact': True, 'full_generation': own,
            'physical_generation': exited, 'every_exit_generation_prefix_mask_equivalence_passed': True})
        print('SMOKE PASSED', unit, sample['question_id'], flush=True)
    gate = {'passed': True, 'manifest_sha256': digest(config), 'tolerances': TOLERANCES,
            'tolerances_changed': False, 'samples': results}
    freeze_json(path, gate)
    return gate


def score(sample, result):
    metrics = generated_score(sample, result['text'])
    key = 'official_generation_correct' if sample['dataset'] == 'scienceqa_img' else 'official_generation_vqa_soft_accuracy'
    return float(metrics[key]), metrics


def checkpoint(path, data):
    temporary = path.with_suffix('.partial')
    with temporary.open('w') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
    os.link(temporary, path)
    temporary.unlink()


def summarize(output, config):
    rows, aggregates, csvrows = [], {}, []
    for unit in UNITS:
        group = []
        for sample in config['samples'][unit]:
            row = json.loads((output/unit/f'{sample["question_id"]}.json').read_text())
            if row['manifest_sha256'] != digest(config) or row['image_sha256'] != sample['image_sha256']:
                raise ValueError('Incomplete/changed checkpoint')
            group.append(row)
            for condition in CONDITIONS:
                result = row['conditions'][condition]
                csvrows.append({'unit': unit, 'question_id': row['question_id'], 'image_sha256': row['image_sha256'],
                    'question': sample['question'], 'gold': json.dumps(sample['answer'], ensure_ascii=False),
                    'condition': condition, 'generated_answer': result['text'], 'score': result['score'],
                    'score_change_vs_full': result['score']-row['conditions']['Full']['score'],
                    'token_ids': json.dumps(result['token_ids']), 'eos_observed': result['eos_observed'],
                    'elapsed_seconds': result['elapsed_seconds']})
        entry = {'questions': len(group), 'unique_image_sha256': len({r['image_sha256'] for r in group}),
                 'scores': {c: sum(r['conditions'][c]['score'] for r in group)/len(group) for c in CONDITIONS}, 'changes': {}}
        for condition in ('K16', 'K24'):
            differences = [(r['conditions'][condition]['score']-r['conditions']['Full']['score'], r) for r in group]
            entry['changes'][condition] = {'score_change': entry['scores'][condition]-entry['scores']['Full'],
                'score_increased': sum(d > 0 for d, r in differences), 'score_decreased': sum(d < 0 for d, r in differences),
                'score_unchanged': sum(d == 0 for d, r in differences),
                'full_correct_to_wrong': sum(r['conditions']['Full']['score'] == 1 and r['conditions'][condition]['score'] == 0 for r in group)
                                         if unit == 'scienceqa_img' else None,
                'full_wrong_to_correct': sum(r['conditions']['Full']['score'] == 0 and r['conditions'][condition]['score'] == 1 for r in group)
                                         if unit == 'scienceqa_img' else None,
                'worst_cases': [{'question_id': r['question_id'], 'score_change': d,
                    'full_answer': r['conditions']['Full']['text'], 'exit_answer': r['conditions'][condition]['text'],
                    'full_score': r['conditions']['Full']['score'], 'exit_score': r['conditions'][condition]['score']}
                    for d, r in sorted(differences, key=lambda x: (x[0], x[1]['question_id'])) if d < 0][:3]}
        aggregates[unit] = entry
        rows.extend(group)
    freeze_json(output/'aggregate.json', {'status': 'complete', 'manifest_sha256': digest(config), 'units': aggregates,
        'sample_count': len(rows), 'paired_conditions': list(CONDITIONS), 'pilot_only': True})
    path = output/'per_sample.csv'
    if not path.exists():
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csvrows[0]))
            writer.writeheader()
            writer.writerows(csvrows)
    lines = ['# 跨数据集 Physical Exit 64题 Pilot', '',
        '固定种子20261029，Phase4冻结500题中各选32张不同SHA256图片；未根据模型结果选题。',
        '同一RTX3090、FP16/eager、Transformers4.49.0、原始Prompt和评分器。',
        '三个条件均Greedy、无KV Cache，每步完整重算，最多32个Token，EOS提前结束。', '',
        '| 数据集 | 题数 | Full | K16 | K24 |', '|---|---:|---:|---:|---:|']
    for unit, entry in aggregates.items():
        lines.append(f'| {unit} | 32 | '+' | '.join(f'{100*entry["scores"][c]:.2f}%' for c in CONDITIONS)+' |')
    lines += ['', '## 分数变化和翻转', '']
    for unit, entry in aggregates.items():
        for condition, changes in entry['changes'].items():
            counts = (f'原对→错{changes["full_correct_to_wrong"]}，原错→对{changes["full_wrong_to_correct"]}' if unit == 'scienceqa_img'
                      else f'软评分上升{changes["score_increased"]}，下降{changes["score_decreased"]}，不变{changes["score_unchanged"]}')
            lines.append(f'- {unit} {condition}：{changes["score_change"]*100:+.2f}个百分点；{counts}。')
    lines += ['', '## 最大退化案例', '', '按评分下降排序，同分按Question ID排序，不事后挑选。']
    for unit, entry in aggregates.items():
        for condition, changes in entry['changes'].items():
            lines.append(f'\n### {unit} / {condition}')
            for case in changes['worst_cases']:
                lines.append(f'- {case["question_id"]}：Full `{case["full_answer"]}`（{case["full_score"]:.2f}）'
                             f' → {condition} `{case["exit_answer"]}`（{case["exit_score"]:.2f}）。')
            if not changes['worst_cases']:
                lines.append('- 无评分退化样本。')
    losses = {unit: max(entry['scores']['Full']-entry['scores'][c] for c in ('K16', 'K24')) for unit, entry in aggregates.items()}
    sensitive = max(losses, key=losses.get)
    lines += ['', '## 结论及边界', '',
        f'本Pilot以最坏平均分下降衡量，{sensitive}更敏感；两个任务评分含义不同，仅描述本批样本，不能推断总体敏感性。',
        '正确性Smoke：每个数据集1题，Full无缓存生成与原生有/无缓存生成Token均一致；K32初始Hidden/Logits精确一致；',
        'K16/K24各Smoke生成前缀均通过原Mask参考和预先冻结数值容差；每步实际物理长度、原文本位置与RoPE复用均审计。',
        '未修改模型权重、基础算法、容差、Prompt、评分器；没有训练、Cache算法或额外干预。',
        '每组仅32题，没有等价性检验，不宣称性能等价或通用提升。实际计时含审计与重算，不作为稳定加速基准。']
    report = output/'report.md'
    if not report.exists():
        report.write_text('\n'.join(lines)+'\n')


def run(output):
    config = verify_config(output)
    cpu = json.loads((output/'cpu_validation.json').read_text())
    if not cpu['passed'] or cpu['source_sha256'] != config['source_sha256']:
        raise ValueError('CPU gate not passed')
    lock = (output/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = Path('/data/lk/ATE-resources/benchmarks/phase4_0/gpu0_smoke.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before_gpu = preflight()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    model, completed = None, 0
    try:
        model, processor, environment = load_model(MODEL_PATH)
        before = state_hash(model)
        if before != json.loads((SUITE/'smoke/validation.json').read_text())['model_weights_buffers_before_sha256']:
            raise AssertionError('Loaded model differs from fixed Phase4 model')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        envpath = output/'environment.json'
        environment.update(model_sha256=before, code_commit=commit, manifest_sha256=digest(config), preflight=before_gpu,
                           started_utc=started_utc)
        if not envpath.exists():
            freeze_json(envpath, environment)
        else:
            old = json.loads(envpath.read_text())
            if any(old[k] != environment[k] for k in ('model_sha256', 'code_commit', 'manifest_sha256')):
                raise ValueError('Resume binding mismatch')
        hooks, versions = hook_footprint(model), {n: p._version for n, p in model.named_parameters()}
        smoke(model, processor, config, output)
        for unit in UNITS:
            folder = output/unit
            folder.mkdir(exist_ok=True)
            for sample in config['samples'][unit]:
                path = folder/f'{sample["question_id"]}.json'
                if path.exists():
                    row = json.loads(path.read_text())
                    if row['manifest_sha256'] != digest(config) or row['image_sha256'] != sample['image_sha256']:
                        raise ValueError('Checkpoint changed')
                else:
                    inputs = actual_inputs(model, processor, sample)
                    hashes = {k: tensor_hash(v) for k, v in inputs.items()}
                    results = {}
                    for condition in CONDITIONS:
                        generated = generate(model, processor, inputs, condition)
                        value, metrics = score(sample, generated)
                        generated.update(score=value, metrics=metrics)
                        results[condition] = generated
                    if hashes != {k: tensor_hash(v) for k, v in inputs.items()}:
                        raise AssertionError('Input mutation')
                    if hooks != hook_footprint(model) or versions != {n: p._version for n, p in model.named_parameters()}:
                        raise AssertionError('Hook/weight version mutation')
                    row = {'question_id': sample['question_id'], 'image_sha256': sample['image_sha256'],
                           'unit': unit, 'manifest_sha256': digest(config), 'conditions': results, 'input_sha256': hashes}
                    checkpoint(path, row)
                completed += 1
                print(json.dumps({'completed': completed, 'unit': unit, 'qid': sample['question_id'],
                    'scores': {c: row['conditions'][c]['score'] for c in CONDITIONS}}, ensure_ascii=False), flush=True)
        after = state_hash(model)
        if before != after or hooks != hook_footprint(model):
            raise AssertionError('Loaded model/buffers or hooks changed')
        summarize(output, config)
        freeze_json(output/'validation.json', {'passed': True, 'completed': completed, 'total': 64,
            'manifest_sha256': digest(config), 'model_before_sha256': before, 'model_after_sha256': after,
            'no_nan_inf_oom': True, 'inputs_hooks_versions_unchanged': True, 'all_step_physical_layouts_checked': True,
            'started_utc': started_utc, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'wall_seconds_including_smoke': time.perf_counter()-started, 'code_commit': commit})
    except BaseException as error:
        freeze_json(output/f'failure_{time.time_ns()}.json', {'status': 'partial', 'completed': completed,
            'error': repr(error), 'traceback': traceback.format_exc(), 'valid_scores_reported': False})
        raise
    finally:
        if model is not None:
            del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        gpu_lock.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        config = freeze(args.output)
        print({unit: [r['question_id'] for r in rows] for unit, rows in config['samples'].items()})
    else:
        run(args.output)


if __name__ == '__main__':
    main()

"""Frozen sampling, smoke, no-cache generation, resumable paired run and report."""
import argparse
from contextlib import ExitStack
import csv
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor

from attention_position_probe import append_answer_inputs, locate_question_tokens
from benchmark_interventions.hooks import ORIGINALS, original_modules
from benchmarks.dataset_registry import digest, freeze_json, load_manifest, sha256
from benchmarks.protocols import MODEL_PATH, prompt_text
from benchmarks.scoring import diagnostic_candidate_metrics, generated_score
from benchmarks.smoke import hook_footprint, preflight, state_hash
from knockout_probe import model_layers, tensor_hash
from physical_exit_pilot import candidate_inputs, sequence_score
from vlm_common import load_model, prepare_inputs
from .hooks import CONDITIONS, LAYERS, ResidualOnly

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/l2_l4_residual_only_pilot'
SUITE = ROOT/'results/phase4_0_dataset_suite'
REL = ROOT/'results/phase3_2_fixed_k_exit/manifest.json'
IMAGES = Path('/data/lk/ATE-resources/data/image-tasks/MMEB/GQA')
UNITS = ('gqa_chooserel', 'scienceqa_img', 'textvqa')
SEED = 20261029


def select(rows, count, balanced=False):
    rng, chosen, used = random.Random(SEED), [], set()
    for label in (('left', 'right') if balanced else (None,)):
        pool = sorted([r for r in rows if label is None or r['answer'] == label], key=lambda r: str(r['question_id']))
        rng.shuffle(pool)
        group = []
        for row in pool:
            if row['image_sha256'] not in used:
                group.append(row)
                used.add(row['image_sha256'])
            if len(group) == (count//2 if balanced else count):
                break
        if len(group) != (count//2 if balanced else count):
            raise ValueError('Insufficient distinct images; do not replace by model results')
        chosen.extend(group)
    return sorted(chosen, key=lambda r: r['question_id'])


def text_prompt(sample):
    return sample['question'] if sample['dataset'] == 'gqa_chooserel' else prompt_text(sample)


def source_hashes():
    paths = [ROOT/name for name in ('vlm_common.py', 'knockout_probe.py', 'attention_position_probe.py',
             'physical_exit_pilot.py', 'benchmarks/scoring.py', 'benchmarks/dataset_registry.py',
             'benchmarks/protocols.py', 'benchmarks/smoke.py', 'benchmark_interventions/hooks.py',
             'benchmarks/_vendor/m4c_evaluator.py', 'cross_dataset_c1_c2/analyze.py',
             'test_knockout.py', 'test_residual_only_pilot.py')]
    paths += sorted((ROOT/'residual_only_pilot').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def freeze(output):
    old = json.loads(REL.read_text())
    relative = select(old['samples'], 32, balanced=True)
    samples = {'gqa_chooserel': []}
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    processor.patch_size = 14
    processor.num_additional_image_tokens = 1
    processor.vision_feature_select_strategy = 'default'
    for row in relative:
        sample = dict(row, dataset='gqa_chooserel', answer_type='closed',
                      image_path=str(IMAGES/f'image_{row["img_id"]}.jpg'))
        if sha256(sample['image_path']) != row['image_sha256']:
            raise ValueError('Relocated ChooseRel image differs')
        conversation = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text_prompt(sample)}]}]
        prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        with Image.open(sample['image_path']) as image:
            inputs = processor(images=image.convert('RGB'), text=prompt, return_tensors='pt')
        ids = inputs['input_ids'][0].tolist()
        questions, audit = locate_question_tokens(processor, sample['question'], ids, 32000)
        sample['metadata'] = {'input_mapping': {'prompt_ids': ids, 'prompt_length': len(ids),
                              'p_minus_1': len(ids)-1, 'question_positions': [r['sequence_index'] for r in questions]},
                              'question_mapping_audit': audit}
        samples['gqa_chooserel'].append(sample)
    parents = {str(REL.relative_to(ROOT)): sha256(REL)}
    for unit in UNITS[1:]:
        path = SUITE/f'manifests/{unit}.json'
        samples[unit] = select(load_manifest(path)['samples'], 16)
        parents[str(path.relative_to(ROOT))] = sha256(path)
    config = {'seed': SEED, 'samples': samples, 'parent_sha256': parents,
              'source_sha256': source_hashes(), 'original_sha256': {k: sha256(p) for k, p in ORIGINALS.items()},
              'protocol_sha256': sha256(SUITE/'evaluation_protocols.json'), 'conditions': list(CONDITIONS),
              'selection': 'String question_id sort, Random(20261029) shuffle, first distinct image SHA256; '
                           'ChooseRel sequential left16/right16, ScienceQA16/TextVQA16. No predictions used.',
              'generation': {'max_new_tokens': 32, 'do_sample': False, 'use_cache': False, 'eos_token_id': 2},
              'interventions': '0-based L2,L3,L4. P1-Joint zeros only original prompt P-1 o_proj and MLP '
                               'outputs on EVERY complete-prefix forward, never the new final position. '
                               'Whole-Skip returns identical input residual for ALL tokens without attention/MLP.',
              'prompt': 'ChooseRel original raw question/chat template; other units unchanged frozen Phase4 prompts.',
              'candidate_protocol': 'Full-vocabulary sum over entire original candidate; no EOS or length normalization; '
                                    'gold minus strongest wrong. No fabricated TextVQA candidates.',
              'smoke_ids': {unit: rows[0]['question_id'] for unit, rows in samples.items()},
              'model_path': MODEL_PATH, 'no_speedup_claim': True}
    freeze_json(output/'manifest.json', config)
    return config


def verify(output):
    config = json.loads((output/'manifest.json').read_text())
    if config['source_sha256'] != source_hashes():
        raise ValueError('Frozen execution source changed')
    for name, expected in config['parent_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Frozen parent changed')
    if sha256(SUITE/'evaluation_protocols.json') != config['protocol_sha256']:
        raise ValueError('Frozen scoring protocol changed')
    for rows in config['samples'].values():
        for sample in rows:
            if sha256(sample['image_path']) != sample['image_sha256']:
                raise ValueError('Frozen image changed')
    return config


def actual_inputs(model, processor, sample):
    inputs, counts = prepare_inputs(model, processor, sample['image_path'], text_prompt(sample))
    mapping = sample['metadata']['input_mapping']
    if inputs['input_ids'][0].tolist() != mapping['prompt_ids'] or counts['visual_tokens'] != 576:
        raise AssertionError('Actual input differs from frozen mapping')
    return inputs


def forward(model, inputs, prompt_ids, condition, zero_query, noop=False, observe=False):
    hashes = {k: tensor_hash(v) for k, v in inputs.items()}
    with torch.inference_mode(), ResidualOnly(model, prompt_ids, condition, zero_query, noop, observe) as trace:
        result = model(**inputs, use_cache=False, return_dict=True)
    if result.past_key_values is not None or not torch.isfinite(result.logits).all():
        raise AssertionError('Cache or nonfinite logits')
    if hashes != {k: tensor_hash(v) for k, v in inputs.items()}:
        raise AssertionError('Input mutation')
    return result, trace


def generate(model, processor, prompt, condition, zero_query, arrays=None):
    original = prompt['input_ids'][0].tolist()
    ids, logp, audits = [], [], []
    for step in range(32):
        inputs = append_answer_inputs(prompt, ids)
        result, trace = forward(model, inputs, original, condition, zero_query)
        logits = result.logits[0, -1].detach().cpu()
        token = int(logits.argmax())
        ids.append(token)
        logp.append(float(logits.double().log_softmax(-1)[token]))
        audits.append(trace.audit())
        if arrays is not None:
            arrays[f'{condition}_generation_{step}'] = logits.numpy()
        del result
        if token == processor.tokenizer.eos_token_id:
            break
    return {'generated_token_ids': ids, 'generated_text': processor.tokenizer.decode(ids, skip_special_tokens=True),
            'generated_token_logp': logp, 'eos_observed': ids[-1] == processor.tokenizer.eos_token_id,
            'length_limit_reached': len(ids) == 32 and ids[-1] != processor.tokenizer.eos_token_id,
            'generation_audits': audits, 'use_cache': False}


def smoke(model, processor, config, zero_query):
    checks = []
    for unit in UNITS:
        sample = config['samples'][unit][0]
        prompt = actual_inputs(model, processor, sample)
        original = prompt['input_ids'][0].tolist()
        base, native_trace = forward(model, prompt, original, 'Full', zero_query, observe=True)
        for condition in CONDITIONS[1:]:
            noop, _ = forward(model, prompt, original, condition, zero_query, noop=True)
            if not torch.equal(noop.logits, base.logits):
                raise AssertionError('No-op not exactly Full')
            del noop
            current, trace = forward(model, prompt, original, condition, zero_query, observe=True)
            if condition == 'P1-Joint' and trace.prefix != native_trace.prefix:
                raise AssertionError('Joint changed causal earlier prefix')
            # Independent computed-zero reference: only for implementation validation.
            with ExitStack() as stack, torch.inference_mode():
                for index in LAYERS:
                    layer = model_layers(model)[index]
                    for module in (layer.self_attn.o_proj, layer.mlp):
                        def zero(module, args, out, all_tokens=condition == 'Whole-Skip'):
                            result = out.clone()
                            if all_tokens:
                                result.zero_()
                            else:
                                result[:, len(original)-1] = 0
                            return result
                        handle = module.register_forward_hook(zero)
                        stack.callback(handle.remove)
                reference = model(**prompt, use_cache=False, return_dict=True)
            if not torch.equal(current.logits, reference.logits):
                raise AssertionError('Residual implementation differs from independent computed-zero reference')
            checks.append({'unit': unit, 'question_id': sample['question_id'], 'condition': condition,
                           'noop_all_logits_exact_full': True, 'independent_zero_reference_all_logits_exact': True,
                           'causal_earlier_prefix_exact': trace.prefix == native_trace.prefix if condition == 'P1-Joint' else None,
                           'audit': trace.audit()})
            del current, reference
        del base
        own = generate(model, processor, prompt, 'Full', zero_query)
        native = {}
        with torch.inference_mode():
            for cached in (False, True):
                sequence = model.generate(**prompt, max_new_tokens=32, do_sample=False, num_beams=1,
                    use_cache=cached, eos_token_id=2, pad_token_id=2, temperature=1., top_p=1.)
                tokens = sequence[0, len(original):].cpu().tolist()
                if tokens != own['generated_token_ids']:
                    raise AssertionError(f'Full tokens differ from native generation, cache={cached}')
                native[str(cached)] = tokens
        checks.append({'unit': unit, 'question_id': sample['question_id'],
                       'full_manual_uncached_matches_native_cached_and_uncached_tokens': True, 'native_token_ids': native})
        print('SMOKE PASSED', unit, sample['question_id'], flush=True)
    return {'passed': True, 'manifest_sha256': digest(config), 'checks': checks, 'tolerances_relaxed': False}


def evaluate(model, processor, sample, zero_query):
    started = time.perf_counter()
    inputs = actual_inputs(model, processor, sample)
    original = inputs['input_ids'][0].tolist()
    versions = {n: p._version for n, p in model.named_parameters()}
    footprints = hook_footprint(model)
    conditions, arrays = {}, {}
    for condition in CONDITIONS:
        scores, candidates, audits = [], [], []
        for i, (current, records) in enumerate(candidate_inputs(processor, inputs, sample['candidates']) if sample['candidates'] else []):
            out, trace = forward(model, current, original, condition, zero_query)
            logits = out.logits[0, [r['query_position'] for r in records]].detach().cpu()
            value = sequence_score(logits, [r['token_id'] for r in records])
            scores.append(value)
            candidates.append({'candidate': sample['candidates'][i], 'logp': value, 'records': records})
            arrays[f'{condition}_candidate_{i}'] = logits.numpy()
            audits.append(trace.audit())
            del out
        result = generate(model, processor, inputs, condition, zero_query, arrays)
        result['metrics'] = generated_score(sample, result['generated_text'])
        if scores:
            result['metrics'].update(diagnostic_candidate_metrics(sample, scores))
        result.update(candidate_scores=candidates, candidate_audits=audits)
        conditions[condition] = result
    if versions != {n: p._version for n, p in model.named_parameters()} or footprints != hook_footprint(model):
        raise AssertionError('Parameter/hook mutation')
    return {'question_id': sample['question_id'], 'dataset': sample['dataset'], 'answer': sample['answer'],
            'image_sha256': sample['image_sha256'], 'prompt_length': len(original), 'p_minus_1': len(original)-1,
            'conditions': conditions, 'elapsed_seconds': time.perf_counter()-started}, arrays


def scalar(result):
    metrics = result['metrics']
    return float(next(metrics[k] for k in ('official_generation_correct', 'official_generation_vqa_soft_accuracy',
                                         'mechanism_generation_literal_exact_match') if k in metrics))


def summarize(output, config):
    from cross_dataset_c1_c2.analyze import verify_raw
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    binding = digest(config)
    aggregate, flat, max_error = {}, [], 0.
    for unit in UNITS:
        rows = []
        for sample in config['samples'][unit]:
            path = output/unit/f'{sample["question_id"]}.json'
            row = json.loads(path.read_text())
            if row['manifest_sha256'] != binding or row['image_sha256'] != sample['image_sha256']:
                raise ValueError('Incomplete/changed checkpoint')
            if sha256(path.with_suffix('.npz')) != row['npz_sha256']:
                raise ValueError('Raw logits integrity failed')
            with np.load(path.with_suffix('.npz')) as archive:
                max_error = max(max_error, verify_raw(row, archive))
            for condition, result in row['conditions'].items():
                if tokenizer.decode(result['generated_token_ids'], skip_special_tokens=True) != result['generated_text']:
                    raise AssertionError('Token decode mismatch')
                recomputed = generated_score(sample, result['generated_text'])
                if any(result['metrics'][k] != v for k, v in recomputed.items()):
                    raise AssertionError('Generated score mismatch')
                if sample['candidates']:
                    expected = diagnostic_candidate_metrics(sample, [c['logp'] for c in result['candidate_scores']])
                    if any(result['metrics'][k] != v for k, v in expected.items()):
                        raise AssertionError('Candidate diagnostic mismatch')
                native = row['conditions']['Full']
                flat.append({'dataset': unit, 'question_id': sample['question_id'], 'question': sample['question'],
                             'gold': json.dumps(sample['answer'], ensure_ascii=False), 'condition': condition,
                             'generated_text': result['generated_text'], 'generated_score': scalar(result),
                             'score_gain': scalar(result)-scalar(native),
                             'margin': result['metrics'].get('diagnostic_candidate_margin'),
                             'margin_gain': result['metrics']['diagnostic_candidate_margin']-native['metrics']['diagnostic_candidate_margin'] if sample['candidates'] else None,
                             'gold_logp': result['metrics'].get('diagnostic_gold_sequence_logp'),
                             'token_ids': json.dumps(result['generated_token_ids']), 'image_sha256': row['image_sha256']})
            rows.append(row)
        summary = {'n': len(rows), 'conditions': {}}
        for condition in CONDITIONS:
            full = np.array([scalar(r['conditions']['Full']) for r in rows])
            score = np.array([scalar(r['conditions'][condition]) for r in rows])
            margins = [r['conditions'][condition]['metrics']['diagnostic_candidate_margin']-
                       r['conditions']['Full']['metrics']['diagnostic_candidate_margin'] for r in rows] if unit != 'textvqa' else []
            summary['conditions'][condition] = {'generated_score': float(score.mean()),
                'score_change': float((score-full).mean()), 'score_increased': int((score > full).sum()),
                'score_decreased': int((score < full).sum()),
                'full_wrong_to_correct': int(((full == 0) & (score == 1)).sum()) if unit != 'textvqa' else None,
                'full_correct_to_wrong': int(((full == 1) & (score == 0)).sum()) if unit != 'textvqa' else None,
                'mean_margin_gain': float(np.mean(margins)) if margins else None,
                'length_limit_count': sum(r['conditions'][condition]['length_limit_reached'] for r in rows)}
        if unit == 'gqa_chooserel':
            summary['gold_counts'] = {side: sum(s['answer'] == side for s in config['samples'][unit]) for side in ('left', 'right')}
        aggregate[unit] = summary
    weight = json.loads((output/'weight_audit.json').read_text())
    if not weight['passed'] or len(flat) != 192:
        raise AssertionError('Incomplete suite or failed weights audit')
    freeze_json(output/'aggregate.json', {'status': 'complete', 'questions': 64, 'condition_rows': 192,
        'manifest_sha256': binding, 'max_raw_logp_recompute_error': max_error, 'independent_decoding_and_scoring_passed': True,
        'code_commit': json.loads((output/'environment.json').read_text())['code_commit'], 'datasets': aggregate})
    with (output/'per_question.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(flat)
    lines = ['# L2-L4 Residual-only Pilot', '',
             '64题，固定种子20261029；ChooseRel左右各16，ScienceQA-IMG16，TextVQA16；无表现筛选。',
             '同一RTX3090、LLaVA-1.5-7B FP16/eager、Transformers4.49.0，全部条件本轮配对。',
             'Full原生；P1-Joint仅原Prompt P-1同时置零L2/L3/L4的o_proj及MLP输出；Whole-Skip全部Token直接传入残差，不运行这些层Attention或MLP。',
             '每一步重新计算完整前缀，无KV Cache，Greedy最多32Token，EOS停止；P1-Joint始终修改原P-1而非新生成位置。', '',
             '| 数据集 | 题数 | Full | P1-Joint | Whole-Skip |', '|---|---:|---:|---:|---:|']
    for unit, data in aggregate.items():
        lines.append(f'| {unit} | {data["n"]} | '+' | '.join(f'{data["conditions"][c]["generated_score"]*100:.2f}%' for c in CONDITIONS)+' |')
    lines += ['', '| 数据集 | 条件 | 评分变化pp | 纠正/软分上升 | 损害/软分下降 | 平均Margin变化nats |', '|---|---|---:|---:|---:|---:|']
    for unit, data in aggregate.items():
        for condition in CONDITIONS[1:]:
            d = data['conditions'][condition]
            margin = f'{d["mean_margin_gain"]:+.6f}' if d['mean_margin_gain'] is not None else 'N/A'
            lines.append(f'| {unit} | {condition} | {d["score_change"]*100:+.2f} | {d["score_increased"]} | {d["score_decreased"]} | {margin} |')
    lines += ['', '## 验证', '',
              '- CPU测试与三数据集真实GPU Smoke通过；Full无Cache手动生成与原生有/无Cache生成Token一致。',
              '- 两干预No-op的全部Logits与Full精确一致；独立计算零输出参考与两实现全部Logits精确一致，未放宽容差。',
              '- P1-Joint每次前向只改固定原P-1的6个子层输出，其他Token局部输出不变；目标位置在L2/L3/L4各自严格保持层输入残差。',
              '- Whole-Skip每层输入输出为同一个Tensor，全部Token精确恒等，目标层Attention/MLP调用为零；正常/异常后forward和Hook恢复。',
              '- 输入和参数版本未变化，模型完整字节哈希前后相同；全部192条生成重新解码/计分，原始Logits独立重算LogP通过。',
              f'- 原始LogP重算最大误差：{max_error:.3g}。逐题审计及答案见各数据集JSON和per_question.csv；原始NPZ仅本地保留。', '',
              '## 边界', '',
              '- 仅描述性Pilot，不推断总体等价或通用提升；没有训练、层扫描或额外消融，不报告无Cache方法加速比。',
              '- ChooseRel复用原始问题和冻结240图来源，来源名含correct_question，不能解释为无偏GQA总体。',
              '- ScienceQA评分为生成选项字母准确率；TextVQA为原10答案软评分，无OCR，软分上升/下降不是二值纠错。',
              '- Margin仅用于ChooseRel/ScienceQA，完整候选全词表LogP求和，gold减最强wrong；TextVQA无虚构候选。',
              '- 本轮Full均为同机无Cache重新运行，未混用历史Cache基线。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def run(output):
    config = verify(output)
    cpu = json.loads((output/'cpu_tests.json').read_text())
    if not cpu['passed'] or cpu['source_sha256'] != config['source_sha256']:
        raise ValueError('CPU gate failed')
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pre = preflight()
        originals = original_modules(config['original_sha256'])
        model, processor, environment = load_model(MODEL_PATH)
        if environment['transformers'] != '4.49.0':
            raise ValueError('Transformers version drift')
        before = state_hash(model)
        environment.update(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                           model_sha256=before, manifest_sha256=digest(config), preflight=pre)
        envpath = output/'environment.json'
        if not envpath.exists():
            freeze_json(envpath, environment)
        else:
            prior = json.loads(envpath.read_text())
            for key in ('code_commit', 'model_sha256', 'manifest_sha256', 'cuda_visible_devices', 'torch', 'transformers'):
                if prior[key] != environment[key]:
                    raise ValueError('Resume environment mismatch')
        try:
            if not (output/'smoke.json').exists():
                freeze_json(output/'smoke.json', smoke(model, processor, config, originals['c2'].zero_query))
            gate = json.loads((output/'smoke.json').read_text())
            if not gate['passed'] or gate['manifest_sha256'] != digest(config):
                raise ValueError('Invalid smoke')
            completed = 0
            for unit in UNITS:
                for sample in config['samples'][unit]:
                    path = output/unit/f'{sample["question_id"]}.json'
                    if path.exists():
                        previous = json.loads(path.read_text())
                        if previous['manifest_sha256'] != digest(config) or sha256(path.with_suffix('.npz')) != previous['npz_sha256']:
                            raise ValueError('Invalid resume checkpoint')
                    else:
                        row, arrays = evaluate(model, processor, sample, originals['c2'].zero_query)
                        row['manifest_sha256'] = digest(config)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_suffix('.partial.npz')
                        np.savez_compressed(temporary, **arrays)
                        os.replace(temporary, path.with_suffix('.npz'))
                        row['npz_sha256'] = sha256(path.with_suffix('.npz'))
                        temporary = path.with_suffix('.partial.json')
                        temporary.write_text(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        os.link(temporary, path)
                        temporary.unlink()
                    completed += 1
                    print('PROGRESS', completed, '/64', unit, sample['question_id'], flush=True)
            after = state_hash(model)
            if before != after:
                raise AssertionError('Model weights changed')
            freeze_json(output/'weight_audit.json', {'passed': True, 'before': before, 'after': after,
                        'completed': completed, 'manifest_sha256': digest(config), 'nan_inf_oom': False})
        except BaseException as exc:
            failure = output/f'failure_{time.time_ns()}.json'
            freeze_json(failure, {'passed': False, 'exception': repr(exc), 'traceback': traceback.format_exc()})
            raise
        finally:
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    summarize(output, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('freeze', 'run', 'summarize'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'freeze':
        freeze(args.output)
    elif args.stage == 'run':
        run(args.output)
    else:
        summarize(args.output, verify(args.output))


if __name__ == '__main__':
    main()

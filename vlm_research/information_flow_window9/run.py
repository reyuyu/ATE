"""Prepare, smoke-test, and run only the requested frozen 9-layer experiment."""
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.dataset_registry import digest, freeze_json, sha256

OUTPUT = ROOT / 'results/gpu1_information_flow_window9'
GPU = 'GPU-b08489af-0a8a-eb48-55ba-8346733af417'
MODEL = '/data/lk/ATE-resources/models/llava-1.5-7b-hf/b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
TASKS = ('gqa_chooserel', 'gqa_chooseattr', 'gqa_queryattr', 'gqa_logicalobj')
SEED = 20261029
BATCH = 4


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT / p for p in
        ('vlm_common.py', 'benchmarks/prepare_datasets.py', 'benchmarks/protocols.py',
         'benchmarks/dataset_registry.py', 'attention_position_probe.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows')
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    from transformers import AutoProcessor
    from benchmarks.prepare_datasets import token_mapping
    from benchmarks.dataset_registry import load_manifest
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    all_samples, sources = [], {}
    for task in TASKS:
        if task == 'gqa_chooserel':
            path = ROOT / 'results/phase3_2_fixed_k_exit/manifest.json'
            old = json.loads(path.read_text())['samples']
            pool = []
            for r in old:
                pool.append({'dataset': task, 'subset': 'phase3_2_frozen240',
                    'split': 'GQA_val_author_filtered', 'question_id': r['question_id'],
                    'image_id': r['img_id'], 'question': r['question'], 'answer': r['answer'],
                    'candidates': r['candidates'], 'answer_type': 'closed',
                    'image_path': f'/data/lk/ATE-resources/data/image-tasks/MMEB/GQA/image_{r["img_id"]}.jpg',
                    'image_sha256': r['image_sha256'], 'source_row_sha256': digest(r),
                    'metadata': {}})
        else:
            path = ROOT / f'results/phase4_0_dataset_suite/manifests/{task}.json'
            pool = load_manifest(path, verify_images=False)['samples']
        sources[task] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'pool_n': len(pool)}
        pool = sorted(pool, key=lambda r: (r['question_id'], r['image_id']))
        selected = random.Random(SEED).sample(pool, 24)
        for sample in sorted(selected, key=lambda r: r['question_id']):
            sample = copy.deepcopy(sample)
            if sha256(sample['image_path']) != sample['image_sha256']:
                raise ValueError(f'Image mismatch {sample["question_id"]}; no substitution allowed')
            mapping = token_mapping(processor, sample)
            previous = sample['metadata'].get('input_mapping')
            if previous is not None and previous != mapping:
                raise ValueError('Previously frozen prompt mapping differs')
            sample['metadata']['input_mapping'] = mapping
            all_samples.append(sample)
    config = {'seed': SEED, 'selection': 'sort(question_id,image_id); independent Random(seed).sample(pool,24); sort selected',
        'tasks': list(TASKS), 'n_per_task': 24, 'window': 9, 'centers': list(range(32)),
        'paths': ['image_to_last', 'question_to_last', 'image_to_question'], 'microbatch': BATCH,
        'batch_policy': 'Four replicas of same image/prompt; four window centers, same batch size for Native Full',
        'gpu_uuid': GPU, 'model_path': MODEL, 'dtype': 'float16', 'attention': 'eager', 'use_cache': False,
        'metric': '100*expm1(masked_logp-full_logp), fixed Native first argmax token; full vocabulary FP32 log_softmax',
        'correct_subset': 'Strict first-token hit: native argmax equals first token of a gold answer, NOT full-answer EM',
        'prompt': 'Original HF chat template; question + newline + Answer the question using a single word or phrase.',
        'chooserel_prompt_note': 'Reuse frozen questions, not historical Full scores; append same short-answer instruction as other tasks',
        'official_code_commit': 'c9a1111dac4baa175f67716fab473789320187c0',
        'base_commit': '7ccb44941cb48e901e6e23e96bae6da173fe2ce5', 'sources': sources,
        'no_prediction_filter': True, 'no_decode_training_or_extra_windows': True}
    freeze_json(OUTPUT / 'config.json', config)
    manifest = {'config_sha256': digest(config), 'samples': all_samples}
    manifest['binding_sha256'] = digest(manifest)
    freeze_json(OUTPUT / 'manifest.json', manifest)
    print('FROZEN', len(all_samples), 'questions', flush=True)


def preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU:
        raise RuntimeError(f'Only physical GPU1 UUID {GPU} may be exposed')
    listing = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                      '--format=csv,noheader'], text=True)
    if any(line.startswith(GPU) for line in listing.splitlines()):
        raise RuntimeError('GPU1 is occupied; do not terminate another process')
    memory = subprocess.check_output(['nvidia-smi', '-i', GPU, '--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    if int(memory.strip()) < 19000:
        raise RuntimeError('GPU1 has less than 19GB free')
    return {'compute_processes_before_load': listing, 'gpu1_free_mib_before_load': int(memory.strip())}


def hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def load_experiment():
    config = json.loads((OUTPUT / 'config.json').read_text())
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    if digest(config) != manifest['config_sha256']:
        raise ValueError('Config binding mismatch')
    if digest({k: v for k, v in manifest.items() if k != 'binding_sha256'}) != manifest['binding_sha256']:
        raise ValueError('Manifest binding mismatch')
    for source in config['sources'].values():
        if sha256(ROOT / source['path']) != source['sha256']:
            raise ValueError('Frozen source manifest changed')
    for r in manifest['samples']:
        if sha256(r['image_path']) != r['image_sha256']:
            raise ValueError('Frozen image changed')
    return config, manifest


def run(smoke=False):
    pre = preflight()
    import torch
    from information_flow_window9.masks import PATHS, intervention, probability_change, window
    from vlm_common import load_model, prepare_inputs, gpu_memory
    config, manifest = load_experiment()
    binding = {'code_commit': git('rev-parse', 'HEAD'), 'source_sha256': source_hashes(),
               'manifest_sha256': sha256(OUTPUT / 'manifest.json'), 'config_sha256': digest(config)}
    if git('status', '--porcelain', '--untracked-files=no', '--', 'information_flow_window9',
           'vlm_common.py', 'benchmarks'):
        raise RuntimeError('Commit source before GPU execution')
    if not smoke:
        smoke_data = json.loads((OUTPUT / 'smoke.json').read_text())
        if smoke_data['binding']['source_sha256'] != binding['source_sha256'] or not smoke_data['passed']:
            raise RuntimeError('Passing GPU smoke with identical source required')
        freeze_json(OUTPUT / 'execution_binding.json', binding)
    model, processor, environment = load_model(MODEL)
    layers = model.language_model.model.layers
    environment.update(pre)
    environment['code_binding'] = binding
    initial_hooks = hooks(model)
    samples = manifest['samples']
    if smoke:
        samples = [next(r for r in samples if r['dataset'] == task) for task in TASKS]
    else:
        freeze_json(OUTPUT / 'environment.json', environment)
    smoke_rows = []
    started = time.time()

    def forward(inputs):
        with torch.inference_mode():
            result = model(**inputs, use_cache=False, output_attentions=False, logits_to_keep=1, return_dict=True)
        if result.past_key_values is not None:
            raise AssertionError('Cache unexpectedly present')
        logits = result.logits[:, -1, :].detach().float()
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError('Nonfinite logits')
        return logits

    try:
        for index, sample in enumerate(samples):
            key = f'{sample["dataset"]}__{sample["question_id"]}'
            checkpoint = OUTPUT / 'checkpoints' / f'{key}.json'
            if not smoke and checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved['binding'] != binding or len(saved['rows']) != 96:
                    raise ValueError('Invalid resume checkpoint')
                print('RESUME', key, flush=True)
                continue
            mapping = sample['metadata']['input_mapping']
            inputs, counts = prepare_inputs(model, processor, sample['image_path'], mapping['text_payload'])
            if inputs['input_ids'][0].tolist() != mapping['prompt_ids']:
                raise AssertionError('Actual expanded prompt positions differ')
            inputs = {k: v.repeat(BATCH, *([1] * (v.ndim - 1))) for k, v in inputs.items()}
            original = {k: v.clone() for k, v in inputs.items()} if smoke else None
            full = forward(inputs)
            if not all(torch.equal(full[0], full[b]) for b in range(BATCH)):
                raise AssertionError('Native replicas differ')
            token = int(full[0].argmax().item())
            full_logp = float(full[0].log_softmax(-1)[token].item())
            gold_ids = mapping['reference_answer_token_ids']
            first_hit = any(ids and ids[0] == token for ids in gold_ids.values())
            baseline = {'dataset': sample['dataset'], 'question_id': sample['question_id'],
                'image_id': sample['image_id'], 'question': sample['question'], 'gold_answer': sample['answer'],
                'target_token_id': token, 'target_token_text': processor.tokenizer.decode([token]),
                'full_logp': full_logp, 'full_probability': math.exp(full_logp),
                'full_first_token_hit': first_hit, 'gold_token_ids': gold_ids,
                'p_minus_1': mapping['p_minus_1'], 'question_positions': mapping['question_positions'],
                'visual_positions': mapping['visual_positions'], 'prompt_length': counts['llm_input_tokens']}
            rows = []
            for path in PATHS:
                centers = [0, 4, 16, 31] if smoke else list(range(32))
                for start in range(0, len(centers), BATCH):
                    chunk = centers[start:start + BATCH]
                    with intervention(layers, path, mapping, chunk, validate=smoke) as checks:
                        masked = forward(inputs)
                    logps = masked.log_softmax(-1)[:, token].cpu().tolist()
                    if smoke:
                        expected_layers = {str(layer) for c in chunk for layer in window(c)}
                        if set(checks['layers']) != expected_layers or checks['target_weights_checked'] != len(expected_layers):
                            raise AssertionError('Window layer coverage differs')
                        smoke_rows.append({'key': key, 'path': path, 'centers': chunk, **checks})
                    for b, center in enumerate(chunk):
                        span = window(center)
                        rows.append({'dataset': sample['dataset'], 'question_id': sample['question_id'],
                            'image_id': sample['image_id'], 'path': path, 'center': center,
                            'window_start': span[0], 'window_end': span[-1], 'window_layers': len(span),
                            'target_token_id': token, 'full_first_token_hit': first_hit,
                            'full_logp': full_logp, 'masked_logp': logps[b], 'delta_logp': logps[b] - full_logp,
                            'full_probability': baseline['full_probability'],
                            'masked_probability': math.exp(logps[b]),
                            'probability_change_pct': probability_change(full_logp, logps[b])})
            restored = forward(inputs)
            if not torch.equal(full, restored) or hooks(model) != initial_hooks:
                raise AssertionError('Native Full or hook footprint not exactly restored')
            if smoke and not all(torch.equal(v, inputs[k]) for k, v in original.items()):
                raise AssertionError('Input mutation')
            if smoke:
                smoke_rows.append({'key': key, 'native_restored_bitwise': True, 'inputs_unchanged': True,
                                   'target': baseline})
            else:
                freeze_json(checkpoint, {'binding': binding, 'baseline': baseline, 'rows': rows,
                                        'native_restored_bitwise': True})
            elapsed = time.time() - started
            print(f'{"SMOKE" if smoke else "RUN"} {index+1}/{len(samples)} {key} '
                  f'target={baseline["target_token_text"]!r} first_hit={first_hit} elapsed={elapsed:.1f}s '
                  f'peak_mib={torch.cuda.max_memory_allocated()/1024**2:.0f}', flush=True)
            del inputs, full, masked, restored, original
        if smoke:
            freeze_json(OUTPUT / 'smoke.json', {'binding': binding, 'passed': True, 'records': smoke_rows,
                        'environment': environment, 'elapsed_seconds': time.time() - started, 'memory': gpu_memory()})
        else:
            checkpoints = [json.loads((OUTPUT / 'checkpoints' / f'{s["dataset"]}__{s["question_id"]}.json').read_text())
                           for s in manifest['samples']]
            if len(checkpoints) != 96 or any(c['binding'] != binding for c in checkpoints):
                raise ValueError('Incomplete or mixed-code run')
            write_csv(OUTPUT / 'per_question.csv', [r for c in checkpoints for r in c['rows']])
            write_csv(OUTPUT / 'baselines.csv', [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                                                for k, v in c['baseline'].items()} for c in checkpoints])
            freeze_json(OUTPUT / 'completion.json', {'status': 'complete', 'questions': 96, 'masked_rows': 9216,
                        'unique_task_question_path_center': 9216, 'native_restored_bitwise_all': True,
                        'elapsed_seconds_this_invocation': time.time() - started, 'memory': gpu_memory(), 'binding': binding})
    finally:
        if hooks(model) != initial_hooks:
            raise AssertionError('Leaked hooks')
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'smoke', 'run'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    else:
        run(smoke=args.stage == 'smoke')


if __name__ == '__main__':
    main()

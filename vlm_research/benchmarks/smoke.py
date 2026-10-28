"""Hard-capped Native Full integration only; no mask/exit/bypass implementation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from .dataset_registry import OUTPUT, digest, freeze_json, load_manifest, sha256
from .protocols import MODEL_PATH, prompt_text
from .scoring import diagnostic_candidate_metrics, generated_score, pope_metrics

GPU0 = 'GPU-a63c6362-dcfd-4a26-c44b-014b40127c89'


def freeze_plan(output):
    plan = {'stage': 'Native Full smoke ONLY', 'selection': 'First two frozen IDs per unit; '
            'ChooseAttr additionally first multi-token candidate case; ScienceQA additionally first maximum-choice case. '
            'Input-derived rules only, before any model output. At most4 samples per unit.', 'datasets': {}, 'manifest_sha256': {}}
    for path in sorted((output/'manifests').glob('*.json')):
        m = load_manifest(path, verify_images=False)
        if m['status'] != 'ready_native_full':
            continue
        ids = list(m['smoke_question_ids'])
        if path.stem == 'gqa_chooseattr':
            extra = next((r for r in m['samples'] if max(r['metadata']['input_mapping']['candidate_lengths'].values()) > 1), None)
            if extra and extra['question_id'] not in ids:
                ids.append(extra['question_id'])
        if path.stem == 'scienceqa_img':
            n = max(len(r['candidates']) for r in m['samples'])
            extra = next(r for r in m['samples'] if len(r['candidates']) == n)
            if extra['question_id'] not in ids:
                ids.append(extra['question_id'])
        if not 2 <= len(ids) <= 4:
            raise ValueError('Smoke sample cap violated')
        plan['datasets'][path.stem] = ids
        plan['manifest_sha256'][path.stem] = sha256(path)
    freeze_json(output/'smoke_plan.json', plan)
    return plan


def preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU0:
        raise ValueError('Expose exactly the prechecked physical GPU0 UUID')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                         '--format=csv,noheader'], text=True)
    if any(line.startswith(GPU0) for line in processes.splitlines()):
        raise ValueError('GPU0 has another compute task; no process will be interrupted')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = int(next(line.split(',')[1] for line in rows.splitlines() if line.startswith(GPU0)))
    if free < 18000:
        raise ValueError('Insufficient free VRAM')
    return {'physical_gpu': GPU0, 'free_mib_before_load': free, 'all_compute_processes_before': processes}


def state_hash(model):
    import torch
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def hook_footprint(model):
    return {name: [len(m._forward_hooks), len(m._forward_pre_hooks), len(m._backward_hooks)]
            for name, m in model.named_modules()}


def run(output, plan):
    import numpy as np
    import torch
    from physical_exit_pilot import candidate_inputs, sequence_score
    from vlm_common import gpu_memory, load_model, prepare_inputs
    directory = output/'smoke'
    if directory.exists():
        raise ValueError('Smoke exists; preserve attempt and inspect, never overwrite/rerun by selection')
    cpu = json.loads((output/'tests/validation.json').read_text())
    if not cpu['passed']:
        raise ValueError('CPU acceptance gate failed')
    code_root = Path(__file__).resolve().parents[1]
    for name, expected in cpu['source_sha256'].items():
        if sha256(code_root/name) != expected:
            raise ValueError(f'Code changed after CPU acceptance: {name}')
    pre = preflight()
    protocols = json.loads((output/'evaluation_protocols.json').read_text())
    directory.mkdir(parents=True)
    rows, model = [], None
    try:
        model, processor, environment = load_model(MODEL_PATH)
        if processor.chat_template != protocols['common']['chat_template']:
            raise AssertionError('Pinned chat template differs')
        freeze_json(directory/'environment.json', dict(environment, preflight=pre,
            code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256=cpu['source_sha256'], plan_sha256=digest(plan),
            model_files_sha256={r['name']: r['sha256'] for r in cpu['model_files']}))
        before = state_hash(model)
        versions = {n: p._version for n, p in model.named_parameters()}
        hooks = hook_footprint(model)
        for name, ids in plan['datasets'].items():
            manifest_path = output/f'manifests/{name}.json'
            if sha256(manifest_path) != plan['manifest_sha256'][name]:
                raise ValueError('Manifest changed after smoke plan')
            manifest = load_manifest(manifest_path)
            lookup = {r['question_id']: r for r in manifest['samples']}
            for qid in ids:
                started = time.perf_counter()
                sample = lookup[qid]
                mapping = sample['metadata']['input_mapping']
                inputs, counts = prepare_inputs(model, processor, sample['image_path'], prompt_text(sample))
                if inputs['input_ids'][0].tolist() != mapping['prompt_ids'] or counts['visual_tokens'] != 576:
                    raise AssertionError('Real GPU input differs from frozen input')
                from knockout_probe import tensor_hash
                input_hashes = {k: tensor_hash(t) for k, t in inputs.items()}
                arrays, scores, candidates_info = {}, [], []
                with torch.inference_mode():
                    for i, (candidate, records) in enumerate(candidate_inputs(processor, inputs, sample['candidates']) if sample['candidates'] else []):
                        out = model(**candidate, use_cache=False, return_dict=True)
                        if not torch.isfinite(out.logits).all():
                            raise AssertionError('Nonfinite native candidate logits')
                        logits = out.logits[0, [r['query_position'] for r in records]].detach().cpu()
                        score = sequence_score(logits, [r['token_id'] for r in records])
                        arrays[f'candidate_{i}_prediction_logits'] = logits.numpy()
                        scores.append(score)
                        candidates_info.append({'candidate': sample['candidates'][i], 'logp': score, 'mapping': records,
                            'length': len(records), 'valid_input_tokens': int(candidate['attention_mask'].sum())})
                        del out
                    generated = model.generate(**inputs, **protocols['common']['generation'],
                        return_dict_in_generate=True, output_scores=True, pad_token_id=processor.tokenizer.eos_token_id)
                    answer_ids = generated.sequences[0, inputs['input_ids'].shape[1]:].detach().cpu().tolist()
                    text = processor.tokenizer.decode(answer_ids, skip_special_tokens=True)
                    gen_logp = []
                    for i, (logits, tid) in enumerate(zip(generated.scores, answer_ids)):
                        if not torch.isfinite(logits).all():
                            raise AssertionError('Nonfinite generation logits')
                        arrays[f'generation_step_{i}_logits'] = logits[0].cpu().numpy()
                        gen_logp.append(float(logits[0].double().log_softmax(-1)[tid]))
                result = {'dataset': name, 'question_id': qid, 'image_id': sample['image_id'],
                    'image_sha256': sample['image_sha256'], 'input_sha256': input_hashes,
                    'prompt_length': mapping['prompt_length'], 'p_minus_1': mapping['p_minus_1'],
                    'question_positions': mapping['question_positions'], 'counts': counts,
                    'candidate_scores': candidates_info,
                    'generated_text': text, 'generated_token_ids': answer_ids, 'generated_token_logp': gen_logp,
                    'eos_observed': processor.tokenizer.eos_token_id in answer_ids,
                    'length_limit_reached': len(answer_ids) == protocols['common']['generation']['max_new_tokens'],
                    'metrics': generated_score(sample, text),
                    'elapsed_seconds': time.perf_counter()-started,
                    'memory': gpu_memory(), 'condition': 'Native Full ONLY'}
                if scores:
                    result['metrics'].update(diagnostic_candidate_metrics(sample, scores))
                if input_hashes != {k: tensor_hash(t) for k, t in inputs.items()}:
                    raise AssertionError('Inputs mutated')
                if versions != {n: p._version for n, p in model.named_parameters()} or hooks != hook_footprint(model):
                    raise AssertionError('Model versions/hooks changed')
                folder = directory/name
                folder.mkdir(exist_ok=True)
                path = folder/f'{qid}.npz'
                np.savez_compressed(path, **arrays)
                result['raw_logits_npz_sha256'] = sha256(path)
                freeze_json(folder/f'{qid}.json', result)
                rows.append(result)
                print(name, qid, repr(text), flush=True)
                del generated
        after = state_hash(model)
        if before != after:
            raise AssertionError('Loaded model parameter/buffer bytes changed')
        aggregates = {}
        for name in plan['datasets']:
            unit = [r for r in rows if r['dataset'] == name]
            manifest = load_manifest(output/f'manifests/{name}.json', verify_images=False)
            lookup = {r['question_id']: r for r in manifest['samples']}
            if name.startswith('pope_'):
                aggregates[name] = pope_metrics([r['generated_text'] for r in unit], [lookup[r['question_id']]['answer'] for r in unit])
        freeze_json(directory/'validation.json', {'passed': True, 'stage': 'Native Full integration smoke, NOT benchmark performance',
            'samples': len(rows), 'per_unit_samples': {k: len(v) for k, v in plan['datasets'].items()},
            'exact_input_mappings': True, 'model_weights_buffers_before_sha256': before,
            'model_weights_buffers_after_sha256': after, 'inputs_versions_hooks_unchanged': True,
            'finite_logits_all': True, 'oom': False, 'official_pope_smoke_metrics': aggregates,
            'candidate_full_vocabulary_logits_saved': True, 'all_rows': rows})
    except Exception as e:
        freeze_json(directory/'failure.json', {'status': 'partial', 'error': repr(e),
            'traceback': traceback.format_exc(), 'completed_samples': len(rows)})
        raise
    finally:
        if model is not None:
            del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--freeze-plan', action='store_true')
    a = p.parse_args()
    plan = freeze_plan(a.output)
    if a.freeze_plan:
        print(json.dumps(plan, indent=2))
        return
    lock = Path('/data/lk/ATE-resources/benchmarks/phase4_0/gpu0_smoke.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run(a.output, plan)


if __name__ == '__main__':
    main()

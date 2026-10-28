"""Immutable-manifest, same-device paired Full/B4/C2 worker and smoke gate."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from benchmarks.dataset_registry import OUTPUT as SUITE, digest, freeze_json, load_manifest, sha256
from .hooks import ORIGINALS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/cross_dataset_b4_c2'
GPUS = ['GPU-a63c6362-dcfd-4a26-c44b-014b40127c89', 'GPU-b08489af-0a8a-eb48-55ba-8346733af417',
        'GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6', 'GPU-96bc853f-ceea-2404-c704-c48a3d1ca185']
CONDITIONS = ['Full', 'B4', 'C2']


def sources():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.glob('*.py'))} | {
        str(p.relative_to(ROOT)): sha256(p) for folder in ('benchmarks', 'benchmark_interventions')
        for p in sorted((ROOT/folder).rglob('*.py'))}


def freeze(output):
    manifests = {p.stem: sha256(p) for p in sorted((SUITE/'manifests').glob('*.json'))}
    jobs = []
    for unit in manifests:
        m = load_manifest(SUITE/f'manifests/{unit}.json')
        if m['status'] != 'ready_native_full':
            raise ValueError(f'Frozen unit not ready: {unit}')
        for row in sorted(m['samples'], key=lambda x: x['question_id']):
            jobs.append({'unit': unit, 'question_id': row['question_id'], 'image_sha256': row['image_sha256']})
    for i, job in enumerate(jobs):
        job['shard'] = i % 4
    originals = {key: sha256(path) for key, path in ORIGINALS.items()}
    # The imported original primitives must see exactly their historical helpers.
    for key, names in {'b4': ['directed_suffix.py', 'knockout_probe.py'],
                       'c2': []}.items():
        directory = ORIGINALS[key].parent.parent if key == 'b4' else ORIGINALS[key].parent
        for name in names:
            if sha256(directory/name) != sha256(ROOT/name):
                raise ValueError(f'Original dependency mismatch: {name}')
    config = {'schema_version': 1, 'suite_commit': '7ccb44941cb48e901e6e23e96bae6da173fe2ce5',
        'conditions': CONDITIONS, 'jobs': jobs, 'manifest_sha256': manifests, 'source_sha256': sources(),
        'original_sha256': originals, 'physical_gpus': GPUS,
        'protocol_sha256': sha256(SUITE/'evaluation_protocols.json'),
        'intervention_protocol': {
            'B4': 'Original PathwayMask: L0-L1 fixed prompt P-1 -> real original Question keys blocked.',
            'C2': 'Original zero_query: L2-L4 MLP output at fixed prompt P-1 zeroed before residual.',
            'generation': 'Intervene during first full prefill only, preserve populated native KV cache; '
                          'remove all hooks before every subsequent native decode. Never retarget generated queries.',
            'candidate': 'Full teacher forcing of every complete candidate; edit only fixed prompt P-1, '
                         'not candidate-prediction positions after it. Keep all candidate tokens, no EOS/length norm.',
            'question': 'Use frozen original Question span; exclude hint, options and appended answer instruction.',
            'comparison': 'Each sample: all conditions, same worker/GPU, FP16/eager pinned model; '
                          'generation metrics separate from diagnostic candidate Margin.',
            'resume': 'All three conditions in one atomic per-sample checkpoint; reject changed code/config/resources.'}}
    freeze_json(output/'config.json', config)
    return config


def load_config(output):
    c = json.loads((output/'config.json').read_text())
    if c['source_sha256'] != sources():
        raise ValueError('Code changed after freeze')
    for name, value in c['manifest_sha256'].items():
        if sha256(SUITE/f'manifests/{name}.json') != value:
            raise ValueError('Frozen manifest changed')
    if c['protocol_sha256'] != sha256(SUITE/'evaluation_protocols.json'):
        raise ValueError('Frozen generation/scoring protocol changed')
    return c


def gpu_preflight(shard):
    gpu = GPUS[shard]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise ValueError('Expose exactly the assigned physical UUID')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                        '--format=csv,noheader'], text=True)
    if any(line.startswith(gpu) for line in processes.splitlines()):
        raise ValueError('Assigned GPU busy; never interrupt another process')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = int(next(line.split(',')[1] for line in rows.splitlines() if line.startswith(gpu)))
    if free < 18000:
        raise ValueError('Insufficient free VRAM')
    return {'gpu_uuid': gpu, 'free_mib': free, 'other_processes': processes}


def input_check(inputs, sample):
    mapping = sample['metadata']['input_mapping']
    if inputs['input_ids'][0].tolist() != mapping['prompt_ids'] or inputs['input_ids'].shape[1]-1 != mapping['p_minus_1']:
        raise AssertionError('Actual processor differs from frozen input')
    return mapping


def score_sample(model, processor, sample, originals):
    import numpy as np
    import torch
    from benchmarks.protocols import prompt_text
    from benchmarks.scoring import diagnostic_candidate_metrics, generated_score
    from benchmarks.smoke import hook_footprint
    from knockout_probe import tensor_hash
    from physical_exit_pilot import candidate_inputs, sequence_score
    from vlm_common import prepare_inputs, gpu_memory
    from .hooks import PrefillIntervention

    start = time.perf_counter()
    inputs, counts = prepare_inputs(model, processor, sample['image_path'], prompt_text(sample))
    mapping = input_check(inputs, sample)
    before = {k: tensor_hash(v) for k, v in inputs.items()}
    hooks = hook_footprint(model)
    versions = {n: p._version for n, p in model.named_parameters()}
    protocols = json.loads((SUITE/'evaluation_protocols.json').read_text())
    results, arrays, prefix_reference = {}, {}, {}
    for condition in CONDITIONS:
        scores, candidate_rows, audits = [], [], []
        with torch.inference_mode():
            plans = candidate_inputs(processor, inputs, sample['candidates']) if sample['candidates'] else []
            for i, (candidate, records) in enumerate(plans):
                label = sample['candidates'][i]
                with PrefillIntervention(model, candidate, mapping, condition, originals) as trace:
                    out = model(**candidate, use_cache=False, return_dict=True)
                if not torch.isfinite(out.logits).all():
                    raise AssertionError('Nonfinite candidate logits')
                selected = out.logits[0, [r['query_position'] for r in records]].detach().cpu()
                score = sequence_score(selected, [r['token_id'] for r in records])
                arrays[f'{condition}_candidate_{i}'] = selected.numpy()
                scores.append(score)
                candidate_rows.append({'candidate': label, 'logp': score, 'records': records})
                audit = {'kind': f'candidate_{i}', 'prefix_sha256': trace.prefix, 'edits': trace.records, 'mass': trace.masses}
                check_prefix(prefix_reference, condition, audit)
                audits.append(audit)
                del out
            with PrefillIntervention(model, inputs, mapping, condition, originals) as trace:
                generated = model.generate(**inputs, **protocols['common']['generation'], return_dict_in_generate=True,
                                           output_scores=True, pad_token_id=processor.tokenizer.eos_token_id)
            answer_ids = generated.sequences[0, mapping['prompt_length']:].cpu().tolist()
            text = processor.tokenizer.decode(answer_ids, skip_special_tokens=True)
            logp = []
            for i, (logits, tid) in enumerate(zip(generated.scores, answer_ids)):
                if not torch.isfinite(logits).all():
                    raise AssertionError('Nonfinite generated logits')
                arrays[f'{condition}_generation_{i}'] = logits[0].cpu().numpy()
                logp.append(float(logits[0].double().log_softmax(-1)[tid]))
            audit = {'kind': 'generation_prefill', 'prefix_sha256': trace.prefix, 'edits': trace.records, 'mass': trace.masses}
            check_prefix(prefix_reference, condition, audit)
            audits.append(audit)
            metrics = generated_score(sample, text)
            if scores:
                metrics.update(diagnostic_candidate_metrics(sample, scores))
            results[condition] = {'generated_text': text, 'generated_token_ids': answer_ids, 'generated_token_logp': logp,
                'eos_observed': processor.tokenizer.eos_token_id in answer_ids,
                'length_limit_reached': len(answer_ids) == protocols['common']['generation']['max_new_tokens'],
                'candidate_scores': candidate_rows, 'metrics': metrics, 'audits': audits}
            del generated
    if before != {k: tensor_hash(v) for k, v in inputs.items()} or hooks != hook_footprint(model):
        raise AssertionError('Input mutation or hook leak')
    if versions != {n: p._version for n, p in model.named_parameters()}:
        raise AssertionError('Parameter version changed')
    return {'question_id': sample['question_id'], 'image_sha256': sample['image_sha256'],
            'cluster_id': sample['metadata']['cluster_id'], 'answer': sample['answer'],
            'conditions': results, 'input_sha256': before, 'elapsed_seconds': time.perf_counter()-start,
            'memory': gpu_memory(), 'finite_inputs_outputs': True, 'input_versions_hooks_unchanged': True}, arrays


def check_prefix(references, condition, audit):
    if condition == 'Full':
        references[audit['kind']] = audit['prefix_sha256']
    elif references[audit['kind']] != audit['prefix_sha256']:
        raise AssertionError('Causally earlier Question/visual/text states changed')


def run(output, shard, smoke=False):
    import numpy as np
    import torch
    from benchmarks.protocols import MODEL_PATH
    from benchmarks.smoke import state_hash
    from vlm_common import load_model
    from .hooks import original_modules
    config = load_config(output)
    binding = digest(config)
    directory = output/('smoke' if smoke else f'shard{shard}')
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not smoke:
        gate = json.loads((output/'smoke/validation.json').read_text())
        if not gate['passed'] or gate['config_sha256'] != binding:
            raise ValueError('Current cross-task correctness smoke gate missing')
    preflight = gpu_preflight(shard)
    originals = original_modules(config['original_sha256'])
    manifests = {unit: load_manifest(SUITE/f'manifests/{unit}.json') for unit in config['manifest_sha256']}
    lookup = {(unit, s['question_id']): s for unit, m in manifests.items() for s in m['samples']}
    if smoke:
        plan = json.loads((SUITE/'smoke_plan.json').read_text())['datasets']
        jobs = [j for j in config['jobs'] if j['question_id'] in plan[j['unit']]]
    else:
        jobs = [j for j in config['jobs'] if j['shard'] == shard]
    envfile = directory/'environment.json'
    if envfile.exists():
        prior = json.loads(envfile.read_text())
        if prior['config_sha256'] != binding:
            raise ValueError('Resume config mismatch')
    model = None
    completed = 0
    try:
        model, processor, environment = load_model(MODEL_PATH)
        before = state_hash(model)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        if not envfile.exists():
            freeze_json(envfile, dict(environment, code_commit=commit, config_sha256=binding,
                preflight=preflight, model_sha256=before, source_sha256=config['source_sha256']))
        elif prior['model_sha256'] != before or prior['code_commit'] != commit:
            raise ValueError('Resume loaded model or code commit mismatch')
        if smoke:
            from .validate import gpu_correctness
            checks = gpu_correctness(model, processor, lookup, originals)
            freeze_json(directory/'correctness.json', checks)
        for job in jobs:
            unit, qid = job['unit'], job['question_id']
            folder = directory/unit
            folder.mkdir(exist_ok=True)
            path = folder/f'{qid}.json'
            npz = folder/f'{qid}.npz'
            if path.exists():
                row = json.loads(path.read_text())
                if row['config_sha256'] != binding or row['image_sha256'] != job['image_sha256'] or row['npz_sha256'] != sha256(npz):
                    raise ValueError('Resume checkpoint binding mismatch')
            else:
                row, arrays = score_sample(model, processor, lookup[(unit, qid)], originals)
                row.update(unit=unit, config_sha256=binding, shard=shard)
                temporary = npz.with_suffix('.npz.partial')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(npz)
                row['npz_sha256'] = sha256(npz)
                temporary_json = path.with_suffix('.json.partial')
                with temporary_json.open('w') as stream:
                    json.dump(row, stream, ensure_ascii=False, allow_nan=False)
                os.link(temporary_json, path)
                temporary_json.unlink()
            completed += 1
            print(json.dumps({'unit': unit, 'qid': qid, 'done': completed, 'total': len(jobs),
                'seconds': row['elapsed_seconds'], 'answers': {c: row['conditions'][c]['generated_text'] for c in CONDITIONS}}, ensure_ascii=False), flush=True)
            status = {'status': 'running', 'completed': completed, 'total': len(jobs), 'config_sha256': binding}
            tmp = directory/'progress.partial'
            tmp.write_text(json.dumps(status))
            tmp.replace(directory/'progress.json')
        after = state_hash(model)
        if before != after:
            raise AssertionError('Loaded weights/buffers changed')
        freeze_json(directory/'validation.json', {'passed': True, 'completed': completed, 'total': len(jobs),
            'config_sha256': binding, 'model_before_sha256': before, 'model_after_sha256': after,
            'all_prefix_states_bitwise_invariant': True, 'no_nan_inf_oom': True})
    except BaseException as error:
        failure = directory/f'failure_{time.time_ns()}.json'
        freeze_json(failure, {'status': 'partial', 'completed': completed, 'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        if model is not None:
            del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        lock.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--shard', type=int, choices=range(4), default=0)
    a = p.parse_args()
    if a.freeze:
        c = freeze(a.output)
        print(f'Frozen {len(c["jobs"])} paired jobs, four deterministic shards')
    else:
        run(a.output, a.shard, a.smoke)


if __name__ == '__main__':
    main()

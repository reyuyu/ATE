"""Immutable four-GPU data parallel execution with fail-closed data/scorer/smoke gates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import transformers

from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from vlm_common import load_model, prepare_inputs
from .engine import cache_fingerprint, forward_step, generate, discovery, native_keys, step_inputs
from .inventory import ROOT, OUTPUT, RESOURCES
from .protocol import GPUS, MODEL_PATH, PROTOCOL, TIMES, select_windows


def source_hashes():
    paths = sorted((ROOT/'paper_rq1').glob('*.py'))
    paths += [ROOT/p for p in ('vlm_common.py', 'knockout_probe.py', 'attention_position_probe.py',
              'benchmarks/scoring.py', 'benchmarks/protocols.py', 'benchmarks/prepare_datasets.py',
              'benchmarks/dataset_registry.py', 'benchmarks/scienceqa_adapter.py', 'benchmarks/smoke.py',
              'benchmarks/_vendor/m4c_evaluator.py',
              'test_paper_rq1.py')]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def freeze_code():
    data = json.loads((OUTPUT/'data_manifest.json').read_text())
    scorer = json.loads((OUTPUT/'scorer_validation.json').read_text())
    cpu = json.loads((OUTPUT/'cpu_validation.json').read_text())
    if not cpu['passed'] or cpu['source_sha256'] != source_hashes():
        raise ValueError('CPU acceptance must pass on exact final sources')
    if data['status'] != 'ready' or not scorer['passed']:
        raise ValueError('Complete disjoint data and verified official scorers required; otherwise explicitly partial')
    relative = ['vlm_research/'+p for p in source_hashes()]
    tracked = subprocess.check_output(['git', 'ls-files', '--', *relative], text=True, cwd=ROOT.parent).splitlines()
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', *relative], text=True, cwd=ROOT.parent).strip()
    if set(tracked) != set(relative) or changed:
        raise ValueError('Commit final sources before execution freeze')
    model = Path(MODEL_PATH)
    files = sorted([*model.glob('*.safetensors'), *model.glob('*.json'), *model.glob('tokenizer.model')])
    weights = {str(p): dict(sha256=sha256(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in files}
    freeze_json(OUTPUT/'execution_manifest.json', dict(protocol=PROTOCOL, source_sha256=source_hashes(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        cpu_validation_sha256=sha256(OUTPUT/'cpu_validation.json'),
        data_manifest_sha256=sha256(OUTPUT/'data_manifest.json'), scorer_validation_sha256=sha256(OUTPUT/'scorer_validation.json'),
        model_files=weights, gpu_uuids=list(GPUS), stages=['smoke', 'discovery', 'confirmation']))


def verify():
    config = json.loads((OUTPUT/'execution_manifest.json').read_text())
    if config['source_sha256'] != source_hashes() or config['protocol'] != PROTOCOL:
        raise ValueError('Frozen source/protocol changed')
    if config['data_manifest_sha256'] != sha256(OUTPUT/'data_manifest.json'):
        raise ValueError('Frozen data changed')
    if config['scorer_validation_sha256'] != sha256(OUTPUT/'scorer_validation.json'):
        raise ValueError('Frozen scorer validation changed')
    for name, expected in config['model_files'].items():
        stat = Path(name).stat()
        if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError('Model artifact changed since main-process SHA256 freeze')
    return config


def preflight(rank):
    uuid = GPUS[rank]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('Expose only this worker physical GPU UUID')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                        '--format=csv,noheader'], text=True)
    if any(line.startswith(uuid) for line in processes.splitlines()):
        raise ValueError('GPU occupied; no existing process will be interrupted')
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = int(next(line.split(',')[1] for line in memory.splitlines() if line.startswith(uuid)))
    if free < 18000:
        raise ValueError('Insufficient free GPU memory')
    return dict(gpu_uuid=uuid, free_mib=free, all_other_processes=processes)


def samples(rank, split):
    manifest = json.loads((OUTPUT/'data_manifest.json').read_text())
    path = OUTPUT/f'manifests/shard{rank}.json'
    if sha256(path) != manifest['shard_sha256'][str(path.relative_to(OUTPUT))]:
        raise ValueError('Shard content changed')
    values = json.loads(path.read_text())['samples']
    return [r for r in values if r['split'] == split]


def prepared(model, processor, sample):
    if sha256(sample['image_path']) != sample['image_sha256']:
        raise ValueError('Image changed')
    inputs, counts = prepare_inputs(model, processor, sample['image_path'], sample['prompt_text'])
    if inputs['input_ids'][0].tolist() != sample['input_mapping']['prompt_ids'] or counts['visual_tokens'] != 576:
        raise AssertionError('GPU input differs from frozen mapping')
    return inputs


def limit(sample):
    return 64 if sample['dataset'] in ('coco_caption', 'textcaps') else 32


def smoke(model, processor, rows):
    chosen = [next(s for s in rows if s['dataset'] == unit) for unit in ('scienceqa_img', 'coco_caption')]
    checks = []
    for sample in chosen:
        prompt = prepared(model, processor, sample)
        p = prompt['input_ids'].shape[1]-1
        footprint = hook_footprint(model)
        current = generate(model, prompt, limit(sample))
        with torch.inference_mode():
            reference = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=limit(sample),
                                       use_cache=True, eos_token_id=2)
        reference_tokens = reference[0, prompt['input_ids'].shape[1]:].tolist()
        if current['token_ids'] != reference_tokens:
            raise AssertionError('Cached Full token sequence differs from native generate')
        history, prefix, records = None, [], []
        for t, token in enumerate(reference_tokens):
            inputs = step_inputs(prompt, prefix)
            query = p if t == 0 else 0
            base, future, _ = forward_step(model, inputs, history, query)
            if int(base.argmax()) != token:
                raise AssertionError('Native prefix replay differs')
            if t in (0, 1, 4):
                before = cache_fingerprint(history) if history is not None else None
                observed, _, stats = forward_step(model, inputs, history, query,
                    observe=native_keys(prompt, sample['input_mapping']['question_positions'], t))
                if not torch.equal(base, observed) or len(stats) != 32:
                    raise AssertionError('Observer affects native logits or misses layers')
                for start in (2, 29):
                    noop, _, _ = forward_step(model, inputs, history, query, start, noop=True)
                    if not torch.equal(base, noop):
                        raise AssertionError('No-op logits not exact')
                    changed, _, _ = forward_step(model, inputs, history, query, start)
                    again, _, _ = forward_step(model, inputs, history, query)
                    if not torch.equal(base, again):
                        raise AssertionError('Intervention contaminated native replay')
                if history is not None and cache_fingerprint(history) != before:
                    raise AssertionError('Window modified parent native KV cache')
                records.append(dict(t=t, query=query, noop_logits_exact=True, observer_logits_exact=True,
                                    parent_cache_unchanged=True, post_window_native_exact=True))
            history, prefix = future, prefix+[token]
        if hook_footprint(model) != footprint:
            raise AssertionError('Hook leak after smoke')
        checks.append(dict(dataset=sample['dataset'], question_id=sample['question_id'],
                           native_cached_generation_tokens_exact=True, tokens=current['token_ids'], checks=records))
    return dict(passed=True, checks=checks)


def gate(stage):
    statuses = [json.loads((OUTPUT/f'{stage}/shard{i}/completion.json').read_text()) for i in range(4)]
    if any(s['status'] != 'complete' for s in statuses) or len({s['binding'] for s in statuses}) != 1:
        raise ValueError('All four matching frozen workers must complete before next stage')
    if len({s['code_commit'] for s in statuses}) != 1:
        raise ValueError('Worker code versions differ')
    envs = [json.loads((OUTPUT/f'{stage}/shard{i}/environment.json').read_text()) for i in range(4)]
    for field in ('python', 'torch', 'transformers', 'cuda_runtime', 'cudnn', 'gpu', 'model_revision',
                  'model_path', 'parameter_dtypes'):
        if len({digest(e[field]) for e in envs}) != 1:
            raise ValueError(f'Worker environment mismatch: {field}')
    return statuses


def worker(stage, rank):
    config = verify()
    if stage == 'discovery':
        gate('smoke')
    elif stage == 'confirmation':
        gate('discovery')
    binding = digest(config)
    output = OUTPUT/f'{stage}/shard{rank}'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = samples(rank, 'confirmation' if stage == 'confirmation' else 'discovery')
    checkout_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    commit = config['code_commit']
    pre = preflight(rank)
    started = time.perf_counter()
    if transformers.__version__ != '4.49.0':
        raise ValueError('Wrong Transformers version')
    model, processor, env = load_model(MODEL_PATH)
    environment = dict(env, preflight=pre, code_commit=commit, checkout_commit=checkout_commit, binding=binding)
    environment_path = output/'environment.json'
    if environment_path.exists():
        previous = json.loads(environment_path.read_text())
        for key in ('binding', 'code_commit', 'python', 'torch', 'transformers', 'gpu', 'model_revision',
                    'model_path', 'parameter_dtypes'):
            if previous[key] != environment[key]:
                raise ValueError('Resume environment differs: '+key)
        freeze_json(output/'attempts'/f'environment_{time.time_ns()}.json', environment)
    else:
        freeze_json(environment_path, environment)
    versions = {n: p._version for n, p in model.named_parameters()}
    footprint = hook_footprint(model)
    if stage == 'smoke':
        freeze_json(output/'smoke.json', smoke(model, processor, rows))
        expected, completed = 2, 2
    else:
        from .scorers import short_score
        windows = None
        if stage == 'confirmation':
            selected = json.loads((OUTPUT/'frozen_windows.json').read_text())
            if selected['execution_binding'] != binding:
                raise ValueError('Representative window binding mismatch')
            windows = selected['window_starts']
        completed, expected = 0, len(rows)
        for sample in rows:
            name = sample['dataset']+'__'+sample['question_id'].replace(':', '_')
            path = output/'checkpoints'/f'{name}.json'
            if path.exists():
                cached = json.loads(path.read_text())
                if cached['binding'] != binding or cached['sample_sha256'] != digest(sample):
                    raise ValueError('Checkpoint binding differs')
                if stage == 'discovery':
                    artifact = cached['result']
                    if sha256(artifact['native_probability_path']) != artifact['native_probability_sha256']:
                        raise ValueError('Native probability artifact missing/changed')
                completed += 1
                continue
            prompt = prepared(model, processor, sample)
            if stage == 'discovery':
                result, probabilities = discovery(model, prompt, sample['input_mapping']['question_positions'], limit(sample))
                raw = RESOURCES/f'raw/discovery/shard{rank}/{name}.npz'
                raw.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(raw, native_probability=probabilities,
                                    token_ids=np.array(result['token_ids'], dtype=np.int64))
                result.update(native_probability_path=str(raw), native_probability_sha256=sha256(raw))
                result['generated_text'] = processor.tokenizer.decode(result['token_ids'], skip_special_tokens=True).strip()
            else:
                generations = []
                for start, mode in [(None, 'Full')]+[(w, mode) for w in windows for mode in ('Prefill-only', 'Decode-only', 'Combined')]:
                    gen = generate(model, prompt, limit(sample), start=start, mode=mode)
                    text = processor.tokenizer.decode(gen['token_ids'], skip_special_tokens=True).strip()
                    gen.update(window_start=start, mode=mode, generated_text=text, score=short_score(sample, text))
                    generations.append(gen)
                result = dict(generations=generations)
            if hook_footprint(model) != footprint:
                raise AssertionError('Hook leak')
            freeze_json(path, dict(binding=binding, sample_sha256=digest(sample), dataset=sample['dataset'],
                question_id=sample['question_id'], image_identity=sample['image_identity'], result=result))
            completed += 1
            print(stage, rank, completed, '/', expected, name, flush=True)
    if {n: p._version for n, p in model.named_parameters()} != versions:
        raise AssertionError('Model parameters changed')
    freeze_json(output/'completion.json', dict(status='complete' if completed == expected else 'partial',
        binding=binding, code_commit=commit, rank=rank, completed=completed, expected=expected,
        wall_seconds=time.perf_counter()-started, model_versions_unchanged=True, hooks_clean=hook_footprint(model) == footprint))


def freeze_windows():
    gate('discovery')
    config = verify()
    cells = {}
    for rank in range(4):
        for path in (OUTPUT/f'discovery/shard{rank}/checkpoints').glob('*.json'):
            sample = json.loads(path.read_text())
            for row in sample['result']['sensitivity']:
                key = (sample['dataset'], row['time'], row['window_start'])
                cells.setdefault(key, []).append(row['js'])
    scores = {start: float(np.mean([np.mean(values) for (unit, t, s), values in cells.items() if s == start])) for start in range(30)}
    chosen = select_windows(scores)
    freeze_json(OUTPUT/'frozen_windows.json', dict(execution_binding=digest(config), window_starts=chosen,
        selection_rule=PROTOCOL['window_selection'], discovery_mean_js=scores,
        reached_cells=[dict(dataset=u, time=t, start=s, n=len(v)) for (u, t, s), v in sorted(cells.items())],
        confirmation_read=False, confirmation_generated=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'smoke', 'discovery', 'select', 'confirmation'])
    parser.add_argument('--rank', type=int, choices=range(4))
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze_code()
    elif args.stage == 'select':
        freeze_windows()
    elif args.rank is None:
        parser.error('worker stage requires --rank')
    else:
        worker(args.stage, args.rank)


if __name__ == '__main__':
    main()

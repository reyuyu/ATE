"""Fail-closed staged execution; immutable Full/target manifests precede ablation."""
import argparse
import fcntl
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import unittest

import numpy as np
import torch
import transformers
import transformers.models.llama.modeling_llama as llama
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.smoke import hook_footprint
from paper_rq1.run import preflight, prepared
from paper_rq1_single_layer_gqa6.run import atomic_json, tensor_hashes
from paper_rq1.analyze import write_csv
from vlm_common import load_model, gpu_memory
from .protocol import ROOT, OUT, OLD, RAW, MODEL_PATH, PROTOCOL, LAYERS, TYPES
from .engine import step, generate, step_inputs, cache_fingerprint, metrics
from .labels import align, select


def sources():
    names = ['paper_rq1_token_role_decode/' + n + '.py' for n in ['__init__', 'protocol', 'data', 'labels', 'engine', 'run', 'launch']]
    names += ['test_token_role_decode.py', 'paper_rq1/engine.py', 'paper_rq1/run.py', 'paper_rq1/protocol.py',
        'paper_rq1_single_layer_gqa6/engine.py', 'paper_rq1_single_layer_gqa6/run.py',
        'vlm_common.py', 'knockout_probe.py', 'benchmarks/protocols.py', 'benchmarks/dataset_registry.py',
        'benchmarks/smoke.py', 'paper_rq1/analyze.py']
    return {name: sha256(ROOT / name) for name in names}


def cpu():
    if (OUT / 'execution_manifest.json').exists():
        raise ValueError('CPU acceptance already GPU-frozen; use unittest directly for read-only reruns')
    if (OUT / 'cpu_validation.json').exists():
        archive = OUT / 'audit'; archive.mkdir(exist_ok=True)
        version = str(time.time_ns())
        for name in ['cpu_validation.json', 'cpu_tests.txt']:
            os.replace(OUT / name, archive / (version + '_' + name))
    suite = unittest.defaultTestLoader.loadTestsFromName('test_token_role_decode')
    with (OUT / 'cpu_tests.txt').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise ValueError('CPU tests failed; inspect cpu_tests.txt')
    freeze_json(OUT / 'cpu_validation.json', dict(passed=True, tests=result.testsRun,
        source_sha256=sources(), log_sha256=sha256(OUT / 'cpu_tests.txt')))


def freeze():
    validation = json.loads((OUT / 'cpu_validation.json').read_text())
    if not validation['passed'] or validation['source_sha256'] != sources():
        raise ValueError('Require exact-source CPU acceptance')
    names = ['vlm_research/' + p for p in sources()]
    tracked = subprocess.check_output(['git', 'ls-files', '--', *names], cwd=ROOT.parent, text=True).splitlines()
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', *names], cwd=ROOT.parent, text=True)
    if set(tracked) != set(names) or changed.strip():
        raise ValueError('Commit frozen worker sources first')
    prior = json.loads((OLD / 'execution_manifest.json').read_text())['model_files']
    for name, binding in prior.items():
        if sha256(name) != binding['sha256']:
            raise ValueError('Pinned model changed')
    freeze_json(OUT / 'execution_manifest.json', dict(protocol=PROTOCOL, source_sha256=sources(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip(),
        data_sha256=sha256(OUT / 'data_manifest.json'), sample_sha256=sha256(OUT / 'sample_manifest.json'),
        cpu_sha256=sha256(OUT / 'cpu_validation.json'), model_files=prior,
        transformers_source_sha256={inspect.getfile(llama): sha256(inspect.getfile(llama))}))


def verify():
    config = json.loads((OUT / 'execution_manifest.json').read_text())
    if config['source_sha256'] != sources() or config['protocol'] != PROTOCOL:
        raise ValueError('Frozen worker sources/protocol changed')
    for key, filename in [('data_sha256', 'data_manifest.json'), ('sample_sha256', 'sample_manifest.json'), ('cpu_sha256', 'cpu_validation.json')]:
        if sha256(OUT / filename) != config[key]:
            raise ValueError('Frozen input or CPU acceptance changed')
    for name, h in config['transformers_source_sha256'].items():
        if sha256(name) != h:
            raise ValueError('Third-party Transformers source changed')
    for name, bound in config['model_files'].items():
        stat = Path(name).stat()
        if stat.st_size != bound['bytes'] or stat.st_mtime_ns != bound['mtime_ns']:
            raise ValueError('Pinned model artifact changed')
    return config


def gate(stage):
    config = verify(); values = []
    for rank in range(4):
        folder = OUT / f'{stage}/shard{rank}'
        value = json.loads((folder / 'completion.json').read_text())
        if value['status'] != 'complete' or value['binding'] != digest(config):
            raise ValueError('Need four complete same-code workers: ' + stage)
        values.append(value)
    envs = [json.loads((OUT / f'{stage}/shard{rank}/environment.json').read_text()) for rank in range(4)]
    fields = ['python', 'torch', 'transformers', 'cuda_runtime', 'gpu', 'model_revision', 'parameter_dtypes', 'code_commit', 'binding']
    if len({digest({k: e[k] for k in fields}) for e in envs}) != 1:
        raise ValueError('GPU environments differ')
    return values


def raw_save(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('xb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return dict(raw_path=str(path), raw_sha256=sha256(path))


def smoke(model, processor, sample, rank):
    prompt = prepared(model, processor, sample)
    input_hashes = tensor_hashes(prompt)
    images = (prompt['input_ids'][0] == model.config.image_token_index).nonzero().flatten().tolist()
    result, logits = generate(model, prompt)
    with torch.inference_mode():
        reference = model.generate(**prompt, do_sample=False, use_cache=True, max_new_tokens=64)
    reference = reference[0, prompt['input_ids'].shape[-1]:].tolist()
    if reference != result['token_ids']:
        raise AssertionError('Native cached generation disagrees with model.generate')
    history = None; prefix = []; checks = []; timing = []
    for t, token in enumerate(result['token_ids']):
        inputs = step_inputs(prompt, prefix)
        native, future, _ = step(model, inputs, history)
        np.testing.assert_array_equal(native.numpy(), logits[t])
        if t in [1, 4]:
            before = cache_fingerprint(history)
            noop, _, trace = step(model, inputs, history, 'All32', images=images, noop=True)
            np.testing.assert_array_equal(noop.numpy(), native.numpy())
            checks.append(dict(t=t, mode='All32-noop', exact=True, layers=len(trace['edits'])))
            for layer in LAYERS:
                observed, _, audit = step(model, inputs, history, 'Image', layer, images, noop=True, audit=True)
                np.testing.assert_array_equal(observed.numpy(), native.numpy())
                checks.append(dict(t=t, mode='Image-observer', layer=layer, exact=True, decomposition=audit['decomposition']))
                for mode in ['Whole', 'Image']:
                    started = time.perf_counter()
                    altered, changed_cache, record = step(model, inputs, history, mode, layer, images, audit=mode == 'Image')
                    del changed_cache
                    timing.append(dict(t=t, mode=mode, layer=layer, seconds=time.perf_counter()-started))
                    if cache_fingerprint(history) != before:
                        raise AssertionError('Smoke condition contaminated native parent cache')
                    checks.append(dict(t=t, mode=mode, layer=layer, parent_cache_exact=True,
                        **metrics(native, altered), decomposition=record['decomposition']))
            changed, _, _ = step(model, inputs, history, 'All32', images=images)
            if cache_fingerprint(history) != before:
                raise AssertionError('All32 changed native history')
            checks.append(dict(t=t, mode='All32', parent_cache_exact=True, **metrics(native, changed)))
            after, _, _ = step(model, inputs, history)
            np.testing.assert_array_equal(after.numpy(), native.numpy())
        prefix.append(token); history = future
    if tensor_hashes(prompt) != input_hashes:
        raise AssertionError('Smoke changed native input tensors')
    return dict(passed=True, native_generate_tokens_exact=True, replay_all_steps_exact=True,
        generation=result, checks=checks, timings=timing, memory=gpu_memory(),
        **raw_save(RAW / f'smoke/shard{rank}/native.npz', full_logits=logits))


def ablate(model, processor, sample, generation, selected):
    prompt = prepared(model, processor, sample)
    input_hashes = tensor_hashes(prompt)
    images = (prompt['input_ids'][0] == model.config.image_token_index).nonzero().flatten().tolist()
    if sha256(generation['raw_path']) != generation['raw_sha256']:
        raise ValueError('Changed native saved logits')
    with np.load(generation['raw_path']) as f:
        expected = f['full_logits']
    targets = {r['generation_step']: r for r in selected}
    if len(targets) != len(selected) or any(t < 1 for t in targets):
        raise ValueError('Duplicate or Prefill semantic targets')
    history = None; prefix = []; rows = []; raw = []; audit = []
    for t, token in enumerate(generation['token_ids']):
        inputs = step_inputs(prompt, prefix)
        native, future, _ = step(model, inputs, history)
        np.testing.assert_array_equal(native.numpy(), expected[t])
        if int(native.argmax()) != token:
            raise AssertionError('Replay token trajectory differs')
        if t in targets:
            target = targets[t]
            if target['token_id'] != token or int(inputs['input_ids'][0, 0]) != generation['token_ids'][t-1]:
                raise AssertionError('Decode target/consumed-token off-by-one')
            if history.get_seq_length() != prompt['input_ids'].shape[-1] + t - 1:
                raise AssertionError('Wrong unmodified history length')
            before = cache_fingerprint(history)
            shared = dict(image_id=sample['image_identity'], image_sha256=sample['image_sha256'],
                generation_step=t, relative_generation_position=target['relative_generation_position'],
                word=target['word'], word_type=target['word_type'], token_length=target['token_length'],
                target_token_id=token, consumed_token_id=generation['token_ids'][t-1],
                caption_length=generation['length'], char_start=target['char_start'], char_end=target['char_end'])
            rows.append(dict(shared, mode='Full', layer=-1, **metrics(native, native)))
            for mode, layer in [(mode, layer) for layer in LAYERS for mode in ['Whole', 'Image']] + [('All32', -1)]:
                started = time.perf_counter()
                altered, mutated_cache, trace = step(model, inputs, history, mode, None if mode == 'All32' else layer, images)
                del mutated_cache
                rows.append(dict(shared, mode=mode, layer=layer, seconds=time.perf_counter()-started,
                    **metrics(native, altered), image_mass=trace['decomposition'][0]['image_mass'] if mode == 'Image' else None,
                    image_contribution_norm=trace['decomposition'][0]['contribution_norm'] if mode == 'Image' else None))
                raw.append(altered.numpy())
            if cache_fingerprint(history) != before:
                raise AssertionError('Formal interventions contaminated Full history')
            audit.append(dict(t=t, consumed=prefix[-1], predicted=token,
                history_length=history.get_seq_length(), parent_cache_sha256_before=before,
                parent_cache_sha256_after=before, replay_logits_exact=True, conditions=19))
        prefix.append(token); history = future
    if len(audit) != len(selected) or len(rows) != 20*len(selected):
        raise AssertionError('Missing formal positions/conditions')
    if tensor_hashes(prompt) != input_hashes:
        raise AssertionError('Formal experiment changed native inputs')
    return rows, audit, np.stack(raw) if raw else np.empty((0, expected.shape[-1]), dtype=np.float32)


def verify_targets():
    frozen = json.loads((OUT / 'target_freeze.json').read_text())
    if sha256(OUT / 'selected_token_manifest.json') != frozen['selected_sha256'] or frozen['binding'] != digest(verify()):
        raise ValueError('Frozen target manifest changed')
    return frozen


def worker(stage, rank):
    config = verify(); binding = digest(config)
    if stage == 'full': gate('smoke')
    if stage == 'ablation':
        gate('full')
        target_binding = verify_targets()
        chosen = json.loads((OUT / 'selected_token_manifest.json').read_text())
        if chosen['binding'] != binding or chosen['full_stage_digest'] != digest(gate('full')):
            raise ValueError('Target manifest belongs to different Full stage')
    folder = OUT / f'{stage}/shard{rank}'; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pre = preflight(rank)
    if transformers.__version__ != '4.49.0': raise ValueError('Wrong Transformers')
    model, processor, env = load_model(MODEL_PATH)
    if not processor.tokenizer.is_fast: raise ValueError('Frozen offset alignment requires fast tokenizer')
    env.update(binding=binding, code_commit=config['code_commit'], preflight=pre)
    if not (folder / 'environment.json').exists(): freeze_json(folder / 'environment.json', env)
    else:
        prior = json.loads((folder / 'environment.json').read_text())
        if any(prior[k] != env[k] for k in ['binding', 'code_commit', 'torch', 'transformers', 'model_revision']):
            raise ValueError('Resume environment differs')
        freeze_json(folder / f'resume_{time.time_ns()}.json', env)
    original = llama.eager_attention_forward; hooks = hook_footprint(model)
    versions = {n: p._version for n, p in model.named_parameters()}
    manifest = json.loads((OUT / 'sample_manifest.json').read_text())
    samples = [manifest['smoke']] if stage == 'smoke' else [s for s in manifest['samples'] if s['rank'] == rank]
    started = time.perf_counter(); complete = 0
    for sample in samples:
        name = sample['image_identity'].replace(':', '_'); path = folder / 'checkpoints' / (name + '.json')
        if path.exists():
            checkpoint = json.loads(path.read_text())
            if checkpoint['binding'] != binding or checkpoint['sample_sha256'] != digest(sample) or sha256(checkpoint['raw_path']) != checkpoint['raw_sha256']:
                raise ValueError('Invalid resume checkpoint')
            if stage == 'ablation' and checkpoint['selected_manifest_sha256'] != target_binding['selected_sha256']:
                raise ValueError('Resume target manifest differs')
            complete += 1; continue
        try:
            if stage == 'smoke':
                result = smoke(model, processor, sample, rank)
            elif stage == 'full':
                prompt = prepared(model, processor, sample)
                input_hashes = tensor_hashes(prompt)
                result, logits = generate(model, prompt)
                if tensor_hashes(prompt) != input_hashes: raise AssertionError('Full generation changed inputs')
                result.update(caption=processor.tokenizer.decode(result['token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    **raw_save(RAW / f'full/shard{rank}/{name}.npz', full_logits=logits))
            else:
                full_path = OUT / f'full/shard{rank}/checkpoints/{name}.json'
                if sha256(full_path) != chosen['images'][sample['image_identity']]['full_checkpoint_sha256']:
                    raise ValueError('Full checkpoint changed after target freeze')
                generation = json.loads(full_path.read_text())
                targets = chosen['images'][sample['image_identity']]['selected']
                rows, audit, logits = ablate(model, processor, sample, generation, targets)
                result = dict(rows=rows, audit=audit, positions=len(targets), selected_manifest_sha256=sha256(OUT / 'selected_token_manifest.json'),
                    **raw_save(RAW / f'ablation/shard{rank}/{name}.npz', altered_logits=logits))
            if hook_footprint(model) != hooks or llama.eager_attention_forward is not original:
                raise AssertionError('Leaked hook or eager callback')
            result.update(binding=binding, sample_sha256=digest(sample), image_id=sample['image_identity'])
            atomic_json(path, result); complete += 1
            print(stage, rank, complete, '/', len(samples), name, flush=True)
        except BaseException as exc:
            atomic_json(folder / f'failures/{name}_{time.time_ns()}.json', dict(error=repr(exc), traceback=traceback.format_exc(), binding=binding, skipped=False))
            raise
    if versions != {n: p._version for n, p in model.named_parameters()} or hook_footprint(model) != hooks:
        raise AssertionError('Model weights or hooks changed')
    verify()
    if stage == 'ablation': verify_targets()
    atomic_json(folder / 'completion.json', dict(status='complete', binding=binding, rank=rank,
        expected=len(samples), completed=complete, wall_seconds=time.perf_counter()-started,
        weights_versions_unchanged=True, hooks_clean=True, memory=gpu_memory()))


def choose_targets():
    full = gate('full'); config = verify()
    if any((OUT / f'ablation/shard{i}').exists() for i in range(4)):
        raise ValueError('Targets must freeze before any formal ablations')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    manifest = json.loads((OUT / 'sample_manifest.json').read_text()); images = {}; captions = []; token_rows = []
    for sample in manifest['samples']:
        path = OUT / f"full/shard{sample['rank']}/checkpoints/{sample['image_identity'].replace(':', '_')}.json"
        generation = json.loads(path.read_text())
        if sha256(generation['raw_path']) != generation['raw_sha256']: raise ValueError('Changed native probability trajectory')
        annotation = align(tokenizer, generation['token_ids'])
        selected = select(sample['image_sha256'], annotation, generation['length'])
        images[sample['image_identity']] = dict(image_sha256=sample['image_sha256'], full_checkpoint_sha256=sha256(path),
            caption=generation['caption'], annotation=annotation, selected=selected,
            missing_types=[role for role in TYPES if role not in [w['word_type'] for w in selected]])
        with np.load(generation['raw_path']) as f:
            logits = torch.from_numpy(f['full_logits']).float(); lp = logits.log_softmax(-1)
            original = torch.as_tensor(generation['token_ids'])
            per_step_lp = lp.gather(-1, original[:, None]).squeeze(-1).tolist()
            entropies = (-(lp.exp()*lp).sum(-1)).tolist()
        captions.append(dict(image_id=sample['image_identity'], image_sha256=sample['image_sha256'],
            rank=sample['rank'], caption=generation['caption'], token_ids=json.dumps(generation['token_ids']),
            token_logp=json.dumps(per_step_lp), token_entropy=json.dumps(entropies), length=generation['length'],
            eos=generation['eos'], truncated=generation['truncated'], generation_seconds=generation['seconds'],
            alignment_exact=annotation['roundtrip_exact'], selected_positions=len(selected)))
        for r in annotation['tokens']:
            token_rows.append(dict(image_id=sample['image_identity'], **r, full_logp=per_step_lp[r['generation_step']], entropy=entropies[r['generation_step']]))
    freeze_json(OUT / 'selected_token_manifest.json', dict(binding=digest(config), full_stage_digest=digest(full),
        images=images, tokenizer_class=type(tokenizer).__name__, tokenizer_fast=tokenizer.is_fast,
        selection_rule=PROTOCOL['position_selection'], classification=PROTOCOL['labels'],
        formal_ablation_outputs_present=False))
    write_csv(OUT / 'full_caption_generations.csv', captions)
    write_csv(OUT / 'full_token_annotations.csv', token_rows)
    freeze_json(OUT / 'target_freeze.json', dict(selected_sha256=sha256(OUT / 'selected_token_manifest.json'),
        positions=sum(len(v['selected']) for v in images.values()), counts={role: sum(w['word_type'] == role for v in images.values() for w in v['selected']) for role in TYPES},
        images=200, alignment_failures=sum(not v['annotation']['roundtrip_exact'] for v in images.values()),
        full_stage_digest=digest(full), binding=digest(config), frozen_before_formal_ablation=True))
    print('Frozen targets:', json.loads((OUT / 'target_freeze.json').read_text())['counts'], flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['cpu', 'freeze', 'smoke', 'full', 'select', 'ablation']); parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.stage == 'cpu': cpu()
    elif args.stage == 'freeze': freeze()
    elif args.stage == 'select': choose_targets()
    else: worker(args.stage, args.rank)


if __name__ == '__main__': main()

"""Prepare, smoke and execute one immutable shard; never aggregate in a worker."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback
import uuid

import numpy as np
from .protocol import (ROOT, CACHE, HISTORY, PINS, CONDITIONS, atomic, checkpoint,
                       digest, frozen_sources, git, history_k24, prepare, protocol, read, sha)


def packages():
    actual = {name: importlib.metadata.version(name) for name in PINS}
    if actual != PINS or sys.version_info[:2] != (3, 11):
        raise ValueError(f'Fixed environment required: {actual}; Python={platform.python_version()}')
    return actual


def input_digest(inputs):
    return digest({name: dict(dtype=str(value.dtype), shape=list(value.shape),
                             sha256=__import__('hashlib').sha256(value.detach().cpu().numpy().tobytes()).hexdigest())
                   for name, value in inputs.items()})


def evaluated(model, inputs, qpos, apos, observe=False):
    import torch
    from roi_sink_common import forward
    expected = input_digest(inputs)
    captures = []
    positions = torch.where((inputs['input_ids'][0] != model.config.image_token_index)
                            & (inputs['attention_mask'][0] != 0))[0]
    hooks_before = len(model._forward_hooks)
    handle = model.register_forward_hook(lambda m, a, out: captures.append(
        out.logits[0].index_select(0, positions).detach().cpu()))
    try:
        value = forward(model, inputs, qpos, apos, observe=observe)
        if len(captures) != 1 or not torch.isfinite(captures[0]).all():
            raise AssertionError('Missing/nonfinite effective-text logits')
        value['text_logits'] = captures[0]
        value['input_digest'] = expected
        if input_digest(inputs) != expected:
            raise AssertionError('Input contamination')
        return value
    finally:
        handle.remove()
        if len(model._forward_hooks) != hooks_before:
            raise AssertionError('Model hook leaked')


def tensor_sha(tensor):
    import hashlib
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def one_sample(model, processor, frozen, config, binding, folder, physical, smoke=False):
    import torch
    from attention_inheritance import VisualAttentionInheritance
    from attention_inheritance_pilot import native_signature
    from knockout_probe import model_layers
    from physical_exit import cosine_distances
    from roi_sink_probe import inputs_and_history, score
    sample = copy.deepcopy(frozen)
    sample['image_path'] = str(Path(config['image_root']) / f"image_{sample['img_id']}.jpg")
    if sha(sample['image_path']) != sample['image_sha256']:
        raise ValueError('Image changed after protocol freeze')
    prompt, candidates, qpos, apos, history, qrecords, counts = inputs_and_history(model, processor, sample)
    visual = torch.where(prompt['input_ids'][0] == model.config.image_token_index)[0].cpu().tolist()
    if len(visual) != 576 or apos in qpos or max(qpos) >= apos:
        raise AssertionError('Invalid dynamically located prediction query/visual keys')
    if len(candidates) != 2 or any(len(rec) != 1 for _, rec in candidates):
        raise AssertionError('Frozen ChooseRel candidates must be complete one-token sequences')
    modules = [layer.self_attn for layer in model_layers(model)]
    library = native_signature()
    full = [evaluated(model, x, qpos, apos, observe=(i == 0)) for i, (x, _) in enumerate(candidates)]
    if not torch.equal(full[0]['prediction_logits'], full[1]['prediction_logits']):
        raise AssertionError('Candidate leaked into P-1 prediction')
    references = {s: full[0]['rows'][s, :, 1, :].copy() for s in (23, 24)}
    if any(not np.isfinite(v).all() or (v < 0).any() or (v.astype(float).sum(-1) <= 0).any() for v in references.values()):
        raise AssertionError('Invalid per-image Full reference')
    np.savez_compressed(folder / 'references.npz', **{f'full_L{s}': v for s, v in references.items()})
    cache_file = CACHE / f"{sample['question_id']}_task.npz"
    cached_comparison = None
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as old:
            previous = old['attention'][[23, 24], :, 1, :]
        current = np.stack([references[s] for s in (23, 24)])
        cached_comparison = dict(bitwise_exact=np.array_equal(previous, current),
                                 max_abs_difference=float(np.max(np.abs(previous.astype(float)-current))),
                                 cache_sha256=sha(cache_file), used_reference='fresh same-image same-machine Full')
    gold = sample['candidates'].index(sample['answer'])
    wrong = 1-gold
    baseline = [score(o, rec) for o, (_, rec) in zip(full, candidates)]
    full_margin = baseline[gold]-baseline[wrong]
    old_logs = [c['logp'] for c in history['full_candidates']]
    comparison = dict(candidate_logp_difference=[v-o for v, o in zip(baseline, old_logs)],
                      candidate_logp_exact=baseline == old_logs,
                      historical_full_margin=float(physical['full_margin']),
                      same_machine_full_margin=full_margin,
                      full_margin_difference=full_margin-float(physical['full_margin']),
                      cache=cached_comparison)
    rows, audits, measurements = [], {}, []
    raw = {}
    for condition in CONDITIONS:
        name = condition['name']
        local_audits = []
        if name == 'full':
            outputs = full
        else:
            outputs = []
            for i, (inputs, _) in enumerate(candidates):
                ctx = VisualAttentionInheritance(modules, visual, apos, references[condition['source']],
                                                 condition['strength'], range(condition['start'], 32))
                with ctx:
                    value = evaluated(model, inputs, qpos, apos)
                    ctx.assert_complete()
                if any('forward' in m.__dict__ for m in modules):
                    raise AssertionError('Instance forward override leaked')
                if not torch.equal(value['question_hidden'], full[i]['question_hidden']):
                    raise AssertionError('Question hidden causal invariant failed')
                if condition['strength'] == 0 and not torch.equal(value['text_logits'], full[i]['text_logits']):
                    raise AssertionError('No-op differs on effective-text full-vocabulary logits')
                local_audits.append(dict(candidate=sample['candidates'][i], layer_audits=ctx.records,
                                         question_hidden_bitwise_full=True,
                                         question_hidden_sha256=tensor_sha(value['question_hidden']),
                                         effective_text_logits_sha256=tensor_sha(value['text_logits']),
                                         noop_text_bitwise_full=True if condition['strength'] == 0 else None))
                outputs.append(value)
        if not torch.equal(outputs[0]['prediction_logits'], outputs[1]['prediction_logits']):
            raise AssertionError('Candidate P-1 logits differ')
        logs = [score(o, rec) for o, (_, rec) in zip(outputs, candidates)]
        if not np.isfinite(logs).all():
            raise AssertionError('Nonfinite candidate score')
        margin = logs[gold]-logs[wrong]
        ids = [rec[0]['token_id'] for _, rec in candidates]
        vector = outputs[0]['prediction_logits']
        qdrift = cosine_distances(full[0]['question_hidden'], outputs[0]['question_hidden'])
        rows.append(dict(question_id=sample['question_id'], img_id=sample['img_id'], answer=sample['answer'],
                         condition=name, source_layer=condition['source'], target_start=condition['start'],
                         strength=condition['strength'], margin=margin, gain=margin-full_margin,
                         full_margin=full_margin, correct_logp=logs[gold], wrong_logp=logs[wrong],
                         full_correct_logp=baseline[gold], absolute_correct_logp_gain=logs[gold]-baseline[gold],
                         correct_logit=float(vector[ids[gold]]), wrong_logit=float(vector[ids[wrong]]),
                         full_vocabulary_logsumexp=float(torch.logsumexp(vector.double(), -1)),
                         historical_full_margin=float(physical['full_margin']),
                         historical_physical_k24_margin=float(physical['answer_margin']),
                         physical_k24_gain=float(physical['answer_margin'])-float(physical['full_margin']),
                         cached=cache_file.is_file(), discovered_case=sample['question_id'] == '15133706',
                         question_mean_cosine_distance=float(qdrift.mean()),
                         question_max_cosine_distance=float(qdrift.max()),
                         candidate_prediction_logits_sha256=tensor_sha(vector),
                         question_hidden_bitwise_full=True, candidate_logits_bitwise_identical=True,
                         effective_text_logits_sha256=[tensor_sha(o['text_logits']) for o in outputs],
                         forward_seconds=sum(o['measurement']['forward_seconds'] for o in outputs),
                         peak_allocated_mib=max(o['measurement']['gpu_memory']['peak_allocated_mib'] for o in outputs)))
        audits[name] = local_audits
        measurements.extend(o['measurement'] for o in outputs)
        if smoke:
            raw[name] = np.stack([o['prediction_logits'].numpy() for o in outputs])
    after = evaluated(model, candidates[0][0], qpos, apos)
    if not torch.equal(after['text_logits'], full[0]['text_logits']) or not torch.equal(after['question_hidden'], full[0]['question_hidden']):
        raise AssertionError('Final native Full restoration failed')
    if native_signature() != library or any('forward' in m.__dict__ for m in modules):
        raise AssertionError('Third-party library/instance cleanup failed')
    records = [a for group in audits.values() for candidate in group for a in candidate['layer_audits']]
    if len(records) != 124 or not all(a['passed'] for a in records):
        raise AssertionError('Incomplete/failed local layer audits')
    audit = dict(question_id=sample['question_id'], answer_query=apos, visual_indices=visual,
                 prompt_ids=prompt['input_ids'][0].cpu().tolist(), question_mapping=qrecords,
                 candidate_mapping=history['candidate_mapping'], counts=counts,
                 historical_comparison=comparison, conditions=audits, measurements=measurements,
                 original_full_text_sha256=[tensor_sha(v['text_logits']) for v in full],
                 question_hidden_sha256=[tensor_sha(v['question_hidden']) for v in full],
                 post_cleanup_native_exact=True, third_party_source_unchanged=True,
                 input_contamination=False, layer_audits=len(records), passed=True)
    with gzip.open(folder / 'audit.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(audit, stream, allow_nan=False)
    if smoke:
        np.savez_compressed(folder / 'raw_prediction_logits.npz', **raw)
    artifacts = {str(p.relative_to(folder.parent.parent)): sha(p) for p in folder.iterdir() if p.is_file()}
    return dict(binding=binding, sample_sha256=digest(frozen), question_id=sample['question_id'],
                passed=True, rows=rows, artifacts=artifacts, historical_comparison=comparison,
                layer_audits=len(records), max_row_sum_error=max(a['row_sum_max_abs_error'] for a in records),
                max_mass_abs_error=max(a['visual_mass_max_abs_error'] for a in records),
                nominal_threshold_exceeded_audits=sum(a['conditional_nominal_exceeded'] for a in records))


def execute(args, smoke=False):
    config, binding = protocol(args.output_root)
    versions = packages()
    if smoke:
        cpu = read(args.output_root / 'acceptance/validation.json')
        if not cpu.get('passed') or cpu['source_binding'] != digest(config['sources']) or cpu['code_commit'] != config['code_commit']:
            raise ValueError('Fixed-version complete CPU regression must pass before GPU smoke')
    if not smoke and os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.shard):
        raise ValueError('Worker must expose only its assigned physical GPU index')
    if not smoke:
        gate = read(args.output_root / 'smoke/validation.json')
        if not gate.get('passed') or gate['binding'] != binding:
            raise ValueError('Real GPU smoke gate has not passed')
    output = args.output_root / ('smoke' if smoke else f'shards/shard_{args.shard}')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected_ids = config['smoke_ids'] if smoke else config['shards'][args.shard]
        cohort = {s['question_id']: s for s in config['samples']}
        points = output / 'checkpoints'
        points.mkdir(exist_ok=True)
        for path in points.glob('*.json'):
            if path.stem not in expected_ids:
                raise ValueError('Foreign sample in this shard')
        finished = {}
        for qid in expected_ids:
            path = points / f'{qid}.json'
            if path.exists():
                finished[qid] = checkpoint(path, binding, cohort[qid], output)
        if len(finished) != len(expected_ids):
            from knockout_probe import resource_preflight
            from vlm_common import load_model
            resource = resource_preflight()
            if not resource['ready']:
                raise RuntimeError(f'GPU resource gate failed: {resource}')
            model, processor, env = load_model(model_path=config['model_path'])
            attempt = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
            atomic(output / f'environment_{attempt}.json', dict(environment=env, resource=resource,
                   packages=versions, code_commit=git('rev-parse', 'HEAD'), binding=binding,
                   hostname=socket.gethostname(), interpreter=sys.executable, python=platform.python_version()))
            physical = history_k24()
            for qid in expected_ids:
                if qid in finished:
                    continue
                if config['sources'] != frozen_sources():
                    raise ValueError('Source changed during execution')
                folder = output / 'samples' / f'{qid}_{attempt}'
                folder.mkdir(parents=True, exist_ok=False)
                try:
                    body = one_sample(model, processor, cohort[qid], config, binding, folder, physical[qid], smoke)
                    if (points / f'{qid}.json').exists():
                        raise FileExistsError('Refuse to replace an existing completed checkpoint')
                    atomic(points / f'{qid}.json', dict(body=body, sha256=digest(body)))
                    finished[qid] = checkpoint(points / f'{qid}.json', binding, cohort[qid], output)
                    print(json.dumps(dict(shard=args.shard, question_id=qid, complete=len(finished),
                                          expected=len(expected_ids))), flush=True)
                except BaseException as error:
                    atomic(folder / 'failure.json', dict(status='partial', error=repr(error), traceback=traceback.format_exc()))
                    raise
        validation = dict(passed=True, status='completed', binding=binding, code_commit=config['code_commit'],
                          shard=None if smoke else args.shard, question_ids=expected_ids, count=len(finished),
                          layer_audits=sum(v['layer_audits'] for v in finished.values()),
                          checkpoint_sha256={q: sha(points / f'{q}.json') for q in expected_ids})
        atomic(output / 'validation.json', validation)
        print(json.dumps(validation), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'worker'])
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--shard', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.stage == 'prepare':
        if not args.model_path or not args.image_root:
            parser.error('prepare requires fixed model and image paths')
        config = prepare(args.output_root, args.image_root.resolve(), args.model_path.resolve())
        print('FROZEN', config['code_commit'], digest(config))
    else:
        execute(args, smoke=args.stage == 'smoke')


if __name__ == '__main__':
    main()

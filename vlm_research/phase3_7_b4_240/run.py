"""Frozen single-physical-GPU paired Full/B4 runner with sample checkpoints."""
import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import unittest
import uuid

import numpy as np
import torch

from phase3_7.pathways import observed_forward, state_hash, drift
from phase3_7.run import inputs, installed_sources, checkpoint as old_checkpoint
from knockout_probe import tensor_hash
from roi_sink_probe import score
from .protocol import (ROOT, OUTPUT, OLD, GPU_UUID, GROUPS, CONDITIONS, atomic, digest,
                       packages, prepare, protocol, read, sha, source_hashes)
from .observe import forward


def acceptance():
    import vlm_common
    config, binding = protocol()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU tests must hide all GPUs')
    packages()
    vlm_common.default_model = lambda: config['model_path']
    folder = OUTPUT / 'acceptance'
    folder.mkdir(exist_ok=False)
    suites = dict(focused=unittest.defaultTestLoader.loadTestsFromName('test_phase37_b4_240'),
                  all=unittest.defaultTestLoader.discover(str(ROOT), pattern='test_*.py'))
    result = {}
    for name, suite in suites.items():
        expected = suite.countTestCases()
        with (folder / (name + '.txt')).open('x') as stream:
            run = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        result[name] = dict(passed=run.wasSuccessful() and not run.skipped and run.testsRun == expected,
                            expected=expected, count=run.testsRun, failures=len(run.failures),
                            errors=len(run.errors), skips=len(run.skipped))
    for name, args in [('pip_check', ['check']), ('pip_freeze', ['freeze'])]:
        run = subprocess.run([sys.executable, '-m', 'pip', *args], capture_output=True, text=True)
        (folder / (name + '.txt')).write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError('Package consistency failed')
    passed = all(v['passed'] for v in result.values())
    atomic(folder / 'validation.json', dict(passed=passed, binding=binding, code_commit=config['code_commit'],
                                           tests=result, packages=packages(), python=platform.python_version()))
    if not passed:
        raise RuntimeError('CPU acceptance failed; GPU prohibited')
    print('CPU_ACCEPTANCE', result, flush=True)


def checkpoint(path, binding, sample, root):
    envelope = read(path)
    body = envelope['body']
    if digest(body) != envelope['sha256'] or body['binding'] != binding or body['sample_sha256'] != digest(sample):
        raise ValueError('Checkpoint hash/configuration/sample mismatch')
    if not body['passed'] or body['conditions'] != CONDITIONS or body['row']['question_id'] != sample['question_id']:
        raise ValueError('Incomplete or incorrectly identified checkpoint')
    if body['row']['img_id'] != sample['img_id'] or body['row']['cohort'] != sample['cohort']:
        raise ValueError('Checkpoint image/cohort contamination')
    for name, expected in body['artifacts'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError('Checkpoint artifact missing/changed/escaped')
    return body


def evaluate(model, candidates, qpos, apos, bos, condition, noop=False):
    values = [forward(model, x, qpos, apos, bos, condition, noop=noop) for x, _ in candidates]
    for key in ['prediction_logits']:
        if not torch.equal(values[0][key], values[1][key]):
            raise AssertionError('Future candidate leaked into prediction query')
    for key in ['hidden', 'final_hidden']:
        if not np.array_equal(values[0]['arrays'][key], values[1]['arrays'][key]):
            raise AssertionError('Future candidate leaked into Question/P-1 states')
    for key in ['mass', 'local_native_mass']:
        if not np.array_equal(values[0]['diagnostics'][key], values[1]['diagnostics'][key]):
            raise AssertionError('Future candidate changed causal attention diagnostics')
    return values, [score(v, record) for v, (_, record) in zip(values, candidates)]


def audit_value(value):
    return dict(**value['audit'], measurement=value['measurement'],
                prediction_logits_sha256=tensor_hash(value['prediction_logits']),
                effective_text_logits_sha256=tensor_hash(value['text_logits']),
                question_layer_sha256=value['diagnostics']['question_layer_sha256'],
                final_question_sha256=value['diagnostics']['final_question_sha256'])


def write_observation(path, value):
    diagnostic = value['diagnostics']
    np.savez_compressed(path, prediction_logits=value['prediction_logits'].numpy(),
                        answer_attention=diagnostic['attention'], mass=diagnostic['mass'],
                        local_native_mass=diagnostic['local_native_mass'],
                        answer_hidden=value['arrays']['hidden'][:, -1],
                        final_answer_hidden=value['arrays']['final_hidden'][-1])


def compare_old(model, candidates, qpos, apos, values, sample):
    old_envelope = read(OLD / 'manifest.json')
    old_sample = next(s for s in old_envelope['config']['samples'] if s['question_id'] == sample['question_id'])
    old_body = old_checkpoint(OLD / 'pilot/checkpoints' / (sample['question_id'] + '.json'),
                              old_envelope['binding'], old_sample, OLD / 'pilot')
    result = {}
    for condition, old_condition in [('C0', 'B0'), ('C1', 'B4')]:
        current = values[condition]
        for (x, _), actual in zip(candidates, current):
            reference = observed_forward(model, x, qpos, apos, old_condition)
            if not torch.equal(actual['text_logits'], reference['text_logits']):
                raise AssertionError('New diagnostics/runner differ from unmodified Phase3.7 forward')
            for key in ['attention', 'hidden', 'mass', 'final_hidden']:
                if not np.array_equal(actual['arrays'][key], reference['arrays'][key]):
                    raise AssertionError('New diagnostics changed the validated original computation')
        relative = next(p for p in old_body['artifacts'] if p.endswith('/' + old_condition + '.npz'))
        with np.load(OLD / 'pilot' / relative, allow_pickle=False) as saved:
            old_logits = saved['prediction_logits'].astype(float)
        new_logits = current[0]['prediction_logits'].numpy().astype(float)
        old_row = next(r for r in old_body['b']['rows'] if r['condition'] == old_condition)
        result[condition] = dict(same_run_old_implementation_effective_text_exact=True,
                                 archived_phase37_logits_exact=bool(np.array_equal(new_logits, old_logits)),
                                 archived_phase37_logits_max_abs_difference=float(np.abs(new_logits-old_logits).max()),
                                 archived_phase37_logits_different_count=int(np.count_nonzero(new_logits != old_logits)),
                                 archived_margin=old_row['margin'], archived_source_sha256=sha(OLD / 'pilot' / relative))
    return result


def one(model, processor, sample, folder, smoke=False):
    _, candidates, qpos, apos, counts = inputs(model, processor, sample, sample)
    bos = sample['input_mapping']['bos_positions']
    if [i for i, token in enumerate(candidates[0][0]['input_ids'][0].cpu().tolist())
        if token == processor.tokenizer.bos_token_id] != bos:
        raise AssertionError('Frozen actual BOS positions changed')
    full, full_logs = evaluate(model, candidates, qpos, apos, bos, 'C0')
    noop, _ = evaluate(model, candidates, qpos, apos, bos, 'C1', noop=True)
    if not all(torch.equal(a['text_logits'], b['text_logits']) for a, b in zip(full, noop)):
        raise AssertionError('No-op effective-text logits not exactly Native Full')
    b4, b4_logs = evaluate(model, candidates, qpos, apos, bos, 'C1')
    for a, b in zip(full, b4):
        if a['diagnostics']['question_layer_sha256'] != b['diagnostics']['question_layer_sha256']:
            raise AssertionError('Question decoder states changed in B4')
        if not np.array_equal(a['arrays']['final_hidden'][:-1], b['arrays']['final_hidden'][:-1]):
            raise AssertionError('Final RMSNorm Question hidden changed')
    restored = forward(model, candidates[0][0], qpos, apos, bos)
    if not torch.equal(full[0]['text_logits'], restored['text_logits']):
        raise AssertionError('Native Full restoration failed')
    gold = ['left', 'right'].index(sample['answer'])
    wrong = 1-gold
    fmargin = full_logs[gold] - full_logs[wrong]
    margin = b4_logs[gold] - b4_logs[wrong]
    row = dict(question_id=sample['question_id'], img_id=sample['img_id'], answer=sample['answer'], cohort=sample['cohort'],
               historical_full_margin=sample['historical_full_margin'], historical_full_difference=fmargin-sample['historical_full_margin'],
               full_logp_left=full_logs[0], full_logp_right=full_logs[1], b4_logp_left=b4_logs[0], b4_logp_right=b4_logs[1],
               full_margin=fmargin, b4_margin=margin, gain=margin-fmargin,
               full_gold_logp=full_logs[gold], full_wrong_logp=full_logs[wrong],
               b4_gold_logp=b4_logs[gold], b4_wrong_logp=b4_logs[wrong],
               gold_logp_change=b4_logs[gold]-full_logs[gold], wrong_logp_change=b4_logs[wrong]-full_logs[wrong],
               full_correct=fmargin>0, b4_correct=margin>0, full_tie=fmargin==0, b4_tie=margin==0,
               harmed=fmargin>0 and margin<=0, recovered=fmargin<=0 and margin>0,
               full_right_minus_left=full_logs[1]-full_logs[0], b4_right_minus_left=b4_logs[1]-b4_logs[0],
               right_minus_left_change=(b4_logs[1]-b4_logs[0])-(full_logs[1]-full_logs[0]))
    for condition, values in [('full', full), ('b4', b4)]:
        logits = values[0]['prediction_logits'].double()
        ids = [record[0]['token_id'] for _, record in candidates]
        row[condition + '_gold_logit'] = float(logits[ids[gold]])
        row[condition + '_wrong_logit'] = float(logits[ids[wrong]])
        row[condition + '_logsumexp'] = float(torch.logsumexp(logits, -1))
    for layer in (0, 1):
        for j, group in enumerate(GROUPS):
            for label, values in [('full', full), ('b4', b4)]:
                row[f'l{layer}_{label}_{group}_mass'] = float(values[0]['diagnostics']['mass'][layer, :, j].mean())
            native = b4[0]['diagnostics']['local_native_mass'][layer, :, j]
            actual = b4[0]['diagnostics']['mass'][layer, :, j]
            row[f'l{layer}_local_{group}_redistribution'] = float((actual-native).mean())
        hidden = drift(full[0]['arrays']['hidden'][layer, -1], b4[0]['arrays']['hidden'][layer, -1])
        row.update({f'l{layer}_answer_{key}': value for key, value in hidden.items()})
    row.update({f'final_answer_{key}': value for key, value in
                drift(full[0]['arrays']['final_hidden'][-1], b4[0]['arrays']['final_hidden'][-1]).items()})
    if not np.isfinite([v for v in row.values() if isinstance(v, (int, float))]).all():
        raise AssertionError('Nonfinite scores or measurements')
    for condition, values in [('C0', full), ('C1', b4)]:
        write_observation(folder / (condition + '.npz'), values[0])
    old_comparison = compare_old(model, candidates, qpos, apos, dict(C0=full, C1=b4), sample) if smoke else None
    return dict(row=row, conditions={c: [audit_value(v) for v in values] for c, values in [('C0', full), ('C1', b4)]},
                noop_effective_text_bitwise_exact=True, noop_audits=[audit_value(v) for v in noop],
                post_cleanup_native_bitwise_exact=True, restored_audit=audit_value(restored),
                question_all32_layers_and_final_bitwise_exact=True, old_implementation_smoke=old_comparison,
                input_mapping=sample['input_mapping'], counts=counts)


def gpu_preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise ValueError('Expose only the prechecked physical GPU0 UUID')
    identity = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    gpu0 = next(line for line in identity.splitlines() if line.split(',')[0].strip() == '0')
    fields = [value.strip() for value in gpu0.split(',')]
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if fields[1] != GPU_UUID or int(fields[2]) < 16000 or any(GPU_UUID in line for line in active.splitlines()):
        raise RuntimeError('Physical GPU0 changed, is occupied, or lacks 16000MiB free; do not disturb another job')
    from knockout_probe import resource_preflight
    resource = resource_preflight()
    if not resource['ready']:
        raise RuntimeError('Original GPU preflight not ready')
    return dict(physical_gpu0=gpu0, visible_uuid=GPU_UUID, original_preflight=resource)


def execute(stage):
    config, binding = protocol()
    packages()
    cpu = read(OUTPUT / 'acceptance/validation.json')
    if not cpu['passed'] or cpu['binding'] != binding:
        raise ValueError('CPU acceptance not passed for this manifest')
    if stage == 'run':
        smoke = read(OUTPUT / 'smoke/validation.json')
        if not smoke['passed'] or smoke['binding'] != binding or not smoke['old_implementation_exact']:
            raise ValueError('Same-implementation real GPU smoke required first')
    samples = config['samples'] if stage == 'run' else [s for s in config['samples'] if s['question_id'] in config['smoke_ids']]
    root = OUTPUT / stage
    root.mkdir(exist_ok=True)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        points = root / 'checkpoints'
        points.mkdir(exist_ok=True)
        allowed = {s['question_id']: s for s in samples}
        done = {}
        for path in points.glob('*.json'):
            if path.stem not in allowed:
                raise ValueError('Foreign sample checkpoint')
            done[path.stem] = checkpoint(path, binding, allowed[path.stem], root)
        if (root / 'validation.json').exists():
            gate = read(root / 'validation.json')
            if not gate['passed'] or gate['binding'] != binding or set(done) != set(allowed):
                raise ValueError('Completed stage gate inconsistent')
            if any(sha(points / (qid + '.json')) != gate['checkpoints_sha256'][qid] for qid in done):
                raise ValueError('Completed stage checkpoint changed')
            print('ALREADY_COMPLETE', stage, len(done), flush=True)
            return
        resource = gpu_preflight()
        from vlm_common import load_model
        model, processor, environment = load_model(model_path=config['model_path'])
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        libraries = installed_sources()
        atomic(root / ('environment_' + stamp + '.json'), dict(binding=binding, code_commit=config['code_commit'],
               environment=environment, resource=resource, packages=packages(), python=platform.python_version(), installed_sources=libraries))
        weights_before = state_hash(model)
        if weights_before != config['model_runtime_state_sha256']:
            raise AssertionError('Loaded model bytes differ from the validated original Phase3.7 model')
        started = time.perf_counter()
        try:
            for sample in samples:
                if sample['question_id'] in done:
                    continue
                if source_hashes() != config['source_sha256']:
                    raise ValueError('Frozen source changed during execution')
                folder = root / 'samples' / (sample['question_id'] + '_' + stamp)
                folder.mkdir(parents=True, exist_ok=False)
                try:
                    data = one(model, processor, sample, folder, smoke=stage == 'smoke')
                    with gzip.open(folder / 'audit.json.gz', 'wt') as stream:
                        json.dump(data, stream, allow_nan=False)
                    body = dict(passed=True, binding=binding, sample_sha256=digest(sample), conditions=CONDITIONS,
                                weights_reference_sha256=weights_before,
                                row=data['row'], artifacts={str(p.relative_to(root)): sha(p) for p in folder.iterdir() if p.is_file()})
                    path = points / (sample['question_id'] + '.json')
                    if path.exists():
                        raise FileExistsError('Never overwrite a completed checkpoint')
                    atomic(path, dict(body=body, sha256=digest(body)))
                    done[sample['question_id']] = checkpoint(path, binding, sample, root)
                    print(json.dumps(dict(stage=stage, complete=len(done), expected=len(samples),
                                          question_id=sample['question_id'])), flush=True)
                except BaseException as error:
                    atomic(folder / 'failure.json', dict(status='partial', error=repr(error), traceback=traceback.format_exc()))
                    raise
            weights_after = state_hash(model)
            if weights_before != weights_after or installed_sources() != libraries:
                raise AssertionError('Model bytes or installed Transformers source changed')
            if any(body['weights_reference_sha256'] != weights_before for body in done.values()):
                raise AssertionError('Resume checkpoints do not share this exact model byte hash')
            atomic(root / 'validation.json', dict(passed=True, status='completed', binding=binding, code_commit=config['code_commit'],
                   count=len(done), expected=len(samples), question_ids=[s['question_id'] for s in samples],
                   checkpoints_sha256={qid: sha(points / (qid + '.json')) for qid in done},
                   weights_sha256_before=weights_before, weights_sha256_after=weights_after,
                   third_party_source_unchanged=True, old_implementation_exact=stage == 'smoke',
                   wall_seconds=time.perf_counter()-started))
            print('COMPLETE', stage, len(done), flush=True)
        except BaseException as error:
            atomic(root / ('failure_' + stamp + '.json'), dict(status='partial', complete=len(done),
                   error=repr(error), traceback=traceback.format_exc()))
            raise
        finally:
            del model, processor
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'acceptance', 'smoke', 'run'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU preparation must hide all GPUs')
        prepare()
    elif args.stage == 'acceptance':
        acceptance()
    else:
        execute(args.stage)


if __name__ == '__main__':
    main()

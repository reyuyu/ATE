"""One-GPU fixed64 x30 x3 scan; only small smoke, no decoding or weight hash sweep."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from benchmark_interventions.hooks import original_modules, PrefillIntervention
from benchmarks.dataset_registry import digest, freeze_json, sha256
from benchmarks.protocols import MODEL_PATH
from benchmarks.scoring import diagnostic_candidate_metrics
from benchmarks.smoke import hook_footprint, preflight
from cross_dataset_c1_c2.hooks import AttentionPrefill
from physical_exit_pilot import candidate_inputs
from residual_only_pilot.hooks import ResidualOnly
from residual_only_pilot.run import actual_inputs
from vlm_common import load_model
from .hooks import KINDS, WindowTrace

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/p1_sliding_window_gpu0'
PARENT = ROOT/'results/l2_l4_residual_only_pilot'
OLD_RAW = Path('/data/lk/ATE-worktrees/residual-only-pilot/vlm_research/results/l2_l4_residual_only_pilot')
UNITS = ('gqa_chooserel', 'scienceqa_img', 'textvqa')


def sources():
    names = ('vlm_common.py', 'knockout_probe.py', 'physical_exit_pilot.py', 'attention_position_probe.py',
             'benchmarks/dataset_registry.py', 'benchmarks/scoring.py', 'benchmarks/protocols.py',
             'benchmarks/smoke.py', 'benchmark_interventions/hooks.py', 'cross_dataset_c1_c2/hooks.py',
             'residual_only_pilot/hooks.py', 'residual_only_pilot/run.py', 'test_p1_sliding_window.py')
    paths = [ROOT/name for name in names]+sorted((ROOT/'p1_sliding_window').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def freeze(output):
    parent = json.loads((PARENT/'manifest.json').read_text())
    jobs = [{'unit': unit, 'question_id': s['question_id'], 'image_sha256': s['image_sha256'],
             'p_minus_1': s['metadata']['input_mapping']['p_minus_1']}
            for unit in UNITS for s in parent['samples'][unit]]
    if len(jobs) != 64:
        raise ValueError('Must reuse exact64 questions')
    config = {'parent_commit': 'fa959891fb26d1f5572dc26e2395252a1e01f795',
              'parent_manifest_sha256': sha256(PARENT/'manifest.json'), 'jobs': jobs,
              'source_sha256': sources(), 'original_sha256': parent['original_sha256'],
              'windows': [[i, i+1, i+2] for i in range(30)], 'interventions': list(KINDS),
              'model_path': MODEL_PATH, 'gpu': 'physical GPU0 only', 'use_cache': False,
              'protocol': 'Original fixed prompt P-1, three consecutive zero-based layers, stride1. '
                          'A zeros o_proj output, M zeros MLP output, Joint zeros both. '
                          'No masks, token deletion, decoding, training or other interventions.',
              'closed_scoring': 'Complete candidate full-vocabulary sequence LogP; gold minus strongest wrong, ties wrong. '
                                'All48 closed tasks have one-token candidates. Retain first-candidate teacher-forced '
                                'N+1 layout and reuse its causal P-1 full-vocabulary logits for all candidates only '
                                'after exact smoke checks. Never silently shorten multi-token candidates.',
              'textvqa_scoring': 'Fix native Full first argmax token per image; measure its full-vocabulary '
                                  'LogP change and changed first argmax, not VQA answer accuracy.',
              'hidden': 'Window end and L31 outputs after residual, before final RMSNorm; compare each '
                        'with same-input Full at the same depth. Also save normalized final P-1 and cosine drift. '
                        'Use pre-norm final vs window-end to assess attenuation; never treat drift as information quality.',
              'ranking': 'Mean absolute margin change, or fixed-Full-first-token LogP change for TextVQA. '
                         'Top/bottom3 deterministic, ties by smaller start; descriptive post-hoc ranking, no tuning.',
              'smoke_ids': parent['smoke_ids'], 'no_extra_audits': True}
    freeze_json(output/'manifest.json', config)
    return config


def verify(output):
    config = json.loads((output/'manifest.json').read_text())
    if config['source_sha256'] != sources() or config['parent_manifest_sha256'] != sha256(PARENT/'manifest.json'):
        raise ValueError('Frozen source/data changed')
    return config, json.loads((PARENT/'manifest.json').read_text())['samples']


def plans(processor, prompt, sample):
    candidates = candidate_inputs(processor, prompt, sample['candidates']) if sample['candidates'] else []
    if candidates and any(len(records) != 1 for _, records in candidates):
        raise ValueError('Refuse to truncate any multi-token candidate')
    p = sample['metadata']['input_mapping']['p_minus_1']
    if candidates and any(records[0]['query_position'] != p for _, records in candidates):
        raise AssertionError('Candidate scoring not at original P-1')
    return (candidates[0][0] if candidates else prompt), [records[0]['token_id'] for _, records in candidates], candidates


def measured(model, inputs, p, kind='Full', start=0, noop=False, strict_zero=None):
    with torch.inference_mode(), WindowTrace(model, p, kind, start, noop, strict_zero) as trace:
        out = model(**inputs, use_cache=False, return_dict=True)
    logits = out.logits[0, p].detach().cpu().numpy()
    arrays = dict(logits=logits, **trace.arrays())
    if out.past_key_values is not None or any(not np.isfinite(v).all() for k, v in arrays.items() if k != 'layer_indices'):
        raise AssertionError('Cache/nonfinite values')
    del out
    return arrays, trace.edits


def score(sample, logits, candidate_tokens, full_token=None):
    x = torch.from_numpy(logits).double()
    logp = x.log_softmax(-1)
    result = {'first_token_id': int(x.argmax())}
    if candidate_tokens:
        values = [float(logp[t]) for t in candidate_tokens]
        result['candidate_scores'] = dict(zip(sample['candidates'], values))
        result.update(diagnostic_candidate_metrics(sample, values))
    else:
        target = result['first_token_id'] if full_token is None else full_token
        result.update(fixed_full_token_id=target, fixed_full_token_logp=float(logp[target]))
    return result


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    if not denominator:
        raise ValueError('Zero hidden vector')
    return float(1-np.clip(np.dot(a, b)/denominator, -1., 1.))


def smoke(model, processor, config, samples, originals):
    checks = []
    for unit in UNITS:
        sample = samples[unit][0]
        prompt = actual_inputs(model, processor, sample)
        inputs, tokens, candidates = plans(processor, prompt, sample)
        ids = prompt['input_ids'][0].tolist()
        p = len(ids)-1
        full, _ = measured(model, inputs, p)
        mapping = dict(sample['metadata']['input_mapping'],
                       visual_positions=torch.where(prompt['input_ids'][0] == model.config.image_token_index)[0].cpu().tolist())
        with torch.inference_mode(), np.load(OLD_RAW/unit/f'{sample["question_id"]}.npz') as archive:
            if candidates:
                for i, (candidate, records) in enumerate(candidates):
                    out = model(**candidate, use_cache=False)
                    if not np.array_equal(full['logits'], out.logits[0, p].cpu().numpy()):
                        raise AssertionError('Single-token causal vocabulary reuse is not exact')
                    if not np.array_equal(full['logits'], archive[f'Full_candidate_{i}'][0]):
                        raise AssertionError('Historical Full candidate logits differ')
                    del out
            elif not np.array_equal(full['logits'], archive['Full_generation_0']):
                raise AssertionError('Historical Full first-token logits differ')
            for kind in KINDS:
                noop, _ = measured(model, inputs, p, kind, 2, noop=True)
                if not np.array_equal(noop['logits'], full['logits']):
                    raise AssertionError('No-op differs from native Full')
                current, edits = measured(model, inputs, p, kind, 2, strict_zero=originals['c2'].zero_query)
                reference = (AttentionPrefill(model, inputs, mapping, originals) if kind == 'A-Skip' else
                             PrefillIntervention(model, inputs, mapping, 'C2', originals, observe=False) if kind == 'M-Skip' else
                             ResidualOnly(model, ids, 'P1-Joint', originals['c2'].zero_query))
                with reference:
                    out = model(**inputs, use_cache=False)
                if not np.array_equal(current['logits'], out.logits[0, p].cpu().numpy()):
                    raise AssertionError('L2-L4 differs from original A/M/Joint implementation')
                if kind == 'Joint-Skip':
                    key = 'P1-Joint_candidate_0' if candidates else 'P1-Joint_generation_0'
                    old = archive[key][0] if candidates else archive[key]
                    if not np.array_equal(current['logits'], old):
                        raise AssertionError('Historical L2-L4 Joint logits mismatch')
                if any(not r['target_zero'] or not r['other_tokens_bitwise_exact'] or r['query'] != p for r in edits):
                    raise AssertionError('Wrong intervention location')
                checks.append({'dataset': unit, 'question_id': sample['question_id'], 'kind': kind,
                               'window': [2, 3, 4], 'original_p_minus_1': p, 'exact_legacy_logits': True,
                               'noop_exact_full': True, 'other_tokens_locally_exact': True,
                               'historical_full_exact': True, 'historical_joint_exact': kind == 'Joint-Skip'})
                del out
        print('SMOKE PASSED', unit, sample['question_id'], flush=True)
    return {'passed': True, 'manifest_sha256': digest(config), 'checks': checks, 'tolerances_relaxed': False,
            'single_token_complete_candidate_logp_reuse_exact': True}


def save(path, row, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path.with_suffix('.npz'))
    row['raw_npz_sha256'] = sha256(path.with_suffix('.npz'))
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    os.link(temp, path)
    temp.unlink()


def checkpoint(path, binding):
    row = json.loads(path.read_text())
    if row['manifest_sha256'] != binding or sha256(path.with_suffix('.npz')) != row['raw_npz_sha256']:
        raise ValueError('Checkpoint integrity/binding changed')
    with np.load(path.with_suffix('.npz')) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return row, arrays


def run(output):
    config, samples = verify(output)
    cpu = json.loads((output/'cpu_tests.json').read_text())
    if not cpu['passed'] or cpu['source_sha256'] != config['source_sha256']:
        raise ValueError('CPU gate failed')
    with (output/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pre = preflight()
        model, processor, environment = load_model(MODEL_PATH)
        if environment['transformers'] != '4.49.0':
            raise ValueError('Pinned environment changed')
        binding = digest(config)
        environment.update(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                           manifest_sha256=binding, preflight=pre)
        if not (output/'environment.json').exists():
            freeze_json(output/'environment.json', environment)
        else:
            old = json.loads((output/'environment.json').read_text())
            for key in ('code_commit', 'manifest_sha256', 'torch', 'transformers', 'cuda_visible_devices'):
                if old[key] != environment[key]:
                    raise ValueError('Resume code/environment mismatch')
        originals = original_modules(config['original_sha256'])
        versions = {n: p._version for n, p in model.named_parameters()}
        footprint = hook_footprint(model)
        started = time.perf_counter()
        try:
            if not (output/'smoke.json').exists():
                freeze_json(output/'smoke.json', smoke(model, processor, config, samples, originals))
            gate = json.loads((output/'smoke.json').read_text())
            if not gate['passed'] or gate['manifest_sha256'] != binding:
                raise ValueError('Smoke failed')
            completed = 0
            for unit in UNITS:
                for sample in samples[unit]:
                    prompt = actual_inputs(model, processor, sample)
                    inputs, tokens, _ = plans(processor, prompt, sample)
                    p = sample['metadata']['input_mapping']['p_minus_1']
                    directory = output/unit/sample['question_id']
                    fullpath = directory/'Full.json'
                    if fullpath.exists():
                        fullrow, full = checkpoint(fullpath, binding)
                    else:
                        full, _ = measured(model, inputs, p)
                        fullrow = {'dataset': unit, 'question_id': sample['question_id'], 'condition': 'Full',
                                   'image_sha256': sample['image_sha256'], 'p_minus_1': p, 'manifest_sha256': binding,
                                   'metrics': score(sample, full['logits'], tokens)}
                        save(fullpath, fullrow, full)
                    fulltoken = fullrow['metrics']['first_token_id']
                    for start in range(30):
                        for kind in KINDS:
                            path = directory/f'{kind}_L{start:02d}.json'
                            if path.exists():
                                checkpoint(path, binding)
                                continue
                            values, _ = measured(model, inputs, p, kind, start)
                            metrics = score(sample, values['logits'], tokens, fulltoken)
                            indices = list(values['layer_indices'])
                            end = values['layer_states'][indices.index(start+2)]
                            final = values['layer_states'][indices.index(31)]
                            row = {'dataset': unit, 'question_id': sample['question_id'], 'condition': kind,
                                   'window_start': start, 'window_end': start+2, 'image_sha256': sample['image_sha256'],
                                   'p_minus_1': p, 'manifest_sha256': binding, 'metrics': metrics,
                                   'window_end_hidden_cosine_distance': cosine(end, full['layer_states'][start+2]),
                                   'final_pre_norm_hidden_cosine_distance': cosine(final, full['layer_states'][31]),
                                   'final_normalized_hidden_cosine_distance': cosine(values['normalized_final'], full['normalized_final'])}
                            save(path, row, values)
                        if start in (9, 19, 29):
                            print('WINDOW_PROGRESS', unit, sample['question_id'], start+1, '/30', flush=True)
                    completed += 1
                    print('QUESTION_PROGRESS', completed, '/64', unit, sample['question_id'], flush=True)
            if versions != {n: p._version for n, p in model.named_parameters()} or footprint != hook_footprint(model):
                raise AssertionError('Model versions or hooks changed')
            freeze_json(output/'completion.json', {'passed': True, 'questions': completed, 'interventions': 5760,
                'full_baselines': 64, 'parameter_versions_unchanged': True, 'hooks_restored': True,
                'manifest_sha256': binding, 'runtime_seconds': time.perf_counter()-started})
        except BaseException as exc:
            freeze_json(output/f'failure_{time.time_ns()}.json', {'passed': False, 'error': repr(exc), 'traceback': traceback.format_exc()})
            raise
        finally:
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    from .analyze import analyze
    analyze(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('freeze', 'run', 'analyze'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'freeze':
        freeze(args.output)
    elif args.stage == 'run':
        run(args.output)
    else:
        from .analyze import analyze
        analyze(args.output)


if __name__ == '__main__':
    main()

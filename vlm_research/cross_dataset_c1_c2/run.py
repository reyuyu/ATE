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
from benchmark_interventions.run import GPUS, SUITE, gpu_preflight, input_check
from benchmarks.dataset_registry import digest, freeze_json, load_manifest, sha256
from benchmarks.protocols import MODEL_PATH, prompt_text
from benchmarks.scoring import diagnostic_candidate_metrics, generated_score
from benchmarks.smoke import hook_footprint, state_hash
from knockout_probe import tensor_hash
from physical_exit_pilot import candidate_inputs, sequence_score
from vlm_common import load_model, prepare_inputs
from .hooks import AttentionPrefill

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/cross_dataset_c1_c2'
HISTORY = Path('/data/lk/ATE-worktrees/cross-dataset-b4-c2/vlm_research/results/cross_dataset_b4_c2')
LOCAL_HISTORY = ROOT/'results/cross_dataset_b4_c2'
HISTORY_COMMIT = 'ad73b00947b40084326c0cde75298ad8afbcb1f0'
ENV_KEYS = ('python', 'torch', 'transformers', 'cuda_runtime', 'cudnn', 'gpu', 'cuda_visible_devices',
            'model_id', 'model_revision', 'model_path', 'parameter_devices', 'parameter_dtypes', 'hf_device_map')


def sources():
    result = {str(p.relative_to(ROOT)): sha256(p) for folder in ('benchmark_interventions', 'benchmarks', 'cross_dataset_c1_c2')
              for p in sorted((ROOT/folder).rglob('*.py'))}
    for name in ('vlm_common.py', 'physical_exit_pilot.py', 'attention_position_probe.py',
                 'knockout_probe.py', 'directed_suffix.py', 'test_cross_dataset_c1_c2.py'):
        result[name] = sha256(ROOT/name)
    return result


def history_row(job):
    return LOCAL_HISTORY/f'shard{job["shard"]}'/job['unit']/f'{job["question_id"]}.json'


def freeze(output):
    parent = json.loads((LOCAL_HISTORY/'config.json').read_text())
    lookup = {}
    for unit, expected in parent['manifest_sha256'].items():
        path = SUITE/f'manifests/{unit}.json'
        if sha256(path) != expected:
            raise ValueError('Parent manifest drift')
        lookup.update({(unit, s['question_id']): s for s in load_manifest(path)['samples']})
    history_sha = {}
    for job in parent['jobs']:
        path = history_row(job)
        history_sha[str(path.relative_to(LOCAL_HISTORY))] = sha256(path)
    smoke_jobs = {}
    for shard in range(4):
        pool = [j for j in parent['jobs'] if j['shard'] == shard]
        chosen = []
        for unit in parent['manifest_sha256']:
            chosen.append(next(j for j in pool if j['unit'] == unit))
        for unit, predicate in [
            ('gqa_chooseattr', lambda s: max(s['metadata']['input_mapping']['candidate_lengths'].values()) > 1),
            ('scienceqa_img', lambda s: len(s['candidates']) == max(len(lookup[(unit, j['question_id'])]['candidates'])
                                                                 for j in pool if j['unit'] == unit))]:
            extra = next(j for j in pool if j['unit'] == unit and predicate(lookup[(unit, j['question_id'])]))
            if extra not in chosen:
                chosen.append(extra)
        smoke_jobs[str(shard)] = chosen
    config = {'history_result_commit': HISTORY_COMMIT, 'history_root': str(HISTORY),
        'history_config_sha256': sha256(LOCAL_HISTORY/'config.json'), 'history_rows_sha256': history_sha,
        'history_aggregate_sha256': sha256(LOCAL_HISTORY/'aggregate.json'), 'source_sha256': sources(),
        'jobs': parent['jobs'], 'manifest_sha256': parent['manifest_sha256'],
        'original_sha256': parent['original_sha256'], 'protocol_sha256': parent['protocol_sha256'],
        'smoke_jobs': smoke_jobs, 'gpus': GPUS,
        'conditions': ['Full', 'C1', 'C2'],
        'intervention': 'C1: original Phase3.8 zero_query at L2-L4 self_attn.o_proj output; '
                        'only fixed original prompt P-1; residual and all other tokens unchanged. '
                        'First prefill only; hooks removed before native cached decode.',
        'candidate_protocol': 'Original full-sequence teacher forcing, edit only original prompt P-1; '
                              'all candidate tokens, full-vocabulary summed LogP, no EOS/length normalization.',
        'reuse_gate': 'Same assigned physical GPU, original model bytes and software environment; '
                      'all current Full/C2 smoke raw candidate/generation logits, tokens, metrics and '
                      'prefix hashes must be exactly historical. Otherwise rerun all Full/C1/C2 paired.',
        'statistics': '20,000 paired actual-image-SHA cluster bootstrap replicates, seed20261029; '
                      'per-task95% CIs. Direct C1-C2 also Bonferroni simultaneous95% familywise CIs '
                      'over8 units. No post-hoc condition/layer tuning. No equivalence claims.'}
    freeze_json(output/'config.json', config)
    return config


def load_config(output):
    config = json.loads((output/'config.json').read_text())
    if config['source_sha256'] != sources():
        raise ValueError('Frozen code changed')
    if sha256(LOCAL_HISTORY/'config.json') != config['history_config_sha256']:
        raise ValueError('History configuration changed')
    for relative, expected in config['history_rows_sha256'].items():
        if sha256(LOCAL_HISTORY/relative) != expected:
            raise ValueError('History checkpoint changed')
    for unit, expected in config['manifest_sha256'].items():
        if sha256(SUITE/f'manifests/{unit}.json') != expected:
            raise ValueError('Frozen question manifest changed')
    if sha256(SUITE/'evaluation_protocols.json') != config['protocol_sha256']:
        raise ValueError('Generation/scoring protocol changed')
    return config


def normalized_prefix(prefix):
    return {str(k): v for k, v in prefix.items()}


def evaluate(model, processor, sample, originals, condition):
    started = time.perf_counter()
    inputs, _ = prepare_inputs(model, processor, sample['image_path'], prompt_text(sample))
    mapping = input_check(inputs, sample)
    hashes = {k: tensor_hash(v) for k, v in inputs.items()}
    footprint = hook_footprint(model)
    versions = {n: p._version for n, p in model.named_parameters()}
    arrays, candidates, audits, scores = {}, [], [], []

    def context(current):
        if condition == 'C1':
            return AttentionPrefill(model, current, mapping, originals)
        return PrefillIntervention(model, current, mapping, condition, originals)

    def audit(trace, kind):
        return {'kind': kind, 'prefix_sha256': normalized_prefix(trace.prefix),
                'edits': trace.records, 'mass': trace.masses}

    with torch.inference_mode():
        plans = candidate_inputs(processor, inputs, sample['candidates']) if sample['candidates'] else []
        for i, (candidate, records) in enumerate(plans):
            with context(candidate) as trace:
                out = model(**candidate, use_cache=False, return_dict=True)
            if not torch.isfinite(out.logits).all():
                raise AssertionError('Nonfinite candidate logits')
            selected = out.logits[0, [r['query_position'] for r in records]].detach().cpu()
            value = sequence_score(selected, [r['token_id'] for r in records])
            arrays[f'{condition}_candidate_{i}'] = selected.numpy()
            scores.append(value)
            candidates.append({'candidate': sample['candidates'][i], 'logp': value, 'records': records})
            audits.append(audit(trace, f'candidate_{i}'))
            del out
        generation = json.loads((SUITE/'evaluation_protocols.json').read_text())['common']['generation']
        with context(inputs) as trace:
            generated = model.generate(**inputs, **generation, return_dict_in_generate=True,
                                       output_scores=True, pad_token_id=processor.tokenizer.eos_token_id)
        tokens = generated.sequences[0, mapping['prompt_length']:].cpu().tolist()
        text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        logp = []
        for i, (logits, tid) in enumerate(zip(generated.scores, tokens)):
            if not torch.isfinite(logits).all():
                raise AssertionError('Nonfinite generated logits')
            arrays[f'{condition}_generation_{i}'] = logits[0].cpu().numpy()
            logp.append(float(logits[0].double().log_softmax(-1)[tid]))
        audits.append(audit(trace, 'generation_prefill'))
        del generated
    metrics = generated_score(sample, text)
    if scores:
        metrics.update(diagnostic_candidate_metrics(sample, scores))
    if hashes != {k: tensor_hash(v) for k, v in inputs.items()} or footprint != hook_footprint(model):
        raise AssertionError('Input mutation or hook leak')
    if versions != {n: p._version for n, p in model.named_parameters()}:
        raise AssertionError('Model parameter version changed')
    return {'generated_text': text, 'generated_token_ids': tokens, 'generated_token_logp': logp,
        'eos_observed': processor.tokenizer.eos_token_id in tokens,
        'length_limit_reached': len(tokens) == generation['max_new_tokens'], 'candidate_scores': candidates,
        'metrics': metrics, 'audits': audits, 'elapsed_seconds': time.perf_counter()-started,
        'input_sha256': hashes}, arrays


def assert_prefix(full, changed):
    reference = {a['kind']: normalized_prefix(a['prefix_sha256']) for a in full['audits']}
    for audit in changed['audits']:
        if reference[audit['kind']] != normalized_prefix(audit['prefix_sha256']):
            raise AssertionError('Causal earlier visual/Question/text prefix changed')


def exact_historical(current, historical, arrays, archive, condition):
    for key in ('generated_text', 'generated_token_ids', 'generated_token_logp', 'candidate_scores', 'metrics',
                'eos_observed', 'length_limit_reached'):
        if current[key] != historical[key]:
            return False, f'{condition}: {key} differs'
    old_prefix = {a['kind']: normalized_prefix(a['prefix_sha256']) for a in historical['audits']}
    new_prefix = {a['kind']: normalized_prefix(a['prefix_sha256']) for a in current['audits']}
    if old_prefix != new_prefix:
        return False, f'{condition}: prefix hashes differ'
    for name, value in arrays.items():
        reference = archive[name]
        if reference.dtype != value.dtype or reference.shape != value.shape or not np.array_equal(reference, value):
            return False, f'{condition}: raw full-vocabulary logits differ ({name})'
    return True, 'All decoded tokens, scores and raw full-vocabulary logits bitwise identical'


def atomic_json(path, value):
    temporary = path.with_suffix('.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    os.link(temporary, path)
    temporary.unlink()


def decide(output):
    config = load_config(output)
    gates = [json.loads((output/f'smoke/shard{i}/validation.json').read_text()) for i in range(4)]
    weights = [json.loads((output/f'smoke/shard{i}/weight_audit.json').read_text()) for i in range(4)]
    if any(not w['passed'] for w in weights):
        raise ValueError('Smoke weight audit failed')
    if any(not g['correctness_passed'] or g['config_sha256'] != digest(config) for g in gates):
        raise ValueError('C1 correctness gate failed; no evaluation allowed')
    reuse = all(g['historical_baselines_exact'] for g in gates)
    decision = {'config_sha256': digest(config), 'reuse_history': reuse,
                'run_conditions': ['C1'] if reuse else ['Full', 'C1', 'C2'],
                'reason': 'All four same-device history gates exact' if reuse else 'Recompute all baselines paired; no mixing'}
    freeze_json(output/'decision.json', decision)
    print(json.dumps(decision), flush=True)


def run(output, shard, stage):
    config = load_config(output)
    cpu = json.loads((output/'cpu_validation.json').read_text())
    if not cpu['passed'] or cpu['source_sha256'] != config['source_sha256']:
        raise ValueError('Frozen CPU gate failed')
    directory = output/(f'smoke/shard{shard}' if stage == 'smoke' else f'shard{shard}')
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight = gpu_preflight(shard)
    binding = digest(config)
    lookup = {(unit, s['question_id']): s for unit in config['manifest_sha256']
              for s in load_manifest(SUITE/f'manifests/{unit}.json')['samples']}
    originals = original_modules(config['original_sha256'])
    model, completed = None, 0
    try:
        model, processor, environment = load_model(MODEL_PATH)
        before = state_hash(model)
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        history_environment = json.loads((LOCAL_HISTORY/f'shard{shard}/environment.json').read_text())
        same_environment = all(environment[k] == history_environment[k] for k in ENV_KEYS)
        same_environment &= before == history_environment['model_sha256']
        envfile = directory/'environment.json'
        if not envfile.exists():
            freeze_json(envfile, dict(environment, model_sha256=before, config_sha256=binding,
                code_commit=commit, same_environment_as_history=same_environment, preflight=preflight))
        else:
            old = json.loads(envfile.read_text())
            if any(old[k] != v for k, v in {'model_sha256': before, 'code_commit': commit, 'config_sha256': binding}.items()):
                raise ValueError('Resume binding changed')
        if stage == 'smoke':
            from .validate import correctness
            proof = correctness(model, processor, lookup, config['smoke_jobs'][str(shard)], originals)
            freeze_json(directory/'correctness.json', proof)
            checks = []
            for job in config['smoke_jobs'][str(shard)]:
                sample = lookup[(job['unit'], job['question_id'])]
                historical = json.loads(history_row(job).read_text())
                path = HISTORY/history_row(job).relative_to(LOCAL_HISTORY).with_suffix('.npz')
                if sha256(path) != historical['npz_sha256']:
                    raise ValueError('Historical raw logits corrupted')
                with np.load(path) as archive:
                    for condition in ('Full', 'C2'):
                        current, arrays = evaluate(model, processor, sample, originals, condition)
                        passed, reason = exact_historical(current, historical['conditions'][condition], arrays, archive, condition)
                        checks.append({'unit': job['unit'], 'question_id': job['question_id'],
                                       'condition': condition, 'exact': passed, 'reason': reason})
                c1, arrays = evaluate(model, processor, sample, originals, 'C1')
                # Cross-condition invariants require same-machine current Full, not a numerical guess.
                current_full, _ = evaluate(model, processor, sample, originals, 'Full')
                assert_prefix(current_full, c1)
                freeze_json(directory/f'{job["unit"]}_{job["question_id"]}.json', {'C1': c1})
            freeze_json(directory/'validation.json', {'correctness_passed': True, 'config_sha256': binding,
                'same_environment_as_history': bool(same_environment), 'samples': len(config['smoke_jobs'][str(shard)]),
                'historical_baselines_exact': bool(same_environment and all(c['exact'] for c in checks)), 'checks': checks})
        else:
            decision = json.loads((output/'decision.json').read_text())
            if decision['config_sha256'] != binding:
                raise ValueError('Run decision binding changed')
            smoke_environment = json.loads((output/f'smoke/shard{shard}/environment.json').read_text())
            if before != smoke_environment['model_sha256'] or any(environment[k] != smoke_environment[k] for k in ENV_KEYS):
                raise ValueError('Environment changed after gate')
            jobs = [j for j in config['jobs'] if j['shard'] == shard]
            for job in jobs:
                sample = lookup[(job['unit'], job['question_id'])]
                folder = directory/job['unit']
                folder.mkdir(exist_ok=True)
                path = folder/f'{job["question_id"]}.json'
                if path.exists():
                    row = json.loads(path.read_text())
                    if row['config_sha256'] != binding or row['npz_sha256'] != sha256(path.with_suffix('.npz')):
                        raise ValueError('Invalid resume checkpoint')
                else:
                    conditions, arrays = {}, {}
                    for condition in decision['run_conditions']:
                        value, raw = evaluate(model, processor, sample, originals, condition)
                        conditions[condition] = value
                        arrays.update(raw)
                    reference = (json.loads(history_row(job).read_text())['conditions']['Full']
                                 if decision['reuse_history'] else conditions['Full'])
                    assert_prefix(reference, conditions['C1'])
                    if not decision['reuse_history']:
                        assert_prefix(reference, conditions['C2'])
                    temporary = path.with_suffix('.npz.partial')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, **arrays)
                    temporary.replace(path.with_suffix('.npz'))
                    row = {'unit': job['unit'], 'question_id': job['question_id'], 'shard': shard,
                        'image_sha256': sample['image_sha256'], 'cluster_id': sample['metadata']['cluster_id'],
                        'answer': sample['answer'], 'conditions': conditions, 'config_sha256': binding,
                        'history_checkpoint_sha256': sha256(history_row(job)) if decision['reuse_history'] else None,
                        'npz_sha256': sha256(path.with_suffix('.npz')), 'all_causal_prefixes_exact': True}
                    atomic_json(path, row)
                completed += 1
                print(json.dumps({'shard': shard, 'completed': completed, 'total': len(jobs),
                    'unit': job['unit'], 'qid': job['question_id'], 'C1': row['conditions']['C1']['generated_text']}, ensure_ascii=False), flush=True)
            freeze_json(directory/'validation.json', {'passed': True, 'completed': completed, 'config_sha256': binding,
                'all_causal_prefixes_exact': True, 'reuse_history': decision['reuse_history']})
        after = state_hash(model)
        if before != after:
            raise AssertionError('Loaded model weights/buffers changed')
        freeze_json(directory/'weight_audit.json', {'passed': True, 'before_sha256': before, 'after_sha256': after,
                                                   'nan_inf_oom': False})
    except BaseException as error:
        freeze_json(directory/f'failure_{time.time_ns()}.json', {'status': 'partial', 'completed': completed,
            'error': repr(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        if model is not None:
            del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=['freeze', 'smoke', 'decide', 'run'], required=True)
    parser.add_argument('--shard', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    if args.stage == 'freeze':
        c = freeze(args.output)
        print('Frozen', len(c['jobs']), 'jobs; smoke counts', {k: len(v) for k, v in c['smoke_jobs'].items()})
    elif args.stage == 'decide':
        decide(args.output)
    else:
        run(args.output, args.shard, args.stage)


if __name__ == '__main__':
    main()

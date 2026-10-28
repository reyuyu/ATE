"""Post-smoke read-only audit; never resamples or modifies frozen scientific files."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from .dataset_registry import OUTPUT, digest, freeze_json, load_manifest, sha256
from .scoring import diagnostic_candidate_metrics, generated_score
from .validate_datasets import validate


def namespace(sample):
    if sample['dataset'].startswith('gqa_'):
        return 'GQA'
    if sample['dataset'] == 'pope':
        return 'COCO_val2014'
    return sample['dataset']


def image_overlap(a, b):
    """Numeric IDs have meaning only within a dataset; SHA is always comparable."""
    names_a = {(namespace(r), r['image_id']) for r in a}
    names_b = {(namespace(r), r['image_id']) for r in b}
    return {'same_namespace_image_ids': [list(k) for k in sorted(names_a & names_b)],
            'verified_image_sha256_overlap': sorted({r['image_sha256'] for r in a if r['image_sha256']}
                                                    & {r['image_sha256'] for r in b if r['image_sha256']}),
            'raw_id_string_collisions_not_proof_of_same_image': sorted({r['image_id'] for r in a} & {r['image_id'] for r in b})
            if {namespace(r) for r in a} != {namespace(r) for r in b} else []}


def logp(logits, tokens):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != len(tokens) or not np.isfinite(logits).all():
        raise ValueError('Invalid raw candidate vocabulary logits')
    maximum = logits.max(axis=-1)
    normalizer = np.log(np.exp(logits-maximum[:, None]).sum(axis=-1)) + maximum
    return float(sum(logits[i, tid]-normalizer[i] for i, tid in enumerate(tokens)))


def audit(output):
    integrity = validate(output)
    smoke = json.loads((output/'smoke/validation.json').read_text())
    if not smoke['passed'] or smoke['model_weights_buffers_before_sha256'] != smoke['model_weights_buffers_after_sha256']:
        raise ValueError('Native smoke gate failed')
    plan = json.loads((output/'smoke_plan.json').read_text())
    if smoke['samples'] != sum(len(v) for v in plan['datasets'].values()):
        raise ValueError('Smoke denominator differs')
    units = {p.stem: load_manifest(p) for p in (output/'manifests').glob('*.json')}
    for m in units.values():
        for sample in m['samples']:
            if digest(sample['metadata']['raw_row']) != sample['source_row_sha256']:
                raise ValueError('Frozen annotation row hash differs')
    checked, max_error = [], 0.
    for row in smoke['all_rows']:
        name, qid = row['dataset'], row['question_id']
        if qid not in plan['datasets'][name]:
            raise ValueError('Unplanned smoke sample')
        sample = next(r for r in units[name]['samples'] if r['question_id'] == qid)
        if digest(sample['metadata']['raw_row']) != sample['source_row_sha256']:
            raise ValueError('Raw annotation row hash differs')
        path = output/f'smoke/{name}/{qid}.npz'
        if sha256(path) != row['raw_logits_npz_sha256']:
            raise ValueError('Smoke raw logit file changed')
        if json.loads((path.with_suffix('.json')).read_text()) != row:
            raise ValueError('Smoke summary/per-sample mismatch')
        scores = []
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays:
                if not np.isfinite(arrays[key]).all():
                    raise ValueError('Nonfinite saved logit array')
            for i, candidate in enumerate(row['candidate_scores']):
                score = logp(arrays[f'candidate_{i}_prediction_logits'], [r['token_id'] for r in candidate['mapping']])
                error = abs(score-candidate['logp'])
                max_error = max(max_error, error)
                if error > 1e-10:
                    raise ValueError('Independent float64 full-vocabulary LogP recomputation differs')
                scores.append(candidate['logp'])
            for i, (token, saved_logp) in enumerate(zip(row['generated_token_ids'], row['generated_token_logp'])):
                logits = arrays[f'generation_step_{i}_logits']
                if int(np.argmax(logits)) != token or abs(logp(logits[None, :], [token])-saved_logp) > 1e-10:
                    raise ValueError('Native greedy token/vocabulary LogP mismatch')
        metrics = generated_score(sample, row['generated_text'])
        if scores:
            metrics.update(diagnostic_candidate_metrics(sample, scores))
        if metrics != row['metrics']:
            raise ValueError('Scorer replay differs')
        checked.append({'dataset': name, 'question_id': qid, 'passed': True,
                        'candidate_count': len(scores), 'greedy_steps': len(row['generated_token_ids'])})
    expected = {(name, qid) for name, ids in plan['datasets'].items() for qid in ids}
    observed = {(r['dataset'], r['question_id']) for r in checked}
    if observed != expected or len(checked) != len(observed):
        raise ValueError('Missing/duplicated smoke case')
    overlap = {f'{a}__{b}': image_overlap(am['samples'], bm['samples'])
               for a, am in units.items() for b, bm in units.items() if a < b}
    freeze_json(output/'canonical_image_overlap_audit.json', {
        'authoritative': True,
        'legacy_note': 'overlap_audit.json retains literal ID string intersections from preparation. '
                       'Cross-dataset IDs are NOT globally unique. Use this namespaced + SHA256 audit instead.',
        'pairs': overlap,
        'per_unit_unique_image_sha256': {name: len({r['image_sha256'] for r in m['samples']}) for name, m in units.items()},
        'all_units_unique_verified_image_sha256': len({r['image_sha256'] for m in units.values() for r in m['samples']})})
    freeze_json(output/'publication_validation.json', {'passed': True, 'dataset_integrity': integrity,
        'smoke_cases_independently_verified': checked, 'logp_max_abs_recomputation_error_nats': max_error,
        'comparison_atol_nats': 1e-10, 'purpose_of_atol': 'Float64 NumPy-vs-PyTorch summation rounding only; not FP16/model equivalence tolerance',
        'original_frozen_artifacts_unchanged': True,
        'frozen_source_commit': json.loads((output/'smoke/environment.json').read_text())['code_commit'],
        'canonical_overlap_sha256': sha256(output/'canonical_image_overlap_audit.json')})
    return checked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUTPUT)
    p.add_argument('--seal', action='store_true')
    p.add_argument('--verify-seal', action='store_true')
    p.add_argument('--cpu-review', action='store_true')
    p.add_argument('--review-name', default='final_review')
    a = p.parse_args()
    if a.verify_seal:
        seal = json.loads((a.output/'artifact_sha256.json').read_text())
        actual = {str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'}
        if actual != seal['files']:
            raise ValueError('Publication artifact set/hash changed')
        print('Publication seal verified:', len(actual), 'files')
        return
    if a.cpu_review:
        import vlm_common
        from .protocols import MODEL_PATH
        vlm_common.default_model = lambda: MODEL_PATH
        import torch
        if torch.cuda.is_available():
            raise ValueError('Hide all GPUs for CPU review')
        log = io.StringIO()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.discover('.', pattern='test_*.py'))
        if not a.review_name.replace('_', '').isalnum():
            raise ValueError('Invalid review artifact name')
        console = a.output/f'tests/{a.review_name}_console.txt'
        if console.exists():
            raise ValueError('Final CPU review already exists')
        console.write_text(log.getvalue())
        freeze_json(a.output/f'tests/{a.review_name}_validation.json', {'passed': result.wasSuccessful() and not result.skipped,
            'tests_run': result.testsRun, 'skipped': len(result.skipped), 'console_sha256': sha256(console),
            'source_sha256': {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
                             for p in Path(__file__).resolve().parents[1].rglob('*.py') if 'results' not in p.parts},
            'note': 'Original136-test pre-GPU acceptance untouched. Post-smoke publication and download guards; no rerun/resampling of GPU cases.'})
        if not result.wasSuccessful() or result.skipped:
            raise AssertionError('CPU final review failed/skipped')
    checked = audit(a.output)
    if a.seal:
        freeze_json(a.output/'artifact_sha256.json', {'files': {str(p.relative_to(a.output)): sha256(p)
            for p in a.output.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'}})
    print('Independent data/logit/scorer audit passed:', len(checked), 'smoke cases')


if __name__ == '__main__':
    main()

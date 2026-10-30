"""CPU-only independent delivery review; never invokes the model or edits collection."""
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from benchmarks.dataset_registry import freeze_json, sha256
from paper_rq1.analyze import write_csv
from .engine import metrics
from .protocol import ROOT, OUT
from .run import verify, verify_targets, gate


def read(path):
    return json.loads(path.read_text())


def review():
    config = verify()
    verify_targets()
    initial = read(OUT / 'analysis_completion.json')
    archive = OUT / 'audit/pre_delivery'
    for name, expected in initial['files_sha256'].items():
        candidate = OUT / name
        if name in ['report.md', 'figures/figure4_real_caption_case.png',
                    'figures/figure4_real_caption_case.pdf'] and (archive / name).exists():
            candidate = archive / name
        if sha256(candidate) != expected:
            raise ValueError('Changed initial analysis artifact: ' + name)
    for name, expected in initial['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError('Changed initial analysis source: ' + name)
    # Keep the original automated report and its inventory intact for review.
    archive.mkdir(parents=True, exist_ok=True)
    for name in ['analysis_completion.json', 'report.md']:
        target = archive / name
        if target.exists():
            expected = sha256(OUT / name) if name == 'analysis_completion.json' else initial['files_sha256'][name]
            if sha256(target) != expected:
                raise ValueError('Existing delivery archive differs')
        else:
            shutil.copyfile(OUT / name, target)

    stages, environments, raw_manifest, performance = {}, [], [], []
    full, smoke, decompositions = [], [], []
    for stage in ['smoke', 'full', 'ablation']:
        stages[stage] = gate(stage)
        for rank in range(4):
            folder = OUT / f'{stage}/shard{rank}'
            env = read(folder / 'environment.json')
            if env['code_commit'] != config['code_commit'] or env['transformers'] != '4.49.0':
                raise ValueError('Worker code/environment mismatch')
            if env['parameter_dtypes'] != ['torch.float16'] or env['model_revision'] != config['protocol']['model_revision']:
                raise ValueError('Worker model mismatch')
            environments.append(dict(stage=stage, rank=rank, **{k: env[k] for k in
                ['code_commit', 'torch', 'transformers', 'cuda_runtime', 'model_revision', 'cuda_visible_devices']}))
            completion = stages[stage][rank]
            performance.append(dict(stage=stage, rank=rank, completed=completion['completed'],
                worker_wall_seconds=completion['wall_seconds'], model_load_seconds=env['load_seconds'],
                peak_allocated_mib=completion['memory']['peak_allocated_mib'],
                peak_reserved_mib=completion['memory']['peak_reserved_mib']))
            for checkpoint in sorted((folder / 'checkpoints').glob('*.json')):
                record = read(checkpoint)
                raw = Path(record['raw_path'])
                if sha256(raw) != record['raw_sha256']:
                    raise ValueError('Changed raw logits: ' + str(raw))
                raw_manifest.append(dict(stage=stage, rank=rank, image_id=record['image_id'],
                    path=str(raw), sha256=record['raw_sha256'], bytes=raw.stat().st_size))
                if stage == 'full':
                    full.append(record)
                if stage == 'smoke':
                    if not all(record[k] for k in ['passed', 'native_generate_tokens_exact', 'replay_all_steps_exact']):
                        raise ValueError('Failed smoke')
                    if any(not r['exact'] for r in record['checks'] if 'exact' in r):
                        raise ValueError('Failed no-op equality')
                    smoke.append(record)
                    for row in record['checks']:
                        decompositions.extend(row.get('decomposition', []))
    for name in ['torch', 'transformers', 'cuda_runtime', 'model_revision', 'code_commit']:
        if len({env[name] for env in environments}) != 1:
            raise ValueError('Inconsistent worker environment: ' + name)
    if len(full) != 200 or len(smoke) != 4 or len(raw_manifest) != 404:
        raise ValueError('Incomplete native/altered checkpoints')
    if any(not d['denominator_unchanged'] or d['head_max_error_to_bound'] > 1 or
           d['projection_max_error_to_bound'] > 1 for d in decompositions):
        raise ValueError('Invalid original-softmax image decomposition')

    # Recompute raw-logit metrics for one deterministic image on each GPU.
    checked = 0
    for rank in range(4):
        checkpoint = sorted((OUT / f'ablation/shard{rank}/checkpoints').glob('*.json'))[0]
        record = read(checkpoint)
        native = read(OUT / f'full/shard{rank}/checkpoints/{checkpoint.name}')
        with np.load(native['raw_path']) as raw:
            baseline = raw['full_logits']
        with np.load(record['raw_path']) as raw:
            altered = raw['altered_logits']
        rows = [r for r in record['rows'] if r['mode'] != 'Full']
        if len(rows) != len(altered):
            raise ValueError('Raw logits / condition order mismatch')
        for i in [0, len(rows)//2, len(rows)-1]:
            row = rows[i]
            recalculated = metrics(torch.from_numpy(baseline[row['generation_step']]), altered[i])
            for key in ['loss', 'abs_delta_logp', 'js_divergence', 'relative_probability_change', 'full_token_entropy']:
                if recalculated[key] != row[key]:
                    raise ValueError('Raw metric not exact: ' + key)
            if recalculated['top1_flip'] != row['top1_flip']:
                raise ValueError('Raw flip differs')
            checked += 1
    write_csv(OUT / 'raw_logits_manifest.csv', raw_manifest)
    write_csv(OUT / 'worker_performance.csv', performance)
    durations = [r['seconds'] for r in full]
    timing = {}
    for mode in ['Whole', 'Image']:
        values = [r['seconds'] for record in smoke for r in record['timings'] if r['mode'] == mode]
        timing[mode] = dict(n=len(values), mean_seconds=float(np.mean(values)),
            median_seconds=float(np.median(values)), p95_seconds=float(np.quantile(values, .95)))
    review_result = dict(passed=True, frozen_collector_commit=config['code_commit'],
        initial_inventory_entries_verified=len(initial['files_sha256']),
        original_analysis_inventory_sha256=sha256(OUT/'analysis_completion.json'),
        environments=environments, raw_files_verified=len(raw_manifest), raw_bytes=sum(r['bytes'] for r in raw_manifest),
        raw_condition_metrics_recomputed_exact=checked, four_smoke_native_and_noop_exact=True,
        decomposition_checks=len(decompositions),
        max_head_abs_error=max(d['head_max_abs_error'] for d in decompositions),
        max_projection_abs_error=max(d['projection_max_abs_error'] for d in decompositions),
        max_head_error_to_frozen_bound=max(d['head_max_error_to_bound'] for d in decompositions),
        max_projection_error_to_frozen_bound=max(d['projection_max_error_to_bound'] for d in decompositions),
        full_generation=dict(images=200, eos=sum(r['eos'] for r in full), truncated=sum(r['truncated'] for r in full),
            mean_tokens=float(np.mean([r['length'] for r in full])), min_tokens=min(r['length'] for r in full),
            max_tokens=max(r['length'] for r in full), mean_seconds=float(np.mean(durations)),
            median_seconds=float(np.median(durations)), summed_gpu_seconds=sum(durations)),
        smoke_single_cached_step_timing=timing,
        timing_limits='Wall times are per-worker stage times including setup; Full is a whole caption, intervention is one cached step with cache-copy overhead. Not comparable as speedup.',
        stage_max_worker_wall_seconds={k:max(r['wall_seconds'] for r in v) for k,v in stages.items()},
        gpu_resources_released='All workers exited; nvidia-smi separately checked by primary agent.',
        classification_review='20 input-hash-chosen image contexts reviewed for alignment/priority sanity, no relabeling. Not a human gold POS or grounding accuracy measurement.')
    freeze_json(OUT / 'delivery_review.json', review_result)
    print(json.dumps(review_result, indent=2))


def inventory():
    verify()
    verify_targets()
    required = ['report.md', 'research_summary.md', 'README.md', 'delivery_review.json',
                'raw_logits_manifest.csv', 'worker_performance.csv', 'result_validation.json']
    if not all((OUT / name).is_file() for name in required):
        raise ValueError('Incomplete delivery documents')
    if len(list((OUT/'figures').glob('*.png'))) != 5 or len(list((OUT/'figures').glob('*.pdf'))) != 5:
        raise ValueError('Need five PNG/PDF figures')
    freeze_json(OUT / 'final_completion.json', dict(status='complete', images=200, target_positions=591,
        intervention_forwards=11229, collector_commit=verify()['code_commit'],
        selected_manifest_sha256=sha256(OUT/'selected_token_manifest.json'),
        initial_completion_preserved=True,
        analysis_sources={str(p.relative_to(ROOT)):sha256(p) for p in
            Path(__file__).parent.glob('*.py') if p.name in ['analyze.py','statistics.py','label_audit.py','finalize.py','case_figure.py']},
        files_sha256={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.name != 'final_completion.json' and p.suffix not in ['.lock','.tmp']},
        raw_logits_storage='External NPZ SHA256 inventory in raw_logits_manifest.csv; model weights/images are not committed.',
        no_accuracy_or_speedup_claim=True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['review', 'inventory'])
    args = parser.parse_args()
    review() if args.stage == 'review' else inventory()

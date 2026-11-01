"""Independently recompute historical metrics from bound raw vocabulary vectors."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import numpy as np
import torch
from safetensors.torch import load_file

from .common import OUT, QWEN, LLAVA, TASKS, read, write, sha256, csv_write
from .metrics import metrics


def qwen():
    # Historical CPU reductions used the >=8-thread reduction layout.
    torch.set_num_threads(8)
    samples = read(QWEN / 'sample_manifest.json')['samples']
    rows, raw_bindings = [], []
    for number, sample in enumerate(samples, 1):
        key = f"{sample['task']}__{sample['question_id']}"
        native = read(QWEN / 'native_checkpoints' / (key + '.json'))
        scan = read(QWEN / 'scan_checkpoints' / (key + '.json'))
        for record in (native, scan):
            assert record['input_binding'] == sample['input_binding']
            assert record['image_sha256'] == sample['image_sha256']
            assert sha256(record['raw_path']) == record['raw_sha256']
            raw_bindings.append(dict(path=record['raw_path'], sha256=record['raw_sha256']))
        full = load_file(native['raw_path'])['logits']
        off = load_file(scan['raw_path'])['logits']
        assert off.shape[0] == 36 and len(scan['rows']) == 36
        for layer, old in enumerate(scan['rows']):
            assert old['layer'] == layer
            new = metrics(full, off[layer])
            mapping = dict(full_first_token='full_token_id', changed_first_token='off_token_id',
                full_top1_logp='full_top1_logp', ablated_full_top1_logp='off_full_top1_logp',
                signed_logp_loss='signed_loss', absolute_logp_change='absolute_change',
                top1_flip='top1_flip', js_divergence='js_divergence')
            for a, b in mapping.items():
                assert old[a] == new[b], (key, layer, a, old[a], new[b])
            rows.append(dict(task=sample['task'], question_id=sample['question_id'], layer=layer, **new))
        if number % 20 == 0:
            print('QWEN LEGACY', number, '/120 bit-exact', flush=True)
    old_summary = list(csv.DictReader((QWEN / 'task_layer_sensitivity.csv').open()))
    summary_error = 0.
    groups = sorted({r['group'] for r in old_summary})
    assert groups == sorted(TASKS)
    for previous in old_summary:
        group, layer = previous['group'], int(previous['layer'])
        selected = [r for r in rows if r['task'] == group and r['layer'] == layer]
        assert len(selected) == int(previous['n_questions']) == 20
        for current, saved in [('signed_loss', 'mean_signed_logp_loss'), ('absolute_change', 'mean_absolute_logp_change')]:
            a = np.array([r[current] for r in selected], dtype=np.float64)
            error = abs(float(a.mean()) - float(previous[saved]))
            # FP64 reduction order bound, not a GPU-logit or FP32-loss tolerance.
            assert error <= len(a) * np.finfo(np.float64).eps * max(1., float(abs(a).max()))
            summary_error = max(summary_error, error)
    csv_write(OUT / 'qwen_legacy_recomputed.csv', rows)
    write(OUT / 'qwen_legacy_validation.json', dict(passed=True, questions=120, rows=4320,
        raw_metrics_bit_exact=True, cpu_reduction_threads=8, raw_bindings=raw_bindings,
        task_labels_and_layer_indices_valid=True, task_mean_max_fp64_reduction_error=summary_error,
        figure_source_sha256=sha256(QWEN / 'task_layer_sensitivity.csv'),
        source_plot_code_sha256=sha256(QWEN.parents[1] / 'qwen3vl_four_stage_pilot/analyze.py'),
        signed_definition='Full - OFF on fixed Full first argmax; not gold',
        plotting_issue='Historical heatmap is signed mean, not the mean absolute primary heatmap requested here. Raw loss computations reproduced exactly. Do not label old signed heatmap as absolute change.',
        gpu_regression='pending'))
    lines = ['# Qwen legacy audit', '',
        'All 120 original Full and 4320 OFF full-vocabulary vectors verified by SHA256.',
        'Independent FP32 recomputation exactly matches every saved target ID, signed loss, per-question absolute change, flip and JS.',
        'The previous task heatmap explicitly plots mean SIGNED loss, not mean ABSOLUTE change. This is a metric/figure-purpose mismatch if used as the new primary figure, not evidence that the raw logits are invalid.',
        'The prior plotting code uses real task labels and layers 0-35; its relative-depth denominator is checked separately in the new plotter.',
        'New primary heatmaps use mean(abs(Full-OFF)); signed heatmaps remain a separately named supplementary view.',
        'Native GPU regressions at L0, L2, L17, L24, L35 are a separate mandatory gate before formal Qwen collection.', '']
    (OUT / 'qwen_legacy_audit.md').write_text('\n'.join(lines), encoding='utf-8')


def llava():
    torch.set_num_threads(1)
    samples = {(s['task'], s['question_id']): s for s in read(OUT / 'shared_3000_manifest.json')['samples']}
    original = {(s['task'], s['question_id']): s for s in read(LLAVA / 'sample_manifest.json')['samples']}
    paths = sorted((LLAVA / 'first_token').glob('shard*/checkpoints/*.json'))
    assert len(paths) == 3000
    rows, bindings = [], []
    maxerr = defaultdict(float)
    seen = set()
    for number, path in enumerate(paths, 1):
        record = read(path)
        key = record['task'], record['question_id']
        assert key not in seen
        seen.add(key)
        sample = samples[key]
        compact = json.dumps(original[key], sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        assert hashlib.sha256(compact.encode()).hexdigest() == record['sample_sha256']
        r = record['result']
        assert sha256(r['logits_path']) == r['logits_sha256']
        bindings.append(dict(checkpoint=str(path), checkpoint_sha256=sha256(path), raw_path=r['logits_path'], raw_sha256=r['logits_sha256']))
        with np.load(r['logits_path']) as raw:
            full = torch.from_numpy(raw['full_logits'])
            changed = torch.from_numpy(raw['ablated_logits'])
        assert full.dtype == changed.dtype == torch.float32 and changed.shape[0] == 32
        for layer, old in enumerate(r['metrics']):
            assert old['layer'] == layer
            new = metrics(full, changed[layer])
            expected = dict(signed_loss=-old['delta_logp'], absolute_change=abs(old['delta_logp']),
                full_top1_logp=old['full_token_logp'], off_full_top1_logp=old['ablated_full_token_logp'],
                js_divergence=old['js'])
            for k, value in expected.items():
                error = abs(new[k] - value)
                maxerr[k] = max(maxerr[k], error)
                assert np.isclose(new[k], value, atol=2e-6, rtol=2e-5), (key, layer, k, error)
            assert new['full_token_id'] == old['full_token_id'] and new['off_token_id'] == old['ablated_token_id']
            assert new['top1_flip'] == int(old['top1_flip'])
            rows.append(dict(model='llava', task=key[0], question_id=key[1], image_id=sample['image_id'],
                image_sha256=sample['image_sha256'], layer=layer, relative_depth=layer / 31,
                input_length=r['mapping']['query'] + 1, image_tokens=576, **new))
        if number % 100 == 0:
            print('LLAVA RAW REPLAY', number, '/3000', flush=True)
    assert seen == set(samples)
    csv_write(OUT / 'llava_per_question_layer.csv', rows)
    write(OUT / 'llava_metric_compatibility.json', dict(passed=True, questions=3000, raw_vectors=99000,
        rows=len(rows), sign_conversion='historical delta_logp=OFF-Full; new signed_loss=-delta_logp',
        max_errors=dict(maxerr), original_frozen_atol=2e-6, original_frozen_rtol=2e-5,
        recomputed_absolute_before_averaging=True, source_bindings=bindings, no_gpu=True))


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['qwen', 'llava'])
    qwen() if parser.parse_args().model == 'qwen' else llava()

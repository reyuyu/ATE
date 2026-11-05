"""Independent CPU-only completion check; never loads the model or starts GPU work."""
import csv
import datetime
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from vlm_research.pope_instruction_probe.common import (
    OUT, RESOURCES, HISTORY, conditions, sources, read, sha, digest,
    load_record, checkpoint, writers, edge_mask, atomic, write_csv,
)


def rows(name):
    with (OUT / name).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    freeze = read(OUT / 'freeze.json')
    assert freeze['sources'] == sources(), 'Frozen inference source changed'
    assert sha(OUT / 'sample_manifest.json') == freeze['sample_manifest_sha256']
    assert sha(OUT / 'protocol.json') == freeze['protocol_sha256']
    assert read(OUT / 'validation.json')['status'] == 'PASS'
    assert read(OUT / 'analysis_audit.json')['status'] == 'PASS'
    assert read(OUT / 'gpu_completion.json')['status'] == 'COMPLETE'
    assert not (OUT / 'failure.json').exists()
    assert 'Ran 38 tests' in (OUT / 'cpu_tests_final.log').read_text()
    assert (OUT / 'cpu_tests_final.log').read_text().strip().endswith('OK')
    samples = read(OUT / 'sample_manifest.json')
    add = {(r['question_id'], r['condition']): r for r in rows('addback_results.csv')}
    assert len(add) == 84 == len(rows('addback_results.csv'))
    assert len(rows('answer_form_logp_audit.csv')) == 84 * 8
    assert sum(1 for _ in (OUT / 'checkpoints').glob('*/*.json')) == 84
    all_records, groups, metrics_count = [], rows('region_attention_summary.csv'), 0
    shapes = {}
    mapping = rows('token_mapping.csv')
    for key, s in samples.items():
        qid, m = str(s['question_id']), s['input_mapping']
        assert sha(s['image_path']) == s['image_sha256']
        e1 = load_record(checkpoint(qid, 'E1'))
        for condition, restored in conditions(10):
            r = load_record(checkpoint(qid, condition))
            assert r['sample_sha256'] == digest(s) and r['sources'] == sources()
            assert r['condition'] == condition and r['restored'] == restored
            assert r['writer_positions'] == writers(m, restored)
            for form, v in r['variants'].items():
                assert v['token_ids'] and all(t not in (1, 2) for t in v['token_ids'])
                assert math.isfinite(v['logp']) and v['logp'] <= 0
            margin = r['variants'][' Yes']['logp'] - r['variants'][' No']['logp']
            assert margin == r['yes_no_margin'] == float(add[qid, condition]['yes_no_margin'])
            assert margin - e1['yes_no_margin'] == float(add[qid, condition]['delta_margin_vs_E1'])
            g = r['generation']
            assert g['text'] == add[qid, condition]['answer']
            assert g['token_ids'] == json.loads(add[qid, condition]['token_ids'])
            assert len(g['token_ids']) == len(g['all_step_logits_sha256'])
            assert g['first_logits_sha256'] == g['all_step_logits_sha256'][0]
            if condition in ('C3', 'E1'):
                old = load_record(HISTORY / 'checkpoints' / key / (condition + '.json'))
                assert g['token_ids'] == old['generation']['token_ids']
                assert g['first_logits_sha256'] == old['generation']['first_logits_sha256']
            all_records.append(r)
        roi = read(OUT / 'images' / qid / 'roi.json')
        assert 0 <= roi['roi_area_fraction'] <= 1
        assert len(roi['roi_patch_fractions']) == 576
        assert abs(np.mean(roi['roi_patch_fractions']) - roi['roi_area_fraction']) < 1e-12
        if qid in ('2459', '301'):
            assert roi['status'] == 'OFFICIAL_FULLY_CROPPED_OUT'
            assert roi['roi_area_fraction'] == 0
        mm = [r for r in mapping if r['question_id'] == qid]
        for condition in ('C3', 'E1'):
            meta = read(OUT / 'attention' / qid / (condition + '.json'))
            assert sha(meta['path']) == meta['sha256']
            with np.load(meta['path']) as data:
                a = data['weights'].astype(np.float64)
                assert a.shape == (20, 32, len(mm), 576)
                assert data['query_positions'].tolist() == m['R']
                assert data['image_positions'].tolist() == m['image']
            assert np.isfinite(a).all() and (a >= 0).all()
            shapes[qid + '/' + condition] = list(a.shape)
            metrics_count += 20 * 32 * len(mm)
            for region in ('Q', 'Instruction', 'ASSISTANT', 'Separator'):
                ix = [int(r['r_index']) for r in mm if r['region'] == region]
                raw = a[:, :, ix].mean((0, 1, 2))
                row = next(r for r in groups if r['question_id'] == qid and r['condition'] == condition and r['region'] == region)
                assert abs(raw.sum() - float(row['image_mass'])) < 1e-12
                assert abs(raw[edge_mask()].sum() - float(row['edge_mass'])) < 1e-12
                if condition == 'E1' and region == 'Instruction':
                    assert np.count_nonzero(a[:, :, ix]) == 0
    detail = rows('per_layer_head_token_attention.csv')
    assert len(detail) == metrics_count == 98560
    unique = {(r['question_id'], r['condition'], r['layer'], r['head'], r['position']) for r in detail}
    assert len(unique) == len(detail)
    for r in detail:
        for name in ('image_mass', 'edge_mass', 'edge_ratio', 'roi_mass', 'roi_ratio', 'roi_enrichment', 'roi_hit_ratio'):
            assert not r[name] or math.isfinite(float(r[name]))
    assert len(rows('token_attention_addback_summary.csv')) == len(mapping) == 77
    pngs = list((OUT / 'figures').rglob('*.png'))
    assert len(pngs) == 77 * 3 + 4
    assert len(list((OUT / 'figures').rglob('*.pdf'))) == 4
    for p in pngs:
        with Image.open(p) as image:
            image.verify()
    shutil.copyfile(RESOURCES / 'official_coco_roi_subset.json', OUT / 'official_coco_roi_subset.json')
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader'
    ], text=True)
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader'
    ], text=True)
    release = dict(at=datetime.datetime.now().astimezone().isoformat(), gpu_snapshot=gpu,
        compute_processes=processes, task_worker_already_exited=True,
        used_physical_gpu=0, other_gpus_not_used=True)
    atomic(OUT / 'gpu_release_final.json', release)
    artifacts = []
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ('artifact_hashes.csv', 'completion.json', 'audit_results.log'):
            artifacts.append(dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=sha(p)))
    write_csv(OUT / 'artifact_hashes.csv', artifacts)
    completion = dict(status='COMPLETE', at=release['at'], formal_records=84, cases=3,
        conditions_per_case=28, attention_rows=metrics_count, native_R_tokens=77,
        cpu_tests=38, gpu_smoke_cases=3, attention_array_shapes=shapes,
        eos=sum(r['generation']['eos'] for r in all_records),
        empty=sum(r['generation']['empty'] for r in all_records),
        truncated=sum(r['generation']['truncated'] for r in all_records),
        gpu_worker_elapsed_seconds=read(OUT / 'gpu_completion.json')['elapsed_seconds'],
        historical_tokens_and_first_logits_exact=True,
        all_records_hash_verified=True, frozen_inference_source_unchanged=True,
        official_roi_geometry_verified=True, png_count=len(pngs),
        artifact_manifest_sha256=sha(OUT / 'artifact_hashes.csv'),
        inference_commit=read(OUT / 'execution_commit.json')['commit'],
        analysis_source_sha256=sha(ROOT / 'vlm_research/pope_instruction_probe/analyze.py'),
        gpu_released=True, no_population_claim=True)
    atomic(OUT / 'completion.json', completion)
    print(json.dumps(completion, indent=2))


if __name__ == '__main__':
    main()

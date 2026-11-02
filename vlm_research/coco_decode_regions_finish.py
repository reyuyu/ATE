"""Final integrity and resource audit; never performs GPU computation."""
import csv
from pathlib import Path
import subprocess
import time

from benchmarks.dataset_registry import sha256
from paper_rq1_single_layer_gqa6.run import atomic_json
from coco_decode_regions import OUT, read, verify, old
from vlm_common import write_json


def provenance():
    """Disambiguate actual historical devices from the current paired image shards."""
    current = read(OUT/'sample_manifest.json')
    historical = read(old.history.OUT/'sample_manifest.json')
    gpu_table = read(OUT/'protocol.json')['gpus']
    with (OUT/'generation_provenance.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question_id', 'condition', 'reused',
            'paired_shard_physical_gpu', 'actual_generation_physical_gpu', 'actual_generation_gpu_uuid'])
        writer.writeheader()
        for sample in current['samples']:
            q = sample['question_id']; paired = gpu_table[current['assignment'][q]]
            for key in read(OUT/'protocol.json')['conditions']:
                origin = (historical['gpus'][historical['assignments']['caption'][q]] if key == 'Full'
                          else dict(physical_gpu=0, uuid=old.GPU) if key == 'Decode None' else paired)
                writer.writerow(dict(question_id=q, condition=key, reused=key in ['Full', 'Decode None'],
                    paired_shard_physical_gpu=paired['physical_gpu'],
                    actual_generation_physical_gpu=origin['physical_gpu'], actual_generation_gpu_uuid=origin['uuid']))
    files = [path for path in OUT.rglob('*') if path.is_file() and path.name not in ['worker.lock', 'artifact_hashes.json']]
    hashes = {str(path.relative_to(OUT)): sha256(path) for path in sorted(files)}
    write_json(OUT/'artifact_hashes.json', hashes)
    assert all(sha256(OUT/path) == h for path, h in hashes.items())
    print(f'Provenance separated; {len(hashes)} final artifact hashes verified.', flush=True)


def main():
    p = verify(); validation = read(OUT/'validation.json'); assert validation['passed']
    publication = read(OUT/'publication_validation.json'); assert publication['passed']
    assert publication['analysis_source_sha256'] == sha256(OUT.parents[1]/'coco_decode_regions_analysis.py')
    with (old.OUT/'accuracy_and_quality_summary.csv').open() as f:
        historical = {r['condition']: r for r in csv.DictReader(f)}
    with (OUT/'condition_summary.csv').open() as f:
        current = {r['condition']: r for r in csv.DictReader(f)}
    for now, before in [('Full', 'A'), ('Decode None', 'DN')]:
        for metric in ['CIDEr', 'SPICE']:
            assert float(current[now][metric]) == float(historical[before][metric]), 'Official scorer historical parity failed'
    for path, binding in p['model_files'].items(): assert sha256(path) == binding['sha256']
    manifest = read(OUT/'sample_manifest.json')
    assert sorted(list(manifest['assignment'].values()).count(i) for i in range(3)) == [20]*3
    for s in manifest['samples']: assert sha256(s['image_path']) == s['image_sha256']
    for path, h in p['scorer_source'].items(): assert sha256(path) == h
    table, apps = old.history.snapshot()
    worker_pids = [read(OUT/f'shard{i}/preflight.json')['pid'] for i in range(3)]
    assert not any(Path(f'/proc/{pid}').exists() for pid in worker_pids), 'Worker still running'
    assert not any(str(pid) in apps for pid in worker_pids), 'Worker still owns GPU memory'
    atomic_json(OUT/'gpu_release.json', dict(released=True, used_physical_gpus=[0, 2, 3],
        uuids=[g['uuid'] for g in p['gpus']], gpu1_used=False, worker_pids=worker_pids,
        inventory=table, compute_apps=apps, checked_epoch=time.time()))
    atomic_json(OUT/'delivery_audit.json', dict(passed=True, execution_commit=read(OUT/'execution.json')['commit'],
        sources_unchanged=True, all_model_files_rehashed=True, images_rehashed=60,
        official_scorer_rehashed=True, reused_historical_generation_bindings_rehashed=120,
        official_Full_and_DecodeNone_rescoring_exact_historical=True,
        complete_new_checkpoints=600, complete_total_rows=720, gpu_released=True,
        wall_minutes_since_protocol=(time.time()-p['started_epoch'])/60,
        scope='Preparation through final integrity audit; excludes initial repository inspection and subsequent push.',
        main_modified=False, independent_branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()))
    provenance()
    print('All three workers exited, GPUs released.', flush=True)


if __name__ == '__main__': main()

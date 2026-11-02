"""Final CPU integrity and resource audit; no model forward or generation."""
from pathlib import Path
import time

from benchmarks.dataset_registry import sha256, digest
from vlm_common import write_json
from .run import OUT, ROOT, read, verify, sources, history


def main():
    p=verify();v=read(OUT/'validation.json');assert v['passed'] and v['complete_generations']==4620
    assert v['analysis_source_sha256']==sha256(ROOT/'attention_text_regions/analysis.py')
    m=read(OUT/'sample_manifest.json');pool=read(ROOT/'results/paper_rq1_single_layer_gqa6/sample_manifest.json')['samples']
    original={s['question_id']:s for s in pool}
    for s in m['samples']:
        assert sha256(s['image_path'])==s['image_sha256']
        if s['dataset']!='coco_caption':
            base=original[s['question_id']]
            assert digest(base)==s['original_sample_sha256']
            assert {k:v for k,v in s.items() if k not in ['dataset','original_sample_sha256']}==base
    for name,b in p['model_files'].items():assert sha256(name)==b['sha256']
    for name,h in p['scorer_source'].items():assert sha256(name)==h
    assert (OUT/'qualitative_notes_zh.md').exists(), 'Manual image/caption audit is required'
    table,apps=history.history.snapshot()
    pids=[read(OUT/f'shard{i}/preflight.json')['pid'] for i in range(3)]
    assert all(not Path(f'/proc/{pid}').exists() for pid in pids)
    active_pids=[int(line.split(', ')[1]) for line in apps.splitlines() if line.strip()]
    assert not set(pids)&set(active_pids)
    write_json(OUT/'gpu_release.json',dict(released=True,physical_gpus=[0,2,3],
        uuids=[g['uuid'] for g in p['gpus']],gpu1_used=False,
        worker_pids=pids,inventory=table,compute_apps=apps,checked_epoch=time.time()))
    write_json(OUT/'delivery_audit.json',dict(passed=True,generations=4620,
        gqa_questions=600,gqa_distinct_images=m['gqa_images'],coco_images=60,
        source_unchanged=True,original_gqa_samples_exact=True,all_model_files_rehashed=True,
        images_rehashed=660,official_scorer_sources_rehashed=True,gpu_released=True,gpu1_used=False,
        main_modified=False,code_commit=read(OUT/'execution.json')['commit'],
        source_sha256=dict(sources(),**{str(path.relative_to(ROOT)):sha256(path) for path in
            [ROOT/'attention_text_regions/analysis.py',ROOT/'attention_text_regions/finish.py']}),
        wall_minutes_since_protocol=(time.time()-p['started_epoch'])/60,
        clock_scope='Preparation through final publication audit; excludes initial repository/code inspection and subsequent Git push'))
    files=[path for path in OUT.rglob('*') if path.is_file() and path.name not in ['worker.lock','artifact_hashes.json']]
    hashes={str(path.relative_to(OUT)):sha256(path) for path in sorted(files)}
    write_json(OUT/'artifact_hashes.json',hashes)
    assert all(sha256(OUT/name)==h for name,h in hashes.items())
    print(f'Complete: {len(hashes)} artifacts verified, 4620 paired generations, all three workers exited.',flush=True)


if __name__=='__main__':main()

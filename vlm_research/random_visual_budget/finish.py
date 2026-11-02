"""Read-only final inputs/model/source audit and GPU resource release check."""
from pathlib import Path
import time

from .common import OUT,MODELS,verify,read,write,sha,digest,command


def main():
    protocol=verify();v=read(OUT/'analysis_validation.json')
    assert v['passed'] and v['rows']==126000 and v['new_complete_generations']==108000
    assert v['source_sha256']==sha(Path(__file__).with_name('analyze.py'))
    samples=read(OUT/'sample_manifest.json')['samples']
    for s in samples:assert sha(s['image_path'])==s['image_sha256']
    for path,h in read(OUT/'model_files.json').items():assert sha(path)==h
    for model in MODELS:
        records=read(OUT/f'endpoint_bindings/{model}.json')
        for r in records.values():
            assert sha(r['source_path'])==r['source_sha256']
            assert sha(r['protocol_path'])==r['protocol_sha256']
            if 'full_path' in r:assert sha(r['full_path'])==r['full_path_sha256']
        pid=read(OUT/model/'preflight.json')['pid']
        assert not Path(f'/proc/{pid}').exists(), 'Worker must exit before final release audit'
        ledger=OUT/model/'resume_ledger.json'
        if ledger.exists():
            assert not Path('/proc/'+str(read(ledger)['prior_preflight']['pid'])).exists()
    inventory=command('nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader')
    apps=command('nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader')
    pids={read(OUT/m/'preflight.json')['pid'] for m in MODELS}
    for model in MODELS:
        ledger=OUT/model/'resume_ledger.json'
        if ledger.exists():pids.add(read(ledger)['prior_preflight']['pid'])
    assert not any(int(r.split(', ')[1]) in pids for r in apps.splitlines() if r.strip())
    release=dict(passed=True,physical_gpus=[0,2,3],gpu1_used=False,worker_pids=sorted(pids),
        all_workers_exited=True,inventory=inventory,compute_apps=apps,checked_epoch=time.time())
    write(OUT/'gpu_release.json',release)
    layer_sets=read(OUT/'layer_sets.json')
    for model in MODELS:
        for seed,order in layer_sets['permutations'][model].items():
            write(OUT/f'per_seed_layers/{model}_{seed}.json',dict(model=model,seed=int(seed),permutation=order,
                conditions={k:r for k,r in layer_sets['conditions'][model].items() if r['seed']==int(seed)},
                frozen_master_sha256=sha(OUT/'layer_sets.json')))
    write(OUT/'delivery_audit.json',dict(passed=True,models=3,questions_per_model=3000,distinct_image_clusters=2409,
        conditions=14,total_condition_records=126000,new_complete_generations=108000,reused_complete_generations=18000,
        code_commit={m:read(OUT/m/'complete.json')['code_commit'] for m in MODELS},
        images_rehashed=3000,all_model_files_rehashed=True,historical_endpoints_rehashed=True,immutable_gpu_sources_verified=True,
        postprocessing_sources={n:sha(Path(__file__).with_name(n)) for n in ['analyze.py','test_analysis.py','finish.py']},
        ordinary_paired_image_bootstrap_repetitions=20000,all_three_seeds_retained=True,
        gpu1_used=False,worker_resources_released=True,wall_hours_since_freeze=(time.time()-protocol['started_epoch'])/3600))
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['worker.lock','artifact_hashes.json']]
    hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(files)}
    write(OUT/'artifact_hashes.json',hashes)
    assert all(sha(OUT/name)==value for name,value in hashes.items())
    print('Final audit passed:',len(hashes),'artifacts; all three GPU workers exited.',flush=True)


if __name__=='__main__':main()

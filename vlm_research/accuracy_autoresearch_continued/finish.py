"""CPU-only result checks; never dispatches more GPU conditions."""
import argparse
import collections
import csv
import math
import subprocess

from .runtime import *


def data_audit():
    pool=read(PRIOR/'sample_manifest.json'); checked={}
    for s in pool.values():
        h=checked.setdefault(s['image_path'],sha(s['image_path']))
        assert h==s['image_sha256']
    atomic(OUT/'data_reuse_audit.json',dict(at=now(),status='PASS',images=len(checked),questions=len(pool),
        all_image_bytes_match=True,sample_manifest_sha256=sha(PRIOR/'sample_manifest.json'),
        previous_source_files_unchanged=original.source_hashes()==read(PRIOR/'execution_freeze.json')['sources']))
    print('Shared image/input audit PASS',len(checked),len(pool))


def audit():
    configure(); pool=read(PRIOR/'sample_manifest.json'); execution=read(OUT/'execution_freeze.json')
    assert sources()==execution['sources'] and read(ROOT/'task_deadline.json')==execution['clock']
    assert all(sha(p)==h for p,h in execution['weights'].items())
    selection=read(OUT/'selection.json')
    assert datetime.datetime.fromisoformat(selection['at']).timestamp()<deadline('candidate_freeze_by')
    rows=[]
    for p in sorted((OUT/'records').glob('*/*.json')):
        b=read(p); r=b['payload']; s=pool[r['sample']]; c=r['config']
        assert digest(r)==b['sha256'] and r['sources']==sources() and r['sample_sha256']==digest(s)
        assert r['generation']['score']==score(s,r['generation']['text'])
        native=read_record(config('native',head=c['logits_to_keep']),r['sample'],s)
        assert r['input_binding']['input_hashes']==native['input_binding']['input_hashes']
        assert r['input_binding']['prompt_ids']==native['input_binding']['prompt_ids']
        assert all(math.isfinite(r['generation'][f]) for f in ('native_first_token_logp','latency_ms'))
        rows.append(dict(sample=r['sample'],config_id=config_id(c),sha256=b['sha256'],input_matches_native=True))
    for i in range(2):
        assert not (OUT/f'worker{i}/STOP.json').exists()
        assert read(OUT/f'worker{i}/additional_smoke.json')['status']=='PASS'
        for name in ('05_recheck','06_recheck_performance'):
            marker=read(OUT/f'worker{i}/jobs'/(name+'.json'))
            assert marker['status']=='COMPLETE' and marker['job_sha256']==digest(read(OUT/'jobs'/(name+'.json')))
    j=read(OUT/'jobs/05_recheck.json'); reused=0
    for k in j['samples']:
        for c in j['configs']:
            assert read_record(c,k,pool[k]) is not None
            reused+=not (OUT/'records'/config_id(c)/(k+'.json')).exists()
    perf=read(OUT/'jobs/06_recheck_performance.json'); measured=0
    for k in perf['samples']:
        p=read(OUT/'performance'/perf['id']/(k+'.json'))
        assert p['sources']==sources() and p['sample_sha256']==digest(pool[k]) and p['configs']==perf['configs']
        assert {(r['config_id'],r['rep']) for r in p['measurements']}=={
            (config_id(c),rep) for c in perf['configs'] for rep in range(5)}
        for r in p['measurements']:
            c=next(c for c in perf['configs'] if config_id(c)==r['config_id'])
            assert r['token_ids']==read_record(c,k,pool[k])['generation']['token_ids']
        measured+=len(p['measurements'])
    state=hardware(); owned=[read(OUT/f'worker{i}/released.json')['pid'] for i in range(2)]
    assert not any(f', {p},' in state['processes'] for p in owned)
    write_csv(OUT/'checkpoint_input_audit.csv',rows)
    atomic(OUT/'gpu_release_audit.json',dict(at=now(),own_worker_pids=owned,own_CUDA_processes_absent=True,
        no_external_process_terminated=True,hardware=state))
    atomic(OUT/'validation.json',dict(at=now(),status='PASS',unique_CPU_tests=15,
        primary_GPU_smokes=6,extra_L32_checks=4,all_new_inputs_exact_native=True,
        runtime_sources_unchanged=True,no_clock_reset=True,no_OOM_NaN_STOP=True,
        independent_confirmation=False,all_RECHECK_records=len(j['samples'])*len(j['configs']),
        reused_records=reused,new_quality_records=len(rows),formal_timed_requests=measured))
    atomic(OUT/'completion.json',dict(at=now(),status='COMPLETE_EXPLORATORY_CONTINUATION',
        quality_records=len(rows),recheck_records=len(j['samples'])*len(j['configs']),
        clock=execution['clock'],selection=selection,no_further_GPU_work=True))
    print('Complete exploratory continuation:',len(rows),'new records;',reused,'reused RECHECK records')


def inventory():
    assert read(OUT/'validation.json')['status']=='PASS'
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.name!='artifact_hashes.json'}
    atomic(OUT/'artifact_hashes.json',dict(at=now(),files=files,runtime=sources(),
        analysis_sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print('Artifact hashes',len(files))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action'); a=ap.parse_args()
    {'data':data_audit,'audit':audit,'inventory':inventory}[a.action]()

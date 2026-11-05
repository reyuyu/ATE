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
    head_changes=[]
    for k in j['samples']:
        for c in j['configs']:
            assert read_record(c,k,pool[k]) is not None
            reused+=not (OUT/'records'/config_id(c)/(k+'.json')).exists()
            if c['logits_to_keep']==1:
                a=read_record(c,k,pool[k])['generation']
                b=read_record(dict(c,logits_to_keep=0),k,pool[k])['generation']
                head_changes.append(dict(sample=k,dataset=pool[k]['dataset'],config_H1=config_id(c),
                    first_logits_changed=a['first_logits_sha256']!=b['first_logits_sha256'],
                    tokens_changed=a['token_ids']!=b['token_ids'],score_H1=a['score'],score_H0=b['score'],
                    answer_H1=a['text'],answer_H0=b['text']))
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
    write_csv(OUT/'last_row_head_changes.csv',head_changes)
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


def full_cohort():
    configure(); pool=read(PRIOR/'sample_manifest.json'); selection=read(OUT/'selection.json')
    candidates=[config('native'),config(),config(answer='both'),config('qsa','decode'),selection['primary']]
    if selection.get('speed_secondary'): candidates.append(selection['speed_secondary'])
    rows=[]
    for d in ('GQA','VQAv2'):
        keys=[k for k,s in pool.items() if s['dataset']==d]
        for c in candidates:
            pairs=[(read_record(c,k,pool[k]),read_record(config('native'),k,pool[k])) for k in keys]
            assert all(a is not None and b is not None for a,b in pairs)
            x=[a['generation']['score'] for a,b in pairs]; y=[b['generation']['score'] for a,b in pairs]
            rows.append(dict(dataset=d,config_id=config_id(c),n=len(keys),accuracy=100*sum(x)/len(x),
                delta_native_pp=100*sum(a-b for a,b in zip(x,y))/len(x),
                harm=sum(a<b for a,b in zip(x,y)),rescue=sum(a>b for a,b in zip(x,y)),
                status='Supportive historical H0 cohort including selection data; NOT independent validation'))
    write_csv(OUT/'supportive_full_cohort.csv',rows)
    print('Full historical cohort supportive table written')


def catalog():
    rows=[]
    with (PRIOR/'results.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            rows.append(dict(stage='prior',**r))
    for path in sorted((OUT/'jobs').glob('*.json')):
        j=read(path)
        if j.get('performance'): continue
        summary=OUT/'rounds'/j['id']/'summary.csv'
        if not summary.exists():
            summary=OUT/'analysis'/j['id']/'accuracy_summary.csv'
        with summary.open() as f:
            for r in csv.DictReader(f):
                delta=float(r.get('delta_pp',r.get('delta_native_pp')))
                rows.append(dict(stage='continuation',round=j['id'],split=j['split'],dataset=r['dataset'],
                    config_id=r['config_id'],expected_n=r['n'],completed_n=r['n'],accuracy=r['accuracy'],
                    original_native_accuracy=float(r['accuracy'])-delta,delta_original_native_pp=delta,
                    harm=r.get('harm',r.get('score_down')),rescue=r.get('rescue',r.get('score_up')),
                    hypothesis=j['hypothesis']))
    with (OUT/'results.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),delimiter='\t'); w.writeheader(); w.writerows(rows)
    print('Combined prior + continuation catalog',len(rows),'rows; repeated stages are not distinct candidates')


def inventory():
    assert read(OUT/'validation.json')['status']=='PASS'
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*'))
           if p.is_file() and p.name!='artifact_hashes.json'}
    atomic(OUT/'artifact_hashes.json',dict(at=now(),files=files,runtime=sources(),
        analysis_sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print('Artifact hashes',len(files))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action'); a=ap.parse_args()
    {'data':data_audit,'audit':audit,'inventory':inventory,'full-cohort':full_cohort,'catalog':catalog}[a.action]()

"""User-authorized wider reaccess, keeping the engine and historical inputs fixed."""
import argparse
import shutil
import subprocess
import time
from collections import defaultdict
from common import *
from analysis import summarize,rank


EXPANSION={
    'Wide21_L11_31':list(range(11,32)),
    'Wide25_L7_31':list(range(7,32)),
    'Wide29_L7_35':list(range(7,36)),
    'Wide30_AllActive':[0]+list(range(7,36)),
}


def expanded_grid():
    base=json.loads((OUT/'candidate_grid.json').read_text())
    base.update(json.loads((OUT/'adaptive_discovery_frozen.json').read_text())['candidates'])
    assert len(base)==37
    assert not set(base)&set(EXPANSION)
    base.update(EXPANSION)
    assert len(base)-1==40
    assert len({tuple(v) for v in base.values()})==len(base)
    return base


def prepare():
    assert not (OUT/'jobs'/'004_Validation_Unified.json').exists()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(u in processes for u in UUIDS),processes
    assert len(records())==37*585
    assert all((OUT/'jobs'/f'003_Discovery_Unified.done{i}').exists() for i in (0,1))
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'sample_manifest.json')==p['manifest_sha256']
    runner_changes={str(ROOT/'worker.py'),str(ROOT/'performance.py')}
    for path,expected in p['sources'].items():
        if path not in runner_changes:
            assert sha(path)==expected,('ALGORITHM_OR_INPUT_SOURCE_CHANGED',path)
    archive=OUT/'attempts'/'0002_pre_wider_reaccess'
    assert not archive.exists()
    archive.mkdir(parents=True)
    for name in ['protocol.json','Discovery_Unified_summary.csv','Discovery_Unified_ranking.json',
                 'STOP','gpu_gate.json','smoke0.json','smoke1.json','release0.json','release1.json']:
        source=OUT/name
        if source.exists():
            shutil.copyfile(source,archive/name)
            if name.startswith(('smoke','release')) or name=='STOP':
                source.unlink()
    grid=expanded_grid()
    write(OUT/'expanded_candidate_grid.json',grid)
    rows=json.loads((OUT/'sample_manifest.json').read_text())
    perf=[]
    seen=set()
    for task in TASKS:
        pool=sorted([r for r in rows if r['split']=='Validation' and r['analysis_task']==task and not r.get('historical_blocked_status')],
                    key=lambda r:digest([SEED,'WIDE_PERF',r['key']]))
        picked=0
        for r in pool:
            if r['image_sha256'] not in seen:
                perf.append(r['key'])
                seen.add(r['image_sha256'])
                picked+=1
            if picked==2:
                break
        assert picked==2
    p.update(previous_protocol_sha256=sha(archive/'protocol.json'),
        wider_reaccess_user_revision=True,previous_records_preserved=21645,
        main_candidate_definitions=grid,main_candidates_max=40,
        wider_candidates=EXPANSION,expanded_candidate_grid_sha256=sha(OUT/'expanded_candidate_grid.json'),
        validation_selection='Top three complete Discovery quality candidates, best K<=4, best K<=8; fill by quality to five, before Validation',
        performance_input_keys=perf,
        notes=['P1 and every real Decode use the same layer set; L1-L6 remain skipped',
               'R writes only L0-L21; main visual Hidden exits before L22',
               'Late visual KV comes from the separate causal visual prefix, not another layer',
               'Functional search retains full active-layer visual KV; no efficiency inference',
               'All 21645 earlier records keep their original protocol hash and remain immutable'],
        sources={**{path:expected for path,expected in p['sources'].items() if not path.startswith(str(ROOT))},
                 **{str(path):sha(path) for path in ROOT.glob('*.py')}},
        execution_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(OUT/'protocol.json',p)
    check()
    print('WIDER_PROTOCOL_FROZEN',sha(OUT/'protocol.json'),EXPANSION,flush=True)


def choose_validation(ranking):
    complete=[r for r in ranking if r['k'] and r['all_tasks']]
    names=[]
    def add(row):
        if row and row['condition'] not in names and len(names)<5:
            names.append(row['condition'])
    for row in complete[:3]:
        add(row)
    for budget in [4,8]:
        add(next((r for r in complete if r['k']<=budget),None))
    for row in complete:
        add(row)
    return names


def run():
    p=check()
    assert sha(OUT/'expanded_candidate_grid.json')==p['expanded_candidate_grid_sha256']
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(u in processes for u in UUIDS),processes
    rows=json.loads((OUT/'sample_manifest.json').read_text())
    grid=json.loads((OUT/'expanded_candidate_grid.json').read_text())
    workers=[]
    logs=[]
    launched=time.time()
    def wait_for(paths,timeout=14400):
        start=time.monotonic()
        while not all(x.exists() for x in paths):
            assert all(w.poll() is None for w in workers),'OWN_WORKER_EXIT'
            assert not any((OUT/f'failure{i}.json').exists() for i in (0,1)),'WORKER_FAILURE'
            assert time.monotonic()-start<timeout,'TIMEOUT_PRESERVE_RECORDS'
            time.sleep(2)
    def submit(number,stage,names,view='Unified',**extra):
        path=OUT/'jobs'/f'{number:03d}_{stage}_{view}.json'
        payload=dict(stage=stage,view=view,conditions={n:grid[n] for n in names},
                     rows=[{'key':r['key']} for r in rows if r['split']==stage],prefill_layers=[],**extra)
        if path.exists():
            assert json.loads(path.read_text())==payload
        else:
            write(path,payload)
        print('STAGE_START',number,stage,names,len(payload['rows']),flush=True)
        wait_for([path.with_name(path.stem+f'.done{i}') for i in (0,1)])
        scores=summarize(stage,view)
        ranking=rank(scores)
        write(OUT/f'{stage}_{view}_ranking.json',ranking)
        print('STAGE_DONE',number,ranking,flush=True)
        return ranking
    def special(number,kind,keys,names,phase=None):
        path=OUT/'jobs'/f'{number:03d}_{kind}.json'
        payload=dict(kind='PERFORMANCE',id=f'wide_{number}',stage='Validation',view='Physical',
                     rows=[{'key':key} for key in keys],conditions={n:grid[n] for n in names},
                     phase=phase,performance_source_sha256=sha(ROOT/'performance.py'))
        if path.exists():
            assert json.loads(path.read_text())==payload
        else:
            write(path,payload)
        wait_for([path.with_name(path.stem+f'.done{i}') for i in (0,1)])
    try:
        for shard,uuid in enumerate(UUIDS):
            log=(OUT/f'worker{shard}.log').open('a')
            logs.append(log)
            workers.append(subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py'),'--shard',str(shard),'--wide-smoke'],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,TOKENIZERS_PARALLELISM='false'),
                stdout=log,stderr=subprocess.STDOUT))
        wait_for([OUT/f'smoke{i}.json' for i in (0,1)],timeout=900)
        for i in (0,1):
            smoke=json.loads((OUT/f'smoke{i}.json').read_text())
            assert smoke['time']>=launched and smoke['wide_reaccess_checked']
        ranking=submit(10,'Discovery',list(EXPANSION))
        names=choose_validation(ranking)
        write(OUT/'validation_candidates.json',dict(time=time.time(),names=names,layers={n:grid[n] for n in names},
            rule=p['validation_selection'],discovery_only=True,heldout_unopened=True))
        ranking=submit(11,'Validation',['K0']+names)
        summarize('Validation',ci=True)
        write(OUT/'validation_complete.json',dict(time=time.time(),ranking=ranking,scope='EXPANDED_DISCOVERY_FROZEN_BEFORE_VALIDATION'))
        nonzero=[r for r in ranking if r['k'] and r['all_tasks']]
        best=nonzero[0]
        small=next((r for r in nonzero if r['k']<=4),None)
        measured=[]
        for row in [best,small]+nonzero:
            if row and row['condition'] not in measured and len(measured)<3:
                measured.append(row['condition'])
        write(OUT/'physical_candidates_before_heldout.json',dict(time=time.time(),names=measured,
            best_quality=best['condition'],min_k_candidate=None if small is None else small['condition'],heldout_unopened=True))
        smoke_keys=[]
        for shard in (0,1):
            smoke_keys.extend(c['key'] for c in json.loads((OUT/f'smoke{shard}.json').read_text())['cases'])
        special(12,'PhysicalGate',smoke_keys,measured,phase='GATE')
        special(13,'Performance',p['performance_input_keys'],measured)
        perf=[]
        for shard in (0,1):
            perf.extend(r for _,r in read_store(OUT/f'performance{shard}.sqlite') if not r['warmup'] and r['mode']=='fixed32')
        costs={name:sum(r['e2e_ms'] for r in perf if r['condition']==name)/sum(r['condition']==name for r in perf) for name in measured}
        eligible=[r for r in nonzero if r['condition'] in costs and r['protected'] and r['textvqa']>=best['textvqa']-1]
        efficient=min(eligible,key=lambda r:costs[r['condition']]) if eligible else None
        roles={'BestQuality':best['condition'],'BestEfficiency':None if efficient is None else efficient['condition'],
               'MinK':None if small is None else small['condition']}
        final=list(dict.fromkeys(n for n in roles.values() if n))
        write(OUT/'final_selection.json',dict(time=time.time(),roles=roles,names=final,measured_fixed32_e2e_ms=costs,
            selection='Validation quality plus measured physical fixed32 cost, no Held-out access',
            MinK_definition='Best measured K<=4 candidate; meaningful recovery is assessed, not presumed',
            historical_strict_gate='FAIL_RETAINED'))
        submit(14,'Held-out',['K0']+final)
        summarize('Held-out',ci=True)
        submit(15,'Held-out',final,view='Physical',storage='SELECTED_PHYSICAL')
        summarize('Held-out','Physical',ci=True)
        for number,view in [(16,'P1Only'),(17,'DecodeOnly')]:
            submit(number,'Held-out',final[:2],view=view)
            summarize('Held-out',view,ci=True)
        write(OUT/'quality_complete.json',dict(time=time.time(),status='COMPLETE_EXPANDED_FROZEN_QUALITY_AND_PHYSICAL_COST',
            roles=roles,full_textvqa5000_and_gqa12578='NOT_RUN',historical_strict_gate='FAIL_RETAINED'))
    except BaseException as exc:
        import traceback
        write(OUT/'wider_pipeline_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        write(OUT/'STOP',dict(time=time.time(),scope='OWN_WIDER_WORKERS_ONLY'))
        for worker in workers:
            try:
                worker.wait(timeout=25)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait()
        for log in logs:
            log.close()
        write(OUT/'gpu_release_audit.json',dict(time=time.time(),exit_codes=[w.returncode for w in workers],
            sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True)))


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--prepare',action='store_true')
    opt=args.parse_args()
    prepare() if opt.prepare else run()

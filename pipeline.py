"""Finite local coordinator. Never changes a running worker's code or config."""
import subprocess
import time
from common import *
from analysis import summarize,rank


def main():
    p=check()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert not any(uuid in processes for uuid in UUIDS),processes
    rows=json.loads((OUT/'sample_manifest.json').read_text())
    grid=json.loads((OUT/'candidate_grid.json').read_text())
    jobs=[]
    handles=[]
    def wait_for(paths,timeout=7200):
        started=time.monotonic()
        while not all(path.exists() for path in paths):
            assert all(j.poll() is None for j in jobs),('WORKER_EXIT',[j.poll() for j in jobs])
            assert not any((OUT/f'failure{i}.json').exists() for i in [0,1]),'WORKER_FAILURE'
            assert time.monotonic()-started<timeout,'STAGE_TIMEOUT_PRESERVE_CHECKPOINTS'
            time.sleep(2)
    def submit(number,stage,names,view='Unified',prefill_layers=None):
        path=OUT/'jobs'/f'{number:03d}_{stage}_{view}.json'
        pool=[r for r in rows if r['split']==stage]
        payload=dict(stage=stage,view=view,conditions={n:grid[n] for n in names},rows=[{'key':r['key']} for r in pool],prefill_layers=prefill_layers or [])
        assert not path.exists()
        write(path,payload)
        print('STAGE_START',number,stage,names,len(pool),flush=True)
        wait_for([path.with_name(path.stem+f'.done{i}') for i in [0,1]])
        summary=summarize(stage,view)
        ranking=rank(summary)
        write(OUT/f'{stage}_{view}_ranking.json',ranking)
        print('STAGE_DONE',number,ranking,flush=True)
        return ranking
    try:
        for shard,uuid in enumerate(UUIDS):
            log=(OUT/f'worker{shard}.log').open('a')
            handles.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-u',str(ROOT/'worker.py'),'--shard',str(shard)],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,TOKENIZERS_PARALLELISM='false'),stdout=log,stderr=subprocess.STDOUT))
        wait_for([OUT/f'smoke{i}.json' for i in [0,1]],timeout=900)
        write(OUT/'gpu_gate.json',dict(status='PASS_UNIFIED_P1_AND_DECODE_SMOKE',historical_strict_gate='FAIL_RETAINED',time=time.time()))
        ranking=submit(1,'Discovery',['K0','R14','R11','R8','R6','R4','R2','R1'])
        # First comparison is deliberately a checkpoint before broadening the frozen grid.
        write(OUT/'initial_comparison_complete.json',dict(time=time.time(),ranking=ranking))
        if not (OUT/'CONTINUE_SEARCH').exists():
            print('INITIAL_COMPARISON_COMPLETE_AWAITING_LOCAL_REVIEW',flush=True)
            wait_for([OUT/'CONTINUE_SEARCH'],timeout=1800)
        ranking=submit(2,'Discovery',[n for n in grid if n not in ['K0','R14','R11','R8','R6','R4','R2','R1']])
        # Remaining budget: deterministic probes from two Discovery leaders.
        candidates=[r for r in ranking if r['k'] and r['condition']!='K0'][:2]
        available=40-(len(grid)-1)
        seen={tuple(x) for x in grid.values()}
        adaptive={}
        probes=[]
        for leader in candidates:
            layers=leader['layers']
            order=sorted(layers,key=lambda i:(-int(i>=22),digest([SEED,'DROP',leader['condition'],i])))
            probes.append([(leader,i) for i in order])
        while available and any(probes):
            for seq in probes:
                if not seq or not available:
                    continue
                leader,i=seq.pop(0)
                value=[j for j in leader['layers'] if j!=i]
                if tuple(value) in seen:
                    continue
                name=f"Drop_{leader['condition']}_L{i}"
                adaptive[name]=value
                seen.add(tuple(value))
                available-=1
        grid.update(adaptive)
        write(OUT/'adaptive_discovery_frozen.json',dict(time=time.time(),candidates=adaptive,
            leaders=candidates,total_major_nonzero=len(grid)-1,max_major=40,
            limitation='Budget-bounded single-layer probes; untested deletion edges explicitly remain unmeasured'))
        if adaptive:
            ranking=submit(3,'Discovery',list(adaptive))
        nonzero=[r for r in ranking if r['k'] and r['all_tasks']]
        chosen=[]
        def pick(pool):
            if pool and pool[0]['condition'] not in chosen:
                chosen.append(pool[0]['condition'])
        pick(nonzero)
        pick([r for r in nonzero if r['protected'] and not r['late_layers']])
        pick([r for r in nonzero if r['k']<=4])
        pick([r for r in nonzero if r['k']<=8])
        for r in nonzero:
            if len(chosen)>=5:
                break
            pick([r])
        write(OUT/'validation_candidates.json',dict(time=time.time(),names=chosen,layers={n:grid[n] for n in chosen},
            rule='Discovery quality leader, protected pre-exit leader, best K<=4, best K<=8, then quality order'))
        ranking=submit(4,'Validation',['K0']+chosen)
        summarize('Validation',ci=True)
        write(OUT/'validation_complete.json',dict(time=time.time(),ranking=ranking))
        # Physical cost measurements and selection are reviewed locally before held-out is opened.
        wait_for([OUT/'final_selection.json'],timeout=3600)
        final=json.loads((OUT/'final_selection.json').read_text())
        names=list(dict.fromkeys(final['names']))
        write(OUT/'selection_before_heldout.json',final)
        submit(5,'Held-out',['K0']+names)
        summarize('Held-out',ci=True)
        submit(6,'Held-out',names[:2],view='P1Only')
        summarize('Held-out','P1Only',ci=True)
        submit(7,'Held-out',names[:2],view='DecodeOnly')
        summarize('Held-out','DecodeOnly',ci=True)
        write(OUT/'quality_complete.json',dict(status='COMPLETE_FROZEN_SELECTION_HELDOUT',time=time.time()))
    except BaseException as exc:
        import traceback
        write(OUT/'pipeline_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        raise
    finally:
        write(OUT/'STOP',dict(time=time.time()))
        for job in jobs:
            try:
                job.wait(timeout=20)
            except subprocess.TimeoutExpired:
                job.terminate()
                job.wait()
        for handle in handles:
            handle.close()
        write(OUT/'gpu_release_audit.json',dict(time=time.time(),exit_codes=[j.returncode for j in jobs],
            sample=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True)))


if __name__=='__main__':
    main()

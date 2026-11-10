"""Discovery -> three-candidate validation -> once-only frozen held-out."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .freeze import OUT, FIRST, ROOT, CANDIDATES, freeze, check
from .engine import generate
from ..qwen3vl_overnight.common import MODEL, GPUS, TASKS, disk, prepare, digest, sha, write, table, scorers
from ..qwen3vl_overnight.store import Store
from ..qwen3vl_overnight.report import records as first_records
from ..qwen3vl_overnight.metrics import complete_keys, metric, paired
from ..qwen3vl_svh_numerical_repair.engine import generate as dense_reference
from ..qwen3vl_svh_preflight48.run import resource, strict_equal
from ..qwen3vl_early_windows.experiment import tensor_digest


def load(stage):
    check()
    rows=json.loads((OUT/(stage+'_manifest.json')).read_text())
    grid=json.loads((OUT/'candidate_grid.json').read_text())
    if stage=='Discovery':
        names=[n for n,c in grid.items() if c['status']=='ELIGIBLE']
    elif stage=='Validation':
        names=['K0']+json.loads((OUT/'validation_selection.json').read_text())['candidates']
    else:
        frozen=json.loads((OUT/'final_selection.json').read_text())
        names=list(dict.fromkeys(['K0',frozen['quality'],frozen['tradeoff'],frozen['minimum_budget']]))
    return rows,names,grid


def worker(stage,shard):
    rows,names,grid=load(stage)
    selected=names[shard::2]
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPUS[shard]
    before=resource()
    assert GPUS[shard] not in before['processes']
    write(OUT/stage/f'gpu{shard}_before.json',before)
    disk()
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    model=store=None
    try:
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='eager',device_map={'':'cuda:0'}).eval()
        # Zero re-access must equal the original functional reference exactly.
        probe=min(rows,key=lambda r:(r['input_length'],r['key']))
        inputs,layout=prepare(processor,probe)
        baseline=dense_reference(model,processor,inputs,layout,'F',capture=True,audit=True,max_tokens=probe['max_new_tokens'])
        zero=generate(model,processor,inputs,layout,[],capture=True,audit=True,cap=probe['max_new_tokens'])
        assert strict_equal(baseline,zero)
        access=generate(model,processor,inputs,layout,[26],capture=True,audit=True,cap=2)
        assert zero[0]['audit']['prefill_non_target_hashes']==access[0]['audit']['prefill_non_target_hashes']
        assert len(access[0]['audit']['permission_audit'])==36*len(access[0]['token_ids'])
        write(OUT/stage/f'smoke{shard}.json',dict(status='PASS',key=probe['key'],functional_K0_reference_equal=True,
            numeric_mode='RELAXED_NUMERICAL_QUALITY_ONLY',physical_svhr=False,
            non_P1_prefill_hidden_equal=True,read_permissions=access[0]['audit']['permission_audit'],
            K0_tokens=zero[0]['token_ids'],K0_logits_hashes=[tensor_digest(v) for v in zero[1]],protocol_sha256=sha(OUT/'protocol.json')))
        del inputs,baseline,zero,access
        gc.collect()
        torch.cuda.empty_cache()
        while not all((OUT/stage/f'smoke{s}.json').exists() for s in [0,1]):
            assert not (OUT/'STOP_STRUCTURE.json').exists() and not (FIRST/'STOP_USER').exists()
            time.sleep(2)
        comparison=[json.loads((OUT/stage/f'smoke{s}.json').read_text()) for s in [0,1]]
        assert all(r['protocol_sha256']==sha(OUT/'protocol.json') and r['status']=='PASS' for r in comparison)
        write(OUT/stage/f'cross_gpu_check{shard}.json',dict(
            K0_tokens_equal=comparison[0]['K0_tokens']==comparison[1]['K0_tokens'],
            K0_logits_exact=comparison[0]['K0_logits_hashes']==comparison[1]['K0_logits_hashes'],
            same_probe=comparison[0]['key']==comparison[1]['key'],differences_never_relabelled_strict_PASS=True))
        bind=digest(dict(protocol=sha(OUT/'protocol.json'),stage=stage,names=names))
        store=Store(OUT/stage/f'shard{shard}.sqlite',bind)
        store.plan([r['key'] for r in rows],selected)
        scoring=scorers()
        done=0
        for name in selected:
            layers=grid[name]['layers']
            for row in rows:
                if (OUT/'STOP_STRUCTURE.json').exists() or (FIRST/'STOP_USER').exists():
                    raise RuntimeError('Search stopped; checkpoints retained')
                if store.state(row['key'],name)!='PENDING':
                    continue
                disk()
                inputs,layout=prepare(processor,row)
                assert digest(layout)==row['expected_input_binding_sha256']
                store.begin(row['key'],name)
                try:
                    result,_=generate(model,processor,inputs,layout,layers,cap=row['max_new_tokens'])
                    r=dict(status='COMPLETE',text=result['text'],token_ids=result['token_ids'],
                        generated_tokens=result['generated_tokens'],eos=result['eos'],truncated=result['truncated'],empty=result['empty'],
                        peak_allocated=result['peak_allocated'],peak_reserved=result['peak_reserved'],
                        scoring=scoring.score(row,result['text']),structural='PASS',
                        actual_live_KV_bytes=result['audit']['kv']['live_bytes'],actual_capacity_KV_bytes=result['audit']['kv']['capacity_bytes'],
                        actual_functional_image_KV_bytes=result['functional_image_KV_live_bytes'],physical_svhr=False)
                    del result
                except torch.OutOfMemoryError as exc:
                    r=dict(status='SKIPPED_OOM',error=str(exc),traceback=traceback.format_exc(),
                        allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
                r.update(key=row['key'],condition=name,benchmark=row['benchmark'],question_id=row['question_id'],
                    image_sha256=row['image_sha256'],gold=row['gold'],input_binding_sha256=digest(layout),
                    image_tokens=row['image_tokens'],input_length=row['input_length'],layers=layers,
                    mode='FUNCTIONAL_REACCESS_ONLY',stage=stage,gpu_uuid=GPUS[shard],protocol_sha256=sha(OUT/'protocol.json'),
                    strategy=row.get('strategy'),mme_task=row.get('task'),pair_id=row.get('pair_id'),circular_group=row.get('circular_group'))
                store.finish(row['key'],name,r)
                done+=1
                if r['status']=='SKIPPED_OOM':
                    gc.collect()
                    torch.cuda.empty_cache()
                    pi,pl=prepare(processor,probe)
                    generate(model,processor,pi,pl,[],cap=2,audit=True)
                    del pi
                    gc.collect()
                    torch.cuda.empty_cache()
                if done%16==0:
                    state=resource()
                    assert not [line for line in state['processes'].splitlines() if GPUS[shard] in line and f', {os.getpid()},' not in line], 'PAUSED_GPU_EXTERNAL_PROCESS'
                    write(OUT/stage/f'worker{shard}_progress.json',dict(records=done,counts=store.counts(),condition=name,last=row['key'],time=time.time()))
                    store.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                    print('SEARCH',stage,shard,done,name,row['key'],flush=True)
                del inputs
                gc.collect()
        write(OUT/stage/f'worker{shard}_completion.json',dict(status='TERMINAL',counts=store.counts(),names=selected))
    except BaseException as exc:
        failure=dict(error=repr(exc),traceback=traceback.format_exc(),stage=stage,shard=shard)
        write(OUT/stage/f'worker{shard}_failure.json',failure)
        write(OUT/'STOP_STRUCTURE.json',failure)
        raise
    finally:
        if store:
            store.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write(OUT/stage/f'gpu{shard}_after_inprocess.json',resource())


def stage_records(stage):
    import sqlite3,zlib,hashlib
    out=[]
    for path in (OUT/stage).glob('shard*.sqlite'):
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        for blob,expected in db.execute('SELECT payload,payload_sha256 FROM predictions WHERE payload IS NOT NULL'):
            raw=zlib.decompress(blob)
            assert hashlib.sha256(raw).hexdigest()==expected
            out.append(json.loads(raw))
        assert not db.execute('SELECT count(*) FROM predictions WHERE state IN ("PENDING","RUNNING")').fetchone()[0]
        db.close()
    assert len(out)==len({(r['key'],r['condition']) for r in out})
    return out


def summarize(stage):
    rows,names,grid=load(stage)
    output=stage_records(stage)
    assert len(output)==len(rows)*len(names)
    scoring=scorers()
    by_id={r['key']:r for r in rows}
    complete=defaultdict(dict)
    for r in output:
        if r['status']=='COMPLETE':
            assert r['scoring']==scoring.score(by_id[r['key']],r['text'])
            assert r['input_binding_sha256']==by_id[r['key']]['expected_input_binding_sha256']
            complete[r['key']][r['condition']]=r
    first,_=first_records()
    baseline=defaultdict(dict)
    for r in first:
        if r['key'] in by_id and r['status']=='COMPLETE':
            baseline[r['key']][r['condition']]=r
    mapping=json.loads((FIRST/'shards.json').read_text())
    summary=[]
    for task in TASKS:
        pool=[r for r in rows if r['benchmark']==task]
        valid={r['key'] for r in pool if set(complete[r['key']])==set(names) and set(baseline[r['key']])=={'Native','SVH'}}
        valid=complete_keys(task,pool,valid)
        for name in names:
            selected=[complete[k][name] for k in sorted(valid)]
            native=[baseline[k]['Native'] for k in sorted(valid)]
            physical=[baseline[k]['SVH'] for k in sorted(valid)]
            zero=[complete[k]['K0'] for k in sorted(valid)]
            vs_physical=paired(task,physical,selected,mapping,repeats=20000)
            vs_native=paired(task,native,selected,mapping,repeats=20000)
            vs_zero=paired(task,zero,selected,mapping,repeats=20000)
            summary.append(dict(task=task,condition=name,k=grid[name]['k'],n=len(valid),planned=len(pool),
                score=metric(task,selected,mapping),native=metric(task,native,mapping),svh=metric(task,physical,mapping),functional_K0=metric(task,zero,mapping),
                delta_svh=vs_physical['delta'],delta_svh_low=vs_physical['low'],delta_svh_high=vs_physical['high'],
                ci_status_svh=vs_physical.get('ci_status'),bootstrap_valid_svh=vs_physical.get('valid_bootstrap_replicates'),
                image_clusters=vs_physical['clusters'],
                delta_native=vs_native['delta'],delta_native_low=vs_native['low'],delta_native_high=vs_native['high'],
                delta_K0=vs_zero['delta'],delta_K0_low=vs_zero['low'],delta_K0_high=vs_zero['high'],
                harm=sum(x['scoring']['score']<y['scoring']['score'] for x,y in zip(selected,physical)),
                rescue=sum(x['scoring']['score']>y['scoring']['score'] for x,y in zip(selected,physical)),
                harm_K0=sum(x['scoring']['score']<y['scoring']['score'] for x,y in zip(selected,zero)),
                rescue_K0=sum(x['scoring']['score']>y['scoring']['score'] for x,y in zip(selected,zero)),
                same_gpu_as_svh=sum(x['gpu_uuid']==y['gpu_uuid'] for x,y in zip(selected,physical)),
                mean_actual_KV_bytes=sum(r['actual_live_KV_bytes'] for r in selected)/max(len(selected),1),
                physical_svhr=False,mode='FUNCTIONAL_REACCESS_ONLY'))
    ranking=[]
    ranking_tasks=json.loads((OUT/'protocol.json').read_text())['ranking_tasks']
    for name in names:
        own=[r for r in summary if r['condition']==name and r['task'] in ranking_tasks]
        eligible=len(own)==len(ranking_tasks) and all(r['score'] is not None and r['n']>0 for r in own)
        objective=sum(r['score']/(2800 if r['task']=='MME' else 100) for r in own)/len(ranking_tasks) if eligible else None
        ranking.append(dict(condition=name,k=grid[name]['k'],objective=objective,eligible=eligible,ranking_tasks=ranking_tasks))
    ranking.sort(key=lambda r:(not r['eligible'],-(r['objective'] if r['objective'] is not None else -1),r['k'],r['condition']))
    table(OUT/stage/'accuracy_summary.csv',summary)
    table(OUT/stage/'per_question.csv',output)
    write(OUT/stage/'ranking.json',ranking)
    write(OUT/stage/'completion.json',dict(status='TERMINAL',records=len(output),planned=len(rows)*len(names),
        physical_svhr=False,cpu_rescored=sum(r['status']=='COMPLETE' for r in output)))
    from .report import preview
    preview(stage)
    from ..qwen3vl_overnight.archive import commit
    commit(f'SVH-R {stage} CPU-verified',OUT/stage)
    return ranking


def launch(stage):
    if all((OUT/stage/f'worker{s}_completion.json').exists() for s in [0,1]):
        return summarize(stage)
    jobs,logs=[],[]
    try:
        for shard,uuid in enumerate(GPUS):
            folder=OUT/stage
            folder.mkdir(parents=True,exist_ok=True)
            log=(folder/f'worker{shard}.log').open('a')
            logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-u','-m','vlm_research.qwen3vl_svhr_search.run','worker','--stage',stage,'--shard',str(shard)],
                cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=subprocess.STDOUT))
        while any(job.poll() is None for job in jobs):
            write(OUT/'supervisor_progress.json',dict(stage=stage,time=time.time(),workers=[dict(pid=j.pid,exitcode=j.poll()) for j in jobs]))
            if any(job.poll() not in (None,0) for job in jobs):
                write(OUT/'STOP_STRUCTURE.json',dict(stage=stage,error='WORKER_NONZERO_EXIT',codes=[j.poll() for j in jobs]))
            if (FIRST/'STOP_USER').exists() or (OUT/'STOP_STRUCTURE.json').exists():
                raise RuntimeError('Search stopped; do not continue other candidates')
            time.sleep(5)
        codes=[job.wait() for job in jobs]
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
        for job in jobs:
            job.wait()
        for log in logs:
            log.close()
        write(OUT/stage/'gpu_release.json',resource())
    assert not any(codes),('SEARCH_WORKER_FAILED',codes)
    return summarize(stage)


def launch_physical(scope):
    folder=OUT/('Physical' if scope=='Held-out' else 'Physical-Full')
    if all((folder/f'completion{s}.json').exists() for s in [0,1]):
        return json.loads((folder/'exit_codes.json').read_text())
    jobs,logs=[],[]
    try:
        folder.mkdir(parents=True,exist_ok=True)
        for shard,uuid in enumerate(GPUS):
            log=(folder/f'worker{shard}.log').open('a')
            logs.append(log)
            jobs.append(subprocess.Popen([sys.executable,'-u','-m','vlm_research.qwen3vl_svhr_search.run','physical',
                '--shard',str(shard),'--scope',scope],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=subprocess.STDOUT))
        while any(job.poll() is None for job in jobs):
            write(OUT/'supervisor_progress.json',dict(stage='Physical-'+scope,time=time.time(),workers=[dict(pid=j.pid,exitcode=j.poll()) for j in jobs]))
            if any(job.poll() not in (None,0) for job in jobs):
                write(folder/'STOP_STRUCTURE.json',dict(scope=scope,error='WORKER_NONZERO_EXIT',codes=[j.poll() for j in jobs]))
            if (FIRST/'STOP_USER').exists() or (folder/'STOP_STRUCTURE.json').exists():
                for job in jobs:
                    if job.poll() is None:
                        job.terminate()
                break
            time.sleep(5)
        codes=[job.wait() for job in jobs]
        if any(codes) and not (folder/'STOP_STRUCTURE.json').exists():
            write(folder/'STOP_STRUCTURE.json',dict(scope=scope,error='WORKER_NONZERO_EXIT',codes=codes))
        write(folder/'exit_codes.json',codes)
        return codes
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
        for job in jobs:
            job.wait()
        for log in logs:
            log.close()
        write(folder/'gpu_release.json',resource())


def pipeline():
    assert not torch.cuda.is_initialized()
    if not (OUT/'protocol.json').exists():
        freeze()
    discovery=launch('Discovery')
    best=[r['condition'] for r in discovery if r['eligible'] and r['condition']!='K0'][:3]
    assert best,'NO_VALID_SIX_TASK_RANKING; keep partial results'
    selection=dict(candidates=best,discovery_ranking_sha256=sha(OUT/'Discovery/ranking.json'),
        selection='Top three nonbaseline normalized six-task objectives; no new layer conditions')
    if (OUT/'validation_selection.json').exists():
        assert json.loads((OUT/'validation_selection.json').read_text())==selection
    else:
        write(OUT/'validation_selection.json',selection)
    validation=launch('Validation')
    eligible=[r for r in validation if r['eligible']]
    assert eligible
    quality=eligible[0]
    near=[r for r in eligible if r['objective']>=quality['objective']-.003]
    tradeoff=min(near,key=lambda r:(r['k'],-r['objective'],r['condition']))
    selection=dict(quality=quality['condition'],tradeoff=tradeoff['condition'],minimum_budget='K1',
        minimum_budget_rule='Predeclared one-layer comparator, regardless of discovery/validation rank',
        validation_ranking_sha256=sha(OUT/'Validation/ranking.json'),selection_frozen_before_heldout=True,
        physical_svhr=False,physical_implementation_status='FUNCTIONAL_ONLY_PENDING_INDEPENDENT_PORT')
    if (OUT/'final_selection.json').exists():
        assert json.loads((OUT/'final_selection.json').read_text())==selection
    else:
        write(OUT/'final_selection.json',selection)
    heldout=launch('Held-out')
    # Optional independent implementation; a failed port is not a functional PASS.
    codes=launch_physical('Held-out')
    selected=quality['condition']
    physical_folder=OUT/'Physical'
    validated=not any(codes) and not (physical_folder/'STOP_STRUCTURE.json').exists()
    for shard in [0,1]:
        p=physical_folder/f'eligible{shard}.json'
        validated=validated and p.exists() and json.loads(p.read_text()).get(selected)=='STRUCTURAL_PASS_RELAXED_QUALITY_ONLY'
    free=shutil.disk_usage('/data').free
    from ..qwen3vl_overnight.common import MIN_DISK
    # A separate full run is authorized only for the choice frozen before held-out.
    decision=dict(choice=selected,physical_validated=validated,free_bytes=free,
        required_free_bytes=MIN_DISK+2**30,chosen_before_heldout=True,
        status='RUN' if validated and selected!='K0' and free>=MIN_DISK+2**30 else 'NOT_STARTED_RESOURCE_OR_PHYSICAL_GATE')
    write(OUT/'full_physical_decision.json',decision)
    full_codes=launch_physical('Full') if decision['status']=='RUN' else None
    write(OUT/'completion.json',dict(status='FUNCTIONAL_SEARCH_COMPLETE',physical_svhr=False,heldout_ranking=heldout,
        no_layer_retuning_after_heldout=True,physical_port_exit_codes=codes,full_physical_exit_codes=full_codes,
        physical_status='SEE_INDEPENDENT_PHYSICAL_AUDIT_NOT_FUNCTIONAL_RESULTS'))
    from .report import generate as report
    report()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['pipeline','worker','physical'])
    parser.add_argument('--stage',choices=['Discovery','Validation','Held-out'])
    parser.add_argument('--shard',type=int,choices=[0,1])
    parser.add_argument('--scope',choices=['Held-out','Full'],default='Held-out')
    args=parser.parse_args()
    if args.mode=='pipeline':
        pipeline()
    elif args.mode=='physical':
        assert args.shard is not None
        from .physical_run import worker as physical_worker
        physical_worker(args.shard,args.scope)
    else:
        assert args.stage is not None and args.shard is not None
        worker(args.stage,args.shard)

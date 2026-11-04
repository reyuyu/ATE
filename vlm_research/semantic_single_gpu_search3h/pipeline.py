"""Predeclared staged search, no idle-model waiting for an agent turn."""
import argparse,ast,collections,fcntl,importlib.util,io,shutil,subprocess,traceback,unittest
from .common import *
from . import worker as backend
from .cpu_cache import enable
from relay_source_causality.common import hardware
from gqa3000_space_nl_visual_writers.failure_probe import idle_gpus
enable()
CID=['S-assistant-article','S-assistant-noun-article','S','S-assistant','S-article','S-noun-article','S-instruction4','S-question-extra4-role']
CONTENDERS=[c for c in CID if c not in ['S','S-assistant']]

def load(cid):
    d=read(P/'candidates'/cid/'definition.json');file=P/'candidates'/cid/'selection_source.py';assert sha(file)==d['source_sha256'];loader=importlib.util.spec_from_file_location('frozen_'+cid.replace('-','_'),file);m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m);return d,m

from .prepare_search import prepare

def available(q,w):
    sp=spec(q)
    return any(w==v for v in sp['historical_writers'].values()) or result_path(q,w).exists() or any((base/'checkpoints'/q/(canonical(w)+'.json')).exists() for base in [RECENT,ROUND2,OLDP])

def publish(job,cids,ids):
    if (P/'jobs'/job/'manifest.json').exists():
        existing=read(P/'jobs'/job/'requests.json')['requests'];assert {(r['question_id'],r['candidate_id']) for r in existing}=={(q,c) for q in ids for c in cids};return
    guard(gpu=True,reserve=600);tasks={};requests=[]
    for cid in cids:
        d,m=load(cid)
        for q in ids:
            sp=spec(q);w=m.select(sp,d['config']);requests.append(dict(question_id=q,candidate_id=cid,writers=w,selector_sha256=d['source_sha256'],config_sha256=digest(d['config'])))
            if not available(q,w):tasks[q+':'+canonical(w)]=dict(question_id=q,writers=w,worker=0)
    assert remaining(gpu=True)>len(tasks)*2+180,'Insufficient remaining time for this stage'
    D=P/'jobs'/job;D.mkdir(parents=True);atomic_json(D/'requests.json',dict(requests=requests));atomic_json(D/'manifest.json',dict(job_id=job,tasks=sorted(tasks.values(),key=lambda r:(sample(r['question_id'])['image_sha256'],r['question_id'],canonical(r['writers']))),logical_requests=len(requests),new_generations_requested=len(tasks),request_sha256=sha(D/'requests.json'),UTC=utc(),monotonic=time.monotonic(),source_hashes=source_hashes()));print('STAGE_FROZEN',job,len(requests),len(tasks),flush=True)

_indexes={}
def audit_reuse(r):
    source=r.get('source',{});path=Path(source.get('path',''))
    for old,commit,indexfile in [(RECENT,'f1d7d503f4529e1d9820821275c403a87a866290','artifact_index.csv'),(ROUND2,'1df71f42ffa318e38021e26b9ad71027e2e3cf88','artifact_index.csv'),(OLDP,'a1c9d81db89f59142a28295f2f1eeaa2b93e8fe6','artifact_sha_index.csv')]:
        if path.is_relative_to(old):
            if old not in _indexes:
                anchor=old.parents[2] if old==RECENT else old.parents[1];index=old/indexfile;blob=subprocess.check_output(['git','show',commit+':'+str(index.relative_to(anchor.parent))],cwd=ROOT.parent);assert blob==index.read_bytes();_indexes[old]={x['path']:x['sha256'] for x in csv.DictReader(index.open())}
            anchor=old.parents[2] if old==RECENT else old.parents[1];assert sha(path)==_indexes[old][str(path.relative_to(anchor))]

def summarize(job):
    if (P/'jobs'/job/'analysis_completion.json').exists():
        out=list(csv.DictReader((P/'jobs'/job/'summary.csv').open()))
        for r in out:
            for key in r:
                if key!='candidate_id':r[key]=float(r[key]) if key in ['ACC','mean_R_ratio'] else int(r[key])
        return out
    D=P/'jobs'/job;h=History();cache={};rows=[]
    for req in read(D/'requests.json')['requests']:
        q=req['question_id'];key=q+':'+canonical(req['writers'])
        if key not in cache:
            r=score_record(q,req['candidate_id'],req['writers'],h);audit_reuse(r);cache[key]=r
        r=cache[key];g=r['generation'];sp=spec(q);row=dict(question_id=q,task=sample(q)['task'],image_sha256=sample(q)['image_sha256'],question=sp['question'],gold=gold(q),candidate_id=req['candidate_id'],correct=g['correct'],answer=g['text'],token_ids=json.dumps(g['token_ids']),writers=json.dumps(req['writers']),k=len(req['writers']),R=len(sp['R']),R_ratio=len(req['writers'])/len(sp['R']),reused=r.get('reused',False),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'])
        for label,short in [('All-Writer','All'),('Question-Only','Q100'),('Semantic-first-Q80','Q80'),('Semantic-k','S')]:
            b=h.get(q,label)['generation'];row.update({short+'_correct':b['correct'],short+'_answer':b['text'],short+'_harm':int(b['correct'] and not g['correct']),short+'_fix':int(not b['correct'] and g['correct']),short+'_first_flip':int(g['token_ids'][0]!=b['token_ids'][0])})
        rows.append(row)
    h.close();writecsv(D/'results.csv',rows);atomic_json(D/'historical_audit_index.json',h.files);summary=[]
    for cid in sorted({r['candidate_id'] for r in rows}):
        rr=[r for r in rows if r['candidate_id']==cid];summary.append(dict(candidate_id=cid,n=len(rr),correct=sum(r['correct'] for r in rr),ACC=100*np.mean([r['correct'] for r in rr]),mean_R_ratio=np.mean([r['R_ratio'] for r in rr]),All_correct=sum(r['All_correct'] for r in rr),Q100_correct=sum(r['Q100_correct'] for r in rr),Q80_correct=sum(r['Q80_correct'] for r in rr),S_correct=sum(r['S_correct'] for r in rr)))
    writecsv(D/'summary.csv',summary);atomic_json(D/'analysis_completion.json',dict(passed=True,UTC=utc(),rows=len(rows)));print('STAGE_SUMMARY',job,json.dumps(summary),flush=True);return summary

def run():
    guard(reserve=1800);prepare() if not (P/'protocol.json').exists() else None;children=[];logs=[];start=time.monotonic();p=verify_sources();failed=None
    try:
        for rank,g in enumerate(p['gpus']):
            W=P/f'worker{rank}';W.mkdir();log=(W/'run.log').open('a');logs.append(log);env={**os.environ,'CUDA_VISIBLE_DEVICES':g['uuid'],'PYTHONUNBUFFERED':'1'};children.append(subprocess.Popen([os.sys.executable,'-m','semantic_single_gpu_search3h.worker','worker','--rank',str(rank)],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT))
        atomic_json(P/'driver_started.json',dict(pid=os.getpid(),children=[c.pid for c in children],UTC=utc(),monotonic=start))
        def check():
            guard(gpu=True,reserve=30)
            if any(c.poll() is not None for c in children):raise RuntimeError('A numerical worker exited unexpectedly')
        while not all((P/f'worker{i}/READY.json').exists() for i in range(1)):check();time.sleep(.5)
        atomic_json(P/'SMOKE_GATE.json',dict(passed=True,workers=1,UTC=utc()));print('BOTH_GPU_SMOKES_PASSED',flush=True)
        splits=read(P/'split_manifest.json');stages=[('s60',p['accepted_candidates'],splits['dev_stages']['60'])];summary=None
        for index in range(5):
            if index==1:
                eligible=[r for r in summary if r['mean_R_ratio']<.5];top=sorted(eligible,key=lambda r:(-r['correct'],r['mean_R_ratio'],r['candidate_id']))[:8];stages.append(('s180',[r['candidate_id'] for r in top],splits['dev_stages']['180']))
            if index==2:
                top=sorted(summary,key=lambda r:(-r['correct'],r['mean_R_ratio'],r['candidate_id']))[:4];stages.append(('s360',[r['candidate_id'] for r in top],splits['dev_stages']['360']))
            if index==3:
                top=sorted(summary,key=lambda r:(-r['correct'],r['mean_R_ratio'],r['candidate_id']))[:2];stages.append(('check600',[r['candidate_id'] for r in top],splits['ids']['CHECK']))
            if index==4:
                top=sorted(summary,key=lambda r:(-r['correct'],r['mean_R_ratio'],r['candidate_id']))[0];winner=top['candidate_id'];d,m=load(winner);allids=sorted(splits['metadata']);positions=[dict(question_id=q,writers=m.select(spec(q),d['config'])) for q in allids];budget=np.mean([len(r['writers'])/len(spec(r['question_id'])['R']) for r in positions]);assert budget<.5;atomic_json(P/'winner_freeze.json',dict(candidate_id=winner,source_sha256=d['source_sha256'],config=d['config'],selection_summary=summary,all3000_mean_R=float(budget),positions=positions,UTC=utc(),all_data_seen_exploratory=True));stages.append(('all3000',[winner],allids))
            job,cids,ids=stages[index];publish(job,cids,ids)
            while not all((P/'jobs'/job/f'worker{i}_done.json').exists() for i in range(1)):check();time.sleep(.5)
            summary=summarize(job)
        atomic_json(P/'PIPELINE_COMPLETE.json',dict(passed=True,UTC=utc(),winner=winner));atomic_json(P/'END_WORKERS.json',dict(reason='All predeclared stages finished; release immediately.',UTC=utc()))
        for c in children:c.wait(timeout=45)
        assert all(c.returncode==0 for c in children)
    except BaseException:
        failed=traceback.format_exc()
        if not (P/'STOP.json').exists():atomic_json(P/'STOP.json',dict(UTC=utc(),traceback=failed))
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=20)
            except subprocess.TimeoutExpired:c.kill();c.wait()
    finally:
        for log in logs:log.close()
        state=hardware();selected={g['uuid'] for g in p['gpus']};busy=[r for r in state['compute_processes'].splitlines() if r.split(',')[0].strip() in selected];atomic_json(P/'gpu_release.json',dict(**state,UTC=utc(),all_own_children_exited=all(c.poll() is not None for c in children),selected_devices_no_compute_processes=not busy,resident_seconds=time.monotonic()-start));print('MODELS_RELEASED',flush=True)
    if failed:raise RuntimeError(failed)

if __name__=='__main__':
    # Prepare must run before guard can read the new clock.
    if not (P/'protocol.json').exists():prepare()
    run()



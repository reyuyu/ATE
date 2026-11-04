"""Freeze each candidate and job before inference; analyze DEV/TUNE only until freeze."""
import argparse,collections,shutil,importlib.util
from .common import *
from .selection import select,flatten,intact,features
from .cpu_cache import enable as enable_cpu_cache
enable_cpu_cache()

def register(cid,config,hypothesis,parent=None):
    guard(reserve=1800);assert time.monotonic()<read(P/'clock.json')['design_stop_monotonic'],'Candidate design boundary reached'
    d=P/'candidates'/cid;d.mkdir(parents=True,exist_ok=True);payload=dict(candidate_id=cid,config=config,hypothesis=hypothesis,parent=parent,source_sha256=sha(Path(__file__).with_name('selection.py')),source_path='receiver_search_round2/selection.py',created_utc=utc(),created_monotonic=time.monotonic(),input_only=True)
    if (d/'definition.json').exists():
        old=read(d/'definition.json');assert old['config']==config;return old
    atomic_json(d/'definition.json',payload);shutil.copyfile(Path(__file__).with_name('selection.py'),d/'selection_source.py');event('CANDIDATE_FROZEN',candidate_id=cid,config_sha256=digest(config),hypothesis=hypothesis);return payload

def schedule(jobid,requests,kind='candidate',details=None):
    guard(gpu=True,reserve=900);assert (P/'SMOKE_GATE.json').exists(),'No formal jobs before all GPU smoke gates'
    assert shutil.disk_usage(P).free>2*1024**3 and shutil.disk_usage(PRIVATE).free>2*1024**3
    directory=P/'jobs'/jobid;assert not directory.exists();directory.mkdir(parents=True)
    p=read(P/'protocol.json');n=len(p['gpus']);tasks={};hist_count=0;saved_count=0;frozen=[]
    for row in requests:
        q=row['question_id'];authorized(q);sp=spec(q);w=row['writers'];assert w==sorted(set(w)) and set(w)<=set(sp['R']) and intact(w,sp['groups']);frozen.append(row)
        if not row.get('force_new',False) and any(w==v for v in sp['historical_writers'].values()):hist_count+=1;continue
        if result_path(q,w).exists():saved_count+=1;continue
        if old_result_path(q,w).exists():saved_count+=1;continue
        key=q+':'+canonical(w)
        if key not in tasks:tasks[key]=dict(question_id=q,writers=w,worker=int(digest(['worker-v1',sample(q)['image_sha256']]),16)%n)
    assert remaining(gpu=True)>len(tasks)*3/max(n,1)+180,'Insufficient time to finish and audit job'
    atomic_json(directory/'requests.json',dict(kind=kind,requests=frozen,details=details or {}))
    manifest=dict(job_id=jobid,kind=kind,tasks=sorted(tasks.values(),key=lambda r:(sample(r['question_id'])['image_sha256'],r['question_id'],canonical(r['writers']))),logical_requests=len(requests),new_generations_requested=len(tasks),historical_set_reuse=hist_count,existing_checkpoint_reuse=saved_count,request_sha256=sha(directory/'requests.json'),source_hashes=source_hashes(),utc=utc(),monotonic=time.monotonic(),remaining_gpu_seconds=remaining(gpu=True),free_bytes=shutil.disk_usage(P).free)
    atomic_json(directory/'manifest.json',manifest);event('JOB_FROZEN',job_id=jobid,job_kind=kind,requests=len(requests),new_generations=len(tasks));print(json.dumps({k:manifest[k] for k in ['job_id','kind','logical_requests','new_generations_requested','historical_set_reuse','existing_checkpoint_reuse']}),flush=True)

def candidate_job(jobid,cids,ids):
    requests=[]
    for cid in cids:
        d=read(P/'candidates'/cid/'definition.json');source=P/'candidates'/cid/'selection_source.py';assert d['source_sha256']==sha(source)
        loader=importlib.util.spec_from_file_location('frozen_selector_'+d['source_sha256'],source);module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
        for q in ids:
            split=authorized(q)
            if split=='FINAL':assert cids==[read(P/'winner_freeze.json')['candidate_id']],'Only the frozen single Winner may be evaluated on FINAL'
            sp=spec(q);w=module.select(sp,d['config']);requests.append(dict(question_id=q,candidate_id=cid,writers=w,config_sha256=digest(d['config']),selector_sha256=d['source_sha256'],force_new=split=='FINAL'))
    schedule(jobid,requests)

def loo_job(jobid):
    req=[];labels=[]
    for q in read(P/'split_manifest.json')['loo96']:
        sp=spec(q);groups=[g for g in sp['groups'] if g['question']];xx=features(groups,sp['question'])
        for i,g in enumerate(groups):
            w=[p for p in sp['Q'] if p not in g['positions']];cid=f'LOO-group-{i:02d}';req.append(dict(question_id=q,candidate_id=cid,writers=w,group_index=i,removed_group=g,features=xx[i].tolist()))
    schedule(jobid,req,'causal_LOO',dict(questions=96,selection='16 per task; deterministic input SHA; all complete Question groups; no outcome filtering'))

def summarize(jobid):
    start=time.monotonic();D=P/'jobs'/jobid;m=read(D/'manifest.json');assert sha(D/'requests.json')==m['request_sha256'];p=read(P/'protocol.json')
    assert all((D/f'worker{i}_done.json').exists() and read(D/f'worker{i}_done.json')['passed'] for i in range(len(p['gpus'])))
    request=read(D/'requests.json');history=History();rows=[];baselines={}
    for row in request['requests']:
        q=row['question_id'];sp=spec(q)
        if row.get('force_new',False):
            r=read(result_path(q,row['writers']));assert r['writers']==row['writers'] and r['input_hashes']==sp['input_hashes'];r['generation']=dict(r['generation'],correct=int(r['generation']['prediction']==gold(q)));r['reused']=False
        else:r=score_record(q,row['candidate_id'],row['writers'],history)
        from .reuse import validate_old_record
        validate_old_record(r)
        g=r['generation']
        if q not in baselines:baselines[q]={c:history.get(q,c) for c in ['All-Writer','Question-Only','Semantic-first-Q80','Semantic-k']}
        base=baselines[q];v=dict(question_id=q,split=authorized(q),task=sample(q)['task'],image_sha256=sample(q)['image_sha256'],question=sp['question'],gold=gold(q),candidate_id=row['candidate_id'],writers=json.dumps(row['writers']),k=len(row['writers']),R=len(sp['R']),R_ratio=len(row['writers'])/len(sp['R']),k80=sp['k80'],answer=g['text'],prediction=g['prediction'],correct=g['correct'],token_ids=json.dumps(g['token_ids']),reused=r.get('reused',False),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'])
        for c,short in [('All-Writer','All'),('Question-Only','Q100'),('Semantic-first-Q80','Q80'),('Semantic-k','S')]:
            b=base[c]['generation'];v.update({short+'_correct':b['correct'],short+'_answer':b['text'],short+'_harm':int(b['correct'] and not g['correct']),short+'_fix':int(not b['correct'] and g['correct']),short+'_first_flip':int(b['token_ids'][0]!=g['token_ids'][0])})
        if request['kind']=='causal_LOO':v.update(group_index=row['group_index'],removed_group=json.dumps(row['removed_group']),features=json.dumps(row['features']),causal_harm_label=v['Q100_correct']-v['correct'])
        rows.append(v)
    history.close();writecsv(D/'results.csv',rows);atomic_json(D/'historical_audit_index.json',history.files)
    summary=[]
    for cid in sorted({r['candidate_id'] for r in rows}):
        rr=[r for r in rows if r['candidate_id']==cid];summary.append(dict(candidate_id=cid,n=len(rr),correct=sum(r['correct'] for r in rr),accuracy=100*np.mean([r['correct'] for r in rr]),mean_R_ratio=np.mean([r['R_ratio'] for r in rr]),mean_k=np.mean([r['k'] for r in rr]),delta_All_pp=100*np.mean([r['correct']-r['All_correct'] for r in rr]),delta_Q100_pp=100*np.mean([r['correct']-r['Q100_correct'] for r in rr]),delta_Q80_pp=100*np.mean([r['correct']-r['Q80_correct'] for r in rr]),harm_Q80=sum(r['Q80_harm'] for r in rr),fix_Q80=sum(r['Q80_fix'] for r in rr)))
    writecsv(D/'summary.csv',summary);atomic_json(D/'analysis_completion.json',dict(passed=True,rows=len(rows),seconds=time.monotonic()-start,utc=utc(),new_generations=m['new_generations_requested']));event('JOB_ANALYZED',job_id=jobid,seconds=time.monotonic()-start,summary=summary if request['kind']!='causal_LOO' else dict(interventions=len(rows),harm=sum(r['causal_harm_label']==1 for r in rows),fix=sum(r['causal_harm_label']==-1 for r in rows)))
    print(json.dumps(summary if request['kind']!='causal_LOO' else dict(interventions=len(rows),harm=sum(r['causal_harm_label']==1 for r in rows),fix=sum(r['causal_harm_label']==-1 for r in rows)),ensure_ascii=False),flush=True)

def initial():
    configs={
      'role-balanced-q80':dict(family='role',fraction=.8,weights=dict(noun=3,attribute=2,spatial=2,logic=2,wh=1.5,core=.5,relational_context=.8,can_container=3,punctuation=-2,article=-.4,position=.1)),
      'relation-context-q80':dict(family='role',fraction=.8,weights=dict(noun=2,attribute=1.5,spatial=1.5,logic=2,wh=.5,relational_context=2,core=.5,article=.2,punctuation=-2,tail=.5)),
      'target-tail-q80':dict(family='role',fraction=.8,weights=dict(noun=2,attribute=2,spatial=1,logic=1,wh=2,core=.3,tail=1.5,position=.5,punctuation=-2)),
      'noun-content-q80':dict(family='role',fraction=.8,weights=dict(noun=3,attribute=2,core=1,pos_VERB=1.2,logic=1,spatial=.3,wh=.5,article=-1,punctuation=-2)),
    }
    for cid,config in configs.items():register(cid,config,'Question-only reverse pruning with exact historical per-question Q80 k; role/context/position priorities, no STOP changes.')
    candidate_job('r01-role-dev60',list(configs),read(P/'split_manifest.json')['dev_stages']['60'])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['initial','loo','summary','run']);a.add_argument('--job');a.add_argument('--candidates',nargs='*');a.add_argument('--split',default='DEV');a.add_argument('--n',type=int,default=360);args=a.parse_args()
    if args.stage=='initial':initial()
    elif args.stage=='loo':loo_job(args.job or 'r02-loo-dev96')
    elif args.stage=='summary':summarize(args.job)
    else:
        splits=read(P/'split_manifest.json');ids=splits['dev_stages'][str(args.n)] if args.split=='DEV' else splits['ids'][args.split];candidate_job(args.job,args.candidates,ids)

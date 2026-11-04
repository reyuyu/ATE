"""Final round: six receiver options, strict image-fold OOF, one frozen input policy."""
from . import common as c
c.P=c.PHASE1/'phase4_policy';c.PRIVATE=c.PRIVATE.parent/'semantic_policy_phase4'
from .common import *
from .cpu_cache import enable
enable()
import shutil,traceback,collections
from .adaptive import feat,typ,fit,predict,route,calibrate,FEATURES
from .pipeline import audit_reuse
from .search_selection import select
from relay_source_causality.common import hardware
from gqa3000_space_nl_visual_writers.failure_probe import idle_gpus
from gqa5h_receiver_search.analyze import bootstrap
from gqa5h_receiver_search.expanded_statistics import cluster_p
PHASE2=PHASE1/'phase2';COMMIT2='557e114e54c5311d0d6a595667888c9461651463'
OPTIONS=['Q','SA','All','Q-period','Q-phrase','Q-SA']

def writers(q):
    sp=spec(q);Q=sp['Q'];SA=sorted(set(sp['historical_writers']['Semantic-k'])|set(sp['groups'][-1]['positions']));return [Q,SA,sp['R'],select(sp,dict(anchors=['period'])),select(sp,dict(anchors=['phrase'])),sorted(set(Q)|set(SA))]

def record(q,w,h):
    prior=PHASE2/'checkpoints'/q/(canonical(w)+'.json')
    if prior.exists():
        data=read(prior);blob=subprocess.check_output(['git','show',COMMIT2+':'+str(prior.relative_to(ROOT.parent))],cwd=ROOT.parent);assert hashlib.sha256(blob).hexdigest()==sha(prior) and data['writers']==w and data['input_hashes']==spec(q)['input_hashes'] and sha(data['raw']['path'])==data['raw']['sha256'];return data
    r=score_record(q,'policy-option',w,h);audit_reuse(r);return r

def possible(q,w):
    return w in list(spec(q)['historical_writers'].values()) or any((base/'checkpoints'/q/(canonical(w)+'.json')).exists() for base in [P,PHASE2,PHASE1,RECENT,ROUND2,OLDP])

def job(name,requests,child):
    guard(gpu=True,reserve=600);tasks={}
    for q,w in requests:
        if not possible(q,w):tasks[q+canonical(w)]=dict(question_id=q,writers=w,worker=0)
    assert remaining(gpu=True)>len(tasks)*1.5+300
    D=P/'jobs'/name;D.mkdir(parents=True);atomic_json(D/'manifest.json',dict(job_id=name,tasks=list(tasks.values()),logical_requests=len(requests),new_generations_requested=len(tasks),UTC=utc()));print('JOB_FROZEN',name,len(requests),len(tasks),flush=True)
    while not (D/'worker0_done.json').exists():
        guard(gpu=True,reserve=30);assert child.poll() is None,'Owned worker exited';time.sleep(.5)

def main():
    P.mkdir();PRIVATE.mkdir();shutil.copyfile(PHASE1/'clock.json',P/'clock.json');shutil.copyfile(PHASE1/'split_manifest.json',P/'split_manifest.json');guard(reserve=2400);prior=read(PHASE1/'protocol.json');state=hardware();assert prior['gpus'][0] in idle_gpus(state)
    for path,value in prior['shared_sources'].items():assert sha(path)==value
    atomic_json(P/'protocol.json',dict(prior,source_hashes=source_hashes(),split_sha256=sha(P/'split_manifest.json'),clock_sha256=sha(P/'clock.json'),options=OPTIONS,final_mean_R_limit=.5,diagnostic_option_budgets_may_exceed_half=True,training='DEV360 plus CHECK600; outcome fitting only on these 960 IDs',selection='strict image-fold OOF on training IDs; then one frozen policy; no REST outcome selection',policy_source_sha256=sha(__file__),worker_bootstrap_sha256=sha(Path(__file__).with_name('phase4_worker.py')),no_7B_training=True))
    splits=read(P/'split_manifest.json');ids=sorted(splits['metadata']);train=splits['dev_stages']['360']+splits['ids']['CHECK'];assert len(set(train))==960;remaining_ids=sorted(set(ids)-set(train));assert len(remaining_ids)==2040;train_images={sample(q)['image_sha256'] for q in train};clean=[q for q in remaining_ids if sample(q)['image_sha256'] not in train_images]
    atomic_json(P/'training_manifest.json',dict(training=train,remaining=remaining_ids,image_disjoint_remaining=clean,remaining_image_overlap_excluded=len(remaining_ids)-len(clean),all_data_previously_seen=True))
    ww={q:writers(q) for q in ids};cost=np.array([[len(w)/len(spec(q)['R']) for w in ww[q]] for q in ids]);X=np.array([feat(q) for q in ids]);types=[typ(q) for q in ids];ix={q:i for i,q in enumerate(ids)};ti=np.array([ix[q] for q in train]);xd=X[ti];td=[types[i] for i in ti];fold=np.array([int(digest(['phase4-fold',sample(q)['image_sha256']]),16)%5 for q in train]);atomic_json(P/'input_options_manifest.json',dict(options=OPTIONS,positions=ww,costs=cost.tolist(),features=FEATURES,folds={q:int(k) for q,k in zip(train,fold)}))
    W=P/'worker0';W.mkdir();log=(W/'run.log').open('w');env={**os.environ,'CUDA_VISIBLE_DEVICES':prior['gpus'][0]['uuid'],'PYTHONUNBUFFERED':'1'};child=subprocess.Popen([os.sys.executable,'-m','semantic_qrelay_search3h.phase4_worker'],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT);atomic_json(P/'driver_started.json',dict(pid=os.getpid(),children=[child.pid],UTC=utc()));failure=None;h=History();cache={}
    def get(q,w):
        key=q+canonical(w)
        if key not in cache:cache[key]=record(q,w,h)
        return cache[key]
    try:
        while not (W/'READY.json').exists():guard(gpu=True);assert child.poll() is None;time.sleep(.5)
        atomic_json(P/'SMOKE_GATE.json',dict(passed=True,workers=1,UTC=utc()));job('labels960',[(q,w) for q in train for w in ww[q]],child)
        yy=np.array([[int(get(q,w)['generation']['prediction']==gold(q)) for w in ww[q]] for q in train]);Y=yy[:,1:]-yy[:,:1];models=[];results=[]
        for kind,alpha in [('ridge',v) for v in [1,10,100,1000]]+[('type',v) for v in [10,50,200]]:
            folds=[]
            for k in range(5):
                tr=fold!=k;model=fit(xd[tr],Y[tr],kind,alpha,[t for t,z in zip(td,tr) if z]);scores=predict(X,types,model);folds.append(scores)
            full=fit(xd,Y,kind,alpha,td);allscore=predict(X,types,full)
            for cap in [.42,.46,.49]:
                choices=np.zeros(960,dtype=int)
                for k,scores in enumerate(folds):
                    lam=calibrate(scores,cost,cap);mask=fold==k;choices[mask]=route(scores[ti[mask]],cost[ti[mask]],lam)
                lam=calibrate(allscore,cost,cap);allchoices=route(allscore,cost,lam);budget=float(cost[np.arange(3000),allchoices].mean());cid=f'{kind}-{alpha}-cap{int(cap*100)}';val=yy[np.arange(960),choices];results.append(dict(candidate_id=cid,OOF_correct=int(val.sum()),OOF_ACC=100*val.mean(),OOF_R=float(cost[ti,choices].mean()),all3000_input_only_R=budget));models.append(dict(candidate_id=cid,model=full,cap=cap,lambda_cost=lam,choices=allchoices.tolist(),budget=budget))
        for name,ch in [('Question-Only',0),('S-assistant',1)]:results.append(dict(candidate_id=name,OOF_correct=int(yy[:,ch].sum()),OOF_ACC=100*yy[:,ch].mean(),OOF_R=float(cost[ti,ch].mean()),all3000_input_only_R=float(cost[:,ch].mean())));models.append(dict(candidate_id=name,model=None,choices=[ch]*3000,budget=float(cost[:,ch].mean())))
        writecsv(P/'OOF960.csv',results);eligible=[r for r in results if r['all3000_input_only_R']<.5];best=sorted(eligible,key=lambda r:(-r['OOF_correct'],r['OOF_R'],r['candidate_id']))[0];winner=next(m for m in models if m['candidate_id']==best['candidate_id']);atomic_json(P/'winner_freeze.json',dict(UTC=utc(),winner=winner,selection_results=results,models=models,train_ids=train,remaining_ids=remaining_ids,all_ids=ids,remaining_outcomes_not_used_for_selection=True,fold_budget_calibration_uses_fold_model_only=True));print('WINNER_FROZEN',best,flush=True)
        job('final3000',[(q,ww[q][winner['choices'][ix[q]]]) for q in ids],child);atomic_json(P/'END_WORKERS.json',dict(UTC=utc(),reason='All GPU requests completed'));child.wait(timeout=45);assert child.returncode==0
    except BaseException:
        failure=traceback.format_exc();atomic_json(P/'STOP.json',dict(UTC=utc(),traceback=failure))
        if child.poll() is None:child.terminate();child.wait(timeout=30)
    finally:
        log.close();atomic_json(P/'gpu_release.json',dict(UTC=utc(),all_own_children_exited=child.poll() is not None,resource=hardware()))
    if failure:raise RuntimeError(failure)
    # Remaining outcomes are read only after one policy has been frozen.
    basepath=PHASE1/'per_question.csv';assert subprocess.check_output(['git','show','03832c68f7bb09adea0af55b82014344806131b4:'+str(basepath.relative_to(ROOT.parent))],cwd=ROOT.parent)==basepath.read_bytes();base={r['question_id']:r for r in csv.DictReader(basepath.open())};rows=[]
    for q in ids:
        k=winner['choices'][ix[q]];w=ww[q][k];r=get(q,w);g=r['generation'];b=base[q];assert g['prediction']==gqa_prediction(g['text']) and set(spec(q)['historical_writers']['Semantic-k'])<=set(w)<=set(spec(q)['R']);rows.append(dict(question_id=q,image_sha256=sample(q)['image_sha256'],task=sample(q)['task'],question=spec(q)['question'],gold=gold(q),split='TRAIN960' if q in set(train) else 'REST2040',image_disjoint_remaining=q in set(clean),branch=OPTIONS[k],correct=int(g['prediction']==gold(q)),answer=g['text'],token_ids=json.dumps(g['token_ids']),writers=json.dumps(w),R_ratio=len(w)/len(spec(q)['R']),All_correct=int(b['All_correct']),Q100_correct=int(b['Q100_correct']),S_correct=int(b['S_correct']),Q80_correct=int(b['Q80_correct']),P1_correct=int(b['correct'])))
    writecsv(P/'per_question.csv',rows);h.close();atomic_json(P/'historical_audit_index.json',h.files);paired=[];cohorts={'ALL3000':rows,'TRAIN960':[r for r in rows if r['split']=='TRAIN960'],'REST2040':[r for r in rows if r['split']=='REST2040'],'REST_image_disjoint':[r for r in rows if r['image_disjoint_remaining']]};summary=[]
    for cohort,rr in cohorts.items():
        summary.append(dict(cohort=cohort,n=len(rr),images=len({r['image_sha256'] for r in rr}),correct=sum(r['correct'] for r in rr),ACC=100*np.mean([r['correct'] for r in rr]),mean_R_ratio=np.mean([r['R_ratio'] for r in rr]),branch_counts=json.dumps(dict(collections.Counter(r['branch'] for r in rr)))))
        for label,short in [('All-Writer','All'),('Question-Only','Q100'),('Semantic-k','S'),('Semantic-Q80','Q80'),('Phase1-whatwhich','P1')]:
            d=np.array([r['correct']-r[short+'_correct'] for r in rr]);lo,hi=bootstrap(d,[r['image_sha256'] for r in rr]);paired.append(dict(cohort=cohort,baseline=label,n=len(rr),delta_pp=100*d.mean(),CI95_low_pp=lo,CI95_high_pp=hi,harm=int(sum(d==-1)),fix=int(sum(d==1)),cluster_signflip_p=cluster_p(rr,short),Holm_p=1.))
    for cohort in cohorts:
        prev=0.
        for j,r in enumerate(sorted([r for r in paired if r['cohort']==cohort],key=lambda x:x['cluster_signflip_p'])):prev=max(prev,min(1.,(5-j)*r['cluster_signflip_p']));r['Holm_p']=prev
    writecsv(P/'accuracy_summary.csv',summary);writecsv(P/'paired_comparisons.csv',paired);fields=list(rows[0]);writecsv(P/'harm_cases.csv',[r for r in rows if any(r[s+'_correct'] and not r['correct'] for s in ['All','Q100','S','P1'])],fields);writecsv(P/'recovered_cases.csv',[r for r in rows if any(not r[s+'_correct'] and r['correct'] for s in ['All','Q100','S','P1'])],fields)
    cp=[]
    for f in sorted((P/'checkpoints').glob('*/*.json')):
        r=read(f);g=r['generation'];assert r['input_hashes']==spec(r['question_id'])['input_hashes'] and sha(r['raw']['path'])==r['raw']['sha256'] and r['weights_unchanged'] and r['hook_cleanup'] and r['MLP_calls']==[g['length']]*32 and g['audit']['passed'] and g['own_cache'] and g['own_prefix']
        with np.load(r['raw']['path']) as z:assert (z['audit'][:,0]==0).all() and (z['audit'][:,4:]==1).all() and z['token_ids'].tolist()==g['token_ids']
        cp.append(dict(path=str(f.relative_to(ROOT)),sha256=sha(f),raw_path=r['raw']['path'],raw_sha256=r['raw']['sha256']))
    writecsv(P/'checkpoint_index.csv',cp);atomic_json(P/'validation.json',dict(passed=True,UTC=utc(),strict_fold_model_budget_calibration=True,no_remaining_outcomes_used_for_selection=True,new_checkpoints_hashed=True,mask_zero_nontarget_exact=True,all32_MLP_and_cache_verified=True,S_preserved=True,bootstrap=20000,seed=20261104,source_sha256=sha(__file__),smoke=read(W/'smoke.json')));atomic_json(P/'completion.json',dict(status='COMPLETE',UTC=utc(),winner=winner['candidate_id'],summary=summary,new_generations=len(cp),OOF_best=best,elapsed_minutes=(time.monotonic()-read(P/'clock.json')['start_monotonic'])/60));print(json.dumps(read(P/'completion.json')),flush=True)

if __name__=='__main__':main()

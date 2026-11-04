"""No fitting: predefined text predicates and fixed receiver sets, staged ACC experiments."""
from . import common as c
c.P=c.PHASE1/'phase9_cached_fallbacks';c.PRIVATE=c.PRIVATE.parent/'semantic_cached_fallbacks'
from .common import *
from .cpu_cache import enable
enable()
import re,shutil,traceback,collections
from .pipeline import audit_reuse
from .search_selection import select
from relay_source_causality.common import hardware
from gqa3000_space_nl_visual_writers.failure_probe import idle_gpus
from gqa5h_receiver_search.analyze import bootstrap
from gqa5h_receiver_search.expanded_statistics import cluster_p
PRIORCOMMIT='d3ca5225017ff5230e07d284b15b635d04cfa3b3';PRIOR=[PHASE1/'phase8_fixed_recheck',PHASE1/'phase7_fixed_refinements',PHASE1/'phase6_fixed_compositions',PHASE1/'phase5_fixed_rules',PHASE1/'phase4_policy',PHASE1/'phase2']

def old_options(q):
    sp=spec(q);Q=sp['Q'];SA=sorted(set(sp['historical_writers']['Semantic-k'])|set(sp['groups'][-1]['positions']));return dict(Q=Q,SA=SA,All=sp['R'],Qperiod=select(sp,dict(anchors=['period'])),Qphrase=select(sp,dict(anchors=['phrase'])),QSA=sorted(set(Q)|set(SA)))

def options(q):
    o=old_options(q)
    o['BASE15']=o['QSA'] if condition(q,'whatwhich|long15') else o['Q']
    o['BASE18']=o['QSA'] if condition(q,'whatwhich|long18') else o['Q']
    return o

def condition(q,rule):
    if '|' in rule:return any(condition(q,r) for r in rule.split('|'))
    words=re.findall('[a-z]+',spec(q)['question'].lower());s=set(words);n=len(spec(q)['Q'])
    if rule=='always':return True
    if rule=='color':return bool(s&{'color','colour'})
    if rule=='compare':return bool(s&set('same different more less fewer larger smaller bigger taller shorter than'.split()))
    if rule=='spatial':return bool(s&set('left right behind above below under beside front'.split()))
    if rule=='count':return 'many' in s or 'number' in s
    if rule=='whatwhich':return words[0] in ['what','which']
    if rule=='yesno':return words[0] in ['is','are','do','does']
    if rule=='where':return words[0]=='where'
    if rule=='negation':return bool(s&{'not','no','neither','nor'})
    if rule.startswith('long'):return n>=int(rule[4:])
    if rule.startswith('short'):return n<=int(rule[5:])
    raise ValueError(rule)

def record(q,w,h):
    for base in PRIOR:
        path=base/'checkpoints'/q/(canonical(w)+'.json')
        if path.exists():
            r=read(path);blob=subprocess.check_output(['git','show',PRIORCOMMIT+':'+str(path.relative_to(ROOT.parent))],cwd=ROOT.parent);assert hashlib.sha256(blob).hexdigest()==sha(path) and sha(r['raw']['path'])==r['raw']['sha256'] and r['writers']==w and r['input_hashes']==spec(q)['input_hashes'];return r
    r=score_record(q,'fixed-rule',w,h);audit_reuse(r);return r

def available(q,w):
    return w in list(spec(q)['historical_writers'].values()) or any((base/'checkpoints'/q/(canonical(w)+'.json')).exists() for base in [P,*PRIOR,PHASE1,RECENT,ROUND2,OLDP])

def main():
    P.mkdir();PRIVATE.mkdir();shutil.copyfile(PHASE1/'clock.json',P/'clock.json');shutil.copyfile(PHASE1/'split_manifest.json',P/'split_manifest.json');guard(reserve=1800);prior=read(PHASE1/'protocol.json');splits=read(P/'split_manifest.json');ids=sorted(splits['metadata']);configs={}
    # The complete list is fixed before reading any performance in this phase.
    # Finite static all-writer fallback predicates. Every answer is already available
    # for this exact writer set; the phase forbids new GPU generation.
    single=['color','count','negation','long20','long25','long30','long35','long40']
    from itertools import combinations
    rules=single+['|'.join(z) for z in combinations(single,2)]
    for base in ['BASE15','BASE18']:
        configs[base]=dict(base=base,rule='always',hit=base)
        for rule in rules:configs[base+'-All-if-'+rule]=dict(base=base,rule=rule,hit='All')
    oo={q:options(q) for q in ids};positions={};budget=[]
    for cid,conf in configs.items():
        masks={q:oo[q][conf['hit'] if condition(q,conf['rule']) else conf['base']] for q in ids};mean=float(np.mean([len(masks[q])/len(spec(q)['R']) for q in ids]));okay=mean<.5;budget.append(dict(candidate_id=cid,mean_R_ratio=mean,eligible=okay,config=json.dumps(conf,sort_keys=True)))
        if okay:positions[cid]=masks
    writecsv(P/'CPU_budget.csv',budget);atomic_json(P/'rule_freeze.json',dict(UTC=utc(),configs=configs,eligible=list(positions),positions=positions,no_training=True,no_fitted_weights=True,no_Gold_in_rules=True,source_sha256=sha(__file__)));atomic_json(P/'protocol.json',dict(prior,source_hashes=source_hashes(),split_sha256=sha(P/'split_manifest.json'),clock_sha256=sha(P/'clock.json'),no_training=True,no_fitting=True,user_no_training_steering=True,plan=['all eligible DEV360','top5 CHECK600','one frozen ALL3000'],zero_new_GPU_generation_required=True,rule_source_sha256=sha(__file__)))
    h=History();cache={};child=None;log=None
    def get(q,w):
        key=q+canonical(w)
        if key not in cache:cache[key]=record(q,w,h)
        return cache[key]
    def ensure_worker():
        nonlocal child,log
        if child is not None:return
        state=hardware();assert prior['gpus'][0] in idle_gpus(state);W=P/'worker0';W.mkdir();log=(W/'run.log').open('w');env={**os.environ,'CUDA_VISIBLE_DEVICES':prior['gpus'][0]['uuid'],'PYTHONUNBUFFERED':'1'};child=subprocess.Popen([os.sys.executable,'-m','semantic_qrelay_search3h.rules_worker'],cwd=ROOT.parent,env=env,stdout=log,stderr=subprocess.STDOUT);atomic_json(P/'driver_started.json',dict(pid=os.getpid(),children=[child.pid],UTC=utc()))
        while not (W/'READY.json').exists():guard(gpu=True);assert child.poll() is None;time.sleep(.5)
        atomic_json(P/'SMOKE_GATE.json',dict(passed=True,workers=1,UTC=utc()))
    def stage(name,cids,questions):
        guard(gpu=True,reserve=600);D=P/'jobs'/name;D.mkdir(parents=True);requests=[dict(question_id=q,candidate_id=cid,writers=positions[cid][q]) for cid in cids for q in questions];atomic_json(D/'requests.json',dict(requests=requests,UTC=utc()));tasks={}
        for r in requests:
            q=r['question_id'];w=r['writers']
            if not available(q,w):tasks[q+canonical(w)]=dict(question_id=q,writers=w,worker=0)
        assert not tasks, 'CPU-only exact-set reuse phase: unknown masks require stopping, never launching a GPU'
        assert remaining(gpu=True)>300
        if tasks:
            ensure_worker();atomic_json(D/'manifest.json',dict(job_id=name,tasks=list(tasks.values()),logical_requests=len(requests),new_generations_requested=len(tasks),UTC=utc()));print('GENERATING',name,len(tasks),flush=True)
            while not (D/'worker0_done.json').exists():guard(gpu=True);assert child.poll() is None;time.sleep(.5)
        else:atomic_json(D/'CPU_only_reuse.json',dict(all_exact_sets_audited=True,UTC=utc()))
        result=[];summary=[]
        for cid in cids:
            vals=[]
            for q in questions:
                w=positions[cid][q];r=get(q,w);g=r['generation'];correct=int(g['prediction']==gold(q));vals.append(correct);result.append(dict(question_id=q,candidate_id=cid,task=sample(q)['task'],image_sha256=sample(q)['image_sha256'],question=spec(q)['question'],gold=gold(q),answer=g['text'],correct=correct,token_ids=json.dumps(g['token_ids']),writers=json.dumps(w),R_ratio=len(w)/len(spec(q)['R'])))
            summary.append(dict(candidate_id=cid,n=len(questions),correct=sum(vals),ACC=100*np.mean(vals),mean_R_ratio=np.mean([len(positions[cid][q])/len(spec(q)['R']) for q in questions])))
        writecsv(D/'per_question.csv',result);writecsv(D/'summary.csv',summary);print('STAGE',name,json.dumps(summary),flush=True);return summary
    failed=None
    try:
        summary=stage('DEV360',list(positions),splits['dev_stages']['360'])
        top=sorted(summary,key=lambda r:(-r['correct'],r['mean_R_ratio'],r['candidate_id']))[:5]
        summary=stage('CHECK600',[r['candidate_id'] for r in top],splits['ids']['CHECK'])
        winner=sorted(summary,key=lambda r:(-r['correct'],r['mean_R_ratio'],r['candidate_id']))[0]['candidate_id']
        atomic_json(P/'winner_freeze.json',dict(UTC=utc(),candidate_id=winner,config=configs[winner],positions=positions[winner],selection=summary,no_training=True))
        stage('ALL3000',[winner],ids)
    except BaseException:failed=traceback.format_exc();atomic_json(P/'STOP.json',dict(UTC=utc(),traceback=failed))
    finally:
        if child is not None:
            if failed and child.poll() is None:child.terminate()
            elif child.poll() is None:atomic_json(P/'END_WORKERS.json',dict(UTC=utc(),reason='Finished fixed-rule GPU requests'))
            child.wait(timeout=45);log.close()
        atomic_json(P/'gpu_release.json',dict(UTC=utc(),all_own_children_exited=child is None or child.poll() is not None,resources=hardware()))
    if failed:raise RuntimeError(failed)
    rows=list(csv.DictReader((P/'jobs/ALL3000/per_question.csv').open()));base={r['question_id']:r for r in csv.DictReader((PHASE1/'per_question.csv').open())};paired=[]
    for r in rows:
        r['correct']=int(r['correct']);q=r['question_id']
        for short,key in [('All','All_correct'),('Q100','Q100_correct'),('S','S_correct'),('Q80','Q80_correct'),('P1','correct')]:r[short+'_correct']=int(base[q][key])
    for label,short in [('All-Writer','All'),('Question-Only','Q100'),('Semantic-k','S'),('Semantic-Q80','Q80'),('Phase1-whatwhich','P1')]:
        d=np.array([r['correct']-r[short+'_correct'] for r in rows]);lo,hi=bootstrap(d,[r['image_sha256'] for r in rows]);paired.append(dict(baseline=label,delta_pp=100*d.mean(),CI95_low_pp=lo,CI95_high_pp=hi,harm=int(sum(d==-1)),fix=int(sum(d==1)),cluster_signflip_p=cluster_p(rows,short),Holm_p=1.))
    prev=0.
    for j,r in enumerate(sorted(paired,key=lambda r:r['cluster_signflip_p'])):prev=max(prev,min(1.,(5-j)*r['cluster_signflip_p']));r['Holm_p']=prev
    writecsv(P/'per_question.csv',rows);writecsv(P/'paired_comparisons.csv',paired);fields=list(rows[0]);writecsv(P/'harm_cases.csv',[r for r in rows if any(r[s+'_correct'] and not r['correct'] for s in ['All','Q100','S','P1'])],fields);writecsv(P/'recovered_cases.csv',[r for r in rows if any(not r[s+'_correct'] and r['correct'] for s in ['All','Q100','S','P1'])],fields);task=[]
    for t in TASKS:
        rr=[r for r in rows if r['task']==t];task.append(dict(task=t,n=len(rr),correct=sum(r['correct'] for r in rr),ACC=100*np.mean([r['correct'] for r in rr])))
    writecsv(P/'task_accuracy.csv',task);h.close();atomic_json(P/'historical_audit_index.json',h.files);cp=[]
    for f in (P/'checkpoints').glob('*/*.json'):
        r=read(f);g=r['generation'];assert sha(r['raw']['path'])==r['raw']['sha256'] and r['input_hashes']==spec(r['question_id'])['input_hashes'] and r['weights_unchanged'] and r['hook_cleanup'] and r['MLP_calls']==[g['length']]*32 and g['own_cache'] and g['own_prefix']
        with np.load(r['raw']['path']) as z:assert (z['audit'][:,0]==0).all() and (z['audit'][:,4:]==1).all() and z['token_ids'].tolist()==g['token_ids']
        cp.append(dict(path=str(f.relative_to(ROOT)),sha256=sha(f),raw_path=r['raw']['path'],raw_sha256=r['raw']['sha256']))
    writecsv(P/'checkpoint_index.csv',cp) if cp else None
    atomic_json(P/'validation.json',dict(passed=True,UTC=utc(),no_training=True,no_fitted_parameters=True,S_preserved=True,all3000_ACC_recomputed=True,all_new_checkpoints_hashed=True,mask_zero_nontarget_unchanged=True,bootstrap=20000,source_sha256=sha(__file__)));atomic_json(P/'completion.json',dict(status='COMPLETE',UTC=utc(),candidate_id=winner,config=configs[winner],correct=sum(r['correct'] for r in rows),ACC=100*np.mean([r['correct'] for r in rows]),mean_R_ratio=np.mean([float(r['R_ratio']) for r in rows]),new_generations=len(cp),no_training=True,elapsed_minutes=(time.monotonic()-read(P/'clock.json')['start_monotonic'])/60));print(json.dumps(read(P/'completion.json')),flush=True)

if __name__=='__main__':main()

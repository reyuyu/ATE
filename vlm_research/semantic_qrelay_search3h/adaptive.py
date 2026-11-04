"""Image-group OOF CPU uplift models: choose Q, S+ASSISTANT or All-R by input only."""
import re,collections
from .common import *
from .pipeline import load,audit_reuse
from .cpu_cache import enable
from gqa5h_receiver_search.analyze import bootstrap
from gqa5h_receiver_search.expanded_statistics import cluster_p
enable()
D=PHASE1/'phase3_adaptive';D.mkdir(exist_ok=True)
FEATURES=['bias','Qlen','Qlen2','S_R','Q_R','noun','adj','adp','det','pron','aux','neg','compare','spatial','color','count','what','which','where','how','is','are','does','do','otherfirst']
def feat(q):
    s=spec(q);gs=[g for g in s['groups'] if g['question']];words=[w.lower() for g in gs for w in g['words']];ws=set(words);pos=collections.Counter(p for g in gs for p in g.get('pos',[]));n=max(1,len(gs));first=words[0];l=len(s['Q'])/30
    return [1,l,l*l,len(s['historical_writers']['Semantic-k'])/len(s['R']),len(s['Q'])/len(s['R'])]+[pos[p]/n for p in ['NOUN','ADJ','ADP','DET','PRON','AUX']]+[float(bool(ws&z)) for z in [set('not no neither nor'.split()),set('same different more less fewer larger smaller bigger taller shorter both than'.split()),set('left right above below behind front near next under beside'.split()),{'color','colour'},{'many','number','count'}]]+[float(first==w) for w in ['what','which','where','how','is','are','does','do']]+[float(first not in ['what','which','where','how','is','are','does','do'])]
def typ(q):
    w=re.findall('[a-z]+',spec(q)['question'].lower());return 'color' if 'color' in w else 'count' if 'many' in w else w[0] if w[0] in ['what','which','where','how','is','are','does','do'] else 'other'
def fit(X,Y,kind,alpha,types):
    if kind=='ridge':
        mu=X[:,1:].mean(0);sd=np.maximum(X[:,1:].std(0),.05);z=np.column_stack([np.ones(len(X)),(X[:,1:]-mu)/sd]);pen=np.eye(z.shape[1])*alpha;pen[0,0]=1e-8;b=np.linalg.solve(z.T@z+pen,z.T@Y);return dict(kind=kind,mu=mu.tolist(),sd=sd.tolist(),coef=b.tolist())
    prior=Y.mean(0);return dict(kind='type',prior=prior.tolist(),table={t:((Y[np.array(types)==t].sum(0)+alpha*prior)/(sum(x==t for x in types)+alpha)).tolist() for t in sorted(set(types))})
def predict(X,types,model):
    if model['kind']=='ridge':z=np.column_stack([np.ones(len(X)),(X[:,1:]-np.array(model['mu']))/np.array(model['sd'])]);return z@np.array(model['coef'])
    return np.array([model['table'].get(t,model['prior']) for t in types])
def route(scores,cost,lam):
    # Column order Q, SA, All; tiny cost tie-break only, never outcomes.
    gains=np.column_stack([np.zeros(len(scores)),scores]);return np.argmax(gains-(lam+1e-10)*cost,axis=1)
def calibrate(scores,cost,cap):
    lo=0.;hi=100.
    for _ in range(50):
        mid=(lo+hi)/2;ix=route(scores,cost,mid);budget=cost[np.arange(len(ix)),ix].mean()
        if budget>=cap:lo=mid
        else:hi=mid
    return hi

def main():
    assert remaining()>600 and not (D/'completion.json').exists();splits=read(P/'split_manifest.json');ids=sorted(splits['metadata']);dev=splits['dev_stages']['360'];check=splits['ids']['CHECK'];X=np.array([feat(q) for q in ids]);types=[typ(q) for q in ids];lookup={q:i for i,q in enumerate(ids)};xd=X[[lookup[q] for q in dev]];td=[typ(q) for q in dev];fold=np.array([int(digest(['adaptive-image-fold-v1',sample(q)['image_sha256']]),16)%5 for q in dev]);cost=np.array([[len(spec(q)['Q'])/len(spec(q)['R']),(len(spec(q)['historical_writers']['Semantic-k'])+4)/len(spec(q)['R']),1.] for q in ids]);cd=cost[[lookup[q] for q in dev]]
    h=History();sd,sm=load('S-assistant');memo={}
    def outcomes(q):
        if q not in memo:
            qq=h.get(q,'Question-Only');aa=h.get(q,'All-Writer');ss=score_record(q,'S-assistant',sm.select(spec(q),sd['config']),h);audit_reuse(ss);memo[q]=[qq,ss,aa]
        return memo[q]
    yy=np.array([[r['generation'].get('correct',int(r['generation']['prediction']==gold(q))) for r in outcomes(q)] for q in dev]);Y=yy[:,1:]-yy[:,:1];candidates=[];oof=[]
    configs=[('ridge',v) for v in [.1,1,10,100,1000]]+[('type',v) for v in [2,10,50,200]]
    for kind,alpha in configs:
        cv=np.zeros((len(dev),2))
        for k in range(5):
            tr=fold!=k;te=fold==k;model=fit(xd[tr],Y[tr],kind,alpha,[t for t,z in zip(td,tr) if z]);cv[te]=predict(xd[te],[t for t,z in zip(td,te) if z],model)
        model=fit(xd,Y,kind,alpha,td);scores=predict(X,types,model)
        for cap in [.42,.46,.49]:
            lam=calibrate(scores,cost,cap);choices=route(cv,cd,lam);pred=yy[np.arange(len(dev)),choices];cid=f'{kind}-{alpha}-cap{int(cap*100)}';allchoices=route(scores,cost,lam);budget=float(cost[np.arange(3000),allchoices].mean());candidates.append(dict(candidate_id=cid,kind=kind,alpha=alpha,cap=cap,lambda_cost=lam,model=model,choices=allchoices.tolist(),mean_R_ratio=budget));oof.append(dict(candidate_id=cid,n=360,correct=int(pred.sum()),ACC=100*pred.mean(),mean_R_ratio=float(cd[np.arange(360),choices].mean()),all3000_input_only_R=budget))
    for cid,ch in [('Question-Only',0),('S-assistant',1)]:candidates.append(dict(candidate_id=cid,choices=[ch]*3000,mean_R_ratio=float(cost[:,ch].mean()),model=None));oof.append(dict(candidate_id=cid,n=360,correct=int(yy[:,ch].sum()),ACC=100*yy[:,ch].mean(),mean_R_ratio=float(cd[:,ch].mean()),all3000_input_only_R=float(cost[:,ch].mean())))
    writecsv(D/'DEV_image_OOF.csv',oof);rank=sorted(oof,key=lambda r:(-r['correct'],r['mean_R_ratio'],r['candidate_id']));short=[r['candidate_id'] for r in rank[:8]];short=sorted(set(short+['Question-Only','S-assistant']));atomic_json(D/'shortlist_freeze.json',dict(UTC=utc(),shortlist=short,features=FEATURES,folds={q:int(k) for q,k in zip(dev,fold)},training_ids=dev,check_ids=check,candidates=candidates,training_only_outcome_use=True,all3000_budget_calibration_inputs_only=True,source_sha256=sha(__file__),one_GPU_previous_phases_only=True))
    checkY=np.array([[r['generation'].get('correct',int(r['generation']['prediction']==gold(q))) for r in outcomes(q)] for q in check]);checks=[]
    for c in candidates:
        if c['candidate_id'] not in short:continue
        choices=np.array([c['choices'][lookup[q]] for q in check]);vals=checkY[np.arange(len(check)),choices];checks.append(dict(candidate_id=c['candidate_id'],n=600,correct=int(vals.sum()),ACC=100*vals.mean(),mean_R_ratio=float(cost[[lookup[q] for q in check],choices].mean()),all3000_mean_R=c['mean_R_ratio']))
    writecsv(D/'CHECK600.csv',checks);best=sorted(checks,key=lambda r:(-r['correct'],r['mean_R_ratio'],r['candidate_id']))[0];win=next(c for c in candidates if c['candidate_id']==best['candidate_id']);assert win['mean_R_ratio']<.5;atomic_json(D/'winner_freeze.json',dict(UTC=utc(),winner=win,selection_results=checks,all_ids=ids,all_data_previously_seen=True,trained_on_DEV_only=True,model_not_retrained_after_CHECK=True))
    # Only now access the remaining pool's outcomes, for the one frozen policy.
    parent_csv=PHASE1/'per_question.csv';blob=subprocess.check_output(['git','show','03832c68f7bb09adea0af55b82014344806131b4:'+str(parent_csv.relative_to(ROOT.parent))],cwd=ROOT.parent);assert blob==parent_csv.read_bytes();base={r['question_id']:r for r in csv.DictReader(parent_csv.open())};records=[]
    labels=['Question-Only','S-assistant','All-Writer']
    for i,q in enumerate(ids):
        choices=outcomes(q);chosen=int(win['choices'][i]);r=choices[chosen];g=r['generation'];assert g['prediction']==gqa_prediction(g['text']);sp=spec(q);old=base[q];record=dict(question_id=q,task=sample(q)['task'],image_sha256=sample(q)['image_sha256'],question=sp['question'],gold=gold(q),branch=labels[chosen],correct=int(g['prediction']==gold(q)),answer=g['text'],token_ids=json.dumps(g['token_ids']),writers=json.dumps(r['writers']),k=len(r['writers']),R=len(sp['R']),R_ratio=len(r['writers'])/len(sp['R']),All_correct=int(old['All_correct']),Q100_correct=int(old['Q100_correct']),S_correct=int(old['S_correct']),SA_correct=int(gqa_prediction(choices[1]['generation']['text'])==gold(q)),P1_correct=int(old['correct']),Q80_correct=int(old['Q80_correct']))
        assert set(sp['historical_writers']['Semantic-k'])<=set(r['writers'])<=set(sp['R']);assert abs(record['R_ratio']-cost[i,chosen])<1e-12
        for short,bg in [('All',choices[2]['generation']),('Q100',choices[0]['generation']),('SA',choices[1]['generation'])]:record[short+'_answer']=bg['text'];record[short+'_harm']=int(record[short+'_correct'] and not record['correct']);record[short+'_fix']=int(not record[short+'_correct'] and record['correct'])
        records.append(record)
    h.close();atomic_json(D/'historical_audit_index.json',h.files);writecsv(D/'per_question.csv',records);fields=list(records[0]);writecsv(D/'harm_cases.csv',[r for r in records if any(r[s+'_harm'] for s in ['All','Q100','SA'])],fields);writecsv(D/'recovered_cases.csv',[r for r in records if any(r[s+'_fix'] for s in ['All','Q100','SA'])],fields)
    comparisons=[]
    for label,short in [('All-Writer','All'),('Question-Only','Q100'),('S-assistant','SA'),('Semantic-k','S'),('Phase1-whatwhich','P1'),('Semantic-Q80','Q80')]:
        d=np.array([r['correct']-r[short+'_correct'] for r in records]);lo,hi=bootstrap(d,[r['image_sha256'] for r in records]);comparisons.append(dict(baseline=label,delta_pp=100*d.mean(),CI95_low_pp=lo,CI95_high_pp=hi,harm=int(sum(d==-1)),fix=int(sum(d==1)),cluster_signflip_p=cluster_p(records,short),Holm_p=1.))
    prev=0.
    for j,r in enumerate(sorted(comparisons,key=lambda x:x['cluster_signflip_p'])):prev=max(prev,min(1.,(6-j)*r['cluster_signflip_p']));r['Holm_p']=prev
    writecsv(D/'paired_comparisons.csv',comparisons);task=[]
    for t in TASKS:
        rs=[r for r in records if r['task']==t];task.append(dict(task=t,n=len(rs),correct=sum(r['correct'] for r in rs),ACC=100*np.mean([r['correct'] for r in rs]),mean_R_ratio=np.mean([r['R_ratio'] for r in rs])))
    writecsv(D/'task_accuracy.csv',task);result=dict(status='COMPLETE',UTC=utc(),winner=win['candidate_id'],correct=sum(r['correct'] for r in records),ACC=100*np.mean([r['correct'] for r in records]),mean_R_ratio=np.mean([r['R_ratio'] for r in records]),branch_counts=dict(collections.Counter(r['branch'] for r in records)),new_GPU_generations=0,bootstrap=20000,seed=20261104,trained_DEV360_only=True,image_group_OOF=True,model_frozen_before_CHECK=True,policy_frozen_before_all3000=True,Gold_not_in_inference=True,elapsed_minutes=(time.monotonic()-read(P/'clock.json')['start_monotonic'])/60,all_data_exploratory=True);atomic_json(D/'completion.json',result);print(json.dumps(dict(result=result,checks=checks,comparisons=comparisons)),flush=True)

if __name__=='__main__':main()

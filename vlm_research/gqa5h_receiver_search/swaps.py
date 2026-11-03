"""Fixed-k causal receiver swaps, frozen by inputs before any outcome is read."""
import argparse,itertools,collections,time
from .common import *
from .selection import FEATURES,features
from .experiments import schedule,register
from .fit import fit,predict,auc

def prepare():
    start=time.monotonic();split=read(P/'split_manifest.json');ids=[q for t in TASKS for q in sorted((q for q in split['ids']['DEV'] if split['metadata'][q]['task']==t),key=lambda q:digest(['swap120-v1',q,split['metadata'][q]['image_sha256']]))[:20]];requests=[];no_pairs=[]
    for q in ids:
        sp=spec(q);groups=[g for g in sp['groups'] if g['question']];x=features(groups,sp['question']);base=set(sp['historical_writers']['Semantic-first-Q80']);selected=[i for i,g in enumerate(groups) if set(g['positions'])<=base];omitted=[i for i,g in enumerate(groups) if not set(g['positions'])&base]
        singles=[((i,),(j,)) for i in selected for j in omitted if len(groups[i]['positions'])==len(groups[j]['positions'])];singles.sort(key=lambda v:digest(['single-swap-v1',sp['question'],v]))
        chosen=[]
        for j in omitted:
            pair=next((p for p in singles if p[1]==(j,)),None)
            if pair is not None and pair not in chosen:chosen.append(pair)
        for pair in singles:
            if len(chosen)>=6:break
            if pair not in chosen:chosen.append(pair)
        doubles=[(a,b) for a in itertools.combinations(selected,2) for b in itertools.combinations(omitted,2) if sum(len(groups[i]['positions']) for i in a)==sum(len(groups[i]['positions']) for i in b)];doubles.sort(key=lambda v:digest(['double-swap-v1',sp['question'],v]));chosen=chosen[:6]+doubles[:2]
        if not chosen:no_pairs.append(q)
        for index,(remove,add) in enumerate(chosen):
            removed={p for i in remove for p in groups[i]['positions']};added={p for i in add for p in groups[i]['positions']};w=sorted(base-removed|added);assert len(w)==sp['k80']
            diff=sum((x[i]*len(groups[i]['positions']) for i in add),np.zeros(len(FEATURES)))-sum((x[i]*len(groups[i]['positions']) for i in remove),np.zeros(len(FEATURES)))
            requests.append(dict(question_id=q,candidate_id=f'SWAP-{index:02d}',writers=w,removed_group_indices=list(remove),added_group_indices=list(add),removed_groups=[groups[i] for i in remove],added_groups=[groups[i] for i in add],phi_difference=diff.tolist(),swap_type='single' if len(remove)==1 else 'double'))
    schedule('r07-fixed-budget-swap-dev120',requests,'causal_swap',dict(frozen_questions=ids,no_feasible_pairs=no_pairs,rule='20/task input-SHA; up to six equal-token whole-group single swaps covering omitted groups, plus two equal-budget double swaps; no correctness filtering.',feature_names=FEATURES,CPU_prepare_seconds=time.monotonic()-start))

def train():
    started=time.monotonic();D=P/'jobs/r07-fixed-budget-swap-dev120';requests=read(D/'requests.json')['requests'];lookup={(r['question_id'],r['candidate_id']):r for r in requests};rows=list(csv.DictReader((D/'results.csv').open()));x=np.array([lookup[r['question_id'],r['candidate_id']]['phi_difference'] for r in rows]);y=np.array([int(r['correct'])-int(r['Q80_correct']) for r in rows]);images=np.array([r['image_sha256'] for r in rows]);counts=collections.Counter(images);w=np.array([1/counts[im] for im in images]);w*=len(w)/w.sum();folds=np.array([int(digest(['swap-risk-CV-v1',im]),16)%5 for im in images]);metrics=[];models={}
    for family in ['role_no_POS','full']:
        keep=np.array([family=='full' or not n.startswith(('pos_','dep_')) for n in FEATURES]);scored=[]
        for lam in [1.,10.,100.,1000.]:
            pp=np.zeros(len(y))
            for f in range(5):
                test=folds==f;assert set(images[test]).isdisjoint(set(images[~test]));m=fit(x[~test],y[~test],lam,keep,w[~test]);pp[test]=predict(x[test],m)
            mse=float(np.average((pp-y)**2,weights=w));metrics.append(dict(family=family,lambda_value=lam,OOF_MSE=mse,fix_AUROC_descriptive=auc(y==1,pp),harms=int((y==-1).sum()),fixes=int((y==1).sum()),neutral=int((y==0).sum())));scored.append((mse,lam))
        models[family]=fit(x,y,min(scored)[1],keep,w)
    directory=P/'fixed_budget_swap_model';directory.mkdir();writecsv(directory/'feature_ablation.csv',metrics);atomic_json(directory/'models.json',models);atomic_json(directory/'image_folds.json',{str(f):sorted(set(images[folds==f])) for f in range(5)});atomic_json(directory/'training_audit.json',dict(source_sha256=sha(__file__),labels='Swap correct minus same-question exact-k Q80 baseline correct.',features='Token-weighted sum of input-only group features, added minus removed; inference maximizes the corresponding additive score under exact historical k.',n_questions=len(set(r['question_id'] for r in rows)),interventions=len(rows),seconds=time.monotonic()-started,Gold_only_DEV=True,utc=utc()))
    labels=[]
    for r in rows:
        v=lookup[r['question_id'],r['candidate_id']];labels.append(dict(**r,removed_groups=json.dumps(v['removed_groups']),added_groups=json.dumps(v['added_groups']),swap_type=v['swap_type'],phi_difference=json.dumps(v['phi_difference'])))
    writecsv(directory/'intervention_labels.csv',labels)
    cids=[]
    for family in ['role_no_POS','full']:
        for prior in [0.,.005]:
            cid='swap-'+family.replace('_','-')+'-prior'+('0' if prior==0 else '005')+'-q80';register(cid,dict(family='risk',fraction=.8,model=models[family],prior=prior),'Learn a global additive receiver score from real exact-k whole-group swaps; five image-fold ridge validation; no per-question memory and no test answers.');cids.append(cid)
    event('SWAP_MODEL_TRAINED',seconds=time.monotonic()-started,harms=int((y==-1).sum()),fixes=int((y==1).sum()),candidates=cids);print(json.dumps(dict(candidates=cids,metrics=metrics)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);a=p.parse_args();prepare() if a.stage=='prepare' else train()

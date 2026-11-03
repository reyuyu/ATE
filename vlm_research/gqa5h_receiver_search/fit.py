"""Image-separated NumPy ridge learning of signed causal deletion harm."""
import csv,collections,time
from .common import *
from .selection import FEATURES
from .experiments import register

def auc(y,s):
    a=np.asarray(s)[np.asarray(y)==1];b=np.asarray(s)[np.asarray(y)==0]
    return None if not len(a) or not len(b) else float(np.mean((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])))
def fit(x,y,lam,keep,weights):
    mean=x.mean(0);scale=x.std(0);mean[0]=0;scale[scale<1e-8]=1;z=(x-mean)/scale;z[:,0]=1;z[:,~keep]=0
    penalty=np.eye(z.shape[1])*lam;penalty[0,0]=lam*.01;coef=np.linalg.solve(z.T@(z*weights[:,None])+penalty,z.T@(y*weights));return dict(mean=mean.tolist(),scale=scale.tolist(),weights=coef.tolist(),lambda_value=lam,feature_mask=keep.tolist(),feature_names=FEATURES)
def predict(x,m):return ((x-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['weights'])
def main():
    guard(reserve=1800);start=time.monotonic();rows=list(csv.DictReader((P/'jobs/r02-loo-dev96/results.csv').open()));x=np.array([json.loads(r['features']) for r in rows]);y=np.array([int(r['causal_harm_label']) for r in rows]);images=np.array([r['image_sha256'] for r in rows]);validation=np.array([int(digest(['risk-validation-v1',im]),16)%5==0 for im in images]);train=~validation
    assert set(images[train]).isdisjoint(set(images[validation]));counts=collections.Counter(images);weights=np.array([1/counts[i] for i in images]);weights*=len(weights)/weights.sum();metrics=[];models={}
    for family in ['position_only','role_no_POS','full']:
        keep=np.ones(len(FEATURES),dtype=bool)
        if family=='position_only':keep=np.array([n in ['bias','position','position2','tail','head','tokens','Q_tokens'] for n in FEATURES])
        elif family=='role_no_POS':keep=np.array([not n.startswith(('pos_','dep_')) for n in FEATURES])
        candidates=[]
        for lam in [.1,1.,10.,100.]:
            m=fit(x[train],y[train],lam,keep,weights[train]);pred=predict(x[validation],m);mse=float(np.average((pred-y[validation])**2,weights=weights[validation]));r=dict(family=family,lambda_value=lam,validation_MSE=mse,harm_AUROC=auc(y[validation]==1,pred),validation_groups=int(validation.sum()),validation_harms=int((y[validation]==1).sum()),validation_fixes=int((y[validation]==-1).sum()));metrics.append(r);candidates.append((mse,lam))
        lam=min(candidates)[1];models[family]=fit(x,y,lam,keep,weights)
    directory=P/'causal_risk_model';directory.mkdir();atomic_json(directory/'models.json',models);writecsv(directory/'feature_ablation.csv',metrics);writecsv(directory/'intervention_labels.csv',rows)
    atomic_json(directory/'training_audit.json',dict(source_sha256=sha(__file__),features=FEATURES,labels='Q100 correct minus leave-one-group-out correct: +1 harmful deletion, -1 beneficial deletion, zero otherwise.',train_images=sorted(set(images[train])),validation_images=sorted(set(images[validation])),all_DEV_images=sorted(set(images)),interventions=len(rows),harms=int((y==1).sum()),fixes=int((y==-1).sum()),Gold_used_only_on_DEV=True,final_selector_has_no_Gold_or_answers=True,seconds=time.monotonic()-start,utc=utc()))
    for family in ['full','role_no_POS']:
        for fraction in [.8,.9]:register('causal-'+family.replace('_','-')+'-q'+str(int(100*fraction)),dict(family='risk',fraction=fraction,model=models[family],prior=.005),'Learned signed Q100 deletion risk; whole-group fixed-budget pruning; ridge hyperparameter selected only on image-disjoint DEV intervention validation.')
    event('RISK_MODEL_TRAINED',seconds=time.monotonic()-start,interventions=len(rows),harms=int((y==1).sum()),fixes=int((y==-1).sum()));print(json.dumps(dict(interventions=len(rows),harms=int((y==1).sum()),fixes=int((y==-1).sum()),seconds=time.monotonic()-start,validation=metrics)),flush=True)
if __name__=='__main__':main()

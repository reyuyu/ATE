"""Five image-fold validation and within-question centered causal risk learning."""
import collections,time
from .common import *
from .selection import FEATURES
from .fit import fit,predict,auc
from .experiments import register

def main():
    guard(reserve=1800);start=time.monotonic();rows=list(csv.DictReader((P/'jobs/r02-loo-dev96/results.csv').open()));x=np.array([json.loads(r['features']) for r in rows]);raw=np.array([int(r['causal_harm_label']) for r in rows],dtype=float);images=np.array([r['image_sha256'] for r in rows]);folds=np.array([int(digest(['risk-validation-v1',im]),16)%5 for im in images]);counts=collections.Counter(images);weights=np.array([1/counts[i] for i in images]);weights*=len(weights)/weights.sum();metrics=[];models={}
    centers={im:raw[images==im].mean() for im in set(images)};centered=np.array([v-centers[im] for v,im in zip(raw,images)])
    for label,target in [('absolute',raw),('within-question',centered)]:
        for family in ['position_only','role_no_POS','full']:
            keep=np.ones(len(FEATURES),dtype=bool)
            if family=='position_only':keep=np.array([n in ['bias','position','position2','tail','head','tokens','Q_tokens'] for n in FEATURES])
            elif family=='role_no_POS':keep=np.array([not n.startswith(('pos_','dep_')) for n in FEATURES])
            scored=[]
            for lam in [1.,10.,100.,1000.]:
                predictions=np.zeros(len(rows))
                for fold in range(5):
                    train=folds!=fold;test=~train;assert set(images[train]).isdisjoint(set(images[test]));m=fit(x[train],target[train],lam,keep,weights[train]);predictions[test]=predict(x[test],m)
                mse=float(np.average((predictions-target)**2,weights=weights));r=dict(label=label,family=family,lambda_value=lam,OOF_MSE=mse,harm_AUROC_descriptive=auc(raw==1,predictions),n_images=len(set(images)),n_groups=len(rows),harm_groups=int((raw==1).sum()),fix_groups=int((raw==-1).sum()));metrics.append(r);scored.append((mse,lam))
            model=fit(x,target,min(scored)[1],keep,weights);models[label+'-'+family]=model
    D=P/'causal_risk_cv';D.mkdir();writecsv(D/'feature_ablation.csv',metrics);atomic_json(D/'models.json',models);atomic_json(D/'image_folds.json',{str(f):sorted(set(images[folds==f])) for f in range(5)});atomic_json(D/'training_audit.json',dict(source_sha256=sha(__file__),fold_rule='SHA(risk-validation-v1,image)%5; all groups of an image stay together.',labels='Absolute signed deletion harm; centered variant subtracts question mean to learn within-question ranking.',interventions=len(rows),train_input_only_features=FEATURES,Gold_only_DEV=True,seconds=time.monotonic()-start,utc=utc()))
    cids=[]
    for label in ['absolute','within-question']:
        for family in ['role_no_POS','full']:
            cid='cv-'+label+'-'+family.replace('_','-')+'-q80';register(cid,dict(family='risk',fraction=.8,model=models[label+'-'+family],prior=.005),'Five image-fold regularization selection; compare absolute and within-question centered causal deletion risk, same exact Q80 budget.');cids.append(cid)
    event('CAUSAL_CV_TRAINED',seconds=time.monotonic()-start,candidates=cids,metrics=metrics);print(json.dumps(dict(seconds=time.monotonic()-start,candidates=cids,metrics=metrics)),flush=True)
if __name__=='__main__':main()

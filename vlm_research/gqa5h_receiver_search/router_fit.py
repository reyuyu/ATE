"""DEV-trained input-only budget allocation; never runs multiple answers at deployment."""
import collections,time
from .common import *
from .selection import route_features,ROUTE_FEATURES,select
from .experiments import register,candidate_job

def train(x,y,lam):
    mean=x.mean(0);scale=x.std(0);mean[0]=0;scale[scale<1e-8]=1;z=(x-mean)/scale;z[:,0]=1;penalty=np.eye(x.shape[1])*lam;penalty[0,0]=lam*.01;w=np.linalg.solve(z.T@z+penalty,z.T@y);return dict(mean=mean.tolist(),scale=scale.tolist(),weights=w.tolist(),feature_names=ROUTE_FEATURES,lambda_value=lam)
def prediction(x,m):return ((x-np.asarray(m['mean']))/np.asarray(m['scale']))@np.asarray(m['weights'])
def main():
    guard(reserve=1800);started=time.monotonic();ids=read(P/'split_manifest.json')['ids']['DEV'];rows=list(csv.DictReader((P/'baselines/DEV/per_question.csv').open()));lookup={(r['question_id'],r['condition']):r for r in rows};x=np.stack([route_features(spec(q)) for q in ids]);valid=np.array([int(digest(['route-validation-v1',sample(q)['image_sha256']]),16)%5==0 for q in ids]);audit=[];configs=[]
    for low,high,tag in [('Semantic-first-Q80','Question-Only','80-100'),('Semantic-first-Q60','Semantic-first-Q80','60-80')]:
        y=np.array([int(lookup[q,high]['correct'])-int(lookup[q,low]['correct']) for q in ids]);metrics=[]
        for lam in [.1,1.,10.,100.]:
            model=train(x[~valid],y[~valid],lam);mse=float(np.mean((prediction(x[valid],model)-y[valid])**2));metrics.append((mse,lam));audit.append(dict(pair=tag,lambda_value=lam,validation_MSE=mse,train_n=int((~valid).sum()),validation_n=int(valid.sum()),benefit_high=int((y==1).sum()),harm_high=int((y==-1).sum())))
        model=train(x,y,min(metrics)[1]);base=dict(family='router',model=model,low=dict(family='historical',label='Semantic-first-Q100' if low=='Question-Only' else low),high=dict(family='historical',label='Semantic-first-Q100' if high=='Question-Only' else high),threshold=0.)
        cid='adaptive-'+tag+'-gain';register(cid,base,'DEV-only image-separated ridge predicts high-minus-low correctness; deployment sees input text only and executes exactly one selected receiver set.');configs.append(cid)
        scores=prediction(x,model)
        low_budget=np.array([len(json.loads(lookup[q,low]['writers']))/int(lookup[q,low]['R']) for q in ids]);high_budget=np.array([len(json.loads(lookup[q,high]['writers']))/int(lookup[q,high]['R']) for q in ids])
        for cap in ([.35] if tag=='80-100' else [.30]):
            cutoffs=sorted(set(scores.tolist()+[float(scores.max()+1)]));possible=[]
            for threshold in cutoffs:
                budget=float(np.where(scores>=threshold,high_budget,low_budget).mean())
                if budget<=cap:possible.append((budget,threshold))
            _,threshold=max(possible,key=lambda v:(v[0],-v[1]));config={**base,'threshold':threshold};cid='adaptive-'+tag+'-budget'+str(int(cap*100));register(cid,config,'Same text-only benefit model; threshold fixed by DEV input-only mean R budget, without using any TUNE/FINAL outcome.');configs.append(cid)
    d=P/'adaptive_model';d.mkdir(exist_ok=True);writecsv(d/'training_metrics.csv',audit);atomic_json(d/'training_audit.json',dict(features=ROUTE_FEATURES,train_images=sorted({sample(q)['image_sha256'] for q,v in zip(ids,valid) if not v}),validation_images=sorted({sample(q)['image_sha256'] for q,v in zip(ids,valid) if v}),labels='Historical DEV complete-answer correctness high-budget minus low-budget; never exposed at inference.',Gold_only_DEV=True,seconds=time.monotonic()-started,utc=utc()));event('ADAPTIVE_MODEL_TRAINED',seconds=time.monotonic()-started,candidates=configs)
    candidate_job('r04-adaptive-dev360',configs,ids)
if __name__=='__main__':main()

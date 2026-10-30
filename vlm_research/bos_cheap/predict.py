import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from benchmarks.dataset_registry import digest,freeze_json,sha256
from .protocol import OUT,OLD,EPS,REGIONS
from .run import verify,gate

QK=[f'{f}_{s}' for f in ['bos_cos','bos_score'] for s in ['mean','std']]
VALUE=[f'{f}_{s}' for f in ['v_norm','projected_v_norm'] for s in ['mean','std']]
RES=['residual_norm','residual_mean','residual_std','residual_absmax','q_norm_mean','q_norm_std']
EXTRA=[f'{f}_{s}' for f in ['sample_margin','sample_score_mean'] for s in ['mean','std']]
BASE=QK+VALUE+['length','log_length']
SPECS={'Layer-only':([],True),'Length-only':(['length','log_length'],False),
 'BOS-QK+Layer':(QK,True),'BOS-QK+Value':(BASE,True),'BOS-QK+Residual':(BASE+RES,True),
 'Sampled6Keys':(BASE+EXTRA,True),'Global-noLayer':(BASE,False),'Early-separate':(BASE,True),'Region-grouped':(BASE,True)}
P_GRID=[0,.001,.002,.005,.01,.02,.03,.05,.1,.2,.4,.6,.8,1]
ORACLE_GRID=[.5,.6,.7,.75,.8,.85,.9,.95]


def load_rows(part):
    if part not in ['old','new']:raise ValueError('Unknown part')
    gate(part);manifest=json.loads((OUT/'sample_manifest.json').read_text());rows=[]
    for s in manifest[part+'_samples']:
        name=s['task']+'__'+s['question_id'];p=OUT/f"{part}/shard{s['rank']}/checkpoints/{name}.json"
        c=json.loads(p.read_text())
        if c['sample_sha256']!=digest(s):raise ValueError('Changed feature sample')
        if part=='old':metrics=json.loads((OLD/f"first_token/shard{s['rank']}/checkpoints/{name}.json").read_text())['result']['metrics']
        else:metrics=c['metrics']
        with np.load(c['raw_path']) as raw:bos=raw['bos_mass'].mean(-1)
        if len(c['features'])!=32 or len(metrics)!=32:raise ValueError('Incomplete labels/features')
        split=manifest['old_splits'][name] if part=='old' else 'heldout_exploratory'
        for i,(f,m) in enumerate(zip(c['features'],metrics)):
            if f['layer']!=i or m['layer']!=i:raise ValueError('Layer mismatch')
            rows.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],split=split,oracle_bos_mass=float(bos[i]),**f,**{k:m[k] for k in ['delta_logp','top1_flip','js']},abs_delta_logp=abs(m['delta_logp']),logp_loss=max(-m['delta_logp'],0),js_divergence=m['js']))
    return rows


def design(rows,columns,layer):
    x=np.array([[r[c] for c in columns] for r in rows],dtype=np.float64).reshape(len(rows),len(columns))
    if layer:x=np.column_stack([x,np.eye(32)[[r['layer'] for r in rows]]])
    if not np.isfinite(x).all():raise ValueError('Nonfinite cheap design')
    return x


def danger(rows,epsilon):
    return np.array([r['top1_flip'] or -r['delta_logp']>epsilon for r in rows],dtype=np.float64)


def logistic(x,y):
    mean=x.mean(0);sd=x.std(0);sd[sd<1e-10]=1
    xt=torch.as_tensor((x-mean)/sd,dtype=torch.float64);yt=torch.as_tensor(y,dtype=torch.float64)
    w=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    p=float((y.sum()+.5)/(len(y)+1));b=torch.tensor(np.log(p/(1-p)),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([w,b],max_iter=80,line_search_fn='strong_wolfe',tolerance_grad=1e-8)
    def closure():
        optimizer.zero_grad();loss=F.binary_cross_entropy_with_logits(xt@w+b,yt)+.001*w.square().sum()/2;loss.backward();return loss
    start=time.perf_counter();optimizer.step(closure)
    return dict(mean=mean.tolist(),scale=sd.tolist(),weights=w.detach().tolist(),intercept=float(b.detach()),training_seconds=time.perf_counter()-start,parameters=x.shape[1]+1)


def raw_score(x,model):return (x-np.array(model['mean']))/np.array(model['scale'])@np.array(model['weights'])+model['intercept']
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-60,60)))


def region(layer,kind):
    if kind=='Early-separate':return int(layer>=2)
    if kind=='Region-grouped':return next(i for i,(a,b) in enumerate(REGIONS) if a<=layer<=b)
    return 0


def fit_model(train,cal,name,epsilon):
    cols,layer=SPECS[name];model=dict(name=name,columns=cols,layer=layer,groups={})
    for g in sorted({region(r['layer'],name) for r in train}):
        tr=[r for r in train if region(r['layer'],name)==g];ca=[r for r in cal if region(r['layer'],name)==g]
        base=logistic(design(tr,cols,layer),danger(tr,epsilon))
        score=raw_score(design(ca,cols,layer),base)
        base['calibration']=logistic(score[:,None],danger(ca,epsilon))
        model['groups'][str(g)]=base
    return model


def predict(rows,model):
    out=np.zeros(len(rows),dtype=np.float64)
    for g,m in model['groups'].items():
        idx=[i for i,r in enumerate(rows) if region(r['layer'],model['name'])==int(g)]
        if not idx:continue
        x=design([rows[i] for i in idx],model['columns'],model['layer'])
        score=raw_score(x,m);out[idx]=sigmoid(raw_score(score[:,None],m['calibration']))
    return out


def score_rule(rows,model):
    if model['kind']=='logistic':return predict(rows,model['model'])
    return np.array([r[model['column']] for r in rows])


def should_skip_attention(layer,cheap_features,rule):
    """Frozen whole-P-1-row rule. Oracle is deliberately rejected online."""
    if rule['name']=='BOS-Mass-Oracle':raise ValueError('Post-Attention oracle is not a cheap online feature')
    row=dict(cheap_features,layer=layer)
    if rule.get('early_never') and layer<2:return False
    score=score_rule([row],rule)[0]
    threshold=rule.get('per_layer_thresholds',[rule['threshold']]*32)[layer]
    return bool(score<=threshold if rule['direction']=='low' else score>=threshold)


def skips(rows,rule):
    score=score_rule(rows,rule);threshold=np.array([rule.get('per_layer_thresholds',[rule['threshold']]*32)[r['layer']] for r in rows])
    selected=score<=threshold if rule['direction']=='low' else score>=threshold
    if rule.get('early_never'):selected&=np.array([r['layer']>=2 for r in rows])
    return selected


def select_threshold(rows,rules,epsilon):
    y=danger(rows,epsilon);best=None;criteria=None
    for index,r in enumerate(rules):
        s=skips(rows,r);n=int(s.sum());risk=float(y[s].mean()) if n else 0.
        key=(n,-risk,-index)
        if risk<=.01 and (criteria is None or key>criteria):best=dict(r,calibration_skipped=n,calibration_risk=risk);criteria=key
    if best is None:
        best=dict(rules[0],threshold=-1e30 if rules[0]['direction']=='low' else 1e30,calibration_skipped=0,calibration_risk=None)
    return best


def fit():
    verify();gate('old')
    if any((OUT/f'new/shard{i}').exists() for i in range(4)):raise ValueError('Fit must precede any new-image model output')
    rows=load_rows('old');train=[r for r in rows if r['split']=='train'];cal=[r for r in rows if r['split']=='calibration']
    if {r['image_sha256'] for r in train}&{r['image_sha256'] for r in cal}:raise ValueError('Train/cal image leakage')
    output=dict(eps={},candidate_protocol='All candidates, thresholds, calibration decisions and recommendations frozen before new-image features/labels.')
    for epsilon in EPS:
        entry=dict(selected=[],curves=[],recommended=None)
        for name in SPECS:
            model=fit_model(train,cal,name,epsilon)
            candidates=[dict(name=name,kind='logistic',model=model,direction='low',threshold=t) for t in P_GRID]
            entry['curves']+=candidates;entry['selected'].append(select_threshold(cal,candidates,epsilon));print('Fitted',epsilon,name,flush=True)
        for name,column in [('BOS-cos','bos_cos_mean'),('BOS-score','bos_score_mean'),('BOS-Mass-Oracle','oracle_bos_mass')]:
            thresholds=ORACLE_GRID if name=='BOS-Mass-Oracle' else np.unique(np.quantile([r[column] for r in train],np.linspace(0,1,21))).tolist()
            directions=['high'] if name=='BOS-Mass-Oracle' else ['high','low']
            candidates=[dict(name=name,kind='scalar',column=column,direction=d,threshold=float(t)) for d in directions for t in thresholds]
            entry['curves']+=candidates;entry['selected'].append(select_threshold(cal,candidates,epsilon))
        base=next(r for r in entry['selected'] if r['name']=='BOS-QK+Residual')
        candidates=[dict(base,name='Early-never-skip',early_never=True,threshold=t) for t in P_GRID]
        entry['curves']+=candidates;entry['selected'].append(select_threshold(cal,candidates,epsilon))
        cosine=[r for r in entry['curves'] if r['name']=='BOS-cos']
        for name,groups in [('Region-threshold',[list(range(a,b+1)) for a,b in REGIONS]),('Layer-threshold',[[i] for i in range(32)])]:
            per=[];directions=[]
            # Fixed high-cosine rule; thresholds use calibration only within each group.
            for group in groups:
                sub=[r for r in cal if r['layer'] in group];candidates=[r for r in cosine if r['direction']=='high']
                chosen=select_threshold(sub,candidates,epsilon);per.append(chosen['threshold'])
            threshold=[next(per[j] for j,g in enumerate(groups) if i in g) for i in range(32)]
            rule=dict(name=name,kind='scalar',column='bos_cos_mean',direction='high',threshold=0.,per_layer_thresholds=threshold)
            s=skips(cal,rule);rule.update(calibration_skipped=int(s.sum()),calibration_risk=float(danger(cal,epsilon)[s].mean()) if s.any() else None)
            entry['curves'].append(rule);entry['selected'].append(rule)
        cheap=[r for r in entry['selected'] if r['name'] not in ['BOS-Mass-Oracle','Sampled6Keys']]
        eligible=[r for r in cheap if r.get('calibration_risk') is not None and r['calibration_risk']<=.01]
        entry['recommended']=max(eligible,key=lambda r:(r['calibration_skipped'],-len(r.get('model',{}).get('columns',[]))))['name'] if eligible else 'Never-skip'
        output['eps'][str(epsilon)]=entry
    freeze_json(OUT/'frozen_predictors.json',output)
    freeze_json(OUT/'predictor_freeze.json',dict(binding=digest(verify()),models_sha256=sha256(OUT/'frozen_predictors.json'),train_questions=len(train)//32,calibration_questions=len(cal)//32,new_outputs_present=False,chosen_without_test=True))


if __name__=='__main__':fit()

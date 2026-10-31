"""Predeclared paired conditional effects, separate from exploratory A contrasts."""
import numpy as np
from paper_rq1_early_joint_attention.statistics import bootstrap
from .protocol import KEYS, SEED


def interaction(values,source):
    lookup={k:i for i,k in enumerate(KEYS)}
    v=np.asarray(values,dtype=float)
    return v[:,lookup['EarlyOFF']]-v[:,lookup['EarlyOFF_L2'+source]]-v[:,0]+v[:,lookup['L2'+source]]


def summarize(rows,images,group='Overall'):
    scores=np.asarray([[r[k]['score'] for k in KEYS] for r in rows],dtype=float)
    flips=np.asarray([[r[k].get('first_top1_flip',False) for k in KEYS] for r in rows],dtype=float)
    losses=np.asarray([[r[k].get('first_logp_loss',0.) for k in KEYS] for r in rows],dtype=float)
    delta=scores[:,1:]-scores[:,:1]
    harm=(scores[:,:1]==1)&(scores[:,1:]==0)
    gain=(scores[:,:1]==0)&(scores[:,1:]==1)
    d=np.column_stack([interaction(scores,s) for s in ['Image','Question']])
    values=np.column_stack([scores,delta,flips[:,1:],losses[:,1:],harm,gain,d])
    # Per-task intervals remain descriptive, not a global task-by-condition family.
    families=[1]*12+[6 if group=='Overall' and i<6 else 1 for i in range(11)]+[1]*44+[2 if group=='Overall' else 1]*2
    stats=bootstrap(values,np.asarray(images),families,seed=SEED)
    def fields(stat,prefix='',scale=1):
        return {prefix+k:scale*stat[k] for k in ['ci95_low','ci95_high','family95_low','family95_high']}
    summary=[]
    for i,k in enumerate(KEYS):
        r=dict(group=group,condition=k,n=len(rows),image_clusters=len(set(images)),correct=int(scores[:,i].sum()),
            accuracy_pct=100*scores[:,i].mean(),**fields(stats[i],'accuracy_',100),
            delta_full_pp=100*(scores[:,i]-scores[:,0]).mean(),
            correct_to_wrong=int(((scores[:,0]==1)&(scores[:,i]==0)).sum()),
            wrong_to_correct=int(((scores[:,0]==0)&(scores[:,i]==1)).sum()),
            first_token_flips=int(flips[:,i].sum()),mean_logp_loss=float(losses[:,i].mean()),
            empty=sum(not r[k]['prediction'] for r in rows),truncated=sum(r[k]['truncated'] for r in rows))
        for prefix,index,scale in [('delta_',12+i-1,100),('flip_',23+i-1,100),('logp_loss_',34+i-1,1),
                                   ('harm_',45+i-1,100),('rescue_',56+i-1,100)]:
            r.update(fields(stats[index],prefix,scale) if i else {prefix+n:0. for n in ['ci95_low','ci95_high','family95_low','family95_high']})
        summary.append(r)
    interactions=[]
    for j,source in enumerate(['Image','Question']):
        s=stats[67+j]; lookup={k:i for i,k in enumerate(KEYS)}
        normal=scores[:,0]-scores[:,lookup['L2'+source]]
        off=scores[:,lookup['EarlyOFF']]-scores[:,lookup['EarlyOFF_L2'+source]]
        interactions.append(dict(group=group,source=source,n=len(rows),image_clusters=len(set(images)),
            normal_source_harm_pp=float(100*normal.mean()),earlyoff_source_harm_pp=float(100*off.mean()),
            D_pp=100*s['mean'],**fields(s,'D_',100),family=s['family'],bootstrap_repetitions=s['repetitions'],
            degenerate=s['degenerate_bootstrap'],
            descriptive_D_logp_loss=float(-interaction(losses,source).mean()),
            descriptive_D_first_flip_pp=float(-100*interaction(flips,source).mean())))
    return summary,interactions


def intersections(rows,group='Overall'):
    full=np.asarray([r['Full']['score'] for r in rows])
    harms={k:(full==1)&np.asarray([r[k]['score']==0 for r in rows]) for k in KEYS[1:]}
    result=[]
    for i,a in enumerate(KEYS[1:]):
        for b in KEYS[i+2:]:
            both=int((harms[a]&harms[b]).sum()); union=int((harms[a]|harms[b]).sum())
            result.append(dict(group=group,condition_a=a,condition_b=b,n=len(rows),
                errors_a=sum(r[a]['score']==0 for r in rows),errors_b=sum(r[b]['score']==0 for r in rows),
                any_error_intersection=sum(r[a]['score']==0 and r[b]['score']==0 for r in rows),
                full_correct_joint_harm=both,full_correct_harm_union=union,
                harm_jaccard=both/union if union else None))
    return result

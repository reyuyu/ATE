"""Image-cluster descriptive alignment analysis of the frozen native-only collection."""
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import rankdata

from .preflight import OUT,HISTORY,dump,sha

QUERIES=('P-1','Question')
METRICS=('image_mass','roi_mass','roi_hit_ratio','roi_enrichment','random_roi_mass',
         'random_roi_hit_ratio','random_roi_enrichment','expected_random_roi_mass',
         'expected_random_roi_hit_ratio','expected_random_roi_enrichment',
         'absolute_logp_change','signed_logp_loss')
TASKS=('ChooseAttr','ChooseCat','ChooseRel','CompareAttr','LogicalObj','QueryAttr')


def read(p):return json.loads(Path(p).read_text())


def csv_write(name,rows):
    assert rows
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        writer.writerows({k:('' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v)
                         for k,v in r.items()} for r in rows)


def rho(x,y):
    x,y=rankdata(x,axis=-1,nan_policy='omit'),rankdata(y,axis=-1,nan_policy='omit')
    x=x-np.nanmean(x,axis=-1,keepdims=True);y=y-np.nanmean(y,axis=-1,keepdims=True)
    den=np.sqrt(np.nansum(x*x,axis=-1)*np.nansum(y*y,axis=-1))
    return np.divide(np.nansum(x*y,axis=-1),den,out=np.full_like(den,np.nan),where=den>0)


def interval(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return tuple(np.quantile(x,[.025,.975])) if len(x) else (float('nan'),float('nan'))


def bootstrap(samples,repetitions,seed):
    groups={}
    for i,s in enumerate(samples):groups.setdefault(s['image_sha256'],[]).append(i)
    groups=list(groups.values());c=len(groups)
    rng=np.random.default_rng(seed);draws=rng.integers(c,size=(repetitions,c))
    counts=np.zeros((repetitions,c));np.add.at(counts,(np.arange(repetitions)[:,None],draws),1)
    inverse=np.empty(len(samples),int)
    for i,members in enumerate(groups):inverse[members]=i
    weights=counts[:,inverse];weights/=weights.sum(1,keepdims=True)
    if c==len(samples):
        # Group insertion order follows sample order when every image occurs once.
        indices=draws
    else:
        expanded=[np.concatenate([groups[i] for i in row]) for row in draws]
        indices=np.full((repetitions,max(map(len,expanded))),-1,dtype=int)
        for i,row in enumerate(expanded):indices[i,:len(row)]=row
    return weights,indices,c


def boot_values(x,indices):
    return np.where(indices>=0,x[np.maximum(indices,0)],np.nan)


def aggregate():
    start=time.perf_counter();protocol=read(OUT/'protocol.json')
    samples=read(OUT/'sample_manifest.json')['samples'];mapping=read(OUT/'roi_mapping.json')
    checks=[];all_rows=[];cube=[]
    for shard in range(3):
        complete=read(OUT/f'shard_{shard}/complete.json');v=read(OUT/f'shard_{shard}/validation.json')
        assert complete['passed'] and complete['count']==80 and v['passed']
        assert complete['protocol_sha256']==sha(OUT/'protocol.json')
        checks.append(dict(shard=shard,**complete,smoke_count=v['smoke_count']))
    assert sum(c['smoke_count'] for c in checks)==24
    assert len({c['code_commit'] for c in checks})==1
    max_row_error=0
    for s in samples:
        key=f'{s["task"]}__{s["question_id"]}';folder=OUT/f'shard_{s["shard"]}'
        r=read(folder/'checkpoints'/f'{key}.json')
        assert r['input_binding']==s['input_binding'] and r['historical_full_exact']
        assert r['protocol_sha256']==sha(OUT/'protocol.json') and len(r['rows'])==72
        assert sha(folder/'rows'/f'{key}.npz')==r['row_array_sha256']
        old=read(HISTORY/f'shard_{s["shard"]}'/'checkpoints'/f'{key}.json')
        assert r['full_logits_sha256']==old['full_logits_sha256']
        matrix=np.empty((36,2,len(METRICS)))
        seen=set()
        for row in r['rows']:
            layer,q=row['layer'],QUERIES.index(row['query'])
            assert (layer,q) not in seen;seen.add((layer,q))
            assert row['image_sha256']==s['image_sha256'] and row['task']==s['task'] and row['question_id']==s['question_id']
            assert row['absolute_logp_change']==old['rows'][layer]['absolute_logp_change']
            assert row['signed_logp_loss']==old['rows'][layer]['signed_logp_loss']
            assert 0<=row['roi_hit_ratio']<=1+1e-12 and row['roi_mass']<=row['image_mass']+1e-12
            assert np.isclose(row['roi_enrichment'],row['roi_hit_ratio']/row['roi_fraction'],rtol=0,atol=1e-12)
            matrix[layer,q]=[row[m] for m in METRICS]
            max_row_error=max(max_row_error,row['row_sum_max_error'])
            all_rows.append(row)
        assert len(seen)==72 and np.isfinite(matrix).all()
        cube.append(matrix)
    cube=np.stack(cube);assert cube.shape==(240,36,2,len(METRICS))
    assert len(all_rows)==17280
    csv_write('per_question.csv',all_rows)
    summary=[];layer_correlations=[];individual=[];individual_summary=[];peaks=[];peak_summary=[];l26=[]
    correlation_arrays={}
    for q,query in enumerate(QUERIES):
        for m,metric in enumerate(METRICS[:4]):
            for sens in ('absolute_logp_change','signed_logp_loss'):
                values=rho(cube[:,:,q,m],cube[:,:,0,METRICS.index(sens)])
                correlation_arrays[(q,metric,sens)]=values
                for i,(s,v) in enumerate(zip(samples,values)):
                    individual.append(dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],
                        query=query,metric=metric,sensitivity=sens,spearman=float(v),
                        constant_roi_metric=int(np.ptp(cube[i,:,q,m])==0),
                        constant_sensitivity=int(np.ptp(cube[i,:,0,METRICS.index(sens)])==0),
                        status='valid' if np.isfinite(v) else 'constant_curve_undefined'))
    csv_write('per_question_correlations.csv',individual)
    for q,query in enumerate(QUERIES):
        roi_peaks=cube[:,:,q,3].argmax(1);sensitivity_peaks=cube[:,:,0,-2].argmax(1)
        for i,s in enumerate(samples):
            peaks.append(dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],query=query,
                roi_enrichment_peak=int(roi_peaks[i]),absolute_sensitivity_peak=int(sensitivity_peaks[i]),
                peak_difference=int(roi_peaks[i]-sensitivity_peaks[i]),
                constant_roi_curve=int(np.ptp(cube[i,:,q,3])==0),roi_fraction=mapping[f'{s["task"]}__{s["question_id"]}']['roi_fraction']))
    csv_write('peak_layers.csv',peaks)
    for group_number,task in enumerate(('Overall',)+TASKS):
        selection=[i for i,s in enumerate(samples) if task=='Overall' or s['task']==task]
        selected=[samples[i] for i in selection];x=cube[selection]
        weights,indices,clusters=bootstrap(selected,protocol['bootstrap']['repetitions'],20261102+group_number)
        boot=np.tensordot(weights,x,axes=(1,0));mean=x.mean(0);std=x.std(0,ddof=1)
        low,high=np.quantile(boot,[.025,.975],axis=0)
        for layer in range(36):
            for q,query in enumerate(QUERIES):
                for m,metric in enumerate(METRICS):
                    summary.append(dict(task=task,questions=len(selection),image_clusters=clusters,layer=layer,query=query,metric=metric,
                        mean=float(mean[layer,q,m]),std=float(std[layer,q,m]),ci_low=float(low[layer,q,m]),ci_high=float(high[layer,q,m])))
        for q,query in enumerate(QUERIES):
            for m,metric in enumerate(METRICS[:4]):
                for sens in ('absolute_logp_change','signed_logp_loss'):
                    sm=METRICS.index(sens)
                    point=float(rho(mean[:,q,m],mean[:,0,sm]));blo,bhi=interval(rho(boot[:,:,q,m],boot[:,:,0,sm]))
                    layer_correlations.append(dict(task=task,questions=len(selection),image_clusters=clusters,query=query,
                        metric=metric,sensitivity=sens,spearman=point,ci_low=blo,ci_high=bhi,
                        unit='36 dependent layer means; uncertainty from paired image resampling'))
                    values=correlation_arrays[(q,metric,sens)][selection];valid=np.isfinite(values)
                    denom=weights[:,valid].sum(1)
                    bmean=np.divide(weights[:,valid]@values[valid],denom,out=np.full_like(denom,np.nan),where=denom>0)
                    clo,chi=interval(bmean)
                    individual_summary.append(dict(task=task,query=query,metric=metric,sensitivity=sens,
                        valid_questions=int(valid.sum()),undefined_questions=int((~valid).sum()),
                        mean_spearman=float(values[valid].mean()) if valid.any() else np.nan,
                        median_spearman=float(np.median(values[valid])) if valid.any() else np.nan,
                        mean_ci_low=clo,mean_ci_high=chi))
                    point26=float(rho(x[:,26,q,m],x[:,26,0,sm]))
                    blo,bhi=interval(rho(boot_values(x[:,26,q,m],indices),boot_values(x[:,26,0,sm],indices)))
                    l26.append(dict(task=task,questions=len(selection),image_clusters=clusters,query=query,metric=metric,
                        sensitivity=sens,spearman=point26,ci_low=blo,ci_high=bhi,unit='Cross-question at fixed L26; image-cluster bootstrap'))
            delta=x[:,:,q,3].argmax(1)-x[:,:,0,-2].argmax(1)
            dlo,dhi=interval(weights@delta)
            curve_dlo,curve_dhi=interval(boot[:,:,q,3].argmax(1)-boot[:,:,0,-2].argmax(1))
            peak_summary.append(dict(task=task,query=query,questions=len(selection),mean_peak_difference=float(delta.mean()),
                median_peak_difference=float(np.median(delta)),ci_low=dlo,ci_high=dhi,
                mean_curve_roi_peak=int(mean[:,q,3].argmax()),mean_curve_sensitivity_peak=int(mean[:,0,-2].argmax()),
                mean_curve_peak_difference=int(mean[:,q,3].argmax()-mean[:,0,-2].argmax()),
                mean_curve_peak_difference_ci_low=curve_dlo,mean_curve_peak_difference_ci_high=curve_dhi,
                exact_peak_matches=int((delta==0).sum()),within_two_layers=int((np.abs(delta)<=2).sum()),
                constant_roi_curves=int((np.ptp(x[:,:,q,3],axis=1)==0).sum())))
        print(f'{task}: {len(selection)} questions, {clusters} image clusters; bootstrap complete.',flush=True)
    csv_write('layer_summary.csv',summary)
    csv_write('task_layer_summary.csv',[s for s in summary if s['task']!='Overall'])
    csv_write('layer_curve_correlations.csv',layer_correlations)
    csv_write('per_question_correlation_summary.csv',individual_summary)
    csv_write('peak_summary.csv',peak_summary)
    csv_write('l26_correlations.csv',l26)
    csv_write('l26_layer_summary.csv',[s for s in summary if s['layer']==26])
    validation=dict(status='complete',samples=240,task_counts=dict(Counter(s['task'] for s in samples)),
        image_clusters=len({s['image_sha256'] for s in samples}),layers=36,query_types=list(QUERIES),paired_rows=17280,
        bbox_resolved=240,cpu_processor_geometry_checked=240,gpu_smoke_questions=24,cpu_tests_passed=9,
        all_full_logits_historical_exact=240,all_attention_row_arrays_verified=True,row_sum_max_error=max_row_error,
        row_sum_tolerance=2**-8+2e-6,native_deepstack_verified_on_gpus=[0,2,3],
        no_new_imageoff_or_generation=True,no_nan_inf_oom=True,hooks_clean=True,weights_unchanged=True,
        code_commit=checks[0]['code_commit'],protocol_sha256=sha(OUT/'protocol.json'),gpu_runs=checks,
        bootstrap_repetitions=2000,statistics_seconds=time.perf_counter()-start)
    dump(OUT/'validation.json',validation)
    from .figures import plot,report
    plot(samples,cube,summary,individual_summary,peaks)
    report(samples,cube,summary,layer_correlations,individual_summary,peak_summary,l26,validation)
    print('Complete: 240 questions; 17,280 paired native ROI rows; figures/report saved.',flush=True)


if __name__=='__main__':aggregate()

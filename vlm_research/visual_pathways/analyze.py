"""Question-weighted, whole-image paired bootstrap; no selection by result."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .common import ROOT,OUT,QH,TASKS,read,write,sha,digest


def csv_write(path,rows):
    assert rows
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def boot_mean(values,samples,repetitions=10000,seed=20261102):
    values=np.asarray(values,float)
    if values.ndim==1:
        values=values[:,None]
    groups={}
    for i,s in enumerate(samples):
        groups.setdefault(s['image_sha256'],[]).append(i)
    members=list(groups.values());g=len(members)
    totals=np.stack([values[ids].sum(0) for ids in members])
    sizes=np.array([len(ids) for ids in members])
    rng=np.random.default_rng(seed)
    output=[]
    for first in range(0,repetitions,250):
        counts=rng.multinomial(g,np.full(g,1/g),size=min(250,repetitions-first))
        output.append((counts@totals)/(counts@sizes)[:,None])
    draws=np.concatenate(output)
    low,high=np.quantile(draws,[.025,.975],axis=0)
    return values.mean(0),low,high,draws,g


def scan_model(model,allow_partial=False):
    samples=read(OUT/'sample_manifest.json')['samples']
    layers={'qwen':36,'internvl':28}[model]
    selected=[];cube=[];flat=[];missing=[];reused=0;new=0
    for s in samples:
        key=f'{s["task"]}__{s["question_id"]}'
        path=OUT/model/'checkpoints'/f'{key}.json'
        if not path.exists() or not read(path).get('complete'):
            missing.append(key);continue
        record=read(path)
        assert record['sample_sha256']==digest(s) and record['image_sha256']==s['image_sha256']
        assert record['protocol_sha256']==sha(OUT/'protocol.json')
        assert len(record['rows'])==layers and sorted(r['layer'] for r in record['rows'])==list(range(layers))
        x=[]
        for row in sorted(record['rows'],key=lambda r:r['layer']):
            assert row['image_sha256']==s['image_sha256']
            assert row['question_id']==s['question_id'] and row['task']==s['task']
            assert row['absolute_logp_change']==abs(row['signed_logp_loss'])
            assert row['top1_flip']==int(row['full_first_token']!=row['changed_first_token'])
            assert row.get('masked_probability_max',row.get('image_probability_max'))==0
            off_logp=row.get('off_full_top1_logp',row.get('imageoff_full_top1_logp'))
            assert float(np.float32(row['full_top1_logp'])-np.float32(off_logp))==row['signed_logp_loss']
            changed_count=row.get('mask_changed_entries',row.get('changed_entries'))
            assert changed_count==row['image_tokens']
            flat.append(dict(row,model=model,pilot=s['pilot'],reused=int('reused_from' in record),
                off_full_top1_logp=off_logp,mask_changed_entries=changed_count,
                mask_expected_entries=row['image_tokens'],masked_probability_max=row.get('masked_probability_max',row.get('image_probability_max'))))
            x.append([row['absolute_logp_change'],row['signed_logp_loss'],row['top1_flip']])
        assert np.isfinite(x).all()
        selected.append(s);cube.append(x)
        reused+=int('reused_from' in record);new+=int('reused_from' not in record)
    if not selected:
        return dict(model=model,status='stopped',questions=0,missing=len(missing))
    if not allow_partial:
        assert len(selected)==3000, f'{model}: {len(selected)}/3000 only'
    x=np.array(cube)
    folder=OUT/model
    csv_write(folder/'per_question.csv',flat)
    rows=[];peaks=[]
    cohorts=['all','pilot240']
    for cohort in cohorts:
        for task in ('Overall',)+TASKS:
            ids=[i for i,s in enumerate(selected) if (cohort=='all' or s['pilot']) and (task=='Overall' or s['task']==task)]
            if not ids:
                continue
            subs=[selected[i] for i in ids]
            means,low,high,_,groups=boot_mean(x[ids].reshape(len(ids),-1),subs)
            means,low,high=[v.reshape(layers,3) for v in (means,low,high)]
            for layer in range(layers):
                row=dict(model=model,cohort=cohort,task=task,questions=len(ids),image_clusters=groups,layer=layer)
                for j,name in enumerate(('mean_absolute_logp_change','mean_signed_loss','top1_flip_rate')):
                    row[name]=float(means[layer,j]);row[name+'_ci_low']=float(low[layer,j]);row[name+'_ci_high']=float(high[layer,j])
                row['top1_flip_count']=int(x[ids,layer,2].sum())
                rows.append(row)
            rank=np.argsort(-means[:,0],kind='stable')
            total=means[:,0].sum()
            peaks.append(dict(model=model,cohort=cohort,task=task,questions=len(ids),
                absolute_peak_layer=int(rank[0]),absolute_peak_value=float(means[rank[0],0]),
                signed_peak_layer=int(means[:,1].argmax()),top3_layers=[int(i) for i in rank[:3]],
                top1_share=float(means[rank[0],0]/total) if total else 0.,
                top3_share=float(means[rank[:3],0].sum()/total) if total else 0.))
    csv_write(folder/'layer_summary.csv',rows);write(folder/'peaks.json',peaks)
    write(folder/'coverage.json',dict(model=model,status='complete' if len(selected)==3000 else 'partial',
        questions=len(selected),conditions=len(flat),new_questions=new,historical_questions=reused,
        tasks=dict(Counter(s['task'] for s in selected)),pilot_questions=sum(s['pilot'] for s in selected),
        unique_images=len({s['image_sha256'] for s in selected}),missing=missing))
    return read(folder/'coverage.json')


def relay_summary():
    folder=OUT/'llava_relay'
    return generation_summary(folder,('A','B','C','D'),interaction=True)


def rescue_summary(model):
    folder=OUT/(model+'_rescue')
    names=('FullImageAccess','AllImageOFF','KeepSparse')
    summary=generation_summary(folder,names)
    samples=read(folder/'sample_manifest.json')['samples']
    records=[read(folder/'checkpoints'/f'{s["task"]}__{s["question_id"]}.json') for s in samples]
    scores=np.array([[r['conditions'][c]['correct'] for c in names]+[r['native_full_accuracy']] for r in records],float)
    comparisons=[(2,1),(2,0),(0,1),(0,3),(1,3),(2,3)]
    labels=list(names)+['NativeFull']
    delta=np.stack([(scores[:,a]-scores[:,b])*100 for a,b in comparisons],axis=1)
    m,l,h,_,groups=boot_mean(delta,samples)
    rows=[]
    for j,(a,b) in enumerate(comparisons):
        rows.append(dict(comparison=labels[a]+' - '+labels[b],n=240,image_clusters=groups,
            delta_accuracy_pp=float(m[j]),ci_low=float(l[j]),ci_high=float(h[j]),
            reference_correct_to_wrong=int(((scores[:,b]==1)&(scores[:,a]==0)).sum()),
            reference_wrong_to_correct=int(((scores[:,b]==0)&(scores[:,a]==1)).sum()),exploratory=True))
    csv_write(folder/'paired_comparisons.csv',rows)
    native_m,native_l,native_h,_,_=boot_mean(scores[:,3]*100,samples)
    write(folder/'native_full_reference.json',dict(accuracy_percent=float(native_m[0]),ci_low=float(native_l[0]),ci_high=float(native_h[0]),
        questions=240,source='Same-machine Task2 A' if model=='llava' else 'Exact-bound historical Full generation; current native logits verified on every question'))
    print(json.dumps(dict(model=model,summary=summary,comparisons=rows),ensure_ascii=False),flush=True)
    return rows


def generation_summary(folder,names,interaction=False):
    samples=read(folder/'sample_manifest.json')['samples']
    records=[];flat=[]
    for s in samples:
        path=folder/'checkpoints'/f'{s["task"]}__{s["question_id"]}.json'
        r=read(path)
        assert r['complete'] and set(r['conditions'])==set(names)
        assert r['image_sha256']==s['image_sha256'] and r['sample_sha256']==digest(s)
        assert r['protocol_sha256']==sha(folder/'protocol.json')
        records.append(r)
        for c in names:
            value=r['conditions'][c]
            assert value['correct']==int(value['prediction']==s['answer'])
            flat.append(dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],answer=s['answer'],
                condition=c,**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in value.items()}))
    assert len(records)==240
    scores=np.array([[r['conditions'][c]['correct'] for c in names] for r in records],float)
    loss=np.array([[r['conditions'][c]['signed_logp_loss'] for c in names] for r in records])
    matrix=np.concatenate([scores*100,(scores-scores[:,:1])*100,loss],axis=1)
    means,low,high,draws,groups=boot_mean(matrix,samples)
    count=len(names);summary=[];taskrows=[]
    for j,c in enumerate(names):
        vals=[r['conditions'][c] for r in records]
        row=dict(condition=c,n=240,image_clusters=groups,accuracy_percent=float(means[j]),
            accuracy_ci_low=float(low[j]),accuracy_ci_high=float(high[j]),delta_full_pp=float(means[count+j]),
            delta_ci_low=float(low[count+j]),delta_ci_high=float(high[count+j]),
            full_correct_to_wrong=int(((scores[:,0]==1)&(scores[:,j]==0)).sum()),
            full_wrong_to_correct=int(((scores[:,0]==0)&(scores[:,j]==1)).sum()),
            mean_full_token_logp_loss=float(means[2*count+j]),loss_ci_low=float(low[2*count+j]),loss_ci_high=float(high[2*count+j]),
            first_token_flip_count=sum(v['top1_flip'] for v in vals),eos=sum(v['eos'] for v in vals),
            truncated=sum(v['truncated'] for v in vals),empty=sum(v['empty'] for v in vals))
        summary.append(row)
        for task in TASKS:
            ids=[i for i,s in enumerate(samples) if s['task']==task]
            taskrows.append(dict(task=task,condition=c,n=len(ids),accuracy_percent=float(scores[ids,j].mean()*100),
                delta_full_pp=float((scores[ids,j]-scores[ids,0]).mean()*100)))
    csv_write(folder/'per_question.csv',flat);csv_write(folder/'accuracy_summary.csv',summary);csv_write(folder/'task_accuracy.csv',taskrows)
    if interaction:
        # Positive values mean the skip is more harmful with the visual relay OFF.
        skip_on=loss[:,1]-loss[:,0];skip_off=loss[:,3]-loss[:,2]
        logp_interaction=skip_off-skip_on
        acc_on=(scores[:,0]-scores[:,1])*100;acc_off=(scores[:,2]-scores[:,3])*100
        analysis=np.stack([skip_on,skip_off,logp_interaction,acc_on,acc_off,acc_off-acc_on],axis=1)
        m,l,h,_,_=boot_mean(analysis,samples)
        labels=['relay_on_skip_logp_harm','relay_off_skip_logp_harm','logp_interaction',
            'relay_on_skip_accuracy_harm_pp','relay_off_skip_accuracy_harm_pp','accuracy_interaction_pp']
        output=[dict(metric=n,mean=float(m[i]),ci_low=float(l[i]),ci_high=float(h[i])) for i,n in enumerate(labels)]
        csv_write(folder/'interaction.csv',output)
        gate=dict(criterion='Predeclared LogP interaction two-sided image-cluster CI strictly above zero',
            mean=float(m[2]),ci_low=float(l[2]),ci_high=float(h[2]),extend_to_other_models=bool(l[2]>0),
            accuracy_interaction_pp=float(m[5]),accuracy_interaction_ci=[float(l[5]),float(h[5])],
            caveat='A positive LogP interaction is not automatically a final-answer accuracy interaction.')
        write(folder/'extension_decision.json',gate)
        print(json.dumps(dict(summary=summary,interaction=output,gate=gate),ensure_ascii=False),flush=True)
    return summary


def historical_llava():
    history=Path('/data/lk/ATE-worktrees/rq2-single-layer-source-knockout/vlm_research/results/rq2_single_layer_source_240')
    samples=[s for s in read(OUT/'sample_manifest.json')['samples'] if s['pilot']]
    lookup={(s['task'],s['question_id']):s for s in samples}
    cube=np.empty((240,32,3));seen=set();order={k:i for i,k in enumerate(lookup)}
    with (history/'per_question.csv').open() as handle:
        for r in csv.DictReader(handle):
            if r['source']!='Image':
                continue
            key=(r['task'],r['question_id']);s=lookup[key]
            assert r['image_sha256']==s['image_sha256']
            i,l=order[key],int(r['layer']);assert (i,l) not in seen;seen.add((i,l))
            loss=float(r['logp_loss'])
            cube[i,l]=[abs(loss),loss,int(r['top1_flip'].lower() in ('1','true'))]
    assert len(seen)==240*32
    rows=[]
    for task in ('Overall',)+TASKS:
        ids=[i for i,s in enumerate(samples) if task=='Overall' or s['task']==task]
        m,l,h,_,groups=boot_mean(cube[ids].reshape(len(ids),-1),[samples[i] for i in ids])
        m,l,h=[v.reshape(32,3) for v in (m,l,h)]
        for layer in range(32):
            row=dict(model='llava_history',cohort='pilot240',task=task,questions=len(ids),image_clusters=groups,layer=layer)
            for j,name in enumerate(('mean_absolute_logp_change','mean_signed_loss','top1_flip_rate')):
                row[name]=float(m[layer,j]);row[name+'_ci_low']=float(l[layer,j]);row[name+'_ci_high']=float(h[layer,j])
            rows.append(row)
    csv_write(OUT/'llava_historical_imageoff_layer_summary.csv',rows)
    write(OUT/'llava_historical_binding.json',dict(source=str(history),commit='3a7c498c722dab7344419490372c2af5dbb67a4d',
        files_sha256={n:sha(history/n) for n in ('per_question.csv','protocol.json','sample_manifest.json')},
        paired_questions=240,rows=7680,scope='Historical P-1 single-layer Image-source OFF only; not o_proj OFF; not new generation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['relay','qwen','internvl','llava_history','llava_rescue','qwen_rescue']);p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    if args.action=='relay':relay_summary()
    elif args.action=='llava_history':historical_llava()
    elif args.action.endswith('_rescue'):rescue_summary(args.action.split('_')[0])
    else:print(json.dumps(scan_model(args.action,args.allow_partial)),flush=True)

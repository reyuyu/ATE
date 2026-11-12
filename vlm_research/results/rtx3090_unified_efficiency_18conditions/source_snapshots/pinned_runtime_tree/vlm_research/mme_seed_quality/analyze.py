"""Full paired quality reconstruction and image-cluster bootstrap (no GPU use)."""
import collections
import csv
import gzip
import numpy as np
import torch
from .common import *


def write_csv(path,rows):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('');return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def percentile(x):
    return [float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]


def seed_boot(rows,native,physical,seed=20261106):
    tasks=sorted({s['task'] for s in rows});clusters=sorted({s['image_sha256'] for s in rows})
    index={h:i for i,h in enumerate(clusters)};num=np.zeros((len(clusters),len(tasks)+1));den=np.zeros_like(num)
    for s,a,b in zip(rows,native,physical):
        k=index[s['image_sha256']];j=tasks.index(s['task'])+1;diff=int(b)-int(a)
        num[k,[0,j]]+=diff;den[k,[0,j]]+=1
    rng=np.random.default_rng(seed);parts=[]
    for offset in range(0,20000,100):
        w=rng.multinomial(len(clusters),np.ones(len(clusters))/len(clusters),size=min(100,20000-offset))
        d=w@den;assert np.all(d>0)
        parts.append(100*(w@num)/d)
    dist=np.concatenate(parts)
    point=100*num.sum(0)/den.sum(0)
    stats=[dict(task=t,delta_pp=float(point[j]),ci_low=percentile(dist[:,j])[0],ci_high=percentile(dist[:,j])[1],
        bootstrap=20000,unit='Accuracy pp',cluster='Image SHA256',ci='exploratory unadjusted 95%')
        for j,t in enumerate(['Overall']+tasks)]
    return stats,dist


def mme_boot(rows,native,physical,seed=20261106):
    tasks=sum(TASKS.values(),[]);clusters=sorted({s['image_sha256'] for s in rows});idx={h:i for i,h in enumerate(clusters)}
    shape=(len(clusters),len(tasks));den=np.zeros(shape);pairsden=np.zeros(shape);n=np.zeros(shape);c=np.zeros(shape)
    npair=np.zeros(shape);cpair=np.zeros(shape);pairgroups=collections.defaultdict(list)
    for s,a,b in zip(rows,native,physical):
        k=idx[s['image_sha256']];j=tasks.index(s['task']);den[k,j]+=1;n[k,j]+=int(a);c[k,j]+=int(b)
        pairgroups[(s['task'],s['pair_id'])].append((s,bool(a),bool(b)))
    for group in pairgroups.values():
        assert len(group)==2 and group[0][0]['image_sha256']==group[1][0]['image_sha256']
        s=group[0][0];k=idx[s['image_sha256']];j=tasks.index(s['task'])
        pairsden[k,j]+=1;npair[k,j]+=all(t[1] for t in group);cpair[k,j]+=all(t[2] for t in group)
    # Preserve the benchmark's task composition while resampling whole SHA
    # clusters. A SHA shared across subtasks is never split between strata.
    strata=collections.defaultdict(list)
    for k in range(len(clusters)):strata[tuple(np.flatnonzero(den[k]))].append(k)
    rng=np.random.default_rng(seed);acc_parts=[];plus_parts=[];score_parts=[]
    for offset in range(0,20000,100):
        batch=min(100,20000-offset);w=np.zeros((batch,len(clusters)),dtype=np.int64)
        for members in strata.values():
            w[:,members]=rng.multinomial(len(members),np.ones(len(members))/len(members),size=batch)
        d=w@den;pd=w@pairsden;assert np.all(d>0) and np.all(pd>0)
        da=100*(w@(c-n))/d;dp=100*(w@(cpair-npair))/pd
        acc_parts.append(da);plus_parts.append(dp);score_parts.append(da+dp)
    da=np.concatenate(acc_parts);dp=np.concatenate(plus_parts);score=np.concatenate(score_parts)
    an=100*n.sum(0)/den.sum(0);ac=100*c.sum(0)/den.sum(0)
    pn=100*npair.sum(0)/pairsden.sum(0);pc=100*cpair.sum(0)/pairsden.sum(0)
    table=[]
    for j,t in enumerate(tasks):
        alo,ahi=percentile(da[:,j]);plo,phi=percentile(dp[:,j]);slo,shi=percentile(score[:,j])
        table.append(dict(task=t,n=int(den[:,j].sum()),images=int(pairsden[:,j].sum()),
            native_accuracy=float(an[j]),c3_accuracy=float(ac[j]),delta_accuracy_pp=float(ac[j]-an[j]),accuracy_ci_low=alo,accuracy_ci_high=ahi,
            native_accuracy_plus=float(pn[j]),c3_accuracy_plus=float(pc[j]),delta_accuracy_plus_pp=float(pc[j]-pn[j]),accuracy_plus_ci_low=plo,accuracy_plus_ci_high=phi,
            native_score=float(an[j]+pn[j]),c3_score=float(ac[j]+pc[j]),delta_score_points=float(ac[j]+pc[j]-an[j]-pn[j]),
            score_ci_low=slo,score_ci_high=shi,unit='Score Points',bootstrap=20000))
    totals=[]
    for label,tt in dict(**TASKS,Total=tasks).items():
        columns=[tasks.index(t) for t in tt];dif=score[:,columns].sum(1);lo,hi=percentile(dif)
        totals.append(dict(task=label,native_score=float((an+pn)[columns].sum()),
            c3_score=float((ac+pc)[columns].sum()),delta_score_points=float(((ac+pc)-(an+pn))[columns].sum()),
            ci_low=lo,ci_high=hi,unit='Score Points',maximum=2800 if label=='Total' else 2000 if label=='Perception' else 800,
            bootstrap=20000,ci='exploratory unadjusted 95%',resampling='Image SHA clusters, stratified by task-membership signature'))
    return table,totals,dict(accuracy=da,accuracy_plus=dp,score=score)


def main():
    samples=read(OUT/'sample_manifest.json');protocol=read(OUT/'protocol.json')
    summary=[];per=[];negative=[];paired=[];coverage=[];index={};allseed=[];allmme=[]
    for kind in ('llava','qwen'):
        for bench,expected in [('MME',2374),('SEED',14233)]:
            rows=[s for s in samples if s['benchmark']==bench];natives=[];physicals=[];data=[]
            missing=[]
            for s in rows:
                results={}
                for condition in ('Native','Physical-C3'):
                    path=OUT/'checkpoints'/kind/s['key']/(condition+'.json')
                    if not path.exists():missing.append(str(path.relative_to(OUT)));continue
                    q=read(path);assert q['sample_sha256']==digest(s)
                    assert q['protocol_sha256']==sha(OUT/'protocol.json')
                    assert q['source_manifest_sha256']==sha(OUT/'source_manifest.json')
                    assert q['image_sha256']==s['image_sha256']
                    result=q['result'];index[str(path.relative_to(ROOT))]=sha(path)
                    if bench=='MME':
                        pred=mme_parse(result['text']);assert pred==result['prediction']
                    else:
                        means=[float(-torch.tensor(lp,dtype=torch.float32).mean()) for lp in result['token_logps']]
                        assert len(means)==4 and np.isfinite(means).all()
                        assert means==result['mean_nll']
                        pred='ABCD'[min(range(4),key=lambda i:(means[i],i))]
                        assert pred==result['prediction']
                    assert bool(result['correct'])==(pred==s['answer'])
                    results[condition]=result
                if len(results)==2:
                    a=results['Native'];b=results['Physical-C3'];natives.append(a['correct']);physicals.append(b['correct'])
                    data.append(dict(model=kind,benchmark=bench,key=s['key'],record_index=s['record_index'],
                        source_question_id=s['source_question_id'],task=s['task'],image_sha256=s['image_sha256'],
                        question=s['question'],gold=s['answer'],native_prediction=a['prediction'],c3_prediction=b['prediction'],
                        native_correct=a['correct'],c3_correct=b['correct'],delta=int(b['correct'])-int(a['correct']),
                        native_text=a.get('text'),c3_text=b.get('text'),native_token_ids=a.get('token_ids'),c3_token_ids=b.get('token_ids'),
                        answer_changed=a['prediction']!=b['prediction'],native_eos=a.get('eos'),c3_eos=b.get('eos'),
                        native_truncated=a.get('truncated'),c3_truncated=b.get('truncated'),
                        native_empty=a.get('empty'),c3_empty=b.get('empty'),
                        native_repeated_3gram=a.get('repeated_3gram'),c3_repeated_3gram=b.get('repeated_3gram'),
                        native_mean_candidate_nll=a.get('mean_nll'),c3_mean_candidate_nll=b.get('mean_nll')))
            coverage.append(dict(model=kind,benchmark=bench,expected_questions=expected,
                complete_paired_questions=len(data),missing_checkpoints=len(missing),
                status='COMPLETE' if not missing else 'STOPPED_OOM' if kind=='qwen' and (OUT/'qwen_stop_summary.json').exists() else 'PARTIAL'))
            per+=data;negative += [r for r in data if r['delta']<0]
            if missing:continue
            assert len(rows)==expected and len(data)==expected
            harm=sum(a and not b for a,b in zip(natives,physicals));recover=sum(b and not a for a,b in zip(natives,physicals))
            common=dict(model=kind,benchmark=bench,questions=len(rows),image_sha_clusters=len({s['image_sha256'] for s in rows}),
                harm=harm,recovered=recover,unchanged_score=len(rows)-harm-recover,
                answer_changes=sum(r['answer_changed'] for r in data))
            if bench=='SEED':
                stats,dist=seed_boot(rows,natives,physicals)
                np.savez_compressed(OUT/f'{kind}_seed_bootstrap.npz',delta_accuracy_pp=dist)
                for d in stats:
                    relevant=list(range(len(rows))) if d['task']=='Overall' else [i for i,s in enumerate(rows) if s['task']==d['task']]
                    d.update(model=kind,n=len(relevant),native_accuracy=100*np.mean([natives[i] for i in relevant]),
                        c3_accuracy=100*np.mean([physicals[i] for i in relevant]),
                        harm=sum(natives[i] and not physicals[i] for i in relevant),
                        recovered=sum(physicals[i] and not natives[i] for i in relevant))
                    allseed.append(d)
                overall=stats[0]
                summary.append(dict(**common,metric='Accuracy',native=overall['native_accuracy'],c3=overall['c3_accuracy'],
                    delta=overall['delta_pp'],ci_low=overall['ci_low'],ci_high=overall['ci_high'],unit='Accuracy pp'))
                paired.append(dict(**common,comparison='Physical-C3 minus Native',**{k:overall[k] for k in ('delta_pp','ci_low','ci_high','unit')}))
            else:
                table,totals,dist=mme_boot(rows,natives,physicals)
                np.savez_compressed(OUT/f'{kind}_mme_bootstrap.npz',**dist)
                allmme += [dict(model=kind,**t) for t in table]
                for t in totals:
                    summary.append(dict(**common,metric=t['task'],native=t['native_score'],c3=t['c3_score'],
                        delta=t['delta_score_points'],ci_low=t['ci_low'],ci_high=t['ci_high'],unit='Score Points'))
                    paired.append(dict(model=kind,benchmark='MME',comparison=t['task']+' Physical-C3 minus Native',**t))
                official_export(kind,rows,data,table)
    write_csv(OUT/'coverage.csv',coverage);write_csv(OUT/'quality_summary.csv',summary)
    write_csv(OUT/'seed_task_accuracy.csv',allseed);write_csv(OUT/'mme_task_scores.csv',allmme)
    write_csv(OUT/'paired_comparisons.csv',paired);write_csv(OUT/'per_question.csv',per)
    write_csv(OUT/'negative_cases.csv',negative)
    write_csv(OUT/'recovered_cases.csv',[r for r in per if r['delta']>0])
    anomalies=[]
    for kind in ('llava','qwen'):
        mm=[r for r in per if r['model']==kind and r['benchmark']=='MME']
        if not mm:continue
        for condition,prefix in [('Native','native'),('Physical-C3','c3')]:
            anomalies.append(dict(model=kind,benchmark='MME',condition=condition,questions=len(mm),
                eos=sum(bool(r[prefix+'_eos']) for r in mm),
                truncated=sum(bool(r[prefix+'_truncated']) for r in mm),
                empty=sum(bool(r[prefix+'_empty']) for r in mm),
                repeated_3gram=sum(bool(r[prefix+'_repeated_3gram']) for r in mm),
                unparsed=sum(r[prefix+'_prediction'] not in ('yes','no') for r in mm)))
    write_csv(OUT/'output_anomalies.csv',anomalies)
    (OUT/'checkpoint_sha256_index.json.gz').write_bytes(gzip.compress(json.dumps(index,sort_keys=True).encode(),mtime=0))
    atomic(OUT/'analysis_validation.json',dict(at=now(),all_read_checkpoint_hashes_verified=True,
        all_saved_scores_reconstructed=True,score_discrepancies=0,coverage=coverage,
        full_pair_matrices=sum(x['status']=='COMPLETE' for x in coverage),expected_full_matrices=4,
        bootstrap=20000,seed=20261106,checkpoint_count=len(index)))
    print('QUALITY ANALYSIS',coverage,summary,flush=True)


def official_export(kind,rows,data,table):
    import subprocess
    import os
    bytask=collections.defaultdict(list)
    for s,d in zip(rows,data):bytask[s['task']].append((s,d))
    for condition,field in [('Native','native_prediction'),('Physical-C3','c3_prediction')]:
        folder=OUT/'official_mme'/kind/condition;folder.mkdir(parents=True,exist_ok=True)
        for task,records in bytask.items():
            groups=collections.defaultdict(list)
            for s,d in records:groups[s['pair_id']].append((s,d))
            with (folder/(task+'.txt')).open('w') as f:
                for pair in groups.values():
                    assert len(pair)==2
                    for s,d in pair:
                        f.write('\t'.join([s['pair_id'],s['question'],s['answer'],d[field]])+'\n')
        tool=OUT/'official_reference/MME_calculation.py'
        env=os.environ.copy()
        dependencies=PRIVATE/'official_eval_deps'
        env['PYTHONPATH']=str(dependencies)+os.pathsep+env.get('PYTHONPATH','')
        p=subprocess.run([__import__('sys').executable,str(tool),'--results_dir',str(folder)],capture_output=True,text=True,env=env)
        assert p.returncode==0,p.stderr
        measured={}
        for line in p.stdout.splitlines():
            if ' score:' in line:
                task,value=line.strip().split(' score:')
                if task.strip()!='total':measured[task.strip()]=float(value)
        assert set(measured)==set(sum(TASKS.values(),[]))
        scorefield='native_score' if condition=='Native' else 'c3_score'
        for taskrow in table:assert abs(measured[taskrow['task']]-taskrow[scorefield])<1e-8
        (folder/'official_output.txt').write_text(p.stdout);(folder/'official_stderr.txt').write_text(p.stderr)


if __name__=='__main__':main()

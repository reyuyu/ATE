"""CPU reconstruction of Qwen-only paired quality, using frozen prior statistics."""
import collections
import gzip
import numpy as np
import torch
from .common import *
from ..mme_seed_quality import analyze as previous
from ..mme_seed_quality.analyze import write_csv,seed_boot,mme_boot


def main():
    samples=read(OUT/'sample_manifest.json');per=[];summary=[];paired=[];coverage=[];seedtable=[];mmetable=[];index={}
    tag=dict(source_manifest_sha256=sha(OUT/'source_manifest.json'),protocol_sha256=sha(OUT/'protocol.json'),
        sample_manifest_sha256=sha(OUT/'sample_manifest.json'),model_manifest_sha256=sha(OUT/'model_manifests.json'))
    for bench,expected in [('MME',2374),('SEED',14233)]:
        rows=[s for s in samples if s['benchmark']==bench];data=[];native=[];physical=[];validrows=[]
        for s in rows:
            result={}
            for condition in ('Native','Physical-C3'):
                path=OUT/'checkpoints'/s['key']/(condition+'.json')
                if not checkpoint_complete(path,tag,s):continue
                q=read(path);r=q['result'];index[str(path.relative_to(OUT))]=sha(path)
                assert q['input_binding']['input_ids'] and q['image_sha256']==s['image_sha256']
                audit=q['execution_audit']['backend']
                assert audit['vision_backend']=='sdpa' and audit['text_backend']=='eager'
                assert not audit['math_sdpa_enabled'] and not audit['cudnn_sdpa_enabled']
                assert sum(audit['efficient_cuda_operators'].values())==27
                if bench=='MME':assert r['prediction']==mme_parse(r['text'])
                else:
                    means=[float(-torch.tensor(x,dtype=torch.float32).mean()) for x in r['token_logps']]
                    assert means==r['mean_nll'] and np.isfinite(means).all()
                    assert r['prediction']=='ABCD'[min(range(4),key=lambda i:(means[i],i))]
                assert bool(r['correct'])==(r['prediction']==s['answer']);result[condition]=r
            if len(result)!=2:continue
            a=result['Native'];b=result['Physical-C3'];native.append(a['correct']);physical.append(b['correct']);validrows.append(s)
            data.append(dict(model='qwen',benchmark=bench,key=s['key'],record_index=s['record_index'],
                source_question_id=s['source_question_id'],task=s['task'],image_sha256=s['image_sha256'],question=s['question'],
                gold=s['answer'],native_prediction=a['prediction'],c3_prediction=b['prediction'],
                native_correct=a['correct'],c3_correct=b['correct'],delta=int(b['correct'])-int(a['correct']),
                answer_changed=a['prediction']!=b['prediction'],native_text=a.get('text'),c3_text=b.get('text'),
                native_token_ids=a.get('token_ids'),c3_token_ids=b.get('token_ids'),native_first_token=a.get('first_token'),c3_first_token=b.get('first_token'),
                native_eos=a.get('eos'),c3_eos=b.get('eos'),native_empty=a.get('empty'),c3_empty=b.get('empty'),
                native_truncated=a.get('truncated'),c3_truncated=b.get('truncated'),
                native_repeated_3gram=a.get('repeated_3gram'),c3_repeated_3gram=b.get('repeated_3gram'),
                native_mean_candidate_nll=a.get('mean_nll'),c3_mean_candidate_nll=b.get('mean_nll')))
        per+=data;complete=len(data)==expected
        coverage.append(dict(benchmark=bench,expected=expected,paired_complete=len(data),status='COMPLETE' if complete else 'PARTIAL'))
        if not complete:continue # No silently changing the population for full-benchmark scores.
        common=dict(benchmark=bench,questions=len(rows),image_sha_clusters=len({s['image_sha256'] for s in rows}),
            harm=sum(a and not b for a,b in zip(native,physical)),recovered=sum(b and not a for a,b in zip(native,physical)),
            answer_changes=sum(d['answer_changed'] for d in data))
        if bench=='SEED':
            stats,dist=seed_boot(rows,native,physical);np.savez_compressed(OUT/'seed_bootstrap.npz',delta_accuracy_pp=dist)
            for d in stats:
                ix=[i for i,s in enumerate(rows) if d['task']=='Overall' or s['task']==d['task']]
                d.update(n=len(ix),native_accuracy=100*np.mean([native[i] for i in ix]),c3_accuracy=100*np.mean([physical[i] for i in ix]),
                    harm=sum(native[i] and not physical[i] for i in ix),recovered=sum(physical[i] and not native[i] for i in ix))
            seedtable+=stats;d=stats[0]
            summary.append(dict(**common,metric='Accuracy',native=d['native_accuracy'],c3=d['c3_accuracy'],delta=d['delta_pp'],
                ci_low=d['ci_low'],ci_high=d['ci_high'],unit='Accuracy pp'))
            paired.append(dict(**common,comparison='Physical-C3 minus Native',**{k:d[k] for k in ('delta_pp','ci_low','ci_high','unit')}))
        else:
            table,totals,dist=mme_boot(rows,native,physical);mmetable+=table
            np.savez_compressed(OUT/'mme_bootstrap.npz',**dist)
            for d in totals:
                summary.append(dict(**common,metric=d['task'],native=d['native_score'],c3=d['c3_score'],delta=d['delta_score_points'],
                    ci_low=d['ci_low'],ci_high=d['ci_high'],unit='Score Points'))
                paired.append(dict(benchmark=bench,comparison=d['task']+' Physical-C3 minus Native',**d))
            previous.OUT=OUT
            previous.official_export('qwen',rows,data,table) # fixed prior PRIVATE retains isolated official CPU dependencies
    write_csv(OUT/'quality_summary.csv',summary);write_csv(OUT/'paired_comparisons.csv',paired)
    write_csv(OUT/'seed_task_accuracy.csv',seedtable);write_csv(OUT/'mme_task_scores.csv',mmetable)
    write_csv(OUT/'coverage.csv',coverage);write_csv(OUT/'per_question.csv',per)
    write_csv(OUT/'negative_cases.csv',[d for d in per if d['delta']<0])
    write_csv(OUT/'recovered_cases.csv',[d for d in per if d['delta']>0])
    anomalies=[]
    for prefix in ('native','c3'):
        mm=[r for r in per if r['benchmark']=='MME']
        anomalies.append(dict(condition=prefix,questions=len(mm),**{k:sum(bool(r[prefix+'_'+k]) for r in mm) for k in ('eos','empty','truncated','repeated_3gram')},
            unparsed=sum(r[prefix+'_prediction'] not in ('yes','no') for r in mm)))
    write_csv(OUT/'output_anomalies.csv',anomalies)
    atomic(OUT/'analysis_validation.json',dict(status='PASSED_RECONSTRUCTION',coverage=coverage,at=now(),
        verified_checkpoints=len(index),all_scores_recomputed=True,score_discrepancies=0))
    (OUT/'checkpoint_sha256_index.json.gz').write_bytes(gzip.compress(json.dumps(index,sort_keys=True).encode(),mtime=0))
    print('ANALYSIS',summary,coverage,flush=True)


if __name__=='__main__':main()

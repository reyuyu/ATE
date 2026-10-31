"""Fixed image-cluster inference; consume every checkpoint exactly once."""
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1.scorers import gqa_prediction
from paper_rq1_single_layer_gqa6.engine import metric_rows
from paper_rq1_early_joint_attention.statistics import bootstrap
from .run import (OUT, ROOT, BASE, GPUS, GPU_INDICES, SEED, PRIMARY, SECONDARY,
                  CONDITIONS, LABELS, previous, read, verify, populations,
                  native_logits, history, checkpoint_valid)

KEYS=['Full',*CONDITIONS]
COMPARISONS=[('C','Full'),*SECONDARY,('B','Full')]


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def collect():
    c=verify();binding=digest(c);manifest=read(OUT/'sample_manifest.json')
    completed=[read(OUT/f'shard{k}/completion.json') for k in range(len(GPUS))]
    assert all(r['status']=='complete' and r['binding']==binding and r['hooks_clean'] and r['parameters_unchanged'] for r in completed)
    assert sum(r['questions'] for r in completed)==2760 and sum(r['generations'] for r in completed)==11040
    paths=list(OUT.glob('shard*/checkpoints/*/*.json'))
    assert len(paths)==11040
    ss=manifest['samples'];flags=manifest['splits'];assign=manifest['assignments']
    records=[];rows=[];transitions=[];images=[set() for _ in GPUS]
    for i,s in enumerate(ss):
        hist=history()[s['question_id']];is_old=flags[s['question_id']]['exploration']
        fullpath=ROOT/hist['full_path'];assert sha256(fullpath)==hist['full_sha256']
        assert sha256(ROOT/hist['first_token_checkpoint'])==hist['first_checkpoint_sha256']
        fullrec=read(fullpath);full=fullrec['generation'];native=native_logits(s,verify_file=True)
        assert fullrec['sample_sha256']==digest(s) and digest(full)==fullrec['result_sha256']
        current={};harmed=[];fixed=[]
        for key in KEYS:
            if key=='Full':
                gen=full;metric=metric_rows(native,native[None])[0];metric['logp_loss']=0.
                origin=hist['full_path'];origin_sha=hist['full_sha256'];kind='historical_Full_3000'
                source_binding=fullrec['binding'];seconds=0.
            else:
                if is_old:
                    old=hist['reused'][key];path=ROOT/old['path'];assert sha256(path)==old['sha256']
                    rec=read(path);raw=previous.OUT/rec['logits_file'];kind='historical_RQ2.2_240'
                    assert rec['sample_sha256']==digest(s) and rec['binding']==digest(read(previous.OUT/'protocol.json'))
                else:
                    rank=assign[s['question_id']];images[rank].add(s['image_sha256'])
                    path=OUT/f"shard{rank}/checkpoints/{s['task']}__{s['question_id']}/{key}.json"
                    rec=checkpoint_valid(path,binding,s,key);raw=OUT/rec['logits_file'];kind='new_validation_2760'
                    assert rec['physical_gpu']==GPU_INDICES[rank]
                gen=rec['generation'];metric=rec['first_token'];seconds=rec['seconds']
                assert digest(gen)==rec['result_sha256'] and digest(metric)==rec['metric_sha256']
                assert gen['masked_layers']==CONDITIONS[key] and gen['decode_native']
                assert rec['hooks_clean'] and rec['inputs_unchanged']
                assert rec.get('parameters_unchanged',rec.get('parameter_versions_unchanged'))
                assert sha256(raw)==rec['logits_sha256']
                with np.load(raw) as z:first=torch.from_numpy(z['first_logits'].copy())
                exact=metric_rows(native,first[None])[0];exact['logp_loss']=-exact['delta_logp']
                assert exact==metric and int(first.argmax())==gen['token_ids'][0]
                origin=str(path.relative_to(ROOT));origin_sha=sha256(path);source_binding=rec['binding']
            assert gen['score']==int(gqa_prediction(gen['text'])==s['answer'])
            assert gen['score']==int(gen['prediction']==s['answer'])
            assert len(gen['token_ids'])==gen['length']<=32 and 2 not in gen['token_ids'][:-1]
            assert gen['eos']==(gen['token_ids'][-1]==2)
            assert gen['truncated']==(gen['length']==32 and not gen['eos'])
            harm=int(full['score']==1 and gen['score']==0);fix=int(full['score']==0 and gen['score']==1)
            if harm:harmed.append(key)
            if fix:fixed.append(key)
            current[key]=dict(generation=gen,metric=metric)
            rows.append(dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],image_sha256=s['image_sha256'],
                **flags[s['question_id']],condition=key,gold=s['answer'],full_text=full['text'],full_prediction=full['prediction'],
                full_token_ids=json.dumps(full['token_ids']),full_score=full['score'],text=gen['text'],prediction=gen['prediction'],
                token_ids=json.dumps(gen['token_ids']),score=gen['score'],delta_score=gen['score']-full['score'],
                correct_to_wrong=harm,wrong_to_correct=fix,full_first_top1=metric['full_token_id'],first_top1=metric['ablated_token_id'],
                first_top1_flip=metric['top1_flip'],full_top1_logp_loss=metric['logp_loss'],js=metric['js'],
                eos=gen['eos'],truncated=gen['truncated'],empty=not bool(gen['text'].strip()),
                abnormal=not gen['eos'] or gen['truncated'] or not bool(gen['text'].strip()),length=gen['length'],
                seconds=seconds,reuse_source=kind,checkpoint=origin,checkpoint_sha256=origin_sha,
                sample_sha256=digest(s),prompt_sha256=hist['prompt_sha256'],source_binding=source_binding,
                model_revision=c['model_revision'],execution_protocol_sha256=sha256(OUT/'protocol.json')))
        records.append(current)
        transitions.append(dict(task=s['task'],question_id=s['question_id'],image_sha256=s['image_sha256'],
                                harmed_conditions=harmed,fixed_conditions=fixed,**flags[s['question_id']]))
        if (i+1)%500==0:print('Validated',i+1,'/3000; no missing or skipped rows',flush=True)
    assert all(not images[a]&images[b] for a in range(len(GPUS)) for b in range(a))
    assert len(rows)==15000 and sum(len(im) for im in images)==2254
    write_csv(OUT/'per_question.csv',rows)
    freeze_json(OUT/'error_transitions.json',transitions)
    return ss,flags,records,rows,completed


def summarize(ss,flags,records):
    c=read(OUT/'protocol.json');scores=np.array([[r[k]['generation']['score'] for k in KEYS] for r in records],dtype=float)
    index={k:i for i,k in enumerate(KEYS)};groups=populations(ss,flags)
    lookup={s['question_id']:i for i,s in enumerate(ss)}
    paired=[];summary=[];tasks=[]
    for groupno,(population,subset) in enumerate(groups.items()):
        positions=[lookup[s['question_id']] for s in subset];y=scores[positions]
        differences=np.stack([100*(y[:,index[a]]-y[:,index[b]]) for a,b in COMPARISONS],axis=-1)
        stats=bootstrap(np.concatenate([differences,100*y],axis=-1),[s['image_sha256'] for s in subset],
            [1,4,4,4,4,1]+[1]*5,repetitions=c['statistics']['repetitions'],seed=SEED+groupno)
        for j,((a,b),st) in enumerate(zip(COMPARISONS,stats[:6])):
            role=('primary_NI' if j==0 else 'secondary_confirmatory' if 1<=j<=4 else 'descriptive') if population==PRIMARY else 'supporting' if population=='validation' else 'descriptive'
            paired.append(dict(population=population,condition=a,reference=b,n=len(subset),images=st['clusters'],role=role,
                delta_pp=st['mean'],ci95_low_pp=st['ci95_low'],ci95_high_pp=st['ci95_high'],
                family95_low_pp=st['family95_low'],family95_high_pp=st['family95_high'],family=st['family'],
                ni_boundary_pp=-1.,lower_strictly_above_minus1=st['ci95_low']>-1. if j==0 else '',
                confirmatory_NI_eligible=population==PRIMARY and j==0,
                correct_to_wrong=int(((y[:,index[b]]==1)&(y[:,index[a]]==0)).sum()),
                wrong_to_correct=int(((y[:,index[b]]==0)&(y[:,index[a]]==1)).sum()),repetitions=st['repetitions'],
                degenerate_bootstrap=st['degenerate_bootstrap']))
        for j,key in enumerate(KEYS):
            g=[records[i][key]['generation'] for i in positions];m=[records[i][key]['metric'] for i in positions]
            delta=100*float((y[:,j]-y[:,0]).mean())
            reference=next((r for r in paired if r['population']==population and r['condition']==key and r['reference']=='Full'),None)
            summary.append(dict(population=population,condition=key,label=LABELS[key],n=len(subset),images=stats[6+j]['clusters'],
                correct=int(y[:,j].sum()),accuracy_pct=100*float(y[:,j].mean()),accuracy_ci95_low=stats[6+j]['ci95_low'],
                accuracy_ci95_high=stats[6+j]['ci95_high'],delta_pp=delta,
                delta_ci95_low=reference['ci95_low_pp'] if reference else 0.,delta_ci95_high=reference['ci95_high_pp'] if reference else 0.,
                correct_to_wrong=int(((y[:,0]==1)&(y[:,j]==0)).sum()),wrong_to_correct=int(((y[:,0]==0)&(y[:,j]==1)).sum()),
                first_top1_flips=sum(v['top1_flip'] for v in m),mean_logp_loss=float(np.mean([v['logp_loss'] for v in m])),
                mean_js=float(np.mean([v['js'] for v in m])),eos=sum(v['eos'] for v in g),
                truncated=sum(v['truncated'] for v in g),empty=sum(not bool(v['text'].strip()) for v in g),
                mean_length=float(np.mean([v['length'] for v in g]))))
        for task in previous.old.TASKS:
            mask=np.array([s['task']==task for s in subset]);selected=y[mask]
            for j,key in enumerate(KEYS):
                tasks.append(dict(population=population,task=task,condition=key,n=int(mask.sum()),correct=int(selected[:,j].sum()),
                    accuracy_pct=100*float(selected[:,j].mean()),delta_pp=100*float((selected[:,j]-selected[:,0]).mean()),
                    correct_to_wrong=int(((selected[:,0]==1)&(selected[:,j]==0)).sum()),
                    wrong_to_correct=int(((selected[:,0]==0)&(selected[:,j]==1)).sum()),role='descriptive'))
        print('Bootstrap complete:',population,len(subset),'questions',flush=True)
    write_csv(OUT/'accuracy_summary.csv',summary);write_csv(OUT/'paired_comparisons.csv',paired)
    write_csv(OUT/'task_accuracy.csv',tasks)
    return summary,paired,tasks


def main():
    lock=(OUT/'aggregation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify()
    tested=read(OUT/'postprocessing_cpu_validation.json')
    assert tested['passed'] and all(sha256(ROOT/n)==h for n,h in tested['source'].items())
    freeze_json(OUT/'postprocessing_binding.json',dict(protocol_sha256=sha256(OUT/'protocol.json'),
        source={n:sha256(ROOT/n) for n in ['rq2_visual_full/analyze.py','rq2_visual_full/presentation.py']},
        cpu_validation_sha256=sha256(OUT/'postprocessing_cpu_validation.json'),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        statistics='Exact frozen primary population, -1pp strict bound, and Bonferroni4 secondary family. No changes from execution protocol.'))
    started=time.perf_counter();ss,flags,records,rows,completed=collect()
    summary,paired,tasks=summarize(ss,flags,records)
    from .presentation import figures,report
    figures(summary,tasks);report(summary,paired,tasks,completed)
    freeze_json(OUT/'validation.json',dict(status='complete',questions=3000,new_questions=2760,new_generations=11040,
        reused_full=3000,reused_interventions=960,rows=15000,cpu_tests=read(OUT/'cpu_validation.json')['tests'],
        all_gpu_smokes_exact=True,all_raw_logits_metrics_replayed_exact=True,no_missing_or_duplicate_conditions=True,
        all_inputs_and_parameter_versions_unchanged=True,all_hooks_clean=True,no_nan_inf_oom=True,
        protocol_sha256=sha256(OUT/'protocol.json'),split_integrity_sha256=sha256(OUT/'split_integrity.json'),
        formal_wall_seconds=max(r['finished_epoch'] for r in completed)-min(r['started_epoch'] for r in completed),
        worker_seconds={str(GPU_INDICES[k]):r['seconds'] for k,r in enumerate(completed)},
        postprocessing_seconds=time.perf_counter()-started))
    print('Complete all3000; unchanged primary NI criterion',flush=True)


if __name__=='__main__':main()

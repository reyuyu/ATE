"""Derived audit of frozen placebo sets; no new inference or selection."""
import json,csv,sys,collections
from pathlib import Path
import numpy as np
P=Path(sys.argv[1]);plan=json.loads((P/'token_plan.json').read_text());rows=list(csv.DictReader((P/'per_question.csv').open()));lookup={(r['question_id'],r['condition']):r for r in rows};summary=[];taskrows=[];aud=[];pairrows=[];rng=np.random.default_rng(20261104)
for base in ['S','SN','N']:
    a=f'{base}+Placebo-D-20261103';b=f'{base}+Placebo-D-20261107';qs=[q for q in plan['pilot_ids'] if plan['cases'][q]['writers'][a] is not None and plan['cases'][q]['writers'][b] is not None];val={q:(int(lookup[q,a]['correct'])+int(lookup[q,b]['correct']))/2 for q in qs}
    summary.append(dict(condition=f'{base}+Placebo-D-mean',n=len(qs),mean_correct=sum(val.values()),accuracy=100*np.mean(list(val.values())),mean_k_R=np.mean([float(lookup[q,a]['k_R']) for q in qs]),identical_seed_sets=sum(plan['cases'][q]['writers'][a]==plan['cases'][q]['writers'][b] for q in qs)))
    for task in sorted({v['sample']['task'] for v in plan['cases'].values()}):
        tt=[q for q in qs if plan['cases'][q]['sample']['task']==task];taskrows.append(dict(condition=f'{base}+Placebo-D-mean',task=task,n=len(tt),mean_correct=sum(val[q] for q in tt),accuracy=100*np.mean([val[q] for q in tt]) if tt else ''))
    for ref in ['Semantic-k','All-Writer']:
        other=np.array([int(lookup[q,ref]['correct']) for q in qs]);aa=np.array([val[q] for q in qs]);d=aa-other;reps=[]
        for _ in range(20):ii=rng.integers(len(qs),size=(1000,len(qs)));reps.extend(100*d[ii].mean(1))
        pairrows.append(dict(comparison=f'{base}+Placebo-D-mean - {ref}',n=len(qs),delta_pp=100*d.mean(),ci_low=np.percentile(reps,2.5),ci_high=np.percentile(reps,97.5),reference_correct_to_wrong=float(np.maximum(other-aa,0).sum()),reference_wrong_to_correct=float(np.maximum(aa-other,0).sum()),bootstrap=20000))
    for q in qs:aud.append(dict(question_id=q,base=base,seed_sets_identical=plan['cases'][q]['writers'][a]==plan['cases'][q]['writers'][b],two_seed_mean_correct=val[q]))
for name,rr in [('placebo_seed_means.csv',summary),('placebo_task_seed_means.csv',taskrows),('placebo_seed_set_audit.csv',aud),('placebo_mean_paired_comparisons.csv',pairrows)]:
    with (P/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
print(summary)

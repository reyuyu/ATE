"""Independent ACC export audit: score complete answer text and exact integer events."""
import csv
import numpy as np
from paper_rq1.scorers import gqa_prediction
from .common import *

def main():
    p=verify();assert read(OUT/'replay_validation.json')['passed'] and read(OUT/'statistics_validation.json')['passed']
    samples=read(OUT/'sample_manifest.json')['samples'];lookup={key(s):i for i,s in enumerate(samples)};cols={c:i for i,c in enumerate(CONDITIONS)}
    scores=np.full((3000,44),np.nan);rows=0
    for part in read(OUT/'per_question_index.json')['partitions']:
        path=OUT/part['path'];assert sha256(path)==part['sha256']
        with path.open() as f:
            for r in csv.DictReader(f):
                sk=r['task']+'__'+r['question_id'];i=lookup[sk];j=cols[r['condition']];s=samples[i]
                assert np.isnan(scores[i,j]) and r['gold']==s['answer'] and r['image_sha256']==s['image_sha256']
                # This uses the complete generated answer, never first-token correctness or Gold NLL.
                prediction=gqa_prediction(r['text']);correct=int(prediction==s['answer'])
                assert prediction==r['prediction'] and correct==int(r['correct']);scores[i,j]=correct;rows+=1
    assert rows==132000 and np.isfinite(scores).all()
    with np.load(OUT/'metrics.npz') as z:assert np.array_equal(scores,z['accuracy'])
    confirm=set(read(OUT/'split_integrity.json')['confirmatory_keys']);populations={
        'all3000_descriptive':list(range(3000)),
        'image_disjoint_confirmatory':[i for i,s in enumerate(samples) if key(s) in confirm],
        **{f'task_{t}':[i for i,s in enumerate(samples) if s['task']==t] for t in TASKS}}
    with (OUT/'condition_summary.csv').open() as f:summary=list(csv.DictReader(f))
    checks=0
    for pop,ix in populations.items():
        a=scores[ix,cols['All-Writer']];base=int(a.sum());N=len(ix)
        for c,j in cols.items():
            r=next(r for r in summary if r['population']==pop and r['condition']==c);v=scores[ix,j];count=int(v.sum())
            harm=int((a*(1-v)).sum());rescue=int(((1-a)*v).sum())
            assert count-base==rescue-harm
            assert float(r['correct_count_or_seed_mean'])==count and int(r['questions'])==N
            assert float(r['accuracy_percent'])==100*(count/N)
            assert float(r['delta_accuracy_pp'])==100*(count/N-base/N)
            assert float(r['All_correct_to_condition_wrong_count_or_seed_mean'])==harm and float(r['All_wrong_to_condition_correct_count_or_seed_mean'])==rescue
            checks+=1
    atomic_json(OUT/'accuracy_export_validation.json',dict(passed=True,binding=digest(p),logical_answers_rescored=132000,actual_conditions=44,
        population_condition_integer_checks=checks,populations=list(populations),all_accuracy_matrix_cells_exact=True,
        full_answer_GQA_accuracy=True,first_token_accuracy_used=False,teacher_forced_NLL_used_as_accuracy=False,
        exact_count_harm_rescue_and_percentage_checks=True,source_sha256=sha256(__file__),cuda_initialized=torch.cuda.is_initialized()))
    assert not torch.cuda.is_initialized();print('FULL ANSWER ACC AUDIT PASS',rows,'answers,',checks,'population-condition checks',flush=True)

if __name__=='__main__':main()

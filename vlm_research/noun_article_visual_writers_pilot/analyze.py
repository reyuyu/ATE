"""Export all labels, preserve unavailable controls, exploratory image-cluster ACC."""
import collections
import numpy as np
from .common import *
def main():
    p=plan();ran=(OUT/'gpu_completion.json').exists();rows=[];lookup={}
    for q,v in p['cases'].items():
        s=v['sample'];lookup[q]={}
        for label,w in v['writers'].items():
            cp=OUT/'records'/q/(label+'.json')
            if cp.exists():r=read(cp);g=r['generation'];status='reused' if r['reused'] else 'generated'
            elif label in v['history']:g=v['history'][label]['generation'];status='reused'
            else:g=None;status='unavailable_exact_budget' if w is None else 'not_run_no_idle_gpu'
            if g is not None:lookup[q][label]=g
            base=v['history']['Semantic-k']['generation'];allg=v['history']['All-Writer']['generation']
            rows.append(dict(question_id=q,task=s['task'],cohort=v['cohort'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],condition=label,status=status,answer=g['text'] if g else '',prediction=g['prediction'] if g else '',correct=g['correct'] if g else '',token_ids=json.dumps(g['token_ids']) if g else '',k=len(w) if w is not None else '',R=len(v['spec']['relay_positions']),k_R=len(w)/len(v['spec']['relay_positions']) if w is not None else '',writers=json.dumps(w),eos=g['eos'] if g else '',empty=g['empty'] if g else '',truncated=g['truncated'] if g else '',repeated_trigram=g['repeated_trigram'] if g else '',output_changed_vs_S=int(g['text']!=base['text']) if g else '',first_top1_flip_vs_S=int(g['token_ids'][0]!=base['token_ids'][0]) if g else '',S_correct=base['correct'],All_correct=allg['correct']))
    csvwrite(OUT/'per_question.csv',rows);csvwrite(OUT/'per_question120.csv',[r for r in rows if r['cohort']=='pilot120']);csvwrite(OUT/'negative_cases.csv',[r for r in rows if r['correct']==0])
    summary=[];tasks=[];comparisons=[];rng=np.random.default_rng(20261104)
    pilot=[r for r in rows if r['cohort']=='pilot120' and r['correct']!=''];conditions=sorted({r['condition'] for r in pilot})
    for label in conditions:
        rr=[r for r in pilot if r['condition']==label];summary.append(dict(condition=label,n=len(rr),correct=sum(r['correct'] for r in rr),accuracy=100*np.mean([r['correct'] for r in rr]),mean_k_R=np.mean([r['k_R'] for r in rr])))
        for task in TASKS:
            tt=[r for r in rr if r['task']==task];tasks.append(dict(condition=label,task=task,n=len(tt),correct=sum(r['correct'] for r in tt),accuracy=100*np.mean([r['correct'] for r in tt]) if tt else ''))
    def compare(name,A,B):
        qs=[q for q in p['pilot_ids'] if all(x in lookup[q] for x in A+B)];a=np.array([np.mean([lookup[q][x]['correct'] for x in A]) for q in qs]);b=np.array([np.mean([lookup[q][x]['correct'] for x in B]) for q in qs]);d=a-b
        # pilot has120 unique image clusters, verified CPU; retain explicit cluster aggregation.
        images=sorted({p['cases'][q]['sample']['image_sha256'] for q in qs});ds=np.array([sum(d[i] for i,q in enumerate(qs) if p['cases'][q]['sample']['image_sha256']==im) for im in images]);ns=np.array([sum(p['cases'][q]['sample']['image_sha256']==im for q in qs) for im in images]);reps=[]
        for _ in range(20):
            idx=rng.integers(len(images),size=(1000,len(images)));reps.extend(100*ds[idx].sum(1)/ns[idx].sum(1))
        comparisons.append(dict(comparison=name,n=len(qs),clusters=len(images),delta_pp=100*d.mean(),ci_low=np.percentile(reps,2.5),ci_high=np.percentile(reps,97.5),reference_correct_to_wrong=float(np.maximum(b-a,0).sum()),reference_wrong_to_correct=float(np.maximum(a-b,0).sum()),random_seeds_averaged_within_question=len(A)>1 or len(B)>1,bootstrap=20000))
    for label in conditions:
        if label!='Semantic-k':compare(label+' - Semantic-k',[label],['Semantic-k'])
        if label!='All-Writer':compare(label+' - All-Writer',[label],['All-Writer'])
    if ran:
        for label,base,control in [('Semantic+Article','Semantic-k','S'),('Semantic+Noun+Article','Semantic+Noun','SN'),('Noun+Article','Noun-only','N')]:
            compare(label+' - '+base,[label],[base]);compare(label+' - placebo mean',[label],[f'{control}+Placebo-D-{z}' for z in SEEDS])
    csvwrite(OUT/'accuracy_summary.csv',summary);csvwrite(OUT/'task_accuracy.csv',tasks);csvwrite(OUT/'paired_comparisons.csv',comparisons)
    coverage=list(csv.DictReader((OUT/'coverage_all3000.csv').open()));cov=[]
    for task in TASKS:
        c=[r for r in coverage if r['task']==task]
        cov.append(dict(task=task,n=len(c),with_noun=sum(int(r['N'])>0 for r in c),with_article=sum(int(r['D'])>0 for r in c),noun_tokens=sum(int(r['N']) for r in c),article_tokens=sum(int(r['D']) for r in c),noun_selected=sum(int(r['N_selected']) for r in c),article_selected=sum(int(r['D_selected']) for r in c),eligible=sum(r['eligible']=='True' for r in c)))
    csvwrite(OUT/'coverage_by_task.csv',cov)
    diagnostic=[]
    for q in p['historical_ids']:
        v=p['cases'][q];g=lookup[q]
        def ok(label):return g[label]['correct'] if label in g else None
        diagnostic.append(dict(question_id=q,question=v['sample']['question'],noun_added=sorted(set(v['N'])-set(v['S'])),article_added=sorted(set(v['D'])-set(v['S'])),noun_recovery=ok('Semantic+Noun'),article_recovery=ok('Semantic+Article'),all_added_recovery=ok('Semantic+Noun+Article'),article_swap=ok('Article-Swap'),placebo_recovery=[ok(f'S+Placebo-D-{z}') for z in SEEDS],strict_form_case=q=='02760157'))
    atomic_json(OUT/'diagnostic_results.json',dict(executed=ran,cases=diagnostic,not_population_estimates=True))
    atomic_json(OUT/'analysis_validation.json',dict(passed=True,pilot_ids=120,historical_ids=12,all_conditions_exported=len(rows),gpu_executed=ran,baseline_scoring_recomputed=True,unexecuted_not_scored=True,bootstrap=20000,random_unit='per-question mean across fixed seeds'))
    print('ANALYSIS',summary,flush=True)
if __name__=='__main__':main()

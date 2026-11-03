"""Describe one unchanged Winner on the historical 3000 pool; no new selection."""
import collections
from .common import *
from .expansion import activate,E
from .deploy import load_selector
from .analyze import bootstrap,exact_p,BASE_MAP
from .cpu_cache import enable

def main():
    started=time.monotonic();activate();enable();selector=load_selector(P);meta=read(E/'manifest.json')['metadata'];initial=read(P/'split_manifest.json');split_of={q:k for k,ss in initial['ids'].items() for q in ss};search_images={initial['metadata'][q]['image_sha256'] for stage in ['DEV','TUNE'] for q in initial['ids'][stage]};oldimages=set(initial['exclusion_images']);final_images={initial['metadata'][q]['image_sha256'] for q in initial['ids']['FINAL']};history=History();rr=[];tasks=[];summaries=[];paired=[]
    for q in sorted(meta):
        sp=spec(q);w=selector(sp);r=score_record(q,read(P/'winner_freeze.json')['candidate_id'],w,history);g=r['generation'];im=meta[q]['image_sha256'];stage=split_of.get(q,'EXPANSION');row=dict(question_id=q,task=meta[q]['task'],image_sha256=im,original_split=stage,question=sp['question'],gold=gold(q),answer=g['text'],prediction=g['prediction'],correct=g['correct'],writers=json.dumps(w),k=len(w),R=len(sp['R']),R_ratio=len(w)/len(sp['R']),token_ids=json.dumps(g['token_ids']),eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],new_image_vs_DEV_TUNE=im not in search_images,new_image_vs_historical_probes=im not in oldimages,new_image_vs_FINAL=im not in final_images)
        for label,short in BASE_MAP.items():
            b=history.get(q,label)['generation'];row.update({short+'_correct':b['correct'],short+'_answer':b['text'],short+'_first_flip':int(g['token_ids'][0]!=b['token_ids'][0]),short+'_harm':int(b['correct'] and not g['correct']),short+'_fix':int(not b['correct'] and g['correct'])})
        rr.append(row)
    history.close();assert len(rr)==3000;writecsv(E/'per_question.csv',rr);atomic_json(E/'historical_audit_index.json',history.files)
    groups={'ALL3000_mixed_historical_pool':rr,'EXPANSION1800':[r for r in rr if r['original_split']=='EXPANSION'],'EXPANSION_clean_images':[r for r in rr if r['original_split']=='EXPANSION' and r['new_image_vs_DEV_TUNE'] and r['new_image_vs_historical_probes'] and r['new_image_vs_FINAL']]}
    for cohort,records in groups.items():
        assert records;labels=[('Winner',lambda r:r['correct'],lambda r:r['R_ratio'])]+[(short,lambda r,s=short:r[s+'_correct'],lambda r,s=short:len(spec(r['question_id'])['historical_writers'][{'All':'All-Writer','Q100':'Semantic-first-Q100','Q80':'Semantic-first-Q80','S':'Semantic-k'}[s]])/r['R']) for short in BASE_MAP.values()]
        for label,score,budget in labels:
            summaries.append(dict(cohort=cohort,condition=label,n=len(records),unique_images=len({r['image_sha256'] for r in records}),correct=sum(score(r) for r in records),ACC=100*np.mean([score(r) for r in records]),mean_R_ratio=np.mean([budget(r) for r in records])))
            for t in TASKS:
                subset=[r for r in records if r['task']==t];tasks.append(dict(cohort=cohort,condition=label,task=t,n=len(subset),correct=sum(score(r) for r in subset),ACC=100*np.mean([score(r) for r in subset])))
        for label,short in BASE_MAP.items():
            delta=np.asarray([r['correct']-r[short+'_correct'] for r in records]);h=int(sum(delta==-1));fix=int(sum(delta==1));lo,hi=bootstrap(delta,[r['image_sha256'] for r in records]);paired.append(dict(cohort=cohort,baseline=label,n=len(records),delta_pp=float(100*delta.mean()),CI95_low_pp=lo,CI95_high_pp=hi,harm=h,fix=fix,first_token_flips=sum(r[short+'_first_flip'] for r in records),answer_changes=sum(r['answer']!=r[short+'_answer'] for r in records),exact_p=exact_p(h,fix),Holm_p='',interpretation='Post-FINAL frozen-Winner extension; exploratory, not a new pre-search external confirmation.'))
        family=[p for p in paired if p['cohort']==cohort];previous=0.
        for j,r in enumerate(sorted(family,key=lambda r:r['exact_p'])):previous=max(previous,min(1.,(4-j)*r['exact_p']));r['Holm_p']=previous
    writecsv(E/'accuracy_summary.csv',summaries);writecsv(E/'paired_comparisons.csv',paired);writecsv(E/'task_accuracy.csv',tasks)
    writecsv(E/'harm_cases.csv',[r for r in rr if any(r[s+'_harm'] for s in BASE_MAP.values())],list(rr[0]));writecsv(E/'recovered_cases.csv',[r for r in rr if any(r[s+'_fix'] for s in BASE_MAP.values())],list(rr[0]))
    atomic_json(E/'analysis_completion.json',dict(passed=True,seconds=time.monotonic()-started,n=3000,Winner_frozen_unchanged=True,bootstrap=20000,seed=20261104,cohort_sizes={k:len(v) for k,v in groups.items()},utc=utc()));event('FROZEN_WINNER_EXPANSION_ANALYZED',seconds=time.monotonic()-started);print(json.dumps(dict(summary=summaries,paired=paired),ensure_ascii=False),flush=True)

if __name__=='__main__':main()

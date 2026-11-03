"""Cluster-valid extension p-values and descriptive observed frontier.

FINAL has one question per image, so its exact McNemar tests are unchanged.
The 3000 pool repeats images: exchange conditions at complete image clusters.
"""
import shutil,collections
from .common import *
from .analyze import rows,BASE_MAP

def cluster_p(records,short):
    sums=collections.Counter()
    for r in records:sums[r['image_sha256']]+=int(r['correct'])-int(r[short+'_correct'])
    values=np.array([v for v in sums.values() if v],dtype=np.int64);observed=abs(values.sum());rng=np.random.default_rng(20261104);more=0
    for _ in range(20):more+=int((np.abs(((rng.integers(0,2,size=(1000,len(values)))*2-1)*values).sum(1))>=observed).sum())
    return (more+1)/20001

def main():
    e=P/'expanded3000';records=rows(e/'per_question.csv');paired=rows(e/'paired_comparisons.csv');attempt=e/'attempts/question_level_p_firstpass';attempt.mkdir(parents=True,exist_ok=True)
    if not (attempt/'paired_comparisons.csv').exists():shutil.copyfile(e/'paired_comparisons.csv',attempt/'paired_comparisons.csv')
    groups={'ALL3000_mixed_historical_pool':records,'EXPANSION1800':[r for r in records if r['original_split']=='EXPANSION'],'EXPANSION_clean_images':[r for r in records if r['original_split']=='EXPANSION' and all(r[k]=='True' for k in ['new_image_vs_DEV_TUNE','new_image_vs_historical_probes','new_image_vs_FINAL'])]}
    for r in paired:
        r.pop('exact_p',None);r['cluster_signflip_p']=cluster_p(groups[r['cohort']],BASE_MAP[r['baseline']]);r['p_method']='Image-cluster paired condition sign-flip; 20000 randomizations, seed 20261104, plus-one p correction.'
    for cohort in groups:
        family=sorted((r for r in paired if r['cohort']==cohort),key=lambda r:r['cluster_signflip_p']);prev=0.
        for j,r in enumerate(family):prev=max(prev,min(1.,(4-j)*r['cluster_signflip_p']));r['Holm_p']=prev
    writecsv(e/'paired_comparisons.csv',paired);summary=rows(e/'accuracy_summary.csv');frontier=[];limits=[]
    for cohort in groups:
        ss=[r for r in summary if r['cohort']==cohort]
        for r in ss:frontier.append(dict(r,Pareto=not any(int(s['correct'])>=int(r['correct']) and float(s['mean_R_ratio'])<=float(r['mean_R_ratio']) and (int(s['correct'])>int(r['correct']) or float(s['mean_R_ratio'])<float(r['mean_R_ratio'])) for s in ss)))
        for limit in [.30,.35,.42,.50]:
            ss=[r for r in summary if r['cohort']==cohort and (float(r['mean_R_ratio'])<limit if limit==.5 else float(r['mean_R_ratio'])<=limit)]
            if ss:
                best=min(ss,key=lambda r:(-int(r['correct']),float(r['mean_R_ratio'])));limits.append(dict(best,R_limit=limit))
    writecsv(e/'budget_accuracy_frontier.csv',frontier);writecsv(e/'budget_limit_best.csv',limits)
    atomic_json(e/'cluster_statistics_validation.json',dict(passed=True,question_level_firstpass_archived=str(attempt.relative_to(ROOT)),correction='The first extension p-values ignored repeated images. Final p-values randomize paired conditions jointly per Image SHA. Bootstrap CIs already resampled image clusters and are unchanged. Original FINAL exact tests remain valid because all 600 images are distinct.',repetitions=20000,seed=20261104,final_paired_csv_sha256=sha(e/'paired_comparisons.csv'),UTC=utc()));print(json.dumps(dict(corrected=True,cohort_sizes={k:len(v) for k,v in groups.items()})),flush=True)

if __name__=='__main__':main()

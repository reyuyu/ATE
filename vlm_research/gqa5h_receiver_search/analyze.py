"""Recompute descriptive and paired statistics from audited complete answers."""
import collections,math,shutil
from .common import *

BASE_MAP={'All-Writer':'All','Question-Only':'Q100','Semantic-first-Q80':'Q80','Semantic-k':'S'}
def rows(path):return list(csv.DictReader(Path(path).open(encoding='utf-8')))
def number(r,key):return float(r[key])
def flag(r,key):return str(r[key]).lower() in ['1','true']

def bootstrap(d,images,seed=20261104):
    """Resample complete image clusters; retain every question in each draw."""
    unique=sorted(set(images));idx={x:i for i,x in enumerate(unique)}
    sums=np.zeros(len(unique));counts=np.zeros(len(unique))
    for x,v in zip(images,d):sums[idx[x]]+=v;counts[idx[x]]+=1
    rng=np.random.default_rng(seed);values=[]
    for _ in range(20):
        draw=rng.integers(0,len(unique),size=(1000,len(unique)));values.append(sums[draw].sum(1)/counts[draw].sum(1))
    lo,hi=np.percentile(np.concatenate(values),[2.5,97.5]);return float(lo*100),float(hi*100)

def exact_p(harm,fix):
    n=harm+fix
    return min(1.,2*sum(math.comb(n,k) for k in range(min(harm,fix)+1))/2**n) if n else 1.

def main():
    started=time.monotonic();freeze=read(P/'winner_freeze.json');winner=freeze['candidate_id'];allrows=[];summary=[];paired=[];task=[]
    jobdirs=sorted((P/'jobs').glob('*/analysis_completion.json'));registry=[];timeline=[]
    for done in jobdirs:
        d=done.parent;m=read(d/'manifest.json');rr=rows(d/'results.csv');kind=m['kind'];first=rr[0]
        timeline.append(dict(job_id=m['job_id'],kind=kind,start_utc=m['utc'],end_utc=read(done)['utc'],logical_conditions=len(rr),new_generations=m['new_generations_requested'],historical_alias_reuse=m['historical_set_reuse'],existing_checkpoint_reuse=m['existing_checkpoint_reuse'],worker_inference_seconds=sum(read(d/f'worker{i}_done.json')['inference_seconds'] for i in range(2)),CPU_summary_seconds=read(done)['seconds'],split=first['split'],n_unique_questions=len({r['question_id'] for r in rr}),source_manifest_sha256=sha(d/'manifest.json')))
        for r in rr:r.update(job_id=m['job_id'],evaluation_kind=kind)
        # LOO and swap interventions are diagnostic labels, not deployable policies.
        if kind=='candidate':allrows.extend(rr)
    for split in ['DEV','TUNE','FINAL']:
        for r in rows(P/'baselines'/split/'per_question.csv'):
            allrows.append(dict(r,candidate_id=r['condition'],job_id='historical-'+split,evaluation_kind='historical_baseline',reused=True,answer=r['answer']))
    by=collections.defaultdict(list)
    for r in allrows:by[(r['split'],r['job_id'],r['candidate_id'])].append(r)
    for (split,job,cid),rr in by.items():
        if len({r['question_id'] for r in rr})!=len(rr):raise AssertionError((job,cid))
        yy=np.array([number(r,'correct') for r in rr]);bb=np.mean([number(r,'R_ratio') for r in rr]);summary.append(dict(split=split,job_id=job,candidate_id=cid,n=len(rr),correct=int(sum(yy)),ACC=float(100*yy.mean()),mean_R_ratio=float(bb),mean_k=float(np.mean([number(r,'k') for r in rr])),comparison_uses_identical_sample=True))
        for t in TASKS:
            ss=[r for r in rr if r['task']==t]
            if ss:task.append(dict(split=split,job_id=job,candidate_id=cid,task=t,n=len(ss),correct=sum(int(r['correct']) for r in ss),ACC=100*np.mean([int(r['correct']) for r in ss]),mean_R_ratio=np.mean([number(r,'R_ratio') for r in ss])))
        if rr[0]['evaluation_kind']=='candidate' and (len(rr)==360 or split!='DEV'):
            for base,short in BASE_MAP.items():
                d=np.array([int(r['correct'])-int(r[short+'_correct']) for r in rr]);h=sum(d==-1);f=sum(d==1);lo,hi=bootstrap(d,[r['image_sha256'] for r in rr]);paired.append(dict(split=split,job_id=job,candidate_id=cid,baseline=base,n=len(rr),delta_pp=float(100*d.mean()),CI95_low_pp=lo,CI95_high_pp=hi,harm=int(h),fix=int(f),answer_changes=sum(r['answer']!=r[short+'_answer'] for r in rr),first_token_flips=sum(int(r[short+'_first_flip']) for r in rr),exact_p=exact_p(int(h),int(f)),Holm_p='',bootstrap=20000,CI_scope='exploratory_pointwise_unadjusted',cluster='Image SHA256'))
    final=[r for r in paired if r['split']=='FINAL'];assert len(final)==4 and all(r['candidate_id']==winner for r in final)
    prev=0.
    for j,r in enumerate(sorted(final,key=lambda r:r['exact_p'])):
        prev=max(prev,min(1.,(4-j)*r['exact_p']));r['Holm_p']=prev
    for split,name in [('DEV','dev_results.csv'),('TUNE','tune_results.csv'),('FINAL','final_holdout_results.csv')]:writecsv(P/name,[r for r in summary if r['split']==split])
    writecsv(P/'accuracy_summary.csv',summary);writecsv(P/'paired_comparisons.csv',paired);writecsv(P/'task_accuracy.csv',task);writecsv(P/'experiment_timeline.csv',timeline)
    # Heterogeneous diagnostic fields stay in their per-job originals.
    fields=sorted(set().union(*(set(r) for r in allrows)));writecsv(P/'per_question.csv',[{f:r.get(f,'') for f in fields} for r in allrows],fields)
    novel=[r for r in allrows if r['evaluation_kind']=='candidate'];hfields=sorted(set().union(*(set(r) for r in novel)))
    writecsv(P/'harm_cases.csv',[{f:r.get(f,'') for f in hfields} for r in novel if any(int(r[s+'_harm']) for s in BASE_MAP.values())],hfields)
    writecsv(P/'recovered_cases.csv',[{f:r.get(f,'') for f in hfields} for r in novel if any(int(r[s+'_fix']) for s in BASE_MAP.values())],hfields)
    for d in sorted((P/'candidates').iterdir()):
        definition=read(d/'definition.json');assert sha(d/'selection_source.py')==definition['source_sha256'];evaluations=[r for r in summary if r['candidate_id']==d.name]
        registry.append(dict(candidate_id=d.name,family=definition['config']['family'],hypothesis=definition['hypothesis'],config=json.dumps(definition['config'],sort_keys=True),source_sha256=definition['source_sha256'],created_utc=definition['created_utc'],evaluations=json.dumps(evaluations),status='frozen_FINAL_winner' if d.name==winner else 'TUNE_evaluated_not_winner' if any(r['split']=='TUNE' for r in evaluations) else 'DEV_only' if evaluations else 'CPU_defined_not_new_VLM_evaluated',not_selected_reason='Single frozen winner' if d.name==winner else 'See DEV stage comparisons and pre-outcome tune_shortlist.json; no FINAL retuning.'))
    writecsv(P/'candidate_registry.csv',registry)
    frontier=[]
    for split,n in [('DEV',360),('TUNE',240),('FINAL',600)]:
        ss=[r for r in summary if r['split']==split and r['n']==n]
        for r in ss:
            dominated=any(s['correct']>=r['correct'] and s['mean_R_ratio']<=r['mean_R_ratio'] and (s['correct']>r['correct'] or s['mean_R_ratio']<r['mean_R_ratio']) for s in ss)
            frontier.append(dict(r,Pareto=not dominated,frontier_cohort=f'{split}{n}',role='selection' if split=='TUNE' else 'single_frozen_Winner_vs_baselines' if split=='FINAL' else 'exploratory_search_or_training'))
    writecsv(P/'budget_accuracy_frontier.csv',frontier)
    limits=[]
    for split in ['DEV','TUNE','FINAL']:
        for limit in [.30,.35,.42,.50]:
            choices=[r for r in frontier if r['split']==split and (r['mean_R_ratio']<limit if limit==.5 else r['mean_R_ratio']<=limit)]
            if choices:
                best=min(choices,key=lambda r:(-r['correct'],r['mean_R_ratio'],r['candidate_id']));limits.append(dict(split=split,R_limit=limit,**{k:best[k] for k in ['candidate_id','n','correct','ACC','mean_R_ratio']},interpretation='Descriptive best among methods actually evaluated on this split; FINAL did not evaluate all DEV candidates.'))
    writecsv(P/'budget_limit_best.csv',limits)
    token=[]
    for r in novel:
        w=json.loads(r['writers']);s=spec(r['question_id']);assert len(w)==int(r['k']) and w==sorted(set(w)) and set(w)<=set(s['R'])
        from semantic_visual_writers.groups import intact
        assert intact(w,s['groups'])
        token.append(dict(split=r['split'],job_id=r['job_id'],candidate_id=r['candidate_id'],question_id=r['question_id'],k=len(w),R=len(s['R']),R_ratio=len(w)/len(s['R']),Q_selected=len(set(w)&set(s['Q'])),non_Q_selected=len(set(w)-set(s['Q'])),writers=json.dumps(w),whole_group=True,image_or_Pminus1_selected=False))
    writecsv(P/'token_selection_audit.csv',token)
    # Reproducible scientific plots use observations only, never a smooth fit.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
    for ax,split in zip(axs,['DEV','TUNE','FINAL']):
        ss=[r for r in frontier if r['split']==split];ax.scatter([100*r['mean_R_ratio'] for r in ss],[r['ACC'] for r in ss],s=25,c='#8996a5',label='Other observed policies')
        pp=sorted((r for r in ss if r['Pareto']),key=lambda r:r['mean_R_ratio']);ax.plot([100*r['mean_R_ratio'] for r in pp],[r['ACC'] for r in pp],'o-',color='#2260a0',label='Observed Pareto frontier')
        for r in ss:
            if r['candidate_id'] in [winner,'Semantic-k','Semantic-first-Q80','Question-Only','All-Writer']:ax.annotate(r['candidate_id'].replace('q80-late-R-group','Q80 + late ASSISTANT'),(100*r['mean_R_ratio'],r['ACC']),fontsize=7,xytext=(3,4),textcoords='offset points')
        ax.axvline(35,color='#aaa',ls=':',lw=1);ax.axvline(50,color='#aaa',ls='--',lw=1);ax.set_title(split+(' (selection)' if split=='TUNE' else ' (frozen)' if split=='FINAL' else ' (exploration)'));ax.set_xlabel('Mean open R queries (%)');ax.grid(alpha=.2)
    axs[0].set_ylabel('Complete-answer accuracy (%)');fig.tight_layout();fig.savefig(P/'budget_accuracy_frontier.png',dpi=220);fig.savefig(P/'budget_accuracy_frontier.pdf');plt.close(fig)
    atomic_json(P/'statistics_completion.json',dict(passed=True,seconds=time.monotonic()-started,paired_rows=len(paired),winner=winner,FINAL=final,bootstrap=20000,seed=20261104,pointwise_intervals=True,exact_p_Holm_family=4,utc=utc()))
    event('STATISTICS_COMPLETE',seconds=time.monotonic()-started);print(json.dumps(final),flush=True)

if __name__=='__main__':main()

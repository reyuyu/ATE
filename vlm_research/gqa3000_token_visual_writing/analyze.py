"""Full checkpoint replay, question-weighted image-cluster bootstrap and fixed tests."""
import argparse,collections,csv,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from paper_rq1.scorers import gqa_prediction
from query_role_formal.common import repeated_trigram
from .common import *
from .worker import spec_for,paths,validate_saved

DERIVED={
    'Question-Random-k mean3':[f'Question-Random-k-{s}' for s in SEEDS],
    'R-Random-k mean3':[f'R-Random-k-{s}' for s in SEEDS],
    'Question-Stratified-k mean2':[f'Question-Stratified-k-{s}' for s in SEEDS[:2]],
    **{f'{source}-Random mean2-Q{b}':[f'{source}-Random-{s}-Q{b}' for s in SEEDS[:2]] for b in BUDGETS for source in ['Question','R']}}
ANALYSIS_CONDITIONS=CONDITIONS+list(DERIVED)

def write_csv(path,rows):
    rows=iter(rows);first=next(rows,None);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),path
    with path.open('w',encoding='utf-8',newline='') as f:
        if first is None:return
        w=csv.DictWriter(f,fieldnames=list(first));w.writeheader();w.writerow(first);w.writerows(rows)

def cluster_bootstrap(values,images,repetitions=20000,seed=BOOTSTRAP_SEED):
    """Resample image identities; retain every question in each sampled image."""
    values=np.asarray(values,dtype=np.float64);unique,inverse=np.unique(images,return_inverse=True)
    sums=np.zeros((len(unique),values.shape[1]),dtype=float);np.add.at(sums,inverse,values)
    counts=np.bincount(inverse).astype(float);rng=np.random.default_rng(seed);draws=[]
    for start in range(0,repetitions,128):
        ids=rng.integers(0,len(unique),size=(min(128,repetitions-start),len(unique)))
        # A count matrix avoids a large [replicate,image,metric] temporary.
        w=np.zeros((len(ids),len(unique)),dtype=float)
        np.add.at(w,(np.arange(len(ids))[:,None],ids),1)
        draws.append((w@sums)/(w@counts)[:,None])
    return np.concatenate(draws)

def interval(a):return tuple(float(v) for v in np.percentile(a,[2.5,97.5]))

def seed_events(left,right,ix,a,cols):
    ls=DERIVED.get(left,[left]);rs=DERIVED.get(right,[right])
    if len(ls)==1:ls=ls*len(rs)
    if len(rs)==1:rs=rs*len(ls)
    assert len(ls)==len(rs),'Seed pairing must be explicitly specified'
    harm=[];rescue=[]
    for l,r in zip(ls,rs):
        lv=a[ix,cols[l]];rv=a[ix,cols[r]]
        harm.append((rv*(1-lv)).sum());rescue.append(((1-rv)*lv).sum())
    return float(np.mean(harm)),float(np.mean(rescue))

def check_generation(g,s,spec,tok):
    ids=g['token_ids'];n=len(s['input_mapping']['prompt_ids'])
    assert 1<=len(ids)<=32 and g['length']==len(ids) and g['own_cache'] and g['own_prefix'] and g['logits_to_keep']==1
    assert g['text']==tok.decode(ids,skip_special_tokens=True).strip() and g['prediction']==gqa_prediction(g['text'])
    assert g['correct']==int(g['prediction']==s['answer']) and g['eos']==(ids[-1]==2)
    assert g['empty']==(not bool(g['text'])) and g['truncated']==(len(ids)==32 and ids[-1]!=2) and g['repeated_trigram']==repeated_trigram(ids)
    assert g['query_layout']==layout(s) and g['writer_positions'] in list(spec['writers'].values())
    assert len(g['cache_lengths'])==len(ids)
    for t,step in enumerate(g['cache_lengths']):
        assert step==dict(t=t,before=n+t-1 if t else 0,after=n+t,absolute_query=n+t-1,consumed_token=ids[t-1] if t else None)
    assert g['audit']['steps']==len(ids) and g['audit']['calls']==32*len(ids)
    assert g['audit']['prefill_blocked_per_head']==576*(len(spec['relay_positions'])-len(g['writer_positions'])+1)
    assert g['audit']['decode_blocked_per_head']==576 and not g['audit']['noop']

def replay():
    started=time.perf_counter();torch.set_num_threads(4);p=verify();assert_no_stop();assert not (OUT/'replay_validation.json').exists()
    for rank in range(2):
        c=read(OUT/f'shard{rank}/completion.json');assert c['passed'] and c['binding']==digest(p)
    samples=read(OUT/'sample_manifest.json')['samples'];index=read(OUT/'selection_index.json');gold=read(OUT/'gold_manifest.json');hashes=read(OUT/'input_hashes.json');shards=read(OUT/'shards.json')['image_assignments']
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);N=len(samples);C=len(CONDITIONS)
    accuracy=np.empty((N,C));nll=np.empty((N,C));ratio=np.empty((N,C));firstflip=np.empty((N,C));anomaly=np.zeros((N,C,4),dtype=int)
    rowhash={};rawhash={};newcount=0;logical=0;imagehash={};total_bytes=0;csvparts=[];swaps={k for pair in read(OUT/'image_swap_manifest.json')['pairs'] for k in pair['sample_keys']}
    # Partition row tables by the already frozen task, keeping individual GitHub files <100MB.
    writers={};files={};failurefiles={};negativefiles={}
    for task in TASKS:
        path=OUT/'per_question'/(task+'.csv');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();files[task]=path.open('w',encoding='utf-8',newline='')
    harmfile=(OUT/'accuracy_harm_cases.csv').open('x',encoding='utf-8',newline='');allbad=(OUT/'all_incorrect_cases.csv').open('x',encoding='utf-8',newline='');harmwriter=None;badwriter=None
    try:
        for i,s in enumerate(samples):
            sk=key(s);rank=shards[s['image_sha256']];spec=spec_for(s,index);records={};first={}
            if s['image_path'] not in imagehash:imagehash[s['image_path']]=sha256(s['image_path'])
            assert imagehash[s['image_path']]==s['image_sha256']
            actual_files=list((OUT/f'shard{rank}/checkpoints'/sk).glob('*.json'));assert len(actual_files)==len(spec['unique'])
            for uid in spec['unique']:
                path,_=paths(s,uid,rank);r=read(path);validate_saved(r,s,spec,uid,p,rank,gold,hashes);g=r['generation'];tf=r['teacher_forced']
                check_generation(g,s,spec,tok);assert tf['own_cache'] and tf['own_gold_prefix'] and not tf['eos_in_primary_nll'] and tf['logits_to_keep']==1
                assert tf['audit']['steps']==len(gold[sk]['token_ids']) and tf['audit']['calls']==32*len(gold[sk]['token_ids'])
                assert tf['audit']['prefill_blocked_per_head']==g['audit']['prefill_blocked_per_head']
                for t,step in enumerate(tf['cache_lengths']):
                    n=len(s['input_mapping']['prompt_ids'])
                    assert step==dict(t=t,before=n+t-1 if t else 0,after=n+t,absolute_query=n+t-1,consumed_token=gold[sk]['token_ids'][t-1] if t else None,gold_target_id=gold[sk]['token_ids'][t])
                with np.load(r['raw']['path'],allow_pickle=False) as z:
                    first[uid]=z['first_logits'].copy()
                    for metric,label in [(g,'greedy_audit'),(tf,'gold_audit')]:
                        a=z[label];assert len(a)==metric['audit']['calls']
                        assert [float(a[k::32,0].max()) for k in range(32)]==metric['audit']['forbidden_max_by_layer']
                        assert [float(a[k::32,1].max()) for k in range(32)]==metric['audit']['row_sum_error_by_layer']
                    if sk in swaps and uid==spec['aliases']['No-Writer']:assert len(z['greedy_valid_logits'])==g['length']
                rowhash[str(path.relative_to(OUT))]=sha256(path);rawhash[r['raw']['path']]=r['raw']['sha256'];records[uid]=r;newcount+=1;total_bytes+=r['raw']['bytes']
            ref=records[spec['aliases']['All-Writer']];ag=ref['generation'];afirst=first[spec['aliases']['All-Writer']]
            for j,c in enumerate(CONDITIONS):
                uid=spec['aliases'][c];r=records[uid];g=r['generation'];tf=r['teacher_forced'];k=len(spec['writers'][c]);q=len(spec['question_positions']);R=len(spec['relay_positions'])
                accuracy[i,j]=g['correct'];nll[i,j]=tf['nll'];ratio[i,j]=k/R;firstflip[i,j]=int(g['token_ids'][0]!=ag['token_ids'][0])
                anomaly[i,j]=[not g['eos'],g['empty'],g['truncated'],g['repeated_trigram']]
                budget=spec['budgets'][c.rsplit('-Q',1)[1]] if '-Q' in c else None
                row=dict(question_id=s['question_id'],task=s['task'],image_sha256=s['image_sha256'],question=s['question'],gold=s['answer'],condition=c,
                    text=g['text'],prediction=g['prediction'],token_ids=json.dumps(g['token_ids']),correct=g['correct'],All_Writer_text=ag['text'],All_correct=ag['correct'],
                    correct_to_wrong=int(ag['correct'] and not g['correct']),wrong_to_correct=int(not ag['correct'] and g['correct']),
                    gold_token_ids=json.dumps(tf['gold_token_ids']),per_token_gold_nll=json.dumps(tf['per_token_nll']),gold_nll=tf['nll'],delta_gold_nll=tf['nll']-ref['teacher_forced']['nll'],
                    first_top1_flip=int(firstflip[i,j]),sequence_changed=int(g['token_ids']!=ag['token_ids']),length=g['length'],eos=g['eos'],empty=g['empty'],truncated=g['truncated'],repeated_trigram=g['repeated_trigram'],
                    writer_count=k,question_token_count=q,R_token_count=R,R_connection_ratio=k/R,question_budget_ratio=k/q,natural_semantic_k=spec['natural']['k'],
                    nominal_question_budget_percent=budget['nominal_percent'] if budget else '',target_floor_k=budget['target_floor_tokens'] if budget else '',budget_shortfall=budget['token_shortfall'] if budget else '',
                    writer_positions=json.dumps(spec['writers'][c]),canonical_id=uid,within_question_aliases=json.dumps(r['labels']),source_checkpoint=str(paths(s,uid,rank)[0].relative_to(OUT)),
                    raw_logits_path=r['raw']['path'],raw_logits_sha256=r['raw']['sha256'],selection_binding=spec['selection_binding'],rank=rank)
                if s['task'] not in writers:writers[s['task']]=csv.DictWriter(files[s['task']],fieldnames=list(row));writers[s['task']].writeheader()
                writers[s['task']].writerow(row)
                if row['correct_to_wrong']:
                    if harmwriter is None:harmwriter=csv.DictWriter(harmfile,fieldnames=list(row));harmwriter.writeheader()
                    harmwriter.writerow(row)
                if not g['correct']:
                    if badwriter is None:badwriter=csv.DictWriter(allbad,fieldnames=list(row));badwriter.writeheader()
                    badwriter.writerow(row)
                logical+=1
            if (i+1)%100==0:print('REPLAY',i+1,'/3000','canonical',newcount,'seconds',round(time.perf_counter()-started),flush=True)
    finally:
        for f in files.values():f.close()
        harmfile.close();allbad.close()
    assert newcount==p['unique_evaluations'] and logical==p['logical_evaluations']
    for rank in range(2):assert len(list((OUT/f'shard{rank}/checkpoints').glob('*/*.json')))==read(OUT/f'shard{rank}/completion.json')['canonical_evaluations']
    for task in TASKS:
        path=OUT/'per_question'/(task+'.csv');assert path.stat().st_size<95_000_000
        csvparts.append(dict(task=task,path=str(path.relative_to(OUT)),rows=500*C,sha256=sha256(path),bytes=path.stat().st_size))
    atomic_json(OUT/'per_question_index.json',dict(total_rows=logical,partitions=csvparts))
    atomic_npz(OUT/'metrics.npz',accuracy=accuracy,nll=nll,ratio=ratio,firstflip=firstflip,anomalies=anomaly)
    atomic_json(OUT/'checkpoint_hashes.json',rowhash);atomic_json(OUT/'raw_logits_hashes.json',rawhash)
    atomic_json(OUT/'replay_validation.json',dict(passed=True,binding=digest(p),canonical_evaluations=newcount,logical_records=logical,raw_bytes=total_bytes,
        every_raw_sha256_verified=True,every_saved_gold_nll_recomputed_exactly=True,every_actual_forbidden_probability_zero=True,all_non_target_masks_exact=True,
        all_generated_tokens_decoded_and_rescored=True,all_gold_positions_and_cache_steps_exact=True,source_and_input_hashes_exact=True,
        raw_logits_in_GitHub=False,raw_logits_server_root=str(RAW),seconds=time.perf_counter()-started))

def matrices():
    with np.load(OUT/'metrics.npz') as z:a=z['accuracy'];n=z['nll'];r=z['ratio'];f=z['firstflip']
    cols={c:i for i,c in enumerate(CONDITIONS)}
    aa=[a];nn=[n];rr=[r];ff=[f]
    for c,names in DERIVED.items():
        ix=[cols[v] for v in names];aa.append(a[:,ix].mean(1,keepdims=True));nn.append(n[:,ix].mean(1,keepdims=True));rr.append(r[:,ix].mean(1,keepdims=True));ff.append(f[:,ix].mean(1,keepdims=True))
    return tuple(np.concatenate(v,axis=1) for v in [aa,nn,rr,ff])

def statistics():
    torch.set_num_threads(4);p=verify();assert read(OUT/'replay_validation.json')['passed'];a,n,r,f=matrices();C=len(ANALYSIS_CONDITIONS);cols={c:i for i,c in enumerate(ANALYSIS_CONDITIONS)};ai=cols['All-Writer']
    with np.load(OUT/'metrics.npz') as z:flags=z['anomalies'].astype(float)
    flagsets=[flags]
    for names in DERIVED.values():flagsets.append(flags[:,[cols[c] for c in names]].mean(1,keepdims=True))
    flags=np.concatenate(flagsets,axis=1);assert flags.shape==(3000,C,4)
    samples=read(OUT/'sample_manifest.json')['samples'];confirm=set(read(OUT/'split_integrity.json')['confirmatory_keys'])
    populations={'all3000_descriptive':list(range(3000)),'image_disjoint_confirmatory':[i for i,s in enumerate(samples) if key(s) in confirm]}
    populations.update({f'task_{t}':[i for i,s in enumerate(samples) if s['task']==t] for t in TASKS})
    summary=[];paired=[];decisions={};bootstrap_index={};budgetrows=[]
    contrasts=[(c,'All-Writer','vs_All') for c in ANALYSIS_CONDITIONS if c!='All-Writer']
    contrasts += [('Semantic-k',b,'natural_selection') for b in ['Question-Random-k mean3','R-Random-k mean3','Question-Stratified-k mean2','Question-Last-k','R-Last-k','Drop-Semantic']]
    for b in BUDGETS:
        contrasts += [(f'Semantic-first-Q{b}',base,'matched_curve_budget') for base in [f'Question-Last-Q{b}',f'Question-Random mean2-Q{b}',f'R-Random mean2-Q{b}']]
    for policy in ['Semantic-first','Question-Last','Question-Random mean2','R-Random mean2']:
        contrasts += [(f'{policy}-Q{b}',f'{policy}-Q{prior}','successive_nominal_budgets_descriptive_not_token_marginal') for prior,b in zip(BUDGETS,BUDGETS[1:])]
    for pop,ix in populations.items():
        images=[samples[i]['image_sha256'] for i in ix];vals=np.concatenate([a[ix],n[ix]],axis=1);draw=cluster_bootstrap(vals,images);means=vals.mean(0)
        info=atomic_npz(OUT/'bootstrap'/(pop+'.npz'),means=draw);bootstrap_index[pop]=dict(**info,conditions=ANALYSIS_CONDITIONS,columns='Accuracy fraction first, Gold NLL second',questions=len(ix),image_clusters=len(set(images)))
        for c,j in cols.items():
            accci=interval(100*draw[:,j]);nllci=interval(draw[:,C+j]);daci=interval(100*(draw[:,j]-draw[:,ai]));dnci=interval(draw[:,C+j]-draw[:,C+ai]);correct=a[ix,j]
            row=dict(population=pop,condition=c,questions=len(ix),image_clusters=len(set(images)),correct_count_or_seed_mean=float(correct.sum()),accuracy_percent=float(100*means[j]),accuracy_ci_low=accci[0],accuracy_ci_high=accci[1],
                delta_accuracy_pp=float(100*(means[j]-means[ai])),delta_accuracy_ci_low=daci[0],delta_accuracy_ci_high=daci[1],
                gold_nll_nat_per_token=float(means[C+j]),gold_nll_ci_low=nllci[0],gold_nll_ci_high=nllci[1],delta_gold_nll=float(means[C+j]-means[C+ai]),delta_nll_ci_low=dnci[0],delta_nll_ci_high=dnci[1],
                All_correct_to_condition_wrong_count_or_seed_mean=float((a[ix,ai]*(1-correct)).sum()),All_wrong_to_condition_correct_count_or_seed_mean=float(((1-a[ix,ai])*correct).sum()),
                first_token_flip_count_or_seed_mean=float(f[ix,j].sum()),actual_R_budget_mean=float(r[ix,j].mean()),seed_average_within_question=c in DERIVED,
                non_EOS_count_or_seed_mean=float(flags[ix,j,0].sum()),empty_count_or_seed_mean=float(flags[ix,j,1].sum()),truncated_count_or_seed_mean=float(flags[ix,j,2].sum()),repeated_trigram_count_or_seed_mean=float(flags[ix,j,3].sum()),
                ci_role='Ordinary pointwise descriptive95%; primary decisions separately Bonferroni one-sided family')
            summary.append(row)
            if '-Q' in c or c in ['All-Writer','No-Writer']:budgetrows.append(row)
        for left,right,kind in contrasts:
            l=cols[left];rr=cols[right];ad=100*(a[ix,l]-a[ix,rr]);nd=n[ix,l]-n[ix,rr];aci=interval(100*(draw[:,l]-draw[:,rr]));nci=interval(draw[:,C+l]-draw[:,C+rr])
            harm,rescue=seed_events(left,right,ix,a,cols)
            paired.append(dict(population=pop,left=left,right=right,kind=kind,questions=len(ix),image_clusters=len(set(images)),accuracy_delta_pp=float(ad.mean()),accuracy_ci_low=aci[0],accuracy_ci_high=aci[1],
                delta_gold_nll=float(nd.mean()),nll_ci_low=nci[0],nll_ci_high=nci[1],right_correct_to_left_wrong_count_or_seed_mean=harm,right_wrong_to_left_correct_count_or_seed_mean=rescue,
                repetitions=20000,seed=BOOTSTRAP_SEED,ci='Exploratory pointwise95%, whole-image paired, question-weighted'))
        if pop=='image_disjoint_confirmatory':
            sem=cols['Semantic-k'];qr=cols['Question-Random-k mean3'];decisions=confirmatory_decisions(draw[:,C+sem]-draw[:,C+qr],100*(draw[:,sem]-draw[:,ai]))
            decisions.update(population=pop,questions=len(ix),image_clusters=len(set(images)),primary_NLL_difference=float((n[ix,sem]-n[ix,qr]).mean()),primary_accuracy_difference_pp=float(100*(a[ix,sem]-a[ix,ai]).mean()),
                question_random_control_seeds=SEEDS,random_averaging='Within each question before resampling',NLL_test='Semantic-k minus Question-only Random-k mean3, upper97.5% < 0',
                accuracy_test='Semantic-k minus All-Writer, lower2.5% > -2pp',multiplicity=p['statistics']['multiplicity'],accuracy_minus1_is_sensitivity_only=True)
        print('STATISTICS COMPLETE',pop,len(ix),len(set(images)),flush=True)
    write_csv(OUT/'condition_summary.csv',summary);write_csv(OUT/'paired_comparisons.csv',paired);write_csv(OUT/'budget_curve.csv',budgetrows)
    write_csv(OUT/'task_accuracy_nll.csv',[v for v in summary if v['population'].startswith('task_')])
    atomic_json(OUT/'confirmatory_tests.json',decisions);atomic_json(OUT/'bootstrap_index.json',bootstrap_index)
    atomic_json(OUT/'statistics_validation.json',dict(passed=True,binding=digest(p),all_populations=list(populations),repetitions=20000,seed=BOOTSTRAP_SEED,
        pointwise_CI_not_simultaneous=True,no_9000_independent_questions=True,confirmatory_tests_predeclared=True,all44_labels_retained=True))

def distribution(left,right):
    a=torch.from_numpy(left.copy()).double().log_softmax(-1);b=torch.from_numpy(right.copy()).double().log_softmax(-1);m=torch.logaddexp(a,b)-math.log(2)
    exact=torch.equal(torch.from_numpy(left),torch.from_numpy(right))
    return dict(first_logits_exact=exact,first_logits_max_abs=float(np.abs(left.astype(float)-right.astype(float)).max()),
        first_JS_nat=0. if exact else float(.5*((a.exp()*(a-m)).sum()+(b.exp()*(b-m)).sum())),first_TV=0. if exact else float(.5*(a.exp()-b.exp()).abs().sum()))

def swaps():
    p=verify();samples=read(OUT/'sample_manifest.json')['samples'];lookup={key(s):s for s in samples};index=read(OUT/'selection_index.json');shards=read(OUT/'shards.json')['image_assignments'];pairs=read(OUT/'image_swap_manifest.json')['pairs'];rows=[]
    for pair in pairs:
        aa,bb=[lookup[k] for k in pair['sample_keys']];assert aa['input_mapping']['prompt_ids']==bb['input_mapping']['prompt_ids'] and aa['answer']!=bb['answer']
        for c in p['swap']['conditions']:
            records=[];vectors=[];full=[];sources=[]
            for s in [aa,bb]:
                spec=spec_for(s,index);uid=spec['aliases'][c];rank=shards[s['image_sha256']];path,_=paths(s,uid,rank);record=read(path);assert record['binding']==digest(p) and sha256(record['raw']['path'])==record['raw']['sha256'];records.append(record['generation']);sources.append(str(path.relative_to(OUT)))
                with np.load(record['raw']['path'],allow_pickle=False) as z:
                    vectors.append(z['first_logits'].copy())
                    if c=='No-Writer':full.append(z['greedy_valid_logits'].copy())
            a,b=records;dm=distribution(*vectors);all_exact=None
            if c=='No-Writer':
                all_exact=all(torch.equal(torch.from_numpy(x),torch.from_numpy(y)) for x,y in zip(full[0],full[1])) and full[0].shape==full[1].shape
                assert all_exact and a['token_ids']==b['token_ids'] and dm['first_logits_exact'],'No-Writer image independence failed'
            rows.append(dict(pair_id=pair['pair_id'],condition=c,question=pair['question'],question_a=aa['question_id'],question_b=bb['question_id'],image_a=aa['image_sha256'],image_b=bb['image_sha256'],gold_a=aa['answer'],gold_b=bb['answer'],
                output_a=a['text'],output_b=b['text'],prediction_a=a['prediction'],prediction_b=b['prediction'],correct_a=a['correct'],correct_b=b['correct'],both_correct=int(a['correct'] and b['correct']),endpoint_accuracy=(a['correct']+b['correct'])/2,
                answer_changed=int(a['prediction']!=b['prediction']),tokens_changed=int(a['token_ids']!=b['token_ids']),**dm,no_writer_all_valid_logits_exact=all_exact if all_exact is not None else '',
                valid_logits_compared=len(full[0]) if c=='No-Writer' else '',both_images_outside_exploration=pair['both_images_outside_exploration'],source_a=sources[0],source_b=sources[1],reuse='Exact main3000 endpoint records; separate image-pair statistics'))
    conditions=p['swap']['conditions'];metrics=['both_correct','endpoint_accuracy','answer_changed','first_JS_nat','first_TV'];values=np.array([[next(row for row in rows if row['pair_id']==pair['pair_id'] and row['condition']==c)[m] for c in conditions for m in metrics] for pair in pairs])
    # Each pair contains two distinct SHA clusters; no SHA occurs in another pair.
    # Resample these linked two-image blocks to preserve pair-level outcomes.
    draw=cluster_bootstrap(values,[pair['pair_id'] for pair in pairs]);summary=[]
    for j,c in enumerate(conditions):
        row=dict(condition=c,pairs=len(pairs),endpoints=2*len(pairs),both_correct_groups=int(values[:,j*5].sum()),answer_changed_groups=int(values[:,j*5+2].sum()),endpoint_correct=int(2*values[:,j*5+1].sum()))
        for k,m in enumerate(metrics):
            lo,hi=interval(draw[:,j*5+k]);row.update({m:float(values[:,j*5+k].mean()),m+'_ci_low':lo,m+'_ci_high':hi})
        summary.append(row)
    write_csv(OUT/'image_swap_per_pair.csv',rows);write_csv(OUT/'image_swap_summary.csv',summary);atomic_npz(OUT/'bootstrap/image_swap_linked_pairs.npz',means=draw)
    atomic_json(OUT/'image_swap_validation.json',dict(passed=True,binding=digest(p),pairs=len(pairs),endpoint_condition_records=len(rows)*2,
        image_sha_no_reuse_across_pairs=True,all_No_Writer_effective_greedy_logits_strictly_equal=True,EOS_prediction_included=True,
        selection_before_model_outputs=True,reused_main_records_strictly_audited=True,not_independent_new_main_questions=True,
        bootstrap='20000 paired resamples of linked two-image blocks, no SHA shared across blocks',first_JS_units='nat',first_TV_range=[0,1]))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['replay','statistics','swaps']);a=parser.parse_args();globals()[a.stage]()

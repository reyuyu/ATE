"""Unique complete-study aggregation, image-cluster uncertainty and unsmoothed figures."""
from collections import defaultdict
import csv
import fcntl
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.dataset_registry import digest, freeze_json, sha256
from paper_rq1.analyze import write_csv
from paper_rq1.scorers import gqa_prediction
from .engine import metric_rows
from .protocol import OUTPUT, PROTOCOL, SEED, TASKS
from .run import gate, samples, verify


def descriptive_correlations(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or x.shape!=y.shape:
        raise ValueError('Finite aligned observations required')
    if np.ptp(x)==0 or np.ptp(y)==0:return None,None
    def midranks(values):
        unique,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
        ranks=np.cumsum(counts)-counts+(counts-1)/2
        return ranks[inverse]
    return float(np.corrcoef(x,y)[0,1]),float(np.corrcoef(midranks(x),midranks(y))[0,1])


def cluster_bootstrap(values, images, repetitions=10000):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim==1:values=values[:,None]
    if len(values)!=len(images) or not np.isfinite(values).all():
        raise ValueError('Finite paired rows aligned to image IDs required')
    indices=defaultdict(list)
    for i,image in enumerate(images):indices[image].append(i)
    groups=list(indices.values())
    if len(groups)<2:raise ValueError('At least two images needed for image-cluster uncertainty')
    sums=np.array([values[g].sum(0) for g in groups])
    counts=np.array([len(g) for g in groups])
    rng=np.random.default_rng(SEED)
    result=np.empty((repetitions,values.shape[1]))
    for lo in range(0,repetitions,64):
        n=min(64,repetitions-lo)
        draw=rng.integers(0,len(groups),(n,len(groups)))
        result[lo:lo+n]=sums[draw].sum(1)/counts[draw].sum(1)[:,None]
    return np.quantile(result,[.025,.975],axis=0)


def load(stage):
    gate(stage)
    config=verify()
    collected=[];seen=set()
    for rank in range(4):
        expected=[s for s in samples(rank) if stage=='first_token' or s['accuracy_subset']]
        actual=list((OUTPUT/f'{stage}/shard{rank}/checkpoints').glob('*.json'))
        if len(actual)!=len(expected):raise ValueError('Missing/extra checkpoint; refuse full aggregate')
        for sample in expected:
            qid=sample['question_id'];path=OUTPUT/f'{stage}/shard{rank}/checkpoints'/f"{sample['task']}__{qid}.json"
            value=json.loads(path.read_text())
            if qid in seen or value['binding']!=digest(config) or value['sample_sha256']!=digest(sample):
                raise ValueError('Duplicate or changed result binding')
            if (value['task'],value['question_id'],value['image_id'])!=(sample['task'],qid,sample['image_id']):
                raise ValueError('Output identity differs')
            seen.add(qid);collected.append((sample,value['result']))
    if len(collected)!=(3000 if stage=='first_token' else 480):raise ValueError('Incomplete study')
    return collected


def numerical_replay(main,accuracy):
    tol=PROTOCOL['tolerances']
    errors=defaultdict(float)
    logits_count=0
    for index,(s,r) in enumerate(main):
        if [x['layer'] for x in r['metrics']]!=list(range(32)) or [x['layer'] for x in r['bos']]!=list(range(32)):
            raise ValueError('Missing/duplicate single layer')
        if sha256(r['logits_path'])!=r['logits_sha256']:raise ValueError('Raw logit SHA256 differs')
        with np.load(r['logits_path']) as raw:
            full,changed=raw['full_logits'],raw['ablated_logits']
            if full.dtype!=np.float32 or changed.dtype!=np.float32 or changed.shape!=(32,len(full)):
                raise ValueError('Invalid raw FP32 vector layout')
            replay=metric_rows(full,changed)
        logits_count+=33
        for old,new in zip(r['metrics'],replay):
            for key,value in new.items():
                if isinstance(value,(bool,int)):
                    if old[key]!=value:raise ValueError('Token/flip replay differs')
                else:
                    err=abs(old[key]-value);errors[key]=max(errors[key],err)
                    atol=tol['raw_relative_replay_atol'] if key=='relative_delta_p' else tol['raw_metric_replay_atol']
                    if not np.isclose(old[key],value,atol=atol,rtol=tol['raw_metric_replay_rtol']):
                        raise ValueError('FP32 numerical replay exceeds frozen tolerance')
        for row in r['bos']:
            heads=np.array(row['bos_mass_per_head'])
            if not np.isfinite(heads).all() or np.any(heads<0) or np.any(heads>1) or len(heads)!=32:
                raise ValueError('Invalid BOS heads')
            if abs(heads.mean()-row['bos_mass'])>tol['raw_metric_replay_atol'] or row['max_row_sum_error']>tol['row_sum_atol']:
                raise ValueError('BOS mean/row audit failed')
            if row['query']!=s['input_mapping']['p_minus_one'] or row['bos_positions']!=s['input_mapping']['bos_positions']:
                raise ValueError('BOS query mapping differs')
        if (index+1)%500==0:print('Replayed all-vocabulary logits',index+1,'/3000',flush=True)
    for s,r in accuracy:
        generations=r['generations']
        if len(generations)!=33 or [g['layer'] for g in generations]!=[None,*range(32)] or not r['all_33_initial_logits_exact']:
            raise ValueError('Incomplete exploratory generation conditions')
        for g in generations:
            if g['prediction']!=gqa_prediction(g['text']) or g['score']!=float(gqa_prediction(g['text'])==s['answer']):
                raise ValueError('Official generation score replay differs')
            if len(g['token_ids'])!=g['length'] or not 0<g['length']<=32 or (g['token_ids'][-1]==2)!=g['eos'] or 2 in g['token_ids'][:-1]:
                raise ValueError('Invalid generated token length/EOS')
            edits=g['intervention_audit']
            if g['layer'] is None:
                if edits:raise ValueError('Native Full had intervention')
            elif len(edits)!=1 or edits[0]['layer']!=g['layer'] or edits[0]['time']!=0 or edits[0]['query']!=s['input_mapping']['p_minus_one'] or not edits[0]['target_zero'] or not edits[0]['other_queries_exact']:
                raise ValueError('Exploratory generation intervention scope differs')
    value=dict(passed=True,questions_replayed=3000,raw_vocabulary_vectors=logits_count,
        ablated_metric_rows=96000,bos_rows=96000,generation_questions=480,generation_conditions=15840,
        max_absolute_metric_errors=dict(errors),tolerances=tol,official_generation_scores_recomputed=True,
        generation_hooks_prefill_only=True)
    freeze_json(OUTPUT/'numerical_audit.json',value)
    return value


def first_tables(main):
    per_question=[];bos_details=[];summaries=[];bos_summaries=[];correlations=[]
    for task in TASKS:
        items=sorted([(s,r) for s,r in main if s['task']==task],key=lambda pair:pair[0]['question_id'])
        if len(items)!=500:raise ValueError('Incomplete task')
        images=[s['image_id'] for s,r in items]
        features=('relative_delta_p','delta_logp','top1_flip','js')
        matrices={key:np.array([[row[key] for row in r['metrics']] for s,r in items],dtype=float) for key in features}
        bos=np.array([[row['bos_mass'] for row in r['bos']] for s,r in items])
        all_values=np.column_stack([matrices[k] for k in features]+[bos])
        ci=cluster_bootstrap(all_values,images,PROTOCOL['bootstrap']['repetitions'])
        for layer in range(32):
            entry=dict(task=task,layer=layer,n_questions=500,n_images=len(set(images)))
            for f,key in enumerate(features):
                values=matrices[key][:,layer]
                entry.update({f'mean_{key}':float(values.mean()),f'median_{key}':float(np.median(values)),
                    f'{key}_ci95_low':float(ci[0,f*32+layer]),f'{key}_ci95_high':float(ci[1,f*32+layer])})
            entry['top1_flip_count']=int(matrices['top1_flip'][:,layer].sum())
            summaries.append(entry)
            values=bos[:,layer]
            bos_summaries.append(dict(task=task,layer=layer,n_questions=500,n_images=len(set(images)),
                mean=float(values.mean()),median=float(np.median(values)),std=float(values.std(ddof=1)),
                **{f'p{p}':float(np.percentile(values,p)) for p in (5,25,75,95)},
                ci95_low=float(ci[0,128+layer]),ci95_high=float(ci[1,128+layer])))
        for sample,result in items:
            base=dict(task=task,question_id=sample['question_id'],image_id=sample['image_id'],answer=sample['answer'],
                first_token_matches_gold=result['first_token_matches_gold'],gold_is_single_token=result['gold_is_single_token'],
                full_first_token_text=result['full_first_token_text'],accuracy_subset=sample['accuracy_subset'])
            per_question.extend(dict(base,**row) for row in result['metrics'])
            for row in result['bos']:
                bos_details.append(dict(task=task,question_id=sample['question_id'],image_id=sample['image_id'],
                    layer=row['layer'],bos_mass=row['bos_mass'],max_row_sum_error=row['max_row_sum_error'],
                    query=row['query'],bos_positions=json.dumps(row['bos_positions']),
                    **{f'head_{h:02d}':v for h,v in enumerate(row['bos_mass_per_head'])}))
        sensitivity=-matrices['delta_logp']
        for kind,x,y in [('across_32_layer_means',bos.mean(0),sensitivity.mean(0)),
                         ('within_layer_centered_question_rows',(bos-bos.mean(0)).ravel(),(sensitivity-sensitivity.mean(0)).ravel())]:
            pearson,spearman=descriptive_correlations(x,y)
            correlations.append(dict(task=task,scope=kind,pearson=pearson,spearman=spearman,
                sensitivity='negative delta_logp (larger = greater original-token LogP loss)',
                interpretation='Descriptive only; repeated image/layer rows are not independent; no p-value or causal claim'))
    write_csv(OUTPUT/'first_token_per_question.csv',per_question)
    write_csv(OUTPUT/'first_token_layer_summary.csv',summaries)
    write_csv(OUTPUT/'bos_mass_per_question.csv',bos_details)
    write_csv(OUTPUT/'bos_mass_layer_summary.csv',bos_summaries)
    write_csv(OUTPUT/'bos_sensitivity_correlations.csv',correlations)
    return summaries,bos_summaries


def generation_tables(main,accuracy):
    details=[];summaries=[];groups=[]
    main_by_q={s['question_id']:(s,r) for s,r in main}
    for task in TASKS:
        items=sorted([(s,r) for s,r in accuracy if s['task']==task],key=lambda p:p[0]['question_id'])
        if len(items)!=80:raise ValueError('Exploratory subset incomplete')
        images=[s['image_id'] for s,r in items]
        full=np.array([r['generations'][0]['score'] for s,r in items])
        changed=np.array([[g['score'] for g in r['generations'][1:]] for s,r in items])
        delta=changed-full[:,None]
        ci=cluster_bootstrap(np.column_stack([full,delta]),images,PROTOCOL['bootstrap']['repetitions'])
        for layer in range(32):
            gens=[r['generations'][layer+1] for s,r in items]
            summaries.append(dict(task=task,layer=layer,n_questions=80,n_images=len(set(images)),
                accuracy_full=float(full.mean()),accuracy_ablated=float(changed[:,layer].mean()),
                full_ci95_low=float(ci[0,0]),full_ci95_high=float(ci[1,0]),
                delta_accuracy_pp=float(100*delta[:,layer].mean()),
                ci95_low_pp=float(100*ci[0,layer+1]),ci95_high_pp=float(100*ci[1,layer+1]),
                correct_to_wrong=int(((full==1)&(changed[:,layer]==0)).sum()),
                wrong_to_correct=int(((full==0)&(changed[:,layer]==1)).sum()),
                text_changed=int(sum(g['text']!=r['generations'][0]['text'] for g,(s,r) in zip(gens,items))),
                mean_length=float(np.mean([g['length'] for g in gens])),eos_rate=float(np.mean([g['eos'] for g in gens])),
                scope='Exploratory80 from main500, not independent Confirmation; ordinary descriptive95%CI'))
            for (sample,result),g in zip(items,gens):
                f=result['generations'][0]
                details.append(dict(task=task,question_id=sample['question_id'],image_id=sample['image_id'],layer=layer,
                    answer=sample['answer'],full_text=f['text'],ablated_text=g['text'],
                    full_token_ids=json.dumps(f['token_ids']),ablated_token_ids=json.dumps(g['token_ids']),
                    full_score=f['score'],ablated_score=g['score'],correct_to_wrong=f['score']==1 and g['score']==0,
                    wrong_to_correct=f['score']==0 and g['score']==1,text_changed=f['text']!=g['text'],
                    full_length=f['length'],ablated_length=g['length'],full_eos=f['eos'],ablated_eos=g['eos']))
        for correct,label in [(True,'Full-correct'),(False,'Full-wrong')]:
            selected=[(s,main_by_q[s['question_id']][1]) for (s,r),score in zip(items,full) if bool(score)==correct]
            if not selected:continue
            matrix=np.array([[row['delta_logp'] for row in r['metrics']] for s,r in selected])
            image_ids=[s['image_id'] for s,r in selected]
            bounds=cluster_bootstrap(matrix,image_ids,PROTOCOL['bootstrap']['repetitions']) if len(set(image_ids))>=2 else np.full((2,32),np.nan)
            for layer in range(32):
                groups.append(dict(task=task,group=label,layer=layer,n_questions=len(selected),n_images=len(set(image_ids)),
                    mean_delta_logp=float(matrix[:,layer].mean()),
                    ci95_low=None if not np.isfinite(bounds[0,layer]) else float(bounds[0,layer]),
                    ci95_high=None if not np.isfinite(bounds[1,layer]) else float(bounds[1,layer]),
                    scope='Actual complete Full generation correctness on frozen80 only; no resampling/filtering of main500'))
    write_csv(OUTPUT/'full_generation_80_per_question.csv',details)
    write_csv(OUTPUT/'full_generation_80_layer_summary.csv',summaries)
    write_csv(OUTPUT/'first_token_by_full_generation_group_summary.csv',groups)
    return summaries,groups


def save(fig,name):
    folder=OUTPUT/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf'):
        path=folder/f'{name}.{ext}'
        if not path.exists():fig.savefig(path,dpi=200)
    plt.close(fig)


def curve(rows,key,low,high,title,ylabel,name,color='#087f8c',baseline=False,percent=False):
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,task in zip(axes.flat,TASKS):
        rr=sorted([r for r in rows if r['task']==task],key=lambda r:r['layer'])
        scale=100 if percent else 1
        x=[r['layer'] for r in rr];y=[scale*r[key] for r in rr]
        ax.plot(x,y,'o-',ms=3,color=color)
        ax.fill_between(x,[scale*r[low] for r in rr],[scale*r[high] for r in rr],color=color,alpha=.18)
        if baseline:ax.axhline(0,color='#555555',ls='--',lw=.8)
        if key=='mean':ax.set_ylim(0,1)
        ax.set(title=f"{task} (q={rr[0]['n_questions']}, images={rr[0]['n_images']})",
               xlabel='Native Transformer layer' if key=='mean' else 'Single ablated layer',ylabel=ylabel,xlim=(0,31))
        ax.grid(alpha=.2)
    fig.suptitle(title+'\nImage-cluster paired ordinary95% CI; per-task y ranges unless BOS[0,1]; no smoothing')
    save(fig,name)


def figures(first,bos,generation,groups):
    curve(first,'mean_relative_delta_p','relative_delta_p_ci95_low','relative_delta_p_ci95_high',
          'Figure1. Native Full-token relative probability change','Mean relative probability change (%)','figure1_relative_probability',baseline=True)
    curve(first,'mean_delta_logp','delta_logp_ci95_low','delta_logp_ci95_high',
          'Figure2. Single-layer first-token sensitivity','Mean Delta LogP (natural log)','figure2_delta_logp',baseline=True)
    curve(bos,'mean','ci95_low','ci95_high','Figure3. Native Full BOS attention','Mean BOS Attention Mass','figure3_bos_mass',color='#3d7e45')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,task in zip(axes.flat,TASKS):
        b=[r for r in bos if r['task']==task];r=[r for r in first if r['task']==task]
        x=list(range(32));right=ax.twinx()
        ax.plot(x,[v['mean'] for v in b],color='#3d7e45',label='BOS mass')
        ax.fill_between(x,[v['ci95_low'] for v in b],[v['ci95_high'] for v in b],color='#3d7e45',alpha=.12)
        right.plot(x,[v['mean_delta_logp'] for v in r],color='#bc4d31',label='Delta LogP')
        right.fill_between(x,[v['delta_logp_ci95_low'] for v in r],[v['delta_logp_ci95_high'] for v in r],color='#bc4d31',alpha=.12)
        ax.set(title=f'{task} (q=500)',xlabel='Layer',ylabel='BOS mass (left, green)',ylim=(0,1),xlim=(0,31))
        right.set_ylabel('Delta LogP (right, red)');ax.grid(alpha=.15)
    fig.suptitle('Figure4. Native BOS mass and first-token sensitivity: separate labeled scales; ordinary95% CI')
    save(fig,'figure4_bos_vs_delta_logp')
    curve(generation,'delta_accuracy_pp','ci95_low_pp','ci95_high_pp',
          'Figure5. Exploratory complete-generation accuracy (preselected80 from500, not independent)',
          'Accuracy Delta vs Full (pp)','figure5_exploratory_accuracy',color='#74509f',baseline=True)
    curve(first,'mean_top1_flip','top1_flip_ci95_low','top1_flip_ci95_high',
          'Figure6. First-token Top-1 changes (NOT complete-answer errors)',
          'Top-1 flip rate (%)','figure6_top1_flip',color='#bc4d31',percent=True)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,task in zip(axes.flat,TASKS):
        for group,color in [('Full-correct','#087f8c'),('Full-wrong','#bc4d31')]:
            rr=[r for r in groups if r['task']==task and r['group']==group]
            if not rr:continue
            ax.plot([r['layer'] for r in rr],[r['mean_delta_logp'] for r in rr],color=color,label=f"{group} (n={rr[0]['n_questions']})")
            if all(r['ci95_low'] is not None for r in rr):
                ax.fill_between([r['layer'] for r in rr],[r['ci95_low'] for r in rr],[r['ci95_high'] for r in rr],color=color,alpha=.12)
        ax.set(title=task,xlabel='Layer',ylabel='Mean first-token Delta LogP',xlim=(0,31));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Actual Full-generation correctness subgroups on frozen exploratory80 only; ordinary image-cluster95% CI')
    save(fig,'figure7_exploratory_full_correct_wrong')


def report(first,bos,generation):
    data=json.loads((OUTPUT/'data_manifest.json').read_text());config=verify()
    lines=['# Six-Task GQA Single-Layer Attention Output Ablation','','Status: complete.','',
        'This is NOT a reproduction of source-to-target attention-edge knockout in Cross-modal Information '
        'Flow in Multimodal Large Language Models. Only one layer\'s entire Attention o_proj update at '
        'the original prompt P-1 is zeroed. All other queries/layers, MLP, LayerNorm, residual and weights '
        'remain native. This computes the Attention output before zeroing and provides no compute-skipping or speedup evidence.','',
        f"Frozen code: `{config['code_commit']}`; seed20261030; pinned LLaVA-1.5-7B FP16/eager/Transformers4.49.0.",'',
        '## Samples and Provenance','',
        'All six categories use original official GQA Balanced validation, not author correct_question CSVs. '
        'ChooseAttr=chooseAttr; ChooseCat=categoryThatChoose/categoryThisChoose/objThisChoose; '
        'ChooseRel=relChooser; CompareAttr=twoCommon; LogicalObj=existThatOr; QueryAttr=positionQuery. '
        'Sampling and each80-question subset were hash-frozen before GPU output. Missing images were not replaced.','',
        '| Task | Main questions | Main images | Generation subset | Full generation accuracy | Most sensitive layer (mean Delta LogP) | Least changed layer (abs mean Delta LogP) |',
        '|---|---:|---:|---:|---:|---|---|']
    for task in TASKS:
        r=[x for x in first if x['task']==task];g=[x for x in generation if x['task']==task]
        high=min(r,key=lambda x:x['mean_delta_logp']);low=min(r,key=lambda x:abs(x['mean_delta_logp']))
        lines.append(f"| {task} | 500 | {data['per_task'][task]['images']} | 80 | {100*g[0]['accuracy_full']:.2f}% | "
            f"L{high['layer']} ({high['mean_delta_logp']:.5f}) | L{low['layer']} ({low['mean_delta_logp']:.5f}) |")
    lines += ['',f"There are {data['total_images']} unique image IDs globally, with within/across-task overlap. "
        'The exact overlap matrix and unequal per-worker task counts are in data_manifest.json. '
        'Every image stays on one GPU. Bootstrap resamples whole image IDs and uses cluster sum/count ratios '
        'to retain question weighting when images have unequal question counts.','',
        '## Completion and Correctness','',
        '- Main: 3,000 Native Full + 96,000 independent single-layer Prefill conditions; all32 layers per question.',
        '- Generation: 480 Native Full + 15,360 single-layer conditions on the predefined80/task exploratory subset.',
        '- Full native BOS mass: 96,000 layer/question rows, including all32 heads; no ablated attention substituted.',
        '- Raw Full and all32 changed full-vocabulary FP32 logits are stored externally with committed paths/SHA256. '
        'All99,000 vectors were replayed for numerical metrics; see numerical_audit.json.',
        '- No-op/native/observed/cached logits exact; all four GPUs passed the same-question predefined-tolerance '
        'Smoke comparison. Intervention affects one o_proj output and one true query, with32 native MLP calls. '
        'Inputs, parameter versions and Hook cleanup were checked.','',
        '## Curves','']
    for name,title in [('figure1_relative_probability','Relative probability'),('figure2_delta_logp','Delta LogP'),
        ('figure3_bos_mass','BOS mass'),('figure4_bos_vs_delta_logp','BOS and sensitivity'),
        ('figure5_exploratory_accuracy','Exploratory generation accuracy'),('figure6_top1_flip','Top-1 flip'),
        ('figure7_exploratory_full_correct_wrong','Actual Full-correct/wrong exploratory subgroups')]:
        lines += [f'![{title}](figures/{name}.png)','']
    lines += ['## Interpretation Boundaries','',
        '- Primary sensitivity is mean Delta LogP of the SAME native Full Top-1 token. Relative probability '
        'uses stable FP32 expm1; JS is full-vocabulary natural-log divergence. None is full-answer accuracy.',
        '- Gold-first-token matching is recorded, including multi-token answers, but is not labeled full-answer '
        'correctness. Actual Full-correct/wrong plots use only the80 predefined questions with complete Full generation.',
        '- BOS mass uses the tokenizer BOS ID located in each real input, post-softmax native rows, then head means. '
        'Layerwise mean/median/std/P5/P25/P75/P95 and ordinary image-cluster95% CIs are in CSV.',
        '- Pearson/Spearman in bos_sensitivity_correlations.csv are descriptive across layer means and '
        'within-layer centered question rows. Repeated image/layer observations are not independent; '
        'no p-value or causal importance claim is made.',
        '- All curves use real unsmoothed32-layer values with paired ordinary95% CIs. There are192 primary '
        'task-layer comparisons; the displayed CIs do NOT control that family. No multiplicity-significance, '
        'noninferiority, equivalence, or safe-removal claim is made from a zero/small/nonsignificant effect.',
        '- The80-question generation subset is exploratory and drawn from the main500, not independent Confirmation. '
        'Each intervention occurs once at original Prefill P-1; later Decode is native and uses the condition\'s own cache.',
        '- GQA score is verified official primary answer equality after the pinned LLaVA submission converter. '
        'No full GQA validity/plausibility/consistency leaderboard reproduction is claimed.',
        '- Failed/missing checkpoints block aggregation. See failure_inventory.json for every preserved failure '
        'attempt; successful resume does not erase earlier failures. No result-based sample deletion is allowed.','']
    path=OUTPUT/'report.md'
    if path.exists():raise ValueError('Never replace a completed report')
    with path.open('x') as f:f.write('\n'.join(lines))


def main():
    lock=(OUTPUT/'analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUTPUT/'completion.json').exists():raise ValueError('Completed study already exists')
    first_data=load('first_token');accuracy=load('accuracy')
    numerical_replay(first_data,accuracy)
    first,bos=first_tables(first_data)
    generation,groups=generation_tables(first_data,accuracy)
    figures(first,bos,generation,groups)
    failures=[str(p.relative_to(OUTPUT)) for p in OUTPUT.glob('*/shard*/failures/*.json')]
    failures += [str(p.relative_to(OUTPUT)) for p in (OUTPUT/'failures').glob('*.json')]
    cpu_failures=[str(p.relative_to(OUTPUT)) for p in (OUTPUT/'cpu_attempts').glob('*.json')
                  if not json.loads(p.read_text())['passed']]
    freeze_json(OUTPUT/'failure_inventory.json',dict(preserved_failure_records=failures,
        preserved_failed_cpu_attempts=cpu_failures,missing_questions=0,missing_images=0,
        missing_conditions=0,all_expected_questions_complete=True))
    report(first,bos,generation)
    config=verify()
    files={str(p.relative_to(OUTPUT)):sha256(p) for p in OUTPUT.rglob('*')
           if p.is_file() and p.suffix not in ('.lock','.tmp') and not p.name.startswith(('console_','coordinator_console','analysis_console'))}
    freeze_json(OUTPUT/'completion.json',dict(status='complete',main_questions=3000,main_full_conditions=3000,
        main_ablations=96000,generation_questions=480,generation_full_conditions=480,generation_ablations=15360,
        code_commit=config['code_commit'],binding=digest(config),numerical_replay_passed=True,
        files_sha256=files))
    print('Complete:3000 main questions and480 exploratory generation questions. Report:',OUTPUT/'report.md',flush=True)


if __name__=='__main__':main()

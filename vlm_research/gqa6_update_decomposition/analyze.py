from collections import defaultdict
import csv
import fcntl
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.dataset_registry import digest,freeze_json,sha256
from paper_rq1.analyze import write_csv
from paper_rq1_single_layer_gqa6.analyze import cluster_bootstrap,descriptive_correlations
from .engine import measurements
from .protocol import METRICS, OLD, OUTPUT, PROTOCOL, REGIONS, TASKS
from .run import gate,samples,verify

FEATURES=(*METRICS,'bos_mass','delta_logp')
COLORS=dict(attention='#087f8c',mlp='#bc4d31',residual='#555555',net='#74509f',cancel='#3d7e45')


def bounds(values,images):
    return cluster_bootstrap(values,images,PROTOCOL['bootstrap']['repetitions'])


def load_and_replay():
    c=verify();gate('collect');binding=digest(c)
    refs=json.loads((OUTPUT/'historical_bindings.json').read_text())
    items=[];seen=set();max_error=0.;max_old=0.;sum_errors=[0.,0.]
    for rank in range(4):
        expected=samples(rank)
        files=list((OUTPUT/f'collect/shard{rank}/checkpoints').glob('*.json'))
        if len(files)!=len(expected):raise ValueError('Missing/extra per-question checkpoints')
        for s in expected:
            name=s['task']+'__'+s['question_id']
            v=json.loads((OUTPUT/f'collect/shard{rank}/checkpoints/{name}.json').read_text())
            if name in seen or v['binding']!=binding or v['sample_sha256']!=digest(s):raise ValueError('Duplicate/changed sample')
            seen.add(name);r=v['result']
            if r['observed_full_forwards']!=1 or r['nodes_captured']!=160 or sha256(r['raw_path'])!=r['raw_sha256']:
                raise ValueError('Invalid one-forward raw capture')
            with np.load(r['raw_path']) as raw:
                vectors=raw['vectors'];logits=raw['full_logits']
                replay=measurements(vectors)
                if vectors.shape!=(32,5,4096) or logits.dtype!=np.float32 or not np.isfinite(logits).all():
                    raise ValueError('Wrong raw native vector/logit layout')
            if len(r['measurements'])!=32:raise ValueError('Missing layers')
            for old,new in zip(r['measurements'],replay):
                if old!=new:raise ValueError('FP32 raw-vector metric replay differs')
                sum_errors[0]=max(sum_errors[0],new['sum1_fp32_max_error'])
                sum_errors[1]=max(sum_errors[1],new['sum2_fp32_max_error'])
            max_old=max(max_old,r['historical_comparison']['max_abs_logit_difference'])
            joined=[dict(row,**refs[name]['rows'][str(row['layer'])]) for row in replay]
            items.append((s,joined))
    if len(items)!=600:raise ValueError('Incomplete600-question cohort')
    audit=dict(passed=True,questions=600,layers=19200,actual_vectors=96000,
        formal_full_forwards=600,raw_metrics_replayed_exact=True,max_metric_replay_error=max_error,
        max_historical_full_logit_difference=max_old,max_sum1_fp32_error=sum_errors[0],max_sum2_fp32_error=sum_errors[1],
        fp16_rounded_residual_sums_all_exact=True,historical_joins=19200)
    freeze_json(OUTPUT/'numerical_audit.json',audit)
    return items


def matrix(items):
    return np.array([[[np.nan if r[k] is None else r[k] for k in FEATURES] for r in rows] for s,rows in items])


def summarize(items,task):
    values=matrix(items);images=[s['image_id'] for s,r in items]
    if np.isfinite(values).all():ci=bounds(values.reshape(len(items),-1),images).reshape(2,32,len(FEATURES))
    else:
        ci=np.full((2,32,len(FEATURES)),np.nan)
        for layer in range(32):
            for j in range(len(FEATURES)):
                good=np.isfinite(values[:,layer,j]);ids=[im for im,ok in zip(images,good) if ok]
                if len(set(ids))>=2:ci[:,layer,j]=bounds(values[good,layer,j],ids)[:,0]
    output=[]
    for layer in range(32):
        row=dict(task=task,layer=layer,n_questions=len(items),n_images=len(set(images)))
        for j,key in enumerate(FEATURES):
            good=values[:,layer,j][np.isfinite(values[:,layer,j])]
            row.update({f'mean_{key}':float(good.mean()) if len(good) else None,
                f'median_{key}':float(np.median(good)) if len(good) else None,
                f'n_valid_{key}':len(good),f'ci95_low_{key}':float(ci[0,layer,j]) if np.isfinite(ci[0,layer,j]) else None,
                f'ci95_high_{key}':float(ci[1,layer,j]) if np.isfinite(ci[1,layer,j]) else None})
        output.append(row)
    return output


def region_tables(groups):
    summaries=[];contrasts=[]
    for task,items in groups.items():
        values=matrix(items);images=[s['image_id'] for s,r in items]
        for region,layers in REGIONS.items():
            per=np.nanmean(values[:,layers],axis=1)
            # Undefined cosine denominators are not replaced with zero.
            finite=np.isfinite(per).all(0);ci=np.full((2,len(FEATURES)),np.nan)
            if finite.any():ci[:,finite]=bounds(per[:,finite],images)
            row=dict(task=task,region=region,layers=json.dumps(layers),n_questions=len(items),n_images=len(set(images)))
            for j,key in enumerate(FEATURES):
                valid=np.isfinite(per[:,j]);ids=[im for im,ok in zip(images,valid) if ok]
                if not finite[j] and len(set(ids))>=2:ci[:,j]=bounds(per[valid,j],ids)[:,0]
                row.update({f'mean_{key}':float(per[valid,j].mean()) if valid.any() else None,
                    f'n_valid_{key}':int(valid.sum()),
                    f'ci95_low_{key}':float(ci[0,j]) if np.isfinite(ci[0,j]) else None,
                    f'ci95_high_{key}':float(ci[1,j]) if np.isfinite(ci[1,j]) else None})
            summaries.append(row)
        def avg(key,layers):return values[:,layers,FEATURES.index(key)].mean(1)
        definitions={
            'early_attention_ratio_minus_middle':avg('attention_ratio',REGIONS['early'])-avg('attention_ratio',REGIONS['middle']),
            'early_attention_norm_minus_middle':avg('norm_attention',REGIONS['early'])-avg('norm_attention',REGIONS['middle']),
            'early_mlp_minus_attention_ratio':avg('mlp_ratio',REGIONS['early'])-avg('attention_ratio',REGIONS['early']),
            'peak_attention_ratio_minus_flanks':avg('attention_ratio',REGIONS['peak'])-avg('attention_ratio',[12,13,17,18]),
            'deep_attention_ratio_minus_middle':avg('attention_ratio',REGIONS['deep'])-avg('attention_ratio',REGIONS['middle']),
            'deep_mlp_ratio_minus_middle':avg('mlp_ratio',REGIONS['deep'])-avg('mlp_ratio',REGIONS['middle'])}
        joined=np.column_stack(list(definitions.values()));ci=bounds(joined,images)
        for j,(name,per) in enumerate(definitions.items()):
            contrasts.append(dict(task=task,contrast=name,mean=float(per.mean()),median=float(np.median(per)),
                ci95_low=float(ci[0,j]),ci95_high=float(ci[1,j]),n_questions=len(items),n_images=len(set(images))))
    write_csv(OUTPUT/'region_update_summary.csv',summaries);write_csv(OUTPUT/'planned_contrasts.csv',contrasts)
    return summaries,contrasts


def correlations(groups):
    pairs=[('bos_mass','attention_ratio'),('bos_mass','mlp_ratio'),('bos_mass','delta_logp'),('attention_ratio','delta_logp')]
    rows=[]
    for task,items in groups.items():
        v=matrix(items)
        for a,b in pairs:
            x=v[:,:,FEATURES.index(a)];y=v[:,:,FEATURES.index(b)]
            definitions=[('across_32_layer_means',None,np.nanmean(x,0),np.nanmean(y,0))]
            xc=x.copy();yc=y.copy()
            for original_task in TASKS:
                indices=[i for i,(s,r) in enumerate(items) if s['task']==original_task]
                if indices:
                    xc[indices]-=np.nanmean(xc[indices],0);yc[indices]-=np.nanmean(yc[indices],0)
            definitions.append(('within_task_layer_centered_rows',None,xc.ravel(),yc.ravel()))
            definitions.extend(('same_layer_matched_questions',layer,x[:,layer],y[:,layer]) for layer in range(32))
            for scope,layer,xx,yy in definitions:
                good=np.isfinite(xx)&np.isfinite(yy)
                pearson,spearman=descriptive_correlations(xx[good],yy[good]) if good.sum()>=2 else (None,None)
                rows.append(dict(task=task,scope=scope,layer=layer,x=a,y=b,n_paired_values=int(good.sum()),
                    pearson=pearson,spearman=spearman,delta_logp_sign='Original signed ablated-minus-Full LogP; NOT negated.',
                    interpretation='Descriptive; repeated question/image/layer rows are not independent; no causal or significance claim.'))
    write_csv(OUTPUT/'bos_update_correlations.csv',rows)


def historical_curves():
    with (OLD/'first_token_layer_summary.csv').open() as f:old=list(csv.DictReader(f))
    curves={task:[dict(layer=int(r['layer']),mean=float(r['mean_delta_logp']),low=float(r['delta_logp_ci95_low']),
                 high=float(r['delta_logp_ci95_high'])) for r in old if r['task']==task] for task in TASKS}
    rows={};images={}
    with (OLD/'first_token_per_question.csv').open() as f:
        for r in csv.DictReader(f):
            key=r['task']+'__'+r['question_id']
            if key not in rows:rows[key]=np.full(32,np.nan);images[key]=r['image_id']
            layer=int(r['layer'])
            if np.isfinite(rows[key][layer]):raise ValueError('Duplicate original500 sensitivity row')
            rows[key][layer]=float(r['delta_logp'])
    if len(rows)!=3000:raise ValueError('Original500 task cohorts incomplete')
    values=np.stack(list(rows.values()))
    if not np.isfinite(values).all():raise ValueError('Missing old sensitivity layer')
    ci=bounds(values,list(images.values()))
    curves['ALL']=[dict(layer=i,mean=float(values[:,i].mean()),low=float(ci[0,i]),high=float(ci[1,i])) for i in range(32)]
    write_csv(OUTPUT/'historical_500_sensitivity_plot_data.csv',[dict(task=t,n_questions=3000 if t=='ALL' else 500,**r)
              for t,data in curves.items() for r in data])
    return curves


def band(ax,rows,key,label,color,style='-'):
    x=[r['layer'] for r in rows];y=[r[f'mean_{key}'] for r in rows]
    low=[np.nan if r[f'ci95_low_{key}'] is None else r[f'ci95_low_{key}'] for r in rows]
    high=[np.nan if r[f'ci95_high_{key}'] is None else r[f'ci95_high_{key}'] for r in rows]
    ax.plot(x,y,style,color=color,label=label,lw=1.4,marker='.',ms=3)
    ax.fill_between(x,low,high,color=color,alpha=.12)


def highlight(ax):
    for left,right in [(-.3,4.3),(13.7,16.3),(24.7,31.3)]:ax.axvspan(left,right,color='#888888',alpha=.07)


def panel(ax,kind,rows,historical):
    right=None
    if kind==1:
        for key,label,color in [('norm_attention','Attention',COLORS['attention']),('norm_mlp','MLP',COLORS['mlp']),('norm_residual','Input residual',COLORS['residual'])]:
            band(ax,rows,key,label,color)
        ax.set_ylabel('Mean FP32 L2 norm')
    elif kind==2:
        band(ax,rows,'attention_ratio','Attention / input residual',COLORS['attention'])
        band(ax,rows,'mlp_ratio','MLP / SAME input residual',COLORS['mlp']);ax.set_ylabel('Mean update / input norm')
    elif kind==3:
        for key,label,color in [('cos_attention_residual','Attention - residual',COLORS['attention']),('cos_mlp_residual','MLP - residual',COLORS['mlp']),('cos_attention_mlp','Attention - MLP',COLORS['net']),('cos_input_output','Input - output',COLORS['residual'])]:
            band(ax,rows,key,label,color)
        ax.set_ylabel('Mean FP32 cosine');ax.set_ylim(-1.05,1.05)
    elif kind==4:
        band(ax,rows,'net_update_ratio','Net update / input',COLORS['net'])
        right=ax.twinx();band(right,rows,'cancellation_ratio','Cancellation ratio [0,1]',COLORS['cancel'])
        ax.set_ylabel('Net update / input norm');right.set_ylabel('Cancellation ratio');right.set_ylim(0,1.05)
    elif kind==5:
        highlight(ax);band(ax,rows,'attention_ratio','Attention / input',COLORS['attention'])
        right=ax.twinx();band(right,rows,'bos_mass','Matched native BOS mass',COLORS['cancel'])
        ax.set_ylabel('Attention / input norm');right.set_ylabel('Historical native BOS mass');right.set_ylim(0,1.05)
    elif kind==6:
        band(ax,rows,'attention_ratio','Attention ratio: current100/task',COLORS['attention'])
        right=ax.twinx();x=[r['layer'] for r in historical]
        right.plot(x,[r['mean'] for r in historical],color='#222222',label='Delta LogP: original500/task')
        right.fill_between(x,[r['low'] for r in historical],[r['high'] for r in historical],color='#222222',alpha=.1)
        band(right,rows,'delta_logp','Delta LogP: matched100/task','#bc4d31','--')
        ax.set_ylabel('Attention / input norm');right.set_ylabel('Signed ablation Delta LogP')
    ax.set_xlabel('Native layer L0-L31');ax.set_xlim(0,31);ax.grid(alpha=.15)
    handles,labels=ax.get_legend_handles_labels()
    if right:
        hs,ls=right.get_legend_handles_labels();handles+=hs;labels+=ls
    ax.legend(handles,labels,fontsize=7,loc='best')


def figures(summary,historical):
    folder=OUTPUT/'figures';folder.mkdir(exist_ok=True)
    titles=['Attention and MLP update norms','Attention and MLP: shared-denominator ratios',
        'Update directions','Net update and cancellation: separate scales',
        'BOS mass and Attention ratio: shaded L0-4 / L14-16 / L25-31',
        'Update ratio100 and original sensitivity500: matched100 also shown']
    names=['A1_update_norms','A2_update_ratios','A3_update_directions','A4_net_cancellation','A5_bos_update_ratio','A6_ratio_sensitivity']
    for kind,(title,name) in enumerate(zip(titles,names),1):
        fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
        for ax,task in zip(axes.flat,TASKS):
            rows=[r for r in summary if r['task']==task];panel(ax,kind,rows,historical[task])
            ax.set_title(f'{task}: q=100, images={rows[0]["n_images"]}')
        fig.suptitle(f'Figure {kind}: {title}\nOrdinary image-cluster95% CI; no smoothing; task y-scales independent except cosines/mass/cancellation')
        for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    rows=[r for r in summary if r['task']=='ALL']
    fig,axes=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    for kind,ax in enumerate(axes.flat,1):
        panel(ax,kind,rows,historical['ALL']);ax.set_title(f'A{kind}: {titles[kind-1]}',fontsize=10)
    fig.suptitle('Six-task overview: current600 questions,100/task; original sensitivity3000,500/task\nWhole-image cluster95% CI retains shared images across tasks; observational only')
    for ext in ['png','pdf']:fig.savefig(folder/f'A7_six_task_overview.{ext}',dpi=180)
    plt.close(fig)


def reports(summary,regions,contrasts,audit):
    c=verify();d=json.loads((OUTPUT/'data_manifest.json').read_text())
    lines=['# Native Attention / MLP Update Decomposition','','Status: complete. No new ablation or Decode.','',
        f"Frozen code: `{c['code_commit']}`. Baseline: `{PROTOCOL['base']}`.",
        'Model: pinned LLaVA-1.5-7B FP16/eager, Transformers4.49.0; seed20261030.','',
        '## Data and Computation','',
        f"600 input-hash-selected questions,100/task, {d['unique_images']} unique image IDs. Same-image questions remain on one physical GPU. "
        'No performance filtering, missing-image replacement, generation or new intervention.',
        'Each formal question uses exactly ONE native prompt forward, capturing actual decoder input, o_proj output, '
        'post_attention_layernorm input, MLP output and decoder output at its true P-1. '
        'The hooks return None; they do not replace tensors. All numerical statistics are computed in FP32 from actual FP16 captures.',
        'Full-vocabulary logits and all five actual vectors are saved externally with SHA256. '
        'Every metric was replayed exactly. Residual additions equal the correctly FP16-rounded sums bit-exactly; '
        'FP32 sum errors are recorded, not suppressed. Undefined zero-vector cosines are null with valid counts, not imputed zero.','',
        '| Task | Questions | Unique images |','|---|---:|---:|']
    lines.extend(f"| {t} | 100 | {d['per_task'][t]['images']} |" for t in TASKS)
    lines+=['','## Regions','',
        'Layer statistics are averaged within question before region uncertainty is estimated. '
        'All intervals use10,000 whole-image paired bootstrap draws and question-weighted cluster sums/counts. '
        'The pooled600 overview equals the equal-task point mean because all tasks have100 questions; '
        'global image clustering retains across-task image sharing. Layers and heads are not independent samples.',
        'All CIs are ordinary descriptive95% intervals, not multiplicity-adjusted significance or equivalence evidence.','',
        '| Task | Region | Attention norm | MLP norm | Residual norm | Attention ratio | MLP ratio | Net ratio | Cancellation |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in regions:
        if r['region']=='peak':continue
        keys=['norm_attention','norm_mlp','norm_residual','attention_ratio','mlp_ratio','net_update_ratio','cancellation_ratio']
        lines.append(f"| {r['task']} | {r['region']} | "+' | '.join('undefined' if r['mean_'+k] is None else f"{r['mean_'+k]:.5f}" for k in keys)+' |')
    lines+=['','## Correctness','',f"Numerical audit: `{json.dumps(audit,sort_keys=True)}`.",
        'CPU tests precede four-GPU Smoke. Same-run unhooked/observed/unhooked logits must be bit-exact. '
        'Historical Full uses the original frozen cross-run tolerances, with actual differences saved per question. '
        'All32 layers and160 nodes are required. Input hashes, parameter versions and hook cleanup are checked. '
        'Formal collection performs600 forwards total; Smoke alone adds a before/after unhooked reference.',
        'See cpu_validation.json, smoke/, collect/, execution_manifest.json and failure_inventory.json.','',
        '## Historical Joins and Figures','',
        'Historical BOS/Delta LogP join by task/question/layer, cross-checked against the original checkpoint. '
        'A6 distinguishes old500 sensitivity from current100 updates and also shows the matched100 old sensitivity. '
        'All sample-level correlations use only strict matched questions. Signed Delta LogP is NOT negated. '
        'Per-layer and within-task/layer-centered correlations are descriptive and have no causal p-value.']
    for name in ['A1_update_norms','A2_update_ratios','A3_update_directions','A4_net_cancellation','A5_bos_update_ratio','A6_ratio_sensitivity','A7_six_task_overview']:
        lines+=['',f'![{name}](figures/{name}.png)']
    lines+=['','## Interpretation','',
        'See research_findings.md for the measured contrasts. Norms, directions and cancellation are observational; '
        'they cannot establish Attention redundancy, causal MLP compensation, safe whole-layer skipping or acceleration. '
        'This experiment concerns native P-1 Prefill only in one model, not Decode or cross-model generality.']
    with (OUTPUT/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    findings=['# Research Findings: Observational Only','','All planned task/layer results are retained; no result-based retuning.','',
        '## Planned Region Contrasts','',
        'Positive early MLP-minus-Attention means the MLP update is larger using the SAME input-residual denominator. '
        'Negative early-minus-middle means the early update is smaller. Positive peak-minus-flanks means a local '
        'L14-L16 update-ratio increase relative to L12/L13/L17/L18. These are descriptive comparisons, not causal tests.','',
        '| Task | Contrast | Mean [ordinary paired95% CI] |','|---|---|---|']
    for r in contrasts:findings.append(f"| {r['task']} | {r['contrast']} | {r['mean']:+.6f} [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] |")
    findings+=['','## Questions and Boundaries','',
        '1. Early Attention magnitude: compare both early-minus-middle norm and ratio, plus layer curves. '
        'A nonnegative early ratio difference contradicts a simple small-update explanation even if ablation sensitivity is low.',
        '2. Early MLP versus Attention: inspect the paired shared-denominator contrast. Larger norm alone is not evidence of causal dominance.',
        '3. L14-L16 local enhancement: inspect the peak-minus-flanks contrast and A2; do not force the update peak to coincide with the historical sensitivity peak.',
        '4. Sensitivity versus update direction: A3/A6 and the signed matched correlations describe alignment, not necessity. '
        'A large norm or a distinctive cosine does not prove why a perturbation is tolerated.',
        '5. High BOS / low sensitivity: inspect A5 and the four matched correlation pairs, including per-layer and '
        'within-task/layer-centered results. Across-layer correlation alone can be depth-confounded.',
        '6. Late versus middle structure: inspect deep-minus-middle Attention/MLP contrasts, net ratio, cancellation and cosines in the region summary.',
        'Further research should test hypotheses independently rather than infer causal compensation or safe skipping from these observational vectors. '
        'No further GPU experiment was added in this run.','',
        'Raw FP16 captures: `/data/lk/ATE-resources/gqa6_update_decomposition/raw/`. '
        'The primary CSVs and manifest hashes provide the full review trail.']
    with (OUTPUT/'research_findings.md').open('x') as f:f.write('\n'.join(findings)+'\n')


def main():
    lock=(OUTPUT/'analysis.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUTPUT/'completion.json').exists():raise ValueError('Already complete')
    items=load_and_replay();groups={t:[(s,r) for s,r in items if s['task']==t] for t in TASKS};groups['ALL']=items
    for t,group in groups.items():
        if len(group)!=(600 if t=='ALL' else 100):raise ValueError('Wrong selected cohort size')
    details=[dict(task=s['task'],question_id=s['question_id'],image_id=s['image_id'],query=s['input_mapping']['p_minus_one'],**r)
             for s,rows in items for r in rows]
    write_csv(OUTPUT/'per_question_updates.csv',details)
    summary=[]
    for task,group in groups.items():
        summary+=summarize(group,task);print('Summarized',task,flush=True)
    write_csv(OUTPUT/'layer_update_summary.csv',summary)
    regions,contrasts=region_tables(groups);correlations(groups);historical=historical_curves()
    figures(summary,historical)
    failures=[str(p.relative_to(OUTPUT)) for p in OUTPUT.glob('*/shard*/failures/*.json')]
    failures += [str(p.relative_to(OUTPUT)) for p in (OUTPUT/'failures').glob('*.json')]
    cpu=[str(p.relative_to(OUTPUT)) for p in (OUTPUT/'cpu_attempts').glob('*.json') if not json.loads(p.read_text())['passed']]
    freeze_json(OUTPUT/'failure_inventory.json',dict(runtime_failures=failures,failed_cpu_attempts=cpu,missing_questions=0,missing_layers=0))
    audit=json.loads((OUTPUT/'numerical_audit.json').read_text());reports(summary,regions,contrasts,audit)
    c=verify();files={str(p.relative_to(OUTPUT)):sha256(p) for p in OUTPUT.rglob('*') if p.is_file()
        and p.suffix not in ['.lock','.tmp'] and not p.name.startswith(('console_','coordinator_console'))}
    freeze_json(OUTPUT/'completion.json',dict(status='complete',questions=600,layers=19200,formal_full_forwards=600,
        ablations=0,decode_steps=0,code_commit=c['code_commit'],binding=digest(c),files_sha256=files))
    print('Complete:600 native questions,19200 layer records, zero ablations or Decode.',flush=True)


if __name__=='__main__':main()

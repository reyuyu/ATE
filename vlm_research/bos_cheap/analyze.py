import csv
import gzip
import json
import shutil
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.dataset_registry import digest,freeze_json,sha256
from paper_rq1.analyze import write_csv
from paper_rq1_single_layer_gqa6.analyze import descriptive_correlations,cluster_bootstrap
from .protocol import OUT,UPDATES,EPS,HEAD_FEATURES,SEED,TASKS,REGIONS
from .predict import load_rows,danger,score_rule,skips,predict
from .run import verify,gate


def clean(value):
    if isinstance(value,np.generic):value=value.item()
    return value if value is None or np.isfinite(value) else None


def risk_values(rows,selected,epsilon):
    selected=np.asarray(selected,bool);n=int(selected.sum());y=danger(rows,epsilon);delta=np.array([r['delta_logp'] for r in rows]);flip=np.array([r['top1_flip'] for r in rows]);js=np.array([r['js'] for r in rows])
    loss=np.maximum(-delta,0);absolute=np.abs(delta)
    return dict(conditions=len(rows),skipped=n,skip_coverage=n/len(rows),dangerous_skipped=int(y[selected].sum()),
        dangerous_among_skipped=float(y[selected].mean()) if n else None,dangerous_recall=float(y[~selected].sum()/y.sum()) if y.sum() else None,
        mean_delta_logp=float(delta[selected].mean()) if n else None,p95_logp_loss=float(np.quantile(loss[selected],.95)) if n else None,p99_logp_loss=float(np.quantile(loss[selected],.99)) if n else None,
        p95_abs_delta_logp=float(np.quantile(absolute[selected],.95)) if n else None,p99_abs_delta_logp=float(np.quantile(absolute[selected],.99)) if n else None,max_abs_delta_logp=float(absolute[selected].max()) if n else None,
        top1_flip_rate=float(flip[selected].mean()) if n else None,js_divergence=float(js[selected].mean()) if n else None)


def intervals(rows,masks,epsilon,family_size):
    """Resample whole image clusters; never treat32 layers as independent."""
    ids,groups=np.unique([r['image_sha256'] for r in rows],return_inverse=True);n=len(ids);R=len(masks)
    y=danger(rows,epsilon);delta=np.array([r['delta_logp'] for r in rows]);flip=np.array([r['top1_flip'] for r in rows]);js=np.array([r['js'] for r in rows])
    c=np.zeros((n,R,7))
    for j,s in enumerate(masks):np.add.at(c[:,j],groups,np.column_stack([s,s*y,s*flip,s*delta,s*js,y,np.ones(len(rows))]))
    rng=np.random.default_rng(SEED);w=rng.multinomial(n,np.full(n,1/n),size=2000).astype(np.float64)
    b=(w@c.reshape(n,-1)).reshape(2000,R,7);out=[]
    with np.errstate(divide='ignore',invalid='ignore'):
        samples=dict(skip_coverage=b[:,:,0]/b[:,:,6],dangerous_among_skipped=b[:,:,1]/b[:,:,0],
            dangerous_recall=1-b[:,:,1]/b[:,:,5],mean_delta_logp=b[:,:,3]/b[:,:,0],top1_flip_rate=b[:,:,2]/b[:,:,0],js_divergence=b[:,:,4]/b[:,:,0])
    for j in range(R):
        item=dict(n_image_clusters=n)
        for name,values in samples.items():
            good=values[:,j][np.isfinite(values[:,j])]
            lo,hi=np.quantile(good,[.025,.975]) if len(good) else [None,None]
            item.update({name+'_ci95_low':clean(lo),name+'_ci95_high':clean(hi),name+'_bootstrap_valid':len(good)})
        # Independent bounded clusters: valid but deliberately conservative ratio bound.
        # One question/new image gives<=32 conditions; old images may have more.
        maximum=float(c[:,:,6].max());r=np.sqrt(np.log(4*family_size/.05)/(2*n))
        lower=float(c[:,j,0].mean()/maximum)-r;upper=float(c[:,j,1].mean()/maximum)+r
        item['simultaneous_hoeffding_risk_upper95']=min(1.,upper/lower) if lower>0 else 1.
        item['risk_bound_method']='Image-bounded numerator/denominator Hoeffding with fixed-family union bound; conservative, not bootstrap zero-event certification.'
        out.append(item)
    return out


def evaluate(rows,part,frozen):
    curves=[];selected=[];calibration=[];exceptions=[]
    family=sum(len(v['curves'])+len(v['selected']) for v in frozen['eps'].values())
    for epsilon in EPS:
        entry=frozen['eps'][str(epsilon)];rules=entry['curves']+entry['selected'];masks=[];cache={}
        for rule in rules:
            key=digest(rule.get('model',rule.get('column')))
            if key not in cache:cache[key]=score_rule(rows,rule)
            score=cache[key];threshold=np.array([rule.get('per_layer_thresholds',[rule['threshold']]*32)[r['layer']] for r in rows])
            s=score<=threshold if rule['direction']=='low' else score>=threshold
            if rule.get('early_never'):s&=np.array([r['layer']>=2 for r in rows])
            masks.append(s)
        cis=intervals(rows,masks,epsilon,family)
        for j,(rule,s,ci) in enumerate(zip(rules,masks,cis)):
            row=dict(dataset_part=part,epsilon=epsilon,predictor=rule['name'],threshold=rule['threshold'],direction=rule['direction'],selected_on_calibration=j>=len(entry['curves']),recommendation_frozen=rule['name']==entry['recommended'],**risk_values(rows,s,epsilon),**ci)
            curves.append(row)
            if j>=len(entry['curves']):selected.append(row)
        for rule,s in zip(entry['selected'],masks[len(entry['curves']):]):
            if rule['kind']=='logistic':
                probs=cache[digest(rule['model'])];y=danger(rows,epsilon)
                for low in np.linspace(0,.9,10):
                    good=(probs>=low)&(probs<low+.1 if low<.9 else probs<=1)
                    calibration.append(dict(dataset_part=part,epsilon=epsilon,predictor=rule['name'],bin_low=float(low),n_conditions=int(good.sum()),mean_predicted=float(probs[good].mean()) if good.any() else None,observed_dangerous=float(y[good].mean()) if good.any() else None))
                for r in selected:
                    if r['predictor']==rule['name'] and r['epsilon']==epsilon and r['dataset_part']==part:
                        r['brier_score']=float(np.mean((probs-y)**2));r['log_loss']=float(np.mean(-y*np.log(np.maximum(probs,1e-12))-(1-y)*np.log(np.maximum(1-probs,1e-12))))
                        r['parameters']=sum(g['parameters']+g['calibration']['parameters'] for g in rule['model']['groups'].values())
                        r['training_seconds']=sum(g['training_seconds']+g['calibration']['training_seconds'] for g in rule['model']['groups'].values())
            if epsilon==.03:
                groups=[('task',t,np.array([r['task']==t for r in rows])) for t in TASKS]
                groups += [('layer',str(i),np.array([r['layer']==i for r in rows])) for i in range(32)]
                for kind,name,mask in groups:
                    sub=[r for r,ok in zip(rows,mask) if ok];ss=s[mask]
                    metric=risk_values(sub,ss,epsilon);ci=intervals(sub,[ss],epsilon,family*38)[0]
                    exceptions.append(dict(dataset_part=part,predictor=rule['name'],epsilon=epsilon,group_type=kind,group=name,**metric,**ci))
        print('Evaluated',part,epsilon,flush=True)
    return curves,selected,calibration,exceptions


def export_heads():
    if (OUT/'head_artifact_manifest.json').exists():
        manifest=json.loads((OUT/'head_artifact_manifest.json').read_text())
        for r in manifest['lossless_partitions']:
            if sha256(OUT/r['path'])!=r['sha256']:raise ValueError('Changed completed per-head partition')
        return
    if (OUT/'cheap_features_per_head.csv').exists():
        archive=OUT/f'audit/partial_head_export_{time.time_ns()}';archive.mkdir(parents=True)
        shutil.move(OUT/'cheap_features_per_head.csv',archive/'cheap_features_per_head.csv')
        if (OUT/'per_head_compressed').exists():shutil.move(OUT/'per_head_compressed',archive/'per_head_compressed')
    manifest=json.loads((OUT/'sample_manifest.json').read_text());anchor=None;maxdiff=None;headrows=0;chunks=[];kv_sum=None;kv_sum2=None;old_count=0
    plain=(OUT/'cheap_features_per_head.csv').open('x',newline='');columns=['dataset_part','task','question_id','image_id','layer','head',*HEAD_FEATURES,'oracle_bos_mass','oracle_weighted_bos_contribution']
    writer=csv.DictWriter(plain,fieldnames=columns);writer.writeheader()
    for part in ['old','new']:
        for task in TASKS:
            path=OUT/f'per_head_compressed/{part}_{task}.csv.gz';path.parent.mkdir(exist_ok=True)
            with gzip.open(path,'xt',newline='') as f:
                out=csv.DictWriter(f,fieldnames=columns);out.writeheader()
                for s in manifest[part+'_samples']:
                    if s['task']!=task:continue
                    name=s['task']+'__'+s['question_id'];ck=json.loads((OUT/f"{part}/shard{s['rank']}/checkpoints/{name}.json").read_text())
                    if sha256(ck['raw_path'])!=ck['raw_sha256']:raise ValueError('Changed captured vectors')
                    with np.load(ck['raw_path']) as a:
                        head=a['head_features'];mass=a['bos_mass'];kv=a['qkv'][:,1:].astype(np.float32)
                        if part=='old':
                            if anchor is None:
                                anchor=kv.copy();maxdiff=np.zeros((32,2,32),dtype=np.float32);kv_sum=np.zeros_like(kv,dtype=np.float64);kv_sum2=np.zeros_like(kv,dtype=np.float64)
                            maxdiff=np.maximum(maxdiff,np.abs(kv-anchor).max(-1))
                            kv_sum+=kv;kv_sum2+=kv.astype(np.float64)**2;old_count+=1
                    for layer in range(32):
                        for h in range(32):
                            row=dict(dataset_part=part,task=task,question_id=s['question_id'],image_id=s['image_id'],layer=layer,head=h,**dict(zip(HEAD_FEATURES,map(float,head[layer,h]))),oracle_bos_mass=float(mass[layer,h]),oracle_weighted_bos_contribution=float(mass[layer,h]*head[layer,h,9]))
                            writer.writerow(row);out.writerow(row);headrows+=1
            chunks.append(dict(path=str(path.relative_to(OUT)),sha256=sha256(path),bytes=path.stat().st_size))
    plain.close()
    freeze_json(OUT/'head_artifact_manifest.json',dict(rows=headrows,uncompressed_local_csv=dict(path=str(OUT/'cheap_features_per_head.csv'),sha256=sha256(OUT/'cheap_features_per_head.csv'),bytes=(OUT/'cheap_features_per_head.csv').stat().st_size,git='Excluded only the oversized redundant uncompressed CSV; lossless per-task gzip partitions are committed.'),lossless_partitions=chunks))
    std=np.sqrt(np.maximum(kv_sum2/old_count-(kv_sum/old_count)**2,0)).mean(-1)
    records=[dict(layer=l,head=h,key_max_abs_from_first_old=float(maxdiff[l,0,h]),value_max_abs_from_first_old=float(maxdiff[l,1,h]),key_mean_coordinate_std=float(std[l,0,h]),value_mean_coordinate_std=float(std[l,1,h]),old_questions=old_count) for l in range(32) for h in range(32)]
    write_csv(OUT/'bos_kv_variation.csv',records)
    freeze_json(OUT/'bos_anchor_audit.json',dict(max_key_difference=float(maxdiff[:,0].max()),max_value_difference=float(maxdiff[:,1].max()),all_old_exact=bool(np.all(maxdiff==0)),online_anchor_used=False,reason='Always read actual current BOS K/V. No unverified fixed anchor assumed.'))


def mechanisms(oldrows):
    lookup={(r['task'],r['question_id'],r['layer']):r for r in oldrows};joined=[]
    with (UPDATES/'per_question_updates.csv').open() as f:
        for u in csv.DictReader(f):
            key=(u['task'],u['question_id'],int(u['layer']));r=lookup[key]
            joined.append(dict(r,attention_ratio=float(u['attention_ratio']),mlp_ratio=float(u['mlp_ratio']),norm_attention=float(u['norm_attention']),norm_mlp=float(u['norm_mlp']),cos_attention_mlp=float(u['cos_attention_mlp'])))
    if len(joined)!=19200:raise ValueError('Wrong reused update sample count')
    write_csv(OUT/'matched_update_mechanisms.csv',joined);cor=[]
    for task in [*TASKS,'ALL']:
        sub=[r for r in oldrows if task=='ALL' or r['task']==task]
        for region,layers in [('all',range(32)),('L0-L1',range(2)),('L2-L31',range(2,32))]:
            rr=[r for r in sub if r['layer'] in layers]
            for a,b in [('oracle_bos_mass','delta_logp'),('bos_cos_mean','oracle_bos_mass'),('bos_score_mean','delta_logp'),('residual_norm','delta_logp')]:
                x=np.array([r[a] for r in rr]);y=np.array([r[b] for r in rr])
                for scope in ['pooled_rows','within_task_layer_centered']:
                    xx=x.copy();yy=y.copy()
                    if scope.startswith('within'):
                        for t in TASKS:
                            for l in layers:
                                idx=[i for i,r in enumerate(rr) if r['task']==t and r['layer']==l]
                                if idx:xx[idx]-=xx[idx].mean();yy[idx]-=yy[idx].mean()
                    pearson,spearman=descriptive_correlations(xx,yy)
                    cor.append(dict(task=task,region=region,scope=scope,x=a,y=b,pearson=pearson,spearman=spearman,n_rows=len(rr),interpretation='Descriptive, repeated image/layer rows not independent; no causal p-value.'))
    write_csv(OUT/'descriptive_correlations.csv',cor)
    return joined


def save(fig,name):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/f'{name}.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)


def plots(oldrows,newrows,curves,selected,calibration,exceptions,joined):
    names=list(dict.fromkeys(r['predictor'] for r in curves));colors={n:plt.get_cmap('tab20')(i/20) for i,n in enumerate(names)}
    fig,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
    for ax,epsilon in zip(axes,EPS):
        for name in names:
            rs=sorted([r for r in curves if r['dataset_part']=='new_exploratory' and r['epsilon']==epsilon and r['predictor']==name and r['skipped']],key=lambda r:r['skip_coverage'])
            if rs:ax.plot([r['skip_coverage'] for r in rs],[r['dangerous_among_skipped'] for r in rs],'.-',ms=3,lw=1,color=colors[name],label=name)
        ax.axhline(.01,color='black',ls='--',lw=.7);ax.set_title(f'epsilon={epsilon}; exploratory new images');ax.set_xlabel('Skip coverage');ax.set_ylabel('Dangerous among skipped');ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=5,loc='outside lower center',fontsize=8)
    save(fig,'figure1_risk_coverage')
    for number,xkey,ykey,title in [(2,'bos_cos_mean','oracle_bos_mass','Cheap BOS cosine vs true post-Attention BOS mass'),(3,'bos_score_mean','delta_logp','Cheap BOS QK score vs signed ablation Delta LogP')]:
        fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
        for ax,task in zip(axes.flat,TASKS):
            rows=[r for r in oldrows if r['task']==task];x=np.array([r[xkey] for r in rows]);y=np.array([r[ykey] for r in rows]);l=[r['layer'] for r in rows]
            points=ax.scatter(x,y,c=l,cmap='viridis',s=3,alpha=.25,rasterized=True)
            edges=np.unique(np.quantile(x,np.linspace(0,1,16)));bx=[];by=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                mask=(x>=lo)&(x<=hi)
                if mask.any():bx.append(x[mask].mean());by.append(y[mask].mean())
            ax.plot(bx,by,'o-',color='#c13735',ms=3);ax.set_title(f'{task}: original500, all32 layers');ax.set_xlabel(xkey);ax.set_ylabel(ykey)
        fig.colorbar(points,ax=axes,label='Layer');fig.suptitle(title+'; red=descriptive quantile-bin means, not a fitted causal curve');save(fig,f'figure{number}_cheap_relationship')
    # Every task/layer is retained, rather than showing only favorable aggregate points.
    for task in TASKS:
        fig,axes=plt.subplots(4,8,figsize=(24,12),layout='constrained')
        for layer,ax in enumerate(axes.flat):
            rows=[r for r in oldrows if r['task']==task and r['layer']==layer]
            ax.scatter([r['bos_score_mean'] for r in rows],[abs(r['delta_logp']) for r in rows],s=3,alpha=.3,rasterized=True);ax.set_title(f'L{layer}, n=500',fontsize=9);ax.tick_params(labelsize=6)
        fig.suptitle(f'{task}: BOS score vs absolute Delta LogP; all32 layers, no tail clipping');save(fig,f'figure3_detail_{task}')
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ranges=[(0,1),(2,5),(14,16),(25,31)];labels=[f'L{a}-L{b}' for a,b in ranges]
    for ax,key in zip(axes.flat,['oracle_bos_mass','bos_cos_mean','bos_score_mean','delta_logp']):
        means=[];low=[];high=[]
        for a,b in ranges:
            data={};images={}
            for r in oldrows:
                if a<=r['layer']<=b:
                    keyid=r['task']+'__'+r['question_id'];data.setdefault(keyid,[]).append(r[key]);images[keyid]=r['image_sha256']
            vals=np.array([np.mean(v) for v in data.values()]);ci=cluster_bootstrap(vals,list(images.values()),2000)[:,0]
            means.append(vals.mean());low.append(vals.mean()-ci[0]);high.append(ci[1]-vals.mean())
        ax.errorbar(labels,means,yerr=[low,high],fmt='o',capsize=4);ax.set_ylabel(key);ax.grid(alpha=.15)
    fig.suptitle('L0-L1 vs later regions: question-level region means, image-cluster95% CI');save(fig,'figure4_early_special_case')
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,key in zip(axes.flat,['v_norm_mean','projected_v_norm_mean','residual_norm','attention_ratio','mlp_ratio','cos_attention_mlp']):
        vals=[]
        for layer in range(32):
            rs=[r for r in joined if r['layer']==layer];v=np.array([r[key] for r in rs]);ci=cluster_bootstrap(v,[r['image_sha256'] for r in rs],2000)[:,0];vals.append((v.mean(),*ci))
        vals=np.array(vals);ax.plot(range(32),vals[:,0],'.-');ax.fill_between(range(32),vals[:,1],vals[:,2],alpha=.2);ax.set_xlabel('Layer');ax.set_ylabel(key)
    fig.suptitle('Strictly matched prior update600: native observations only; not causal compensation');save(fig,'figure5_value_residual_update')
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for part,color in [('calibration','#327c97'),('new_exploratory','#bc4c37')]:
        rr=[r for r in calibration if r['predictor']=='BOS-QK+Residual' and r['epsilon']==.03 and r['dataset_part']==part and r['n_conditions']]
        axes[0].plot([r['mean_predicted'] for r in rr],[r['observed_dangerous'] for r in rr],'o-',label=part,color=color)
    axes[0].plot([0,1],[0,1],'k--',lw=1);axes[0].set_xlabel('Mean predicted danger');axes[0].set_ylabel('Observed danger');axes[0].legend()
    for ax,group in zip(axes[1:],['layer','task']):
        rr=[r for r in exceptions if r['dataset_part']=='new_exploratory' and r['predictor']=='BOS-QK+Residual' and r['group_type']==group]
        ax.bar([r['group'] for r in rr],[0 if r['dangerous_among_skipped'] is None else r['dangerous_among_skipped'] for r in rr]);ax.tick_params(axis='x',labelrotation=90,labelsize=7);ax.set_ylabel('Danger among skipped (see CSV denominators/CIs)');ax.set_title(group)
    fig.suptitle('Calibration/error analysis, epsilon=.03; no-selected groups have undefined risk, bars shown at0 only');save(fig,'figure6_calibration_errors')
    fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
    for part,marker in [('calibration','o'),('new_exploratory','x')]:
        rs=[r for r in selected if r['dataset_part']==part and r['epsilon']==.03]
        x=np.arange(len(rs));axes[0].scatter(x,[r['skip_coverage'] for r in rs],label=part,marker=marker)
        axes[1].scatter(x,[np.nan if r['dangerous_among_skipped'] is None else r['dangerous_among_skipped'] for r in rs],label=part,marker=marker)
        if part=='new_exploratory':axes[1].scatter(x,[r['simultaneous_hoeffding_risk_upper95'] for r in rs],marker='_',label='Conservative simultaneous upper95')
    for ax in axes:ax.set_xticks(x,[r['predictor'] for r in rs],rotation=65,ha='right',fontsize=8);ax.legend();ax.grid(alpha=.15)
    axes[0].set_ylabel('Frozen-rule skip coverage');axes[1].set_ylabel('Risk / upper bound');axes[1].axhline(.01,color='black',ls='--',lw=.8)
    fig.suptitle('Thresholds selected on calibration only; new set is exploratory due task shortfall');save(fig,'figure7_threshold_comparison')


def report(selected):
    d=json.loads((OUT/'data_manifest.json').read_text());f=json.loads((OUT/'frozen_predictors.json').read_text());c=verify()
    lines=['# BOS-Guided Cheap Bypass Predictor Discovery','','Status: completed exploratory study; NOT independent confirmation.','',
      f"Code `{c['code_commit']}`. Original3000 training/calibration questions; {d['new_questions']} unused-image exploratory questions instead of required600. LogicalObj shortfall is explicit, no replacement, resampling or synthetic labels.",
      'All original96000 single-layer labels reused. Old questions receive only one new native feature forward. New questions receive Full plus32 original single-layer P-1 Attention-output ablations. No LLaVA training, Decode study, joint gate or custom accelerated kernel.',
      'True BOS mass, Attention/MLP output norms are post-computation oracle/mechanism features, excluded from online predictor inputs. All risks concern Full original top1, not answer accuracy. All policies and thresholds froze before new-image model outputs.','',
      '## Frozen Thresholds and Risk','',
      'Primary epsilon=.03 below; all .01/.03/.05 results retained in CSV. Maximum calibration coverage under observed1% risk is exploration, not a guarantee. Ordinary2000 image-cluster bootstrap intervals do not certify zero-event risk. Additional bounded-cluster simultaneous Hoeffding upper bounds assume independent image clusters and are conservative; they are not unconditional population safety guarantees. Task shortfall also prevents the requested confirmation claim.','',
      '| Predictor | Threshold | New skip coverage | Dangerous/skipped | Risk | Bootstrap95%CI | Simultaneous upper95 |',
      '|---|---:|---:|---:|---:|---|---:|']
    for r in selected:
        if r['dataset_part']!='new_exploratory' or r['epsilon']!=.03:continue
        risk='undefined' if r['dangerous_among_skipped'] is None else f"{100*r['dangerous_among_skipped']:.3f}%"
        lo=r['dangerous_among_skipped_ci95_low'];hi=r['dangerous_among_skipped_ci95_high'];ci='undefined' if lo is None else f'[{100*lo:.3f}%,{100*hi:.3f}%]'
        lines.append(f"| {r['predictor']} | {r['threshold']:.6g} | {100*r['skip_coverage']:.2f}% | {r['dangerous_skipped']}/{r['skipped']} | {risk} | {ci} | {100*r['simultaneous_hoeffding_risk_upper95']:.2f}% |")
    lines+=['','Calibration-only recommended names (never chosen from new-set performance):',f'`{json.dumps({e:v["recommended"] for e,v in f["eps"].items()})}`','',
      '## Boundaries','',
      'Single-layer labels cannot be added to infer joint-gate harm. Joint dynamic paths and full-answer accuracy were intentionally not executed. No safe skipping, actual computation omission, end-to-end latency improvement or cross-model generality is claimed. A P-1 row is a small fraction of full Prefill. Microbenchmarks report PyTorch operations and dispatch, not a custom kernel speedup.','',
      '## Artifacts','',
      'cheap_features_per_question.csv retains train/calibration/new-exploratory labels, all32 layers and online fields. Full per-head CSV remains locally with SHA256; identical lossless gzip task partitions are in Git. Per-question raw q/k/v, sampled Keys, residuals and full/ablated logits remain in external resources with SHA256.']
    for p in sorted((OUT/'figures').glob('figure[1-7]_*.png')):lines+=['',f'![{p.stem}](figures/{p.name})']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'research_summary.md').write_text('# Research Summary\n\nExploratory only; see report.md and the post-run Chinese interpretation.\n\nAll six tasks, all32 layers, all thresholds and negative results are retained. No independent600-question confirmation, joint-gate result or actual speedup is claimed.\n')


def main():
    verify();gate('old');gate('new')
    if (OUT/'completion.json').exists():raise ValueError('Already complete')
    frozen=json.loads((OUT/'frozen_predictors.json').read_text());oldrows=load_rows('old');newrows=load_rows('new');cal=[r for r in oldrows if r['split']=='calibration']
    write_csv(OUT/'cheap_features_per_question.csv',oldrows+newrows)
    curves=[];selected=[];calibration=[];exceptions=[]
    for rows,part in [(cal,'calibration'),(newrows,'new_exploratory')]:
        a,b,c,d=evaluate(rows,part,frozen);curves+=a;selected+=b;calibration+=c;exceptions+=d
    write_csv(OUT/'risk_coverage.csv',curves);write_csv(OUT/'predictor_comparison.csv',selected)
    write_csv(OUT/'heldout_validation.csv',[r for r in selected if r['dataset_part']=='new_exploratory'])
    write_csv(OUT/'candidate_thresholds.csv',[dict(epsilon=float(e),predictor=r['name'],threshold=r['threshold'],direction=r['direction'],per_layer_thresholds=json.dumps(r.get('per_layer_thresholds')),early_never=r.get('early_never',False),calibration_skipped=r['calibration_skipped'],calibration_risk=r['calibration_risk']) for e,v in frozen['eps'].items() for r in v['selected']])
    write_csv(OUT/'calibration_bins.csv',calibration);write_csv(OUT/'layer_exception_analysis.csv',exceptions)
    cost=[]
    for rank in range(4):
        p=next((OUT/f'smoke/shard{rank}/checkpoints').glob('*.json'));ck=json.loads(p.read_text())
        cost.extend(dict(rank=rank,**r) for r in ck['checks']['cost'])
        for p in (OUT/f'new/shard{rank}').glob('actual_predictor_cost_*.json'):cost.extend(dict(rank=rank,**r) for r in json.loads(p.read_text()))
    write_csv(OUT/'compute_cost_microbench.csv',cost)
    freeze_json(OUT/'theoretical_cost.json',dict(heads=32,head_dim=128,
      bos_score_per_query='O(H*d) dot products plus projection/normalization. Does not know the Softmax denominator.',
      sampled_score_per_query='O(7*H*d), up to6 extra Keys; feature-reproduction projection cost is measured separately.',
      full_attention_query='QK and AV each approximately2*T*H*d FLOPs, Softmax O(T*H), Key+Value reads O(2*T*H*d).',
      full_prefill='Native dense Attention computes T rows; gating only P-1 cannot remove all other T-1 rows or other projection/MLP work.',
      bos_read='2*32*128*2=16384 bytes for FP16 K/V, not including weights, projected norm Gram, predictor or dispatch.',
      sampled_extra_key_read='Up to6*32*128*2=49152 additional bytes for FP16 Keys.',
      gram='Offline per-layer/per-head W_head.T W_head;64MiB total FP32. No fixed BOS K/V anchor is assumed.',
      actual_acceleration_evidence=False,joint_gating_executed=False))
    export_heads();joined=mechanisms(oldrows);plots(oldrows,newrows,curves,selected,calibration,exceptions,joined);report(selected)
    verify();files={str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and p.suffix not in ['.lock','.tmp'] and not p.name.startswith('console') and p.name!='cheap_features_per_head.csv'}
    freeze_json(OUT/'completion.json',dict(status='complete_exploratory',independent_confirmation=False,historical_labels_reused=96000,old_feature_questions=3000,new_feature_questions=len(newrows)//32,new_single_layer_labels=len(newrows),joint_gate=False,files_sha256=files,code_commit=verify()['code_commit']))


if __name__=='__main__':main()

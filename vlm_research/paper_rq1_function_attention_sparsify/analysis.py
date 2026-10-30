import csv
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.dataset_registry import digest,sha256
from paper_rq1.analyze import write_csv
from paper_rq1_single_layer_gqa6.run import atomic_json
from .protocol import OUT,PROTOCOL,LAYERS,MODES


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def paired_ci(values,seed=20261030):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    estimates=np.concatenate([values[rng.integers(0,len(values),(min(500,10000-i),len(values)))].mean(1) for i in range(0,10000,500)])
    return np.quantile(estimates,[.025,.975]).tolist()


def image_ci(rows,field):
    groups=defaultdict(list)
    for r in rows:groups[r['image_sha256']].append(float(r[field]))
    return paired_ci([np.mean(v) for v in groups.values()])


def figure(fig,name):
    path=OUT/'figures';path.mkdir(exist_ok=True)
    fig.savefig(path/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(path/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def stage_a():
    config=json.loads((OUT/'execution_manifest.json').read_text());binding=digest(config)
    manifest=json.loads((OUT/'sample_manifest.json').read_text());rows=[];checkpoints=[]
    for rank in range(3):
        complete=json.loads((OUT/f'stage_a/shard{rank}/completion.json').read_text())
        if complete['status']!='complete' or complete['binding']!=binding:raise ValueError('Incomplete/unequal shards')
    for s in manifest['development']:
        path=OUT/f"stage_a/shard{s['rank']}/checkpoints/{s['image_identity'].replace(':','_')}.json"
        r=json.loads(path.read_text())
        if r['sample_sha256']!=digest(s) or r['binding']!=binding or sha256(r['raw_path'])!=r['raw_sha256']:raise ValueError('Invalid checkpoint')
        if len(r['rows'])!=21*len(s['selected']):raise ValueError('Missing retained condition')
        rows.extend(r['rows']);checkpoints.append(r)
    if len({r['image_id'] for r in checkpoints})!=400:raise ValueError('Duplicate/incomplete images')
    write_csv(OUT/'source_retention_per_token.csv',rows)
    summaries=[];groups=defaultdict(list)
    for r in rows:groups[(r['role'],r['layer'],r['algorithm'],r['mode'])].append(r)
    for (role,layer,algorithm,mode),rs in groups.items():
        loss=np.array([r['loss'] for r in rs]);lo,hi=image_ci(rs,'loss')
        summaries.append(dict(role=role,layer=layer,algorithm=algorithm,mode=mode,n=len(rs),
            mean_loss=float(loss.mean()),median_loss=float(np.median(loss)),p90_loss=float(np.quantile(loss,.90)),
            p95_loss=float(np.quantile(loss,.95)),p99_loss=float(np.quantile(loss,.99)),max_loss=float(loss.max()),
            mean_abs_delta_logp=float(np.mean([r['abs_delta_logp'] for r in rs])),top1_flips=sum(r['top1_flip'] for r in rs),
            flip_rate=float(np.mean([r['top1_flip'] for r in rs])),mean_js=float(np.mean([r['js_divergence'] for r in rs])),
            mean_retained_ratio=float(np.mean([r['retained_keys']/r['full_keys'] for r in rs])),ci95_low=lo,ci95_high=hi))
    write_csv(OUT/'source_retention_summary.csv',summaries)
    key=next(r for r in summaries if r['role']=='Function' and r['layer']==31 and r['algorithm']=='A2' and r['mode']=='BOS+History')
    limits=PROTOCOL['risk_gate'];passed=key['mean_loss']<=limits['function_l31_bos_history_max_mean_loss'] and key['p95_loss']<=limits['max_p95_loss'] and key['flip_rate']<=limits['max_flip_rate']
    atomic_json(OUT/'stage_a_gate.json',dict(dynamic_source_gate_passed=bool(passed),observed=key,predeclared_limits=limits,
        stage_a_complete=True,images=400,targets=len(rows)//21,interventions=len(rows),
        historical_max_logit_difference=max(r['historical_max_logit_difference'] for r in checkpoints),
        reasoning='All three conservative exploratory gates must pass; not a safety conclusion.',
        next='Train/calibrate frozen cheap classifiers. Dynamic free generation only if both source and classifier gates pass.'))
    write_csv(OUT/'severe_source_cases.csv',sorted(rows,key=lambda r:r['loss'],reverse=True)[:100])
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    colors=['#176b91','#b24039','#4b7951']
    for ai,algorithm in enumerate(['A1','A2']):
        for mi,mode in enumerate(MODES):
            rs=[next(r for r in summaries if r['role']=='Function' and r['layer']==layer and r['algorithm']==algorithm and r['mode']==mode) for layer in LAYERS]
            for ax,field in zip(axes,['mean_loss','mean_abs_delta_logp','flip_rate']):
                ax.plot(LAYERS,[r[field] for r in rs],marker='o',linestyle='--' if algorithm=='A1' else '-',color=colors[mi],label=algorithm+' '+mode)
    for ax,label in zip(axes,['Mean Full-token LogP loss (nat)','Mean absolute LogP change','Top-1 flip rate']):
        ax.set(xlabel='Single target layer',ylabel=label,xticks=LAYERS);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Function source retention: original denominator vs sparse renormalization');figure(fig,'figure1_function_source_retention')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,mode in zip(axes,MODES):
        one={(r['image_id'],r['t']):r for r in rows if r['layer']==31 and r['mode']==mode and r['algorithm']=='A1'}
        for role,color in zip(['Function','Visual','Relation'],colors):
            rs=[r for r in rows if r['layer']==31 and r['mode']==mode and r['algorithm']=='A2' and r['role']==role]
            ax.scatter([one[(r['image_id'],r['t'])]['loss'] for r in rs],[r['loss'] for r in rs],s=9,alpha=.45,label=role,color=color)
        ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7);ax.set(title=mode,xlabel='A1 original denominator loss',ylabel='A2 sparse loss (nat)');ax.legend(fontsize=8)
    figure(fig,'figure2_subtraction_vs_sparse')
    print(json.dumps(json.loads((OUT/'stage_a_gate.json').read_text()),indent=2),flush=True)


if __name__=='__main__':stage_a()

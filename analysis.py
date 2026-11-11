"""Question-weighted, image-cluster paired exploratory summaries."""
from collections import defaultdict
import numpy as np
from common import *


def contribution(task,rows):
    if task!='MMBench':
        return [(r['image_sha256'],float(r['score'])) for r in rows]
    groups=defaultdict(list)
    for r in rows:
        groups[r['circular_group']].append(r)
    return [(g[0]['image_sha256'],float(all(x['score']==1 for x in g))) for g in groups.values()]


def paired(task,baseline,candidate,repeats=20000):
    a,b=contribution(task,baseline),contribution(task,candidate)
    assert [x[0] for x in a]==[x[0] for x in b]
    groups=defaultdict(lambda:np.zeros(3))
    for (cluster,x),(_,y) in zip(a,b):
        groups[cluster]+=np.array([x,y,1])
    values=np.array(list(groups.values()))
    delta=100*(values[:,1].sum()-values[:,0].sum())/values[:,2].sum()
    rng=np.random.default_rng(SEED)
    draws=[]
    for start in range(0,repeats,128):
        w=rng.multinomial(len(values),np.full(len(values),1/len(values)),size=min(128,repeats-start))
        aggregate=w@values
        draws.extend(100*(aggregate[:,1]-aggregate[:,0])/aggregate[:,2])
    lo,hi=np.quantile(draws,[.025,.975]) if draws else [None,None]
    return dict(delta_pp=float(delta),ci_low=None if lo is None else float(lo),ci_high=None if hi is None else float(hi),
        harm=sum(y<x for (_,x),(_,y) in zip(a,b)),rescue=sum(y>x for (_,x),(_,y) in zip(a,b)),
        unit_n=len(a),clusters=len(values),bootstrap_repeats=repeats)


def summarize(split,view='Unified',ci=False):
    rows=[r for r in json.loads((OUT/'sample_manifest.json').read_text()) if r['split']==split]
    expected={r['key']:r for r in rows}
    all_records=records()
    methods=defaultdict(dict)
    for r in all_records.values():
        if r.get('stage')==split and r.get('view')==view:
            methods[r['condition']][r['key']]=r
    hist=dict(read_store(OUT/'historical.sqlite'))
    summaries=[]
    for name,by_key in sorted(methods.items()):
        for task in TASKS+PROTECTION:
            planned=[r for r in rows if r['analysis_task']==task]
            if not planned:
                continue
            if not all(r['key'] in by_key for r in planned):
                continue
            valid=[r for r in planned if 'score' in by_key[r['key']]]
            if task in ['MMBench','MME']:
                complete_groups=defaultdict(list)
                for r in planned:
                    complete_groups[r['circular_group' if task=='MMBench' else 'pair_id']].append(r)
                valid=[r for g in complete_groups.values() if all('score' in by_key[x['key']] for x in g) for r in g]
            selected=[by_key[r['key']] for r in valid]
            if not selected:
                continue
            baseline=[]
            native=[]
            for row in valid:
                for label,destination in [('SVH',baseline),('Native',native)]:
                    r=dict(hist[row['key']+'|'+label])
                    r.update(score=r.get('score',r.get('scoring',{}).get('score')))
                    destination.append(r)
            if task=='MME':
                from vlm_research.qwen3vl_overnight.metrics import metric as mme_metric,paired as mme_paired
                mapping={r['key']:{'cluster':r['image_sha256']} for r in valid}
                def convert(pool):
                    return [dict(r,scoring={'score':r['score']},mme_task=expected[r['key']]['task'],pair_id=expected[r['key']]['pair_id']) for r in pool]
                a,b,n=convert(baseline),convert(selected),convert(native)
                stats=mme_paired('MME',a,b,mapping,repeats=20000 if ci else 0)
                summaries.append(dict(split=split,view=view,condition=name,task=task,planned_rows=len(planned),valid_rows=len(valid),
                    blocked_rows=len(planned)-len(valid),k=len(selected[0]['layers']),layers=selected[0]['layers'],
                    accuracy=mme_metric('MME',b,mapping),svh=mme_metric('MME',a,mapping),native=mme_metric('MME',n,mapping),
                    delta_pp=stats['delta'],ci_low=stats['low'],ci_high=stats['high'],unit='MME_POINTS_NOT_PERCENT',
                    ci_status=stats.get('ci_status'),harm=sum(y['score']<x['score'] for x,y in zip(baseline,selected)),
                    rescue=sum(y['score']>x['score'] for x,y in zip(baseline,selected))))
                continue
            vals=contribution(task,selected)
            sv=contribution(task,baseline)
            nv=contribution(task,native)
            stats=paired(task,baseline,selected,20000 if ci else 0)
            native_stats=paired(task,native,selected,20000 if ci else 0)
            summaries.append(dict(split=split,view=view,condition=name,task=task,planned_rows=len(planned),valid_rows=len(valid),
                blocked_rows=len(planned)-len(valid),k=len(selected[0]['layers']),layers=selected[0]['layers'],
                accuracy=100*float(np.mean([x[1] for x in vals])),svh=100*float(np.mean([x[1] for x in sv])),
                native=100*float(np.mean([x[1] for x in nv])),
                delta_native_pp=native_stats['delta_pp'],native_ci_low=native_stats['ci_low'],native_ci_high=native_stats['ci_high'],
                first_token_flip=sum(r['first_token_flip_vs_svh'] for r in selected),
                eos_rate=float(np.mean([r['eos'] for r in selected])),truncated=sum(r['truncated'] for r in selected),
                empty=sum(r['empty'] for r in selected),**stats))
    table(OUT/f'{split}_{view}_summary.csv',summaries)
    return summaries


def rank(summaries):
    by=defaultdict(dict)
    for row in summaries:
        by[row['condition']][row['task']]=row
    out=[]
    for name,tasks in by.items():
        if 'TextVQA' not in tasks:
            continue
        tv=tasks['TextVQA']
        protected=[r['delta_pp'] for k,r in tasks.items() if k!='TextVQA']
        out.append(dict(condition=name,textvqa=tv['accuracy'],delta_svh=tv['delta_pp'],k=tv['k'],layers=tv['layers'],
            late_layers=sum(i>=22 for i in tv['layers']),protection_min=min(protected) if protected else None,
            protected=bool(protected) and min(protected)>=-1,all_tasks=len(tasks)==len(TASKS)))
    return sorted(out,key=lambda r:(-r['textvqa'],r['k'],r['condition']))

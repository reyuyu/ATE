"""Predeclared continuation rounds; no prior FINAL scores enter selection."""
import argparse
import collections
import csv
import numpy as np

from .runtime import *
from ..accuracy_autoresearch.dispatch import with_native
from ..accuracy_autoresearch import analyze as analysis


def data():
    return read(PRIOR/'sample_manifest.json'),read(PRIOR/'split_manifest.json')


def keys_for(split):
    pool,parts=data()
    return sorted(k for k,s in pool.items() if parts['image_split'][s['image_sha256']]==split)


def job(name,split,configs,samples=None,performance=False,finish=False):
    ensure_time(True); p=read(OUT/'execution_freeze.json'); path=OUT/'jobs'/(name+'.json')
    assert not path.exists()
    value=dict(id=name,split=split,configs=with_native(configs),samples=samples or keys_for(split),
        sources=sources(),samples_sha256=p['samples_sha256'],created_at=now(),performance=performance,
        finish_after=finish,hypothesis='Deeper All-R writing may recover quality; preserve P-1 text Attention. Post-selection continuation, not an independent test.')
    if split=='RECHECK': value['selection_sha256']=digest(read(OUT/'selection.json'))
    atomic(path,value); print('DISPATCH',name,len(value['samples']),len(value['configs']),flush=True)


def init():
    configure(); assert time.time()<deadline('candidate_freeze_by')
    assert not (OUT/'execution_freeze.json').exists()
    old=read(PRIOR/'execution_freeze.json'); state=hardware(); devices=[]
    for d in old['devices']:
        line=next(x for x in state['gpus'].splitlines() if x.startswith(str(d['index'])+','))
        used,free,util=[float(x.strip()) for x in line.split(',')[-3:]]
        if d['uuid'] not in state['processes'] and used<200 and free>22000 and util==0: devices.append(d)
    # Old image shards bind these two exact physical devices; do not move history across GPUs.
    assert devices==old['devices'],'Stop if the original paired devices are not both idle'
    assert all(sha(p)==h for p,h in old['weights'].items())
    assert 'OK' in (OUT/'logs/cpu_tests.txt').read_text()
    atomic(OUT/'resource_preflight.json',state)
    atomic(OUT/'execution_freeze.json',dict(at=now(),sources=sources(),devices=devices,
        samples_sha256=sha(PRIOR/'sample_manifest.json'),splits_sha256=sha(PRIOR/'split_manifest.json'),
        weights=old['weights'],clock=read(ROOT/'task_deadline.json'),previous_commit='60d262f83a981e49a614ef8213e5311c3c7c6469',
        sample_manifest_reference=str(PRIOR/'sample_manifest.json'),split_manifest_reference=str(PRIOR/'split_manifest.json')))
    atomic(OUT/'protocol.json',dict(at=now(),clock=old['clock'],
        previous_final_is_already_used=True,selection_uses_only_DEV_CHECK=True,
        next_evaluation_label='RECHECK: historical and previously inspected cohort, not independent FINAL',
        added_candidates=[config('all',mode,depth) for depth in (28,32) for mode in ('none','decode')],
        promote='Top two DEV worst-dataset losses; <=0.25pp ties prefer shallower then Decode-only',
        selection='Best CHECK worst-dataset loss among two promoted and frozen L24 All-R Decode-only reference; <=0.25pp ties use minimum matched-sequence E2E speed then KV',
        no_additional_prompt_or_token_search=True,no_new_untouched_confirmatory_sample_claim=True,
        max_original_candidates=16,forbidden_direct_Image='All candidate P-1 and all real Decode at every layer'))
    configs=[config('all',mode,depth) for depth in (28,32) for mode in ('none','decode')]
    job('01_coarse','DEV',configs,read(PRIOR/'split_manifest.json')['coarse'])
    job('02_dev','DEV',configs)


def load_round(name):
    configure(); j=read(OUT/'jobs'/(name+'.json')); pool,_=data(); records={}
    for k in j['samples']:
        records[k]={config_id(c):read_record(c,k,pool[k]) for c in j['configs']}
        assert all(v is not None for v in records[k].values()),k
    return j,pool,records


def summary(name):
    j,pool,r=load_round(name); rows=[]
    for d in ('GQA','VQAv2'):
        keys=[k for k in j['samples'] if pool[k]['dataset']==d]
        for c in j['configs']:
            b=config('native',head=c['logits_to_keep']); x=np.array([r[k][config_id(c)]['generation']['score'] for k in keys])
            y=np.array([r[k][config_id(b)]['generation']['score'] for k in keys])
            rows.append(dict(dataset=d,config_id=config_id(c),n=len(keys),accuracy=100*x.mean(),
                delta_pp=100*(x-y).mean(),harm=int(sum(x<y)),rescue=int(sum(x>y))))
    write_csv(OUT/'rounds'/name/'summary.csv',rows)
    for row in rows: print(row,flush=True)
    return rows


def promote():
    rows=summary('02_dev'); configs=[c for c in read(OUT/'jobs/02_dev.json')['configs'] if c['writer']!='native']
    def loss(c): return max(-r['delta_pp'] for r in rows if r['config_id']==config_id(c))
    remaining=list(configs); chosen=[]
    while len(chosen)<2:
        best=min(map(loss,remaining)); tied=[c for c in remaining if loss(c)<=best+.25]
        c=min(tied,key=lambda c:(c['cutoff'],c['answer_attention']!='decode'))
        chosen.append(c); remaining.remove(c)
    atomic(OUT/'promotion.json',dict(at=now(),configs=chosen,DEV_only=True))
    baseline=config('all','decode',24)
    job('03_check','CHECK',[config(),config(answer='both'),config('qsa','decode'),baseline]+chosen)
    job('04_check_performance','CHECK',[baseline]+chosen,
        read(PRIOR/'jobs/08_check_performance.json')['samples'],performance=True)


def analyze_round(name,perf=False,offline=False):
    configure(); analysis.OUT=OUT; analysis.load_round=load_round
    analysis.source_hashes=sources; analysis.read_record=read_record
    if perf: analysis.performance_summary(name)
    else: analysis.analyze(name,offline=offline)


def freeze():
    assert time.time()<deadline('candidate_freeze_by') and not (OUT/'selection.json').exists()
    rows=summary('03_check'); promoted=read(OUT/'promotion.json')['configs']
    candidates=[config('all','decode',24)]+promoted
    paired=list(csv.DictReader((OUT/'analysis/04_check_performance/performance_paired.csv').open()))
    def loss(c): return max(-r['delta_pp'] for r in rows if r['config_id']==config_id(c))
    def speed(c):
        subset=[r for r in paired if r['condition']==config_id(c) and r['baseline']==config_id(config('native'))]
        assert len(subset)==2 and all(int(r['same_sequence_n'])>=6 for r in subset)
        return min(float(r['same_sequence_e2e_speedup']) for r in subset)
    best=min(map(loss,candidates)); tied=[c for c in candidates if loss(c)<=best+.25]
    primary=max(tied,key=speed)
    faster=[]
    for c in candidates:
        if c==primary: continue
        compared=[r for r in paired if r['condition']==config_id(c) and r['baseline']==config_id(primary)]
        if len(compared)==2 and all(int(r['same_sequence_n'])>=6 and float(r['same_sequence_e2e_speedup'])>=1.02 for r in compared):
            faster.append(c)
    secondary=max(faster,key=speed) if faster else None
    selected=[primary]+([secondary] if secondary else [])
    selection=dict(at=now(),primary=primary,speed_secondary=secondary,
        primary_is_capacity_diagnostic=primary['cutoff']==32,
        secondary_rule='Inherited original 2% faster on BOTH matched-sequence dataset subsets, at least six each; addendum before continuation timing scores',
        ranking=[dict(config=c,worst_loss_pp=loss(c),speed=speed(c)) for c in candidates],
        prior_FINAL_not_used_for_selection=True,remaining_original_clock=read(ROOT/'task_deadline.json'),
        recheck_is_not_independent_confirmation=True)
    atomic(OUT/'selection.json',selection)
    configs=[]
    for head in (0,1):
        configs.extend([config('native',head=head),config(head=head),config(answer='both',head=head)]+
                       [dict(c,logits_to_keep=head) for c in selected])
    job('05_recheck','RECHECK',configs,keys_for('FINAL'))
    job('06_recheck_performance','RECHECK',configs,
        read(PRIOR/'jobs/10_final_performance.json')['samples'],performance=True,finish=True)
    print('FROZEN',selection,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action'); ap.add_argument('name',nargs='?')
    ap.add_argument('--offline',action='store_true',help='CPU-only recomputation from completed checkpoints; never dispatches GPU work')
    a=ap.parse_args()
    if a.offline and a.action not in ('analyze','performance'):
        ap.error('--offline is restricted to CPU-only analysis')
    if a.action=='init': init()
    elif a.action=='summary': summary(a.name)
    elif a.action=='promote': promote()
    elif a.action=='freeze': freeze()
    elif a.action=='analyze': analyze_round(a.name,offline=a.offline)
    elif a.action=='performance': analyze_round(a.name,True)

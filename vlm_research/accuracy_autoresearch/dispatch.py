"""Small immutable round manifests and descriptive selection summaries."""
import argparse
import collections
import csv
import numpy as np

from .common import *
from .worker import read_record


def keys_for(split):
    pool=read(OUT/'sample_manifest.json'); parts=read(OUT/'split_manifest.json')['image_split']
    return sorted(k for k,s in pool.items() if parts[s['image_sha256']]==split)


def with_native(configs):
    natives=[]
    for c in configs:
        for prompt in ('original',c['prompt_variant']):
            n=config('native',prompt=prompt,head=c['logits_to_keep'])
            if n not in natives:
                natives.append(n)
    return natives+[c for c in configs if c not in natives]


def job(name,split,configs,samples=None,hypothesis='',performance=False):
    ensure_time(True)
    assert len([c for c in configs if c['writer']!='native'])<=18
    p=read(OUT/'execution_freeze.json')
    configs=with_native(configs)
    path=OUT/'jobs'/(name+'.json')
    assert not path.exists()
    value=dict(id=name,split=split,configs=configs,samples=samples or keys_for(split),
        hypothesis=hypothesis,created_at=now(),sources=p['sources'],
        samples_sha256=p['samples_sha256'],performance=performance)
    if split=='FINAL':
        value['final_freeze_sha256']=digest(read(OUT/'final_candidate_freeze.json'))
    atomic(path,value)
    print('DISPATCH',name,split,len(value['samples']),'questions',len(configs),'conditions',flush=True)


def summary(name):
    j=read(OUT/'jobs'/(name+'.json')); pool=read(OUT/'sample_manifest.json')
    counts=collections.defaultdict(list); missing=[]; rows=[]
    for k in j['samples']:
        s=pool[k]
        for c in j['configs']:
            r=read_record(c,k,s)
            if r is None:
                missing.append((k,config_id(c))); continue
            native=config('native',prompt=c['prompt_variant'],head=c['logits_to_keep'])
            b=read_record(native,k,s)['generation']; g=r['generation']
            data=dict(sample=k,dataset=s['dataset'],task=s['task'],answer_type=s.get('answer_type',s['task']),
                config_id=config_id(c),score=g['score'],native_score=b['score'],
                delta=g['score']-b['score'],score_down=g['score']<b['score'],score_up=g['score']>b['score'],
                native_first_token_logp_loss=b['native_first_token_logp']-g['native_first_token_logp'],
                first_token_flip=b['token_ids'][0]!=g['token_ids'][0],
                answer=g['text'],native_answer=b['text'],tokens=g['token_ids'],native_tokens=b['token_ids'],
                sequence_changed=g['token_ids']!=b['token_ids'],eos=g['eos'],truncated=g['truncated'],empty=g['empty'])
            counts[s['dataset'],config_id(c)].append(data); rows.append(data)
    for (dataset,cid),records in sorted(counts.items()):
        rowscore=dict(round=name,dataset=dataset,config_id=cid,n=len(records),
            accuracy=100*np.mean([r['score'] for r in records]),native=100*np.mean([r['native_score'] for r in records]),
            delta_pp=100*np.mean([r['delta'] for r in records]),
            score_down=sum(r['score_down'] for r in records),score_up=sum(r['score_up'] for r in records),
            first_token_flips=sum(r['first_token_flip'] for r in records))
        rows.append(dict(summary=rowscore))
    tables=[r['summary'] for r in rows if 'summary' in r]
    if not missing:
        write_csv(OUT/'rounds'/name/'summary.csv',tables)
        write_csv(OUT/'rounds'/name/'per_question.csv',[r for r in rows if 'summary' not in r])
    print('ROUND',name,'missing',len(missing),flush=True)
    for r in tables:
        print(r['dataset'],r['config_id'],f"{r['accuracy']:.3f}",f"delta {r['delta_pp']:+.3f}",
              'harm/rescue',r['score_down'],r['score_up'],flush=True)
    return tables,missing


def freeze_execution():
    p=read(OUT/'data_freeze.json')
    state=hardware(); devices=[]
    for line in state['gpus'].splitlines():
        index,uuid,name,used,free,util=[s.strip() for s in line.split(',')]
        if int(index) in (2,3) and uuid not in state['processes'] and int(used)<200 and int(free)>22000 and int(util)==0:
            devices.append(dict(index=int(index),uuid=uuid,name=name))
    assert devices,'No authorized idle device; do not wait or preempt'
    if len(devices)==1:
        parts=read(OUT/'split_manifest.json')
        parts['image_shard']=dict.fromkeys(parts['image_shard'],0)
        atomic(OUT/'split_manifest.json',parts)
        p['splits_sha256']=sha(OUT/'split_manifest.json'); atomic(OUT/'data_freeze.json',p)
    assert all(sha(path)==h for path,h in p['weights'].items())
    assert 'OK' in (OUT/'logs/cpu_tests.txt').read_text()
    atomic(OUT/'resource_preflight.json',state)
    atomic(OUT/'execution_freeze.json',dict(at=now(),sources=source_hashes(),devices=devices,
        samples_sha256=p['samples_sha256'],splits_sha256=p['splits_sha256'],weights=p['weights'],
        clock=read(ROOT/'task_deadline.json'),cpu_test_log_sha256=sha(OUT/'logs/cpu_tests.txt')))
    candidates=[config(w,a) for w in ('all','qsa') for a in ('none','both','decode')]
    job('01_coarse','DEV',candidates,read(OUT/'split_manifest.json')['coarse'],
        'H1: Always Q+ASSISTANT can improve compressed writing without the old 50% cap. H2: Keeping P-1 text Attention while bypassing Decode may preserve first-token accuracy. Frozen 2x3 matrix; visual Prefill updates always retained.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action'); ap.add_argument('name',nargs='?'); a=ap.parse_args()
    if a.action=='freeze-execution': freeze_execution()
    elif a.action=='summary': summary(a.name)

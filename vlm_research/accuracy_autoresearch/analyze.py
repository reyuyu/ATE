"""CPU-only paired analysis. Never chooses a candidate using FINAL scores."""
import argparse
import collections
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import *
from .dispatch import keys_for
from .worker import read_record


def bootstrap(values, images, seed=SEED, family=1):
    values=np.asarray(values,dtype=float)
    if values.ndim==1:
        values=values[:,None]
    groups=sorted(set(images)); index={h:i for i,h in enumerate(groups)}
    sums=np.zeros((len(groups),values.shape[1])); sizes=np.zeros(len(groups))
    for h,v in zip(images,values):
        sums[index[h]]+=v; sizes[index[h]]+=1
    rng=np.random.default_rng(seed); chunks=[]
    for start in range(0,20000,256):
        picks=rng.integers(len(groups),size=(min(256,20000-start),len(groups)))
        chunks.append(sums[picks].sum(1)/sizes[picks].sum(1)[:,None])
    draws=np.concatenate(chunks)
    return dict(mean=values.mean(0),low=np.quantile(draws,.025,axis=0),high=np.quantile(draws,.975,axis=0),
        corrected_low=np.quantile(draws,.025/max(family,1),axis=0),
        corrected_high=np.quantile(draws,1-.025/max(family,1),axis=0),
        ni_lower=np.quantile(draws,.05/max(family,1),axis=0),images=len(groups))


def load_round(name):
    j=read(OUT/'jobs'/(name+'.json')); pool=read(OUT/'sample_manifest.json')
    records={}
    for k in j['samples']:
        records[k]={}
        for c in j['configs']:
            r=read_record(c,k,pool[k]); assert r is not None,(k,config_id(c))
            records[k][config_id(c)]=r
    return j,pool,records


def analyze(name,offline=False):
    if not offline: ensure_time()
    j,pool,records=load_round(name); folder=OUT/'analysis'/name
    configs=j['configs']; native=config('native')
    if j['split']=='FINAL':
        frozen=read(OUT/'final_candidate_freeze.json')
        assert digest(frozen)==j['final_freeze_sha256']
        family=frozen['noninferiority_family_size']
        ni_configs={config_id(dict(c,logits_to_keep=h)) for c in frozen['selected_candidates'] for h in (0,1)}
    else:
        family=max(1,len(configs)-1)*2
        ni_configs=set()
    allrows=[]; summaries=[]; pairs=[]; groups=[]
    for dataset in ('GQA','VQAv2'):
        keys=[k for k in j['samples'] if pool[k]['dataset']==dataset]
        images=[pool[k]['image_sha256'] for k in keys]
        for c in configs:
            cid=config_id(c)
            n=config('native',prompt=c['prompt_variant'],head=c['logits_to_keep'])
            nid=config_id(n)
            g=[records[k][cid]['generation'] for k in keys]
            base=[records[k][nid]['generation'] for k in keys]
            original=[records[k][config_id(config('native',head=c['logits_to_keep']))]['generation'] for k in keys]
            scores=np.array([x['score'] for x in g]); basescores=np.array([x['score'] for x in base])
            originalscores=np.array([x['score'] for x in original])
            s=bootstrap(np.column_stack((scores,100*(scores-basescores),100*(scores-originalscores))),images,family=family)
            summary=dict(round=name,split=j['split'],dataset=dataset,config_id=cid,n=len(keys),image_clusters=s['images'],
                accuracy=100*s['mean'][0],accuracy_ci_low=100*s['low'][0],accuracy_ci_high=100*s['high'][0],
                delta_native_pp=s['mean'][1],delta_ci_low=s['low'][1],delta_ci_high=s['high'][1],
                corrected_low=s['corrected_low'][1],corrected_high=s['corrected_high'][1],
                ni_one_sided_lower=s['ni_lower'][2],ni_comparison_predeclared=cid in ni_configs,
                ni_minus1_supported=bool(s['ni_lower'][2]>-1) if cid in ni_configs else None,
                delta_original_native_pp=s['mean'][2],original_delta_ci_low=s['low'][2],original_delta_ci_high=s['high'][2],
                original_corrected_low=s['corrected_low'][2],original_corrected_high=s['corrected_high'][2],
                score_down_original_native=int(sum(scores<originalscores)),score_up_original_native=int(sum(scores>originalscores)),
                perfect_to_nonperfect_original=int(sum((originalscores==1)&(scores<1))),
                nonperfect_to_perfect_original=int(sum((originalscores<1)&(scores==1))),
                score_down=int(sum(x['score']<y['score'] for x,y in zip(g,base))),
                score_up=int(sum(x['score']>y['score'] for x,y in zip(g,base))),
                first_token_flips=int(sum(x['token_ids'][0]!=y['token_ids'][0] for x,y in zip(g,base))),
                mean_signed_first_token_logp_loss=float(np.mean([y['native_first_token_logp']-x['native_first_token_logp'] for x,y in zip(g,base)])),
                mean_absolute_first_token_logp_change=float(np.mean([abs(y['native_first_token_logp']-x['native_first_token_logp']) for x,y in zip(g,base)])),
                sequence_changes=int(sum(x['token_ids']!=y['token_ids'] for x,y in zip(g,base))),
                eos=sum(x['eos'] for x in g),truncated=sum(x['truncated'] for x in g),empty=sum(x['empty'] for x in g),
                repeated=sum(x['repeated'] for x in g),mean_content_tokens=float(np.mean([x['content_tokens'] for x in g])))
            summaries.append(summary)
            for k,x,y in zip(keys,g,base):
                sample=pool[k]
                allrows.append(dict(sample=k,question_id=sample['question_id'],image_sha256=sample['image_sha256'],
                    dataset=dataset,task=sample['task'],answer_type=sample.get('answer_type',sample['task']),
                    config_id=cid,gold=sample.get('answer',sample.get('answers')),prediction=x['text'],native_prediction=y['text'],
                    score=x['score'],native_score=y['score'],delta=x['score']-y['score'],
                    matched_native_top1_token=y['token_ids'][0],
                    native_first_token_logp_loss=y['native_first_token_logp']-x['native_first_token_logp'],
                    score_down=x['score']<y['score'],score_up=x['score']>y['score'],
                    token_ids=x['token_ids'],native_token_ids=y['token_ids'],first_token_flip=x['token_ids'][0]!=y['token_ids'][0],
                    eos=x['eos'],truncated=x['truncated'],empty=x['empty'],repeated=x['repeated']))
            categories=sorted(set(pool[k]['task'] if dataset=='GQA' else pool[k]['answer_type'] for k in keys))
            for category in categories:
                mask=[i for i,k in enumerate(keys) if (pool[k]['task'] if dataset=='GQA' else pool[k]['answer_type'])==category]
                groups.append(dict(dataset=dataset,category=category,config_id=cid,n=len(mask),
                    accuracy=100*float(scores[mask].mean()),delta_native_pp=100*float((scores-basescores)[mask].mean()),
                    score_down=int(sum(scores[i]<basescores[i] for i in mask)),score_up=int(sum(scores[i]>basescores[i] for i in mask))))
        for i,a in enumerate(configs):
            for b in configs[:i]:
                x=np.array([records[k][config_id(a)]['generation']['score'] for k in keys])
                y=np.array([records[k][config_id(b)]['generation']['score'] for k in keys])
                s=bootstrap(100*(x-y),images,family=family)
                pairs.append(dict(dataset=dataset,condition=config_id(a),baseline=config_id(b),n=len(keys),
                    delta_pp=s['mean'][0],low=s['low'][0],high=s['high'][0],corrected_low=s['corrected_low'][0],
                    corrected_high=s['corrected_high'][0],harm=int(sum(x<y)),rescue=int(sum(x>y)),family=family))
    write_csv(folder/'accuracy_summary.csv',summaries); write_csv(folder/'paired_comparisons.csv',pairs)
    write_csv(folder/'task_accuracy.csv',groups); write_csv(folder/'per_question.csv',allrows)
    write_csv(folder/'negative_cases.csv',[r for r in allrows if r['score_down']])
    atomic(folder/'analysis_binding.json',dict(job_sha256=digest(j),sources=source_hashes(),at=now(),
        bootstrap_replicates=20000,seed=SEED,family=family,unit='Image SHA256 cluster, question weighted'))
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,dataset in zip(axes,('GQA','VQAv2')):
        selected=[s for s in summaries if s['dataset']==dataset]
        x=np.arange(len(selected)); y=[s['delta_original_native_pp'] for s in selected]
        err=np.array([[s['delta_original_native_pp']-s['original_delta_ci_low'] for s in selected],
                      [s['original_delta_ci_high']-s['delta_original_native_pp'] for s in selected]])
        ax.errorbar(x,y,yerr=err,fmt='o',capsize=3,color='#146f64')
        ax.axhline(0,color='0.4',lw=.8); ax.axhline(-1,color='#b13a36',ls='--',lw=1)
        ax.set_xticks(x,[s['config_id'].replace('-original-H0','').replace('-original-H1',' H1') for s in selected],rotation=55,ha='right',fontsize=8)
        ax.set_title(dataset); ax.set_ylabel('Accuracy difference vs matched Native (pp)')
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.2)
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'quality_comparison.png',dpi=220); fig.savefig(folder/'quality_comparison.pdf'); plt.close(fig)
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


def performance_summary(name):
    j,pool,records=load_round(name); folder=OUT/'analysis'/name
    files=[read(OUT/'performance'/name/(k+'.json')) for k in j['samples']]
    measured=collections.defaultdict(list); flops=collections.defaultdict(list)
    for file in files:
        sample=file['sample']; dataset=pool[sample]['dataset']
        for row in file['measurements']:
            measured[(dataset,row['config_id'],sample)].append(row)
        for row in file['flops']:
            flops[(dataset,row['config_id'],sample)].append(row)
    medians={}
    fields=['vision_ms','language_prefill_ms','ttft_ms','latency_ms','decode_ms','decode_steps','peak_allocated','peak_reserved','content_tokens']
    for key,rows in measured.items():
        assert len(rows)==5
        medians[key]={f:float(np.median([r[f] for r in rows])) for f in fields}
        medians[key]['decode_ms_per_token']=medians[key]['decode_ms']/max(1,medians[key]['decode_steps'])
        medians[key]['kv_live_mib']=rows[0]['kv']['live_bytes']/2**20
        medians[key]['kv_capacity_mib']=rows[0]['kv']['storage_bytes']/2**20
        shape_rows=flops[key][0]['rows']
        for phase in ('vision','prefill','decode'):
            medians[key][phase+'_flops']=sum(r['estimated_flops'] for r in shape_rows if r['phase']==phase)
        medians[key]['total_flops']=sum(r['estimated_flops'] for r in shape_rows)
        medians[key]['algorithmic_tflops_per_gpu_model_second']=medians[key]['total_flops']/1e9/float(np.median([r['gpu_model_ms'] for r in rows]))
    summaries=[]; pairs=[]; per=[]
    for dataset in ('GQA','VQAv2'):
        keys=[k for k in j['samples'] if pool[k]['dataset']==dataset]
        for c in j['configs']:
            cid=config_id(c); values=[medians[dataset,cid,k] for k in keys]
            summary=dict(dataset=dataset,config_id=cid,n=len(keys),**{f:float(np.mean([v[f] for v in values])) for f in values[0]})
            summaries.append(summary)
            for k,v in zip(keys,values): per.append(dict(sample=k,dataset=dataset,config_id=cid,**v))
            for b in j['configs']:
                bid=config_id(b)
                same=[k for k in keys if records[k][cid]['generation']['token_ids']==records[k][bid]['generation']['token_ids']]
                pair=dict(dataset=dataset,condition=cid,baseline=bid,n=len(keys),same_sequence_n=len(same))
                for label,subset in (('all',keys),('same_sequence',same)):
                    if not subset: continue
                    pair[label+'_e2e_speedup']=float(np.mean([medians[dataset,bid,k]['latency_ms'] for k in subset])/np.mean([medians[dataset,cid,k]['latency_ms'] for k in subset]))
                    pair[label+'_e2e_reduction_percent']=100*(1-1/pair[label+'_e2e_speedup'])
                pairs.append(pair)
    write_csv(folder/'latency_flops_kv.csv',summaries); write_csv(folder/'performance_paired.csv',pairs)
    write_csv(folder/'performance_per_question.csv',per)
    atomic(folder/'performance_protocol.json',dict(warmup=3,repeats=5,interleaved=True,cold_KV=True,
        vision_recomputed=True,processor_and_IO_excluded=True,request_median_then_question_mean=True,
        hardware_tflops='N/A: no hardware counters',hardware_bandwidth='N/A: no hardware counters',
        flops='Executed matrix shapes; 2 FLOPs/MAC; excludes nonlinear/norm/softmax/RoPE; not masked-edge savings'))
    print(json.dumps(summaries,indent=2))


def historical_audit():
    pool=read(OUT/'sample_manifest.json'); old_pool=read(V2_ROOT/'sample_manifest.json')
    old_specs=read(V2_ROOT/'input_specs.json'); rows=[]
    for k,s in pool.items():
        if s['dataset']!='VQAv2' or s['question_id'] not in old_pool:
            continue
        q=s['question_id']; previous=read(V2_ROOT/'checkpoints'/(q+'.json'))
        assert digest(previous['payload'])==previous['sha256']
        for label,c in [('C0',config('native')),('C1',config()),('C3-V2',config(answer='both'))]:
            current=read_record(c,k,s); assert current is not None
            old=previous['payload']['conditions'][label]['quality']; g=current['generation']
            assert current['input_binding']['input_hashes']==old_specs[q]['input_hashes']
            assert s['image_sha256']==old_pool[q]['image_sha256']
            rows.append(dict(question_id=q,historical_condition=label,current_config=config_id(c),
                input_hashes_exact=True,full_tokens_exact=g['token_ids']==old['token_ids'],
                first_logits_hash_exact=g['first_logits_sha256']==old['first_logits_sha256'],
                score_delta=g['score']-old['score'],historical_checkpoint_sha256=sha(V2_ROOT/'checkpoints'/(q+'.json'))))
    assert len(rows)==180 and all(r['full_tokens_exact'] and r['first_logits_hash_exact'] and r['score_delta']==0 for r in rows)
    write_csv(OUT/'history_binding_audit.csv',rows)
    print('Historical V2 audit: 180/180 input, first logits and complete token bindings exact. No old latency reused.')


def historical_gqa_audit():
    old=git_json(BASE,'vlm_research/results/gqa3000_p1_layerwise_accuracy/llava/baselines.json')
    pool=read(OUT/'sample_manifest.json'); rows=[]
    for k in keys_for('DEV'):
        s=pool[k]
        if s['dataset']!='GQA': continue
        r=read_record(config('native'),k,s)['generation']; b=old[s['task']+'__'+s['question_id']]
        rows.append(dict(sample=k,current_score=r['score'],historical_score=b['correct'],
            tokens_exact=r['token_ids']==b['token_ids'],first_logp_difference=r['native_first_token_logp']-b['first_logp']))
    write_csv(OUT/'historical_gqa_native_DEV_audit.csv',rows)
    print('GQA DEV Native',len(rows),'exact tokens',sum(r['tokens_exact'] for r in rows),
          'current/historical accuracy',np.mean([r['current_score'] for r in rows]),np.mean([r['historical_score'] for r in rows]))


def catalog():
    pool=read(OUT/'sample_manifest.json'); rows=[]
    for path in sorted((OUT/'jobs').glob('*.json')):
        j=read(path)
        if j['split']=='FINAL':
            assert (OUT/'final_candidate_freeze.json').exists()
        for c in j['configs']:
            for dataset in ('GQA','VQAv2'):
                keys=[k for k in j['samples'] if pool[k]['dataset']==dataset]
                values=[]
                for k in keys:
                    r=read_record(c,k,pool[k]); n=read_record(config('native',head=c['logits_to_keep']),k,pool[k])
                    if r is not None and n is not None:
                        values.append((r['generation']['score'],n['generation']['score']))
                if values:
                    x,y=np.asarray(values).T
                    rows.append(dict(round=j['id'],split=j['split'],dataset=dataset,config_id=config_id(c),
                        expected_n=len(keys),completed_n=len(values),accuracy=100*x.mean(),
                        original_native_accuracy=100*y.mean(),delta_original_native_pp=100*(x-y).mean(),
                        harm=int(sum(x<y)),rescue=int(sum(x>y)),hypothesis=j['hypothesis']))
    fields=list(rows[0]); path=OUT/'results.tsv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields,delimiter='\t'); w.writeheader(); w.writerows(rows)
    print('Catalog rows',len(rows))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action'); ap.add_argument('name',nargs='?')
    ap.add_argument('--offline-audit',action='store_true',help='CPU-only reanalysis; does not extend or dispatch GPU work')
    a=ap.parse_args()
    if a.action=='quality': analyze(a.name,offline=a.offline_audit)
    elif a.action=='performance': performance_summary(a.name)
    elif a.action=='history': historical_audit()
    elif a.action=='history-gqa': historical_gqa_audit()
    elif a.action=='catalog': catalog()

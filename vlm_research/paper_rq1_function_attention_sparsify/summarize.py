import argparse
import difflib
import json
from pathlib import Path
from collections import Counter,defaultdict
import textwrap
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from benchmarks.dataset_registry import digest,sha256
from paper_rq1.analyze import write_csv
from paper_rq1.scorers import caption_scores
from paper_rq1_token_role_decode.labels import align,classify,WORD
from paper_rq1_single_layer_gqa6.run import atomic_json
from .protocol import OUT,MODEL_PATH,PROTOCOL
from .predictor import category
from .analysis import paired_ci,read_csv,figure
from .train import measures,precision_interval
import matplotlib.pyplot as plt


def records(stage,manifest):
    config=json.loads((OUT/'quality_execution_manifest.json').read_text());binding=digest(config)
    for rank in range(3):
        complete=json.loads((OUT/f'{stage}/shard{rank}/completion.json').read_text())
        if complete['status']!='complete' or complete['binding']!=binding:raise ValueError('Incomplete policy stage')
    result=[]
    for s in manifest['holdout']:
        p=OUT/f"{stage}/shard{s['rank']}/checkpoints/{s['image_identity'].replace(':','_')}.json";r=json.loads(p.read_text())
        if r['binding']!=binding or r['sample_sha256']!=digest(s):raise ValueError('Policy checkpoint binding')
        if r.get('raw_path') and sha256(r['raw_path'])!=r['raw_sha256']:raise ValueError('Raw policy results changed')
        result.append(r)
    if len(result)!=200 or len({r['image_id'] for r in result})!=200:raise ValueError('Incomplete or duplicate holdout')
    return result


def lexical_delta(full,changed):
    a=[w.lower() for w in WORD.findall(full)];b=[w.lower() for w in WORD.findall(changed)]
    matcher=difflib.SequenceMatcher(a=a,b=b);deleted=[];added=[]
    for tag,i,j,k,l in matcher.get_opcodes():
        if tag!='equal':deleted+=a[i:j];added+=b[k:l]
    counts=Counter(classify(w)[0] for w in deleted+added)
    colors=set('red blue green yellow orange purple pink brown black white gray grey silver golden gold'.split())
    numbers=set('one two three four five six seven eight nine ten several many multiple single few pair couple 1 2 3 4 5 6 7 8 9 10'.split())
    return dict(removed_words=' '.join(deleted),added_words=' '.join(added),visual_word_changes=counts['Visual'],relation_word_changes=counts['Relation'],
        function_word_changes=counts['Function'],unknown_word_changes=counts['Unknown']+counts['Ambiguous'],
        color_word_changes=sum(w in colors for w in deleted+added),number_word_changes=sum(w in numbers for w in deleted+added),
        interpretation='Lexical differences only; not ground-truth object/semantic error annotations')


def quality():
    manifest=json.loads((OUT/'sample_manifest.json').read_text());config=json.loads((OUT/'quality_execution_manifest.json').read_text())
    quality=records('quality',manifest);risks=records('risk',manifest)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True,use_fast=True)
    samples={s['image_identity']:s for s in manifest['holdout']};refs={r['image_id']:samples[r['image_id']]['answer'] for r in quality}
    scores={};rows=[];steps=[];summaries=[]
    for strategy in config['active_strategies']:
        predictions={r['image_id']:r['runs'][strategy]['caption'] for r in quality}
        cache=OUT/f'official_scores_{strategy}.json'
        binding=digest(dict(refs=refs,predictions=predictions))
        if cache.exists():
            result=json.loads(cache.read_text())
            if result['binding']!=binding:raise ValueError('Official scoring cache input mismatch')
            total,per=result['total'],result['per_image']
        else:
            total,per=caption_scores(refs,predictions)
            atomic_json(cache,dict(binding=binding,total=total,per_image=per,official_scorer='pycocoevalcap1.2 PTBTokenizer/CIDEr/SPICE'))
        scores[strategy]=(total,per)
        for r in quality:
            current=r['runs'][strategy];full=r['runs']['S0'];ann=align(tokenizer,current['token_ids'])
            word_roles={t:category(ann,t) for t in range(len(current['token_ids']))}
            decode=[x for x in current['steps'] if x['t']>=1];pruned=[x for x in decode if x['pruned']]
            rows.append(dict(image_id=r['image_id'],strategy=strategy,caption=current['caption'],full_caption=full['caption'],
                changed=current['token_ids']!=full['token_ids'],length=current['length'],eos=current['eos'],truncated=current['truncated'],
                pruned_decode_steps=len(pruned),decode_steps=len(decode),pruned_function_steps=sum(word_roles[x['t']]=='Function' for x in pruned),
                pruned_nonfunction_steps=sum(word_roles[x['t']]!='Function' for x in pruned),
                retained_keys=sum(x['retained_keys'] for x in decode),full_keys=sum(x['full_keys'] for x in decode),
                **per[r['image_id']],**lexical_delta(full['caption'],current['caption'])))
            steps.extend(dict(image_id=r['image_id'],strategy=strategy,emitted_category=word_roles[x['t']],**x) for x in current['steps'])
    full_scores=scores['S0'][1]
    for strategy,(total,per) in scores.items():
        rs=[r for r in rows if r['strategy']==strategy]
        summary=dict(strategy=strategy,n=200,**total,changed_captions=sum(r['changed'] for r in rs),
            mean_length=float(np.mean([r['length'] for r in rs])),eos_count=sum(r['eos'] for r in rs),truncated_count=sum(r['truncated'] for r in rs),
            pruned_decode_steps=sum(r['pruned_decode_steps'] for r in rs),decode_steps=sum(r['decode_steps'] for r in rs),
            pruned_function_steps=sum(r['pruned_function_steps'] for r in rs),pruned_nonfunction_steps=sum(r['pruned_nonfunction_steps'] for r in rs),
            retained_key_ratio=sum(r['retained_keys'] for r in rs)/sum(r['full_keys'] for r in rs))
        for metric in ['CIDEr','SPICE']:
            delta=np.array([per[i][metric]-full_scores[i][metric] for i in refs]);lo,hi=paired_ci(delta)
            margin=PROTOCOL['quality'][metric+'_raw_noninferiority_margin' if metric=='CIDEr' else 'SPICE_noninferiority_margin']
            summary.update({metric+'_delta':float(delta.mean()),metric+'_delta_ci95_low':lo,metric+'_delta_ci95_high':hi,
                metric+'_noninferiority_margin':margin,metric+'_bootstrap_noninferior':bool(lo>-margin),
                metric+'_improved_images':int((delta>0).sum()),metric+'_harmed_images':int((delta<0).sum())})
        summaries.append(summary)
    write_csv(OUT/'caption_per_image.csv',rows);write_csv(OUT/'caption_generation_steps.csv',steps);write_csv(OUT/'caption_quality_summary.csv',summaries)
    predictions=[p for r in risks for p in r['predictions']];risk_rows=[p for r in risks for p in r['rows']]
    write_csv(OUT/'holdout_token_predictions.csv',predictions);write_csv(OUT/'oracle_and_predicted_single_step_risk.csv',risk_rows)
    y=np.array([r['category']=='Function' for r in predictions]);images=np.array([r['image_id'] for r in predictions]);clf=[];category_errors=[];curves=[];bins=[]
    freezes=json.loads((OUT/'predictor_freeze.json').read_text())
    for kind in ['A','B','C']:
        p=np.array([r[kind+'_probability'] for r in predictions]);threshold=freezes['thresholds'][kind];selected=p>=threshold
        lo,hi=precision_interval(selected,y,images) if selected.any() else [None,None]
        clf.append(dict(predictor=kind,split='independent_holdout',precision_ci95_low=lo,precision_ci95_high=hi,**measures(p,y,threshold)))
        for threshold0 in PROTOCOL['classifier']['threshold_grid']:curves.append(dict(predictor=kind,**measures(p,y,threshold0)))
        for c in sorted({r['category'] for r in predictions}):
            mask=np.array([r['category']==c for r in predictions]);category_errors.append(dict(predictor=kind,category=c,n=int(mask.sum()),false_function_predictions=int((mask&selected).sum()),rate=float((mask&selected).sum()/mask.sum())))
        for lo in np.linspace(0,.9,10):
            mask=(p>=lo)&(p<lo+.1)
            if mask.any():bins.append(dict(predictor=kind,bin_low=float(lo),n=int(mask.sum()),probability=float(p[mask].mean()),function_rate=float(y[mask].mean())))
    write_csv(OUT/'holdout_classifier_summary.csv',clf);write_csv(OUT/'holdout_classifier_category_errors.csv',category_errors)
    write_csv(OUT/'holdout_precision_coverage.csv',curves);write_csv(OUT/'holdout_calibration_bins.csv',bins)
    risk_summary=[]
    for group in ['oracle_function','predicted_function']:
        for mode in ['No-Image','BOS+History']:
            rs=[r for r in risk_rows if r[group] and r['mode']==mode]
            if not rs:continue
            loss=np.array([r['loss'] for r in rs]);risk_summary.append(dict(group=group,mode=mode,n=len(rs),images=len(set(r['image_id'] for r in rs)),mean_loss=float(loss.mean()),p95_loss=float(np.quantile(loss,.95)),p99_loss=float(np.quantile(loss,.99)),mean_abs_change=float(np.mean([r['abs_delta_logp'] for r in rs])),top1_flips=sum(r['top1_flip'] for r in rs),severe_gt1=int((loss>1).sum()),false_function_positions=sum(r['category']!='Function' for r in rs)))
    write_csv(OUT/'holdout_pruning_risk_summary.csv',risk_summary)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    colors=['#176b91','#b24039','#4b7951']
    for kind,color in zip(['A','B','C'],colors):
        rs=[r for r in curves if r['predictor']==kind and r['precision'] is not None]
        axes[0].plot([r['coverage'] for r in rs],[r['precision'] for r in rs],marker='o',color=color,label=kind)
        rs=[r for r in bins if r['predictor']==kind];axes[1].plot([r['probability'] for r in rs],[r['function_rate'] for r in rs],marker='o',color=color,label=kind)
        rs=[r for r in category_errors if r['predictor']==kind and r['category']!='Function'];axes[2].plot(range(len(rs)),[r['rate'] for r in rs],marker='o',color=color,label=kind)
    axes[0].set(xlabel='Held-out token coverage',ylabel='Function precision',ylim=(0,1.02));axes[0].axhline(.95,color='grey',ls=':')
    axes[1].plot([0,1],[0,1],color='grey',ls=':');axes[1].set(xlabel='Predicted probability',ylabel='Observed Function rate')
    axes[2].set(xticks=range(len(rs)),xticklabels=[r['category'] for r in rs],ylabel='False Function rate');axes[2].tick_params(axis='x',rotation=65)
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    figure(fig,'figure3_function_identification_holdout')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for metric,ax in zip(['CIDEr','SPICE'],axes):
        for s in summaries:
            if s['strategy']=='S0':continue
            delta=s[metric+'_delta'];lo=s[metric+'_delta_ci95_low'];hi=s[metric+'_delta_ci95_high']
            ax.errorbar(1-s['retained_key_ratio'],delta,yerr=[[max(0,delta-lo)],[max(0,hi-delta)]],fmt='o',label=s['strategy'],capsize=3)
        ax.axhline(0,color='grey',lw=.7);ax.axhline(-(.02 if metric=='CIDEr' else .01),color='red',ls=':',label='Frozen noninferiority margin')
        ax.set(xlabel='Fraction L31 Decode QK/AV keys removed',ylabel=metric+' change vs Full (raw)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    figure(fig,'figure4_quality_sparsity')
    cases=sorted([r for r in rows if r['strategy']!='S0'],key=lambda r:r['CIDEr']-full_scores[r['image_id']]['CIDEr'])
    chosen=cases[:2]+list(reversed(cases[-2:]));write_csv(OUT/'caption_case_studies.csv',chosen)
    fig,axes=plt.subplots(4,2,figsize=(13,12),gridspec_kw={'width_ratios':[1,2]})
    for index,r in enumerate(chosen):
        axes[index,0].imshow(Image.open(samples[r['image_id']]['image_path']));axes[index,0].axis('off')
        delta=r['CIDEr']-full_scores[r['image_id']]['CIDEr']
        text=f"{r['image_id']} | {r['strategy']} | CIDEr delta {delta:+.4f}\n\nFull: {r['full_caption']}\n\nPruned: {r['caption']}\n\nRemoved: {r['removed_words']}\nAdded: {r['added_words']}\nPruned Decode steps: {r['pruned_decode_steps']}/{r['decode_steps']}"
        axes[index,1].text(0,1,'\n'.join(textwrap.fill(line,90) for line in text.splitlines()),va='top',fontsize=10);axes[index,1].axis('off')
    fig.suptitle('Actual free-running Caption examples: score losses and gains (not safety proof)');figure(fig,'figure6_caption_case_studies')
    atomic_json(OUT/'quality_analysis_completion.json',dict(status='complete',images=200,strategies=config['active_strategies'],classifier_tokens=len(predictions),oracle_risk_rows=len(risk_rows),scores=summaries,source_sha256=sha256(Path(__file__))))
    print(json.dumps(summaries,indent=2),flush=True)


def latency():
    folder=OUT/'latency/shard0';done=json.loads((folder/'completion.json').read_text())
    if done['status']!='complete' or done['physical_gpu']!=1:raise ValueError('Fixed GPU latency incomplete')
    manifest=json.loads((OUT/'sample_manifest.json').read_text());rows=[];profiles=[]
    for s in manifest['holdout'][:10]:
        r=json.loads((folder/f"checkpoints/{s['image_identity'].replace(':','_')}.json").read_text());rows.extend(r['rows']);profiles.extend(dict(image_id=r['image_id'],**p) for p in r['profiles'])
    if len(set(r['image_id'] for r in rows))!=10:raise ValueError('Benchmark missing images')
    write_csv(OUT/'latency_repetitions.csv',rows);write_csv(OUT/'operator_latency.csv',profiles)
    full={(r['image_id'],r['repeat']):r for r in rows if r['strategy']=='S0'};groups=defaultdict(list)
    for r in rows:groups[(r['strategy'],r['implementation'])].append(r)
    summaries=[]
    for (strategy,implementation),rs in groups.items():
        if len(rs)!=50:raise ValueError('Need10 images by5 repeats')
        per=defaultdict(list);times=defaultdict(list);per_token=defaultdict(list)
        for r in rs:
            per[r['image_id']].append(r['wall_seconds']-full[(r['image_id'],r['repeat'])]['wall_seconds']);times[r['image_id']].append(r['wall_seconds'])
            f=full[(r['image_id'],r['repeat'])]
            per_token[r['image_id']].append(r['decode_cuda_ms']/max(1,r['length']-1)-f['decode_cuda_ms']/max(1,f['length']-1))
        lo,hi=paired_ci([np.mean(v) for v in per.values()]);mean=float(np.mean([r['wall_seconds'] for r in rs]));fullmean=float(np.mean([r['wall_seconds'] for r in full.values()]))
        per_token_lo,per_token_hi=paired_ci([np.mean(v) for v in per_token.values()])
        ps=[r for r in profiles if r['strategy']==strategy and r['implementation']==implementation]
        summaries.append(dict(strategy=strategy,implementation=implementation,images=10,repeats=5,
            mean_caption_seconds=mean,paired_time_delta=mean-fullmean,paired_time_delta_ci95_low=lo,paired_time_delta_ci95_high=hi,
            measured_speed_ratio=fullmean/mean if implementation=='A2' else None,speed_supported=hi<0 and per_token_hi<0 and implementation=='A2',
            decode_per_token_delta_ci95_low=per_token_lo,decode_per_token_delta_ci95_high=per_token_hi,
            mean_prefill_cuda_ms=float(np.mean([r['prefill_cuda_ms'] for r in rs])),mean_decode_cuda_ms=float(np.mean([r['decode_cuda_ms'] for r in rs])),
            mean_length=float(np.mean([r['length'] for r in rs])),mean_tokens_per_second=float(np.mean([r['tokens_per_second'] for r in rs])),peak_mib=max(r['peak_mib'] for r in rs),
            mean_l31_attention_operator_ms=float(np.mean([r['milliseconds'] for r in ps if r['operator']=='l31_attention'])) if any(r['operator']=='l31_attention' for r in ps) else None,
            mean_classifier_host_ms=float(np.mean([r['milliseconds'] for r in ps if r['operator']=='classifier_host_including_sync'])) if any(r['operator']=='classifier_host_including_sync' for r in ps) else 0.,
            retained_key_ratio=sum(r['retained_keys'] for r in rs)/sum(r['full_keys'] for r in rs),
            interpretation='A1 mechanism simulation, full QK/AV: no acceleration claim' if implementation=='A1' else 'Actual sparse path, paired fixed GPU1; different generation lengths remain part of true Caption latency'))
    write_csv(OUT/'latency_summary.csv',summaries)
    qs=read_csv(OUT/'caption_quality_summary.csv');quality_by={r['strategy']:r for r in qs}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for r in summaries:
        if r['implementation']=='A1':continue
        for metric,ax in zip(['CIDEr','SPICE'],axes):
            ax.scatter(r['mean_caption_seconds'],float(quality_by[r['strategy']][metric+'_delta']),s=60)
            ax.annotate(r['strategy'],(r['mean_caption_seconds'],float(quality_by[r['strategy']][metric+'_delta'])),xytext=(4,4),textcoords='offset points')
            ax.set(xlabel='Actual Caption model wall time, GPU1 (s)',ylabel=metric+' change (held-out200)');ax.axhline(0,color='grey',lw=.7);ax.grid(alpha=.2)
    fig.suptitle('Latency:10 fixed images x5 repeats; quality:independent200 images');figure(fig,'figure5_latency_quality')
    atomic_json(OUT/'latency_analysis_completion.json',dict(status='complete',images=10,physical_gpu=1,rows=len(rows),summaries=summaries))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['quality','latency']);a=p.parse_args();globals()[a.stage]()

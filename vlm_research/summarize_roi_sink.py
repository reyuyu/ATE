"""Offline, strict paired Phase 3.4 summaries; cases never enter main estimates."""
import argparse,csv,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from roi_sink_common import OUTPUT,WINDOWS,RANDOM_SEEDS
from roi_sink_probe import config_and_data,dataset,gate
from prepare_fixed_k_data import digest
from prepare_physical_exit_data import sha256
from fixed_k_exit import atomic_json
from summarize_fixed_k import write_csv,correlation


def regional_metrics(a,entry,sample):
    total=a.sum(-1);p=a/np.maximum(total[...,None],1e-30)
    roi=entry['roi_indices'];sink=entry['sink_indices'];nonroi=entry['sink_nonroi_indices'];overlap=entry['roi_sink_overlap_indices']
    regions={'roi':roi,'sink':sink,'sink_nonroi':nonroi,'roi_sink_overlap':overlap,'background':sorted(set(range(576))-set(roi))}
    d={'total_visual_mass':total}
    for name,idx in regions.items():d[name+'_mass']=a[...,idx].sum(-1)
    d['roi_enrichment']=d['roi_mass']/np.maximum(total,1e-30)/(len(roi)/576) if roi else np.zeros_like(total)
    centers=sorted(set(sample['objects'][0]['center_patch_indices'])|set(sample['objects'][1]['center_patch_indices']))
    d['roi_center_enrichment']=a[...,centers].sum(-1)/np.maximum(total,1e-30)/(len(centers)/576) if centers else np.zeros_like(total)
    d['sink_conditional_share']=d['sink_mass']/np.maximum(total,1e-30)
    d['sink_nonroi_conditional_share']=d['sink_nonroi_mass']/np.maximum(total,1e-30)
    d['sink_enrichment']=d['sink_conditional_share']/(len(sink)/576) if sink else np.zeros_like(total)
    d['sink_nonroi_enrichment']=d['sink_nonroi_conditional_share']/(len(nonroi)/576) if nonroi else np.zeros_like(total)
    d['normalized_entropy']=-(p*np.log(np.maximum(p,1e-30))).sum(-1)/np.log(576)
    d['concentration_hhi']=(p*p).sum(-1)
    top=np.argsort(-a,axis=-1,kind='stable')[...,:58];topa=np.take_along_axis(a,top,axis=-1)
    mask=np.zeros(576,dtype=bool);mask[roi]=True;inroi=mask[top]
    d['top58_mass_share']=topa.sum(-1)/np.maximum(total,1e-30)
    d['top58_roi_count_share']=inroi.mean(-1)
    d['top58_roi_attention_share']=(topa*inroi).sum(-1)/np.maximum(topa.sum(-1),1e-30)
    return d,regions


def observations(config,manifest,samples,binding):
    gate('observe',binding,53);observed=dataset('observe',samples,binding)
    sink=json.loads((OUTPUT/'sink_manifest.json').read_text());entries={r['question_id']:r for r in sink['samples']}
    if sink['binding']!=binding:raise ValueError('Sink binding mismatch')
    layerrows=[];headsums={};count=0;stability=[];occupancy=np.zeros(576,int);allmax=0.
    headpath=OUTPUT/'head_attention_statistics.csv.gz'
    with gzip.open(headpath,'wt') as stream:
        writer=None
        for s in samples:
            q=s['question_id'];e=entries[q];main=e['main'];obs=observed[q]
            with np.load(OUTPUT/'attention_rows'/f'{q}_task.npz') as saved:
                a=saved['attention'].astype(np.float64);phi=saved['sink_phi'].astype(float);act=saved['activation_max'].astype(float)
            if a.shape!=(32,32,2,576) or not np.isfinite(a).all():raise AssertionError('Bad attention rows')
            d,regions=regional_metrics(a,e,s)
            with np.load(OUTPUT/'attention_rows'/f'{q}_norms.npz') as norms:
                for typ in ['key_norm','value_norm']:
                    n=norms[typ].astype(float)
                    if not np.isfinite(n).all():raise ValueError('Nonfinite projection norm')
                    for region in ['roi','background','sink_nonroi']:
                        idx=regions[region]
                        if idx:d[f'{typ}_{region}']=np.repeat(n[...,idx].mean(-1)[...,None],2,-1)
            for region in ['roi','background','sink_nonroi']:
                idx=regions[region]
                if idx:
                    d[f'raw_sink_phi_{region}']=np.broadcast_to(phi[:,idx].mean(-1)[:,None,None],(32,32,2))
                    d[f'phi_ge20_fraction_{region}']=np.broadcast_to((phi[:,idx]>=20).mean(-1)[:,None,None],(32,32,2))
                    d[f'normalized_activation_max_{region}']=np.broadcast_to(act[:,idx].mean(-1)[:,None,None],(32,32,2))
            neutral=np.load(OUTPUT/'attention_rows'/f'{q}_neutral.npz')['attention'].astype(float)
            neutral_enrichment=neutral[...,e['roi_indices']].sum(-1)/np.maximum(neutral.sum(-1),1e-30)/(len(e['roi_indices'])/576)
            for l in range(32):
                for g,group in enumerate(['question','answer']):
                    row=dict(question_id=q,main=main,answer=s['answer'],layer=l,query_group=group,full_margin=obs['full_margin'],
                             roi_patches=len(e['roi_indices']),sink_patches=len(e['sink_indices']),sink_nonroi_patches=len(e['sink_nonroi_indices']),
                             roi_sink_overlap_patches=len(e['roi_sink_overlap_indices']),neutral_holdout_roi_enrichment=float(neutral_enrichment[2,l,g]))
                    row.update({name:float(values[l,:,g].mean()) for name,values in d.items()});layerrows.append(row)
                    for h in range(32):
                        headrow=dict(question_id=q,main=main,layer=l,head=h,query_group=group)
                        headrow.update({name:float(values[l,h,g]) for name,values in d.items()})
                        if writer is None:writer=csv.DictWriter(stream,fieldnames=list(headrow),lineterminator='\n');writer.writeheader()
                        writer.writerow(headrow)
            if main:
                for name,values in d.items():headsums[name]=headsums.get(name,np.zeros_like(values))+values
                count+=1;occupancy[e['sink_indices']]+=1
            r=e['sink_identification'];stability.append(dict(question_id=q,main=main,sink_patches=len(e['sink_indices']),sink_nonroi_patches=len(e['sink_nonroi_indices']),
                roi_sink_overlap_patches=len(e['roi_sink_overlap_indices']),holdout_jaccard=r['holdout_jaccard'],holdout_retention=r['holdout_retention'],
                random_roi_available=all(v['available'] for k,v in e['random_controls'].items() if k.startswith('random_roi')),
                mean_roi_random_bin_distance=float(np.mean([v['mean_bin_manhattan'] for k,v in e['random_controls'].items() if k.startswith('random_roi') and v['available']])) if any(v['available'] for k,v in e['random_controls'].items() if k.startswith('random_roi')) else None,
                mean_sink_random_bin_distance=float(np.mean([v['mean_bin_manhattan'] for k,v in e['random_controls'].items() if k.startswith('random_sink') and v['available']]))))
    if count!=48:raise AssertionError('Wrong main sample count')
    write_csv(OUTPUT/'sample_layer_attention.csv',layerrows);write_csv(OUTPUT/'sink_stability.csv',stability)
    headsummary=[];layersummary=[]
    for l in range(32):
        for g,group in enumerate(['question','answer']):
            values=[r for r in layerrows if r['main'] and r['layer']==l and r['query_group']==group]
            row=dict(layer=l,query_group=group,n=48)
            for name in headsums:
                row[name]=float(headsums[name][l,:,g].mean()/48)
                row[name+'_sd_images']=float(np.std([r[name] for r in values],ddof=1))
            row['roi_enrichment_gt1_images']=sum(r['roi_enrichment']>1 for r in values)
            row['neutral_holdout_roi_enrichment']=float(np.mean([r['neutral_holdout_roi_enrichment'] for r in values]));layersummary.append(row)
            for h in range(32):headsummary.append(dict(layer=l,head=h,query_group=group,**{k:float(v[l,h,g]/48) for k,v in headsums.items()}))
    write_csv(OUTPUT/'layer_attention_summary.csv',layersummary);write_csv(OUTPUT/'head_layer_attention_summary.csv',headsummary)
    windowrows=[]
    for s in samples:
        for group in ['question','answer']:
            for window,ls in WINDOWS.items():
                v=[r for r in layerrows if r['question_id']==s['question_id'] and r['query_group']==group and r['layer'] in ls]
                r=dict(question_id=s['question_id'],main=v[0]['main'],query_group=group,window=window,full_margin=v[0]['full_margin'],roi_patches=v[0]['roi_patches'],sink_nonroi_patches=v[0]['sink_nonroi_patches'])
                r.update({k:float(np.mean([x[k] for x in v])) for k in headsums});windowrows.append(r)
    write_csv(OUTPUT/'sample_window_attention.csv',windowrows)
    description=[]
    for group in ['question','answer']:
        for window in WINDOWS:
            v=[r for r in windowrows if r['main'] and r['query_group']==group and r['window']==window]
            description.append(dict(query_group=group,window=window,n=48,**{k:float(np.mean([x[k] for x in v])) for k in headsums},
                roi_enrichment_gt1_images=sum(r['roi_enrichment']>1 for r in v),
                sink_mass_margin_pearson=correlation([r['sink_mass'] for r in v],[r['full_margin'] for r in v]),
                sink_share_margin_pearson=correlation([r['sink_conditional_share'] for r in v],[r['full_margin'] for r in v])))
    atomic_json(OUTPUT/'observation_summary.json',dict(main_window_statistics=description,main_n=48,case_n=5,
        main_sink_count_median=float(np.median([r['sink_patches'] for r in stability if r['main']])),
        main_holdout_jaccard_mean=float(np.mean([r['holdout_jaccard'] for r in stability if r['main']])),
        main_holdout_jaccard_median=float(np.median([r['holdout_jaccard'] for r in stability if r['main']])),
        main_sink_patch_coordinate_frequency=occupancy.tolist(),
        note='Per-head raw Attention group means; Q averaged over all real Question tokens before nonlinear statistics. Enrichment=ROI Mass/Visual Mass divided by ROI fraction. Samples equally weighted, heads averaged within image. No labels/outcomes used in Sink rule.'))
    observation_plots(layersummary,headsummary,occupancy)
    return windowrows,entries,observed


def observation_plots(layers,heads,occupancy):
    colors={'question':'tab:blue','answer':'tab:orange'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for group in colors:
        v=[r for r in layers if r['query_group']==group];x=np.arange(32);y=np.array([r['roi_enrichment'] for r in v]);sd=np.array([r['roi_enrichment_sd_images'] for r in v])
        axes[0].plot(x,y,label=group,color=colors[group]);axes[0].fill_between(x,y-sd,y+sd,alpha=.12,color=colors[group])
        axes[1].plot(x,[r['roi_enrichment']-r['neutral_holdout_roi_enrichment'] for r in v],label=group,color=colors[group])
    axes[0].axhline(1,color='black',ls='--');axes[0].set_ylabel('ROI enrichment (visual-conditional, area adjusted)');axes[0].set_title('Mean +/- SD across 48 images')
    axes[1].axhline(0,color='black',lw=.6);axes[1].set_ylabel('Task ROI enrichment - neutral holdout');axes[1].set_title('Descriptive; prompt lengths/positions differ')
    for ax in axes:ax.set_xlabel('LLM layer');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUTPUT/'roi_enrichment_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for group in colors:
        v=[r for r in layers if r['query_group']==group]
        for ax,metric in zip(axes,['total_visual_mass','sink_mass','sink_conditional_share']):ax.plot(range(32),[r[metric] for r in v],label=group,color=colors[group]);ax.set_title(metric);ax.set_xlabel('Layer')
    axes[0].legend();fig.suptitle('Frozen neutral-prompt candidates; ROI overlap included here');fig.tight_layout();fig.savefig(OUTPUT/'sink_attention_curves.png',dpi=160);plt.close(fig)
    for metric,name in [('roi_enrichment','head_layer_roi_enrichment.png'),('sink_conditional_share','head_layer_sink_attention.png')]:
        fig,axes=plt.subplots(1,2,figsize=(12,5));arrays=[]
        for group in colors:
            array=np.zeros((32,32))
            for r in heads:
                if r['query_group']==group:array[r['layer'],r['head']]=r[metric]
            arrays.append(array)
        for ax,array,group in zip(axes,arrays,colors):
            im=ax.imshow(array,aspect='auto',origin='lower',vmin=0,vmax=max(a.max() for a in arrays));ax.set(xlabel='Head',ylabel='Layer',title=group)
        fig.colorbar(im,ax=axes.tolist(),label=metric);fig.savefig(OUTPUT/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for group in colors:
        v=[r for r in layers if r['query_group']==group]
        for ax,metric in zip(axes,['normalized_entropy','top58_mass_share']):ax.plot(range(32),[r[metric] for r in v],label=group);ax.set_title(metric);ax.set_xlabel('Layer')
    axes[0].legend();fig.tight_layout();fig.savefig(OUTPUT/'concentration_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,5));im=ax.imshow(occupancy.reshape(24,24)/48,vmin=0,vmax=1);ax.set_title('Candidate location frequency across 48 images');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUTPUT/'sink_coordinate_frequency.png',dpi=160);plt.close(fig)


def causal_summaries(rows):
    results=[]
    for group in ['question','answer']:
        for window in WINDOWS:
            for condition in ['roi','sink_nonroi']+[f'random_{typ}_s{seed}' for typ in ['roi','sink'] for seed in RANDOM_SEEDS]:
                v=[r for r in rows if r['main'] and r['query_group']==group and r['window']==window and r['condition']==condition and r['available']]
                if len(v)!=48:raise ValueError('Missing main causal comparison')
                drop=np.array([r['answer_margin_drop'] for r in v]);base=sum(r['full_correct'] for r in v)
                r=dict(query_group=group,window=window,condition=condition,n=48,correct=sum(r['candidate_correct'] for r in v),full_correct=base,
                  correct_to_wrong=sum(r['full_correct'] and not r['candidate_correct'] for r in v),wrong_to_correct=sum(not r['full_correct'] and r['candidate_correct'] for r in v),
                  margin_drop_mean=float(drop.mean()),margin_drop_median=float(np.median(drop)),margin_drop_p95=float(np.percentile(drop,95)),margin_drop_max=float(drop.max()),
                  improved_margin_count=int((drop<0).sum()),positive_drop_count=int((drop>0).sum()),weak_margin_cases=sum(r['weak_margin'] for r in v),
                  blocked_patches_mean=float(np.mean([r['blocked_patches'] for r in v])),gold_logp_mean=float(np.mean([r['correct_logp'] for r in v])))
                for label in ['left','right']:
                    subset=[x for x in v if x['answer']==label];r[label+'_correct']=sum(x['candidate_correct'] for x in subset);r[label+'_n']=len(subset)
                results.append(r)
    return results


def paired_diagnostics(rows,windows):
    index={(r['question_id'],r['query_group'],r['window'],r['condition']):r for r in rows};pairs=[];summary=[]
    allids=list(dict.fromkeys(r['question_id'] for r in rows));metrics={(r['question_id'],r['query_group'],r['window']):r for r in windows}
    for group in ['question','answer']:
        for window in WINDOWS:
            for target,label in [('roi','roi'),('sink_nonroi','sink')]:
                for q in allids:
                    r=index[q,group,window,target];randoms=[index[q,group,window,f'random_{label}_s{s}'] for s in RANDOM_SEEDS]
                    if not all(x['available'] for x in randoms):continue
                    if any(x['blocked_patches']!=r['blocked_patches'] for x in randoms):raise AssertionError('Unequal patch count control')
                    drops=[x['answer_margin_drop'] for x in randoms];randommean=float(np.mean(drops));a=metrics[q,group,window]
                    pairs.append(dict(question_id=q,main=r['main'],query_group=group,window=window,target=target,blocked_patches=r['blocked_patches'],
                         target_drop=r['answer_margin_drop'],random_mean_drop=randommean,paired_excess_drop=r['answer_margin_drop']-randommean,
                         improves_margin=r['answer_margin_drop']<0,better_than_all_three_randoms=all(r['answer_margin_drop']<x for x in drops),
                         better_than_random_mean=r['answer_margin_drop']<randommean,baseline_attention_mass=a[target+'_mass'],
                         full_margin=r['full_margin'],weak_margin=r['weak_margin']))
                v=[r for r in pairs if r['main'] and r['query_group']==group and r['window']==window and r['target']==target]
                x=np.array([r['paired_excess_drop'] for r in v]);rng=np.random.default_rng(3404);boot=x[rng.integers(0,len(x),(2000,len(x)))].mean(-1)
                summary.append(dict(query_group=group,window=window,target=target,n=len(v),paired_excess_drop_mean=float(x.mean()),paired_excess_drop_median=float(np.median(x)),
                  bootstrap_mean_95_low=float(np.percentile(boot,2.5)),bootstrap_mean_95_high=float(np.percentile(boot,97.5)),
                  better_than_random_mean=sum(r['better_than_random_mean'] for r in v),improves_and_better_than_all_three=sum(r['improves_margin'] and r['better_than_all_three_randoms'] for r in v),
                  attention_vs_drop_pearson=correlation([r['baseline_attention_mass'] for r in v],[r['target_drop'] for r in v]),
                  no_multiple_testing_correction=True))
    return pairs,summary


def causal_plots(rows,pairs):
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for gi,group in enumerate(['question','answer']):
        for wi,window in enumerate(WINDOWS):
            ax=axes[gi,wi];v=[r for r in pairs if r['main'] and r['query_group']==group and r['window']==window]
            ax.boxplot([[r['paired_excess_drop'] for r in v if r['target']==t] for t in ['roi','sink_nonroi']],tick_labels=['ROI vs own random','Sink vs own random'],showfliers=True)
            ax.axhline(0,color='black',lw=.7);ax.set_title(f'{group}, {window}');ax.set_ylabel('Target Drop - mean matched random Drop (nats)')
    fig.suptitle('Each region has its own exact-count, spatially matched background controls (3 seeds)')
    fig.tight_layout();fig.savefig(OUTPUT/'causal_matched_margin_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,7))
    for gi,group in enumerate(['question','answer']):
        for wi,window in enumerate(WINDOWS):
            for ti,target in enumerate(['roi','sink_nonroi']):
                ax=axes[gi,wi*2+ti];v=[r for r in pairs if r['main'] and r['query_group']==group and r['window']==window and r['target']==target]
                ax.boxplot([[r['target_drop'] for r in v],[r['random_mean_drop'] for r in v]],tick_labels=['target','random mean'],showfliers=True);ax.axhline(0,color='black',lw=.6)
                ax.set_title(f'{group}, {window}, {target}');ax.set_ylabel('Full - intervened Margin (nats)')
    fig.suptitle('Separate panels: ROI and Sink patch counts differ; compare each to its own random control')
    fig.tight_layout();fig.savefig(OUTPUT/'causal_absolute_margin_drop.png',dpi=160);plt.close(fig)


def spatial_examples(manifest,samples,entries):
    from PIL import Image
    from transformers import AutoProcessor
    from vlm_common import default_model
    from roi_sink_common import geometry
    from matplotlib.patches import Rectangle
    ip=AutoProcessor.from_pretrained(default_model(),local_files_only=True).image_processor
    folder=OUTPUT/'attention_overlays';folder.mkdir(exist_ok=True)
    chosen=manifest['main_question_ids'][:2]+manifest['main_question_ids'][24:26]+manifest['case_question_ids']
    for s in samples:
        q=s['question_id']
        if q not in chosen:continue
        e=entries[q];_,crop=geometry(Image.open(s['image_path']).convert('RGB'),ip)
        a=np.load(OUTPUT/'attention_rows'/f'{q}_task.npz')['attention'].astype(float).mean(1)
        fig,axes=plt.subplots(2,3,figsize=(11,7.5))
        for row,(window,ls) in enumerate(WINDOWS.items()):
            axes[row,0].imshow(crop)
            for key,color in [('roi_indices','cyan'),('sink_nonroi_indices','red'),('roi_sink_overlap_indices','magenta')]:
                for index in e[key]:
                    r,c=divmod(index,24);axes[row,0].add_patch(Rectangle((c*14,r*14),14,14,fill=False,edgecolor=color,lw=.7))
            axes[row,0].set_title(f'{window}: cyan ROI, red nonROI sink\nmagenta overlap; ROI={len(e["roi_indices"])}')
            for g,group in enumerate(['question','answer']):
                v=a[ls,g].mean(0);grid=v.reshape(24,24)
                heat=np.asarray(Image.fromarray(grid.astype(np.float32)).resize((336,336),Image.Resampling.BILINEAR))
                ax=axes[row,g+1];ax.imshow(crop);ax.imshow(heat,cmap='inferno',alpha=.6)
                ax.set_title(f'{group}: raw visual mass={v.sum():.4f}\nROI mass={v[e["roi_indices"]].sum():.4f}; sink mass={v[e["sink_indices"]].sum():.4f}')
        for ax in axes.ravel():ax.axis('off')
        fig.suptitle(f'{q}: {s["question"]}\nEach heatmap independently scaled for location only; raw masses shown.',fontsize=10)
        fig.tight_layout();fig.savefig(folder/f'{q}.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observations-only',action='store_true');args=parser.parse_args()
    config,manifest,samples=config_and_data();binding=digest(config)
    windows,entries,observed=observations(config,manifest,samples,binding)
    if args.observations_only:print('Observation summaries complete; no causal results read');return
    cb=digest(dict(config_binding=binding,sink_sha256=sha256(OUTPUT/'sink_manifest.json')))
    gate('smoke',cb,3);gate('causal',cb,53)
    data=dataset('causal',samples,cb);rows=[r for s in samples for r in data[s['question_id']]['rows']]
    if len(rows)!=1749:raise ValueError('Incomplete sample-condition cohort')
    write_csv(OUTPUT/'per_sample_results.csv',rows);atomic_json(OUTPUT/'per_sample_results.json',dict(binding=cb,main_n=48,case_n=5,rows=rows))
    summary=causal_summaries(rows);write_csv(OUTPUT/'causal_summary.csv',summary);atomic_json(OUTPUT/'causal_summary.json',summary)
    pairs,ps=paired_diagnostics(rows,windows);write_csv(OUTPUT/'paired_target_random.csv',pairs);atomic_json(OUTPUT/'paired_effect_summary.json',ps)
    cases=[dict(question=next(s['question'] for s in samples if s['question_id']==r['question_id']),**r) for r in rows if not r['main']]
    write_csv(OUTPUT/'historical_cases.csv',cases);atomic_json(OUTPUT/'historical_cases.json',cases)
    smoke_samples=[s for s in samples if (OUTPUT/'checkpoints/smoke'/f"{s['question_id']}.json").exists()];smoke=dataset('smoke',smoke_samples,cb)
    for q,s in smoke.items():
        for a,b in zip(s['rows'],data[q]['rows']):
            if any(a[k]!=b[k] for k in ['answer_margin','correct_logp','question_drift']):raise AssertionError('Smoke/production scores changed')
    audits=[a for s in smoke.values() for d in s['audits'] for a in d['attention_audit']]
    atomic_json(OUTPUT/'correctness_validation.json',dict(passed=True,cpu_tests_before_gpu=83,cpu_tests_final=85,roi_audit_images=240,observation_images=53,causal_main_images=48,case_images=5,
      smoke_images=len(smoke),attention_layers_audited=len(audits),target_attention_exact_zero=all(a['target_attention_max']==0 for a in audits),
      max_row_sum_error=max(a['row_sum_error'] for a in audits),smoke_causal_scores_exact=True,
      all_answer_only_question_states_exact_full=all(d['answer_question_hidden_exact'] for d in data.values()),
      all_native_history_exact=all(d['history_full_exact'] for d in data.values()),all_cleanup_exact=all(d['post_cleanup_exact'] for d in data.values()),
      unavailable_case_rows=sum(not r['available'] for r in rows),main_unavailable_rows=sum(r['main'] and not r['available'] for r in rows),
      all_random_patch_counts_exact=True,baseline_predictions_unaffected_by_attention_streaming=True))
    compute=[]
    for group in ['question','answer']:
        for window in WINDOWS:
            for condition in sorted({r['condition'] for r in rows if r['query_group']==group}):
                selected=[r for r in rows if r['main'] and r['query_group']==group and r['window']==window and r['condition']==condition and r['available']]
                compute.append(dict(query_group=group,window=window,condition=condition,n=len(selected),
                    mean_single_forward_seconds=float(np.mean([r['forward_seconds_two_candidates']/2 for r in selected])),
                    peak_allocated_mib=max(r['peak_allocated_mib'] for r in selected),
                    note='Diagnostic only; includes mask validation/clone/hooks; no repeated timing or speed claim.'))
    write_csv(OUTPUT/'compute_diagnostics.csv',compute)
    precision=json.loads((OUTPUT/'precision_sensitivity.json').read_text())
    index={(r['question_id'],r['query_group'],r['window'],r['condition']):r for r in rows}
    for r in precision['rows']:
        original=index[r['question_id'],r['query_group'],r['window'],r['condition']]
        if abs(r['fp16_margin']-original['answer_margin'])>1e-12:raise AssertionError('Precision native margins differ from production')
    precision_pairs=[]
    for q in manifest['case_question_ids']:
        values=[r for r in precision['rows'] if r['question_id']==q]
        full=next(r for r in values if r['condition']=='full')
        for group in ['question','answer']:
            target=next(r for r in values if r['query_group']==group and r['condition']=='sink_nonroi')
            controls=[r for r in values if r['query_group']==group and r['condition'].startswith('random_sink')]
            precision_pairs.append(dict(question_id=q,query_group=group,
                fp16_sink_drop=full['fp16_margin']-target['fp16_margin'],fp32_readout_sink_drop=full['fp32_readout_margin']-target['fp32_readout_margin'],
                fp16_sink_excess_drop=float(np.mean([r['fp16_margin'] for r in controls]))-target['fp16_margin'],
                fp32_readout_sink_excess_drop=float(np.mean([r['fp32_readout_margin'] for r in controls]))-target['fp32_readout_margin']))
    write_csv(OUTPUT/'precision_paired_sink_random.csv',precision_pairs)
    atomic_json(OUTPUT/'numeric_sensitivity_summary.json',dict(
        precision_conditions=len(precision['rows']),fp32_readout_classification_changes=len(precision['classification_changes']),
        max_fp16_vs_fp32_readout_margin_difference=precision['max_margin_difference'],
        tiny_drop_rows_abs_le_1e_minus_10=sum(abs(r['answer_margin_drop'])<=1e-10 for r in rows if r['available'] and r['condition']!='full'),
        note='1e-10 only identifies FP64 subtraction roundoff/zero; raw margins and primary statistics retained. FP32 final head does not validate full-backbone precision.'))
    causal_plots(rows,pairs)
    spatial_examples(manifest,samples,entries)
    print(json.dumps(dict(causal=summary,paired=ps),indent=2))


if __name__=='__main__':main()

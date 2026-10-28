"""Validate frozen checkpoints and derive paired Phase 3.2 statistics/figures."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prepare_fixed_k_data import OUTPUT, digest
from fixed_k_exit import CUTOFFS, configuration, verify_manifest, check_directory, check_gate, atomic_json


def correlation(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def summarize_rows(rows, cutoffs=CUTOFFS):
    result=[]
    for k in cutoffs:
        values=[r for r in rows if r['cutoff']==k]
        if not values: continue
        base=[r for r in values if r['full_correct']]
        drop=np.array([r['answer_margin_drop'] for r in values])
        drift=np.array([r['question_mean_cosine_distance'] for r in values])
        correct=sum(r['candidate_correct'] for r in values)
        ctw=sum(not r['candidate_correct'] for r in base)
        wtc=sum(r['candidate_correct'] and not r['full_correct'] for r in values)
        d=dict(cutoff=k,n=len(values),correct=correct,accuracy=correct/len(values),full_correct=len(base),
               full_accuracy=len(base)/len(values),accuracy_delta=(correct-len(base))/len(values),
               full_correct_retained=len(base)-ctw,correct_to_wrong=ctw,wrong_to_correct=wtc,
               full_correct_flip_rate=ctw/len(base) if base else None,
               margin_positive_to_negative=sum(r['full_margin']>0 and r['answer_margin']<0 for r in values),
               margin_positive_to_zero=sum(r['full_margin']>0 and r['answer_margin']==0 for r in values),
               margin_drop_mean=float(drop.mean()),margin_drop_median=float(np.median(drop)),
               margin_drop_p90=float(np.percentile(drop,90)),margin_drop_p95=float(np.percentile(drop,95)),
               max_positive_margin_drop=max(0.,float(drop.max())),margin_drop_positive_fraction=float((drop>0).mean()),
               question_drift_mean=float(drift.mean()),question_drift_median=float(np.median(drift)),
               drift_margin_drop_pearson=correlation(drift,drop))
        for label in ('left','right'):
            group=[r for r in values if r['answer']==label]
            d[label+'_correct']=sum(r['candidate_correct'] for r in group)
            d[label+'_n']=len(group)
            d[label+'_accuracy']=d[label+'_correct']/len(group) if group else None
            d[label+'_full_accuracy']=sum(r['full_correct'] for r in group)/len(group) if group else None
            d[label+'_accuracy_delta']=(d[label+'_accuracy']-d[label+'_full_accuracy']) if group else None
        d['full_correct_margin_drop_mean']=float(np.mean([r['answer_margin_drop'] for r in base])) if base else None
        d['predict_left']=sum((r['answer']=='left' and r['answer_margin']>0) or (r['answer']=='right' and r['answer_margin']<0) for r in values)
        d['ties']=sum(r['answer_margin']==0 for r in values)
        result.append(d)
    return result


def write_csv(path,rows):
    if not rows: return
    temporary=Path(str(path)+'.tmp')
    with temporary.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def plots(rows,summary):
    ks=[s['cutoff'] for s in summary]
    fig,ax=plt.subplots(figsize=(8,5))
    for key,label in [('accuracy','All 240'),('left_accuracy','Gold left'),('right_accuracy','Gold right')]:
        ax.plot(ks,[100*s[key] for s in summary],'o-',label=label)
    ax.axhline(100*summary[-1]['accuracy'],color='black',linestyle='--',label='Full overall')
    ax.set(xlabel='Retained visual layers K',ylabel='Candidate accuracy (%)',xticks=ks);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUTPUT/'accuracy_vs_k.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(ks,[100*s['full_correct_flip_rate'] for s in summary],'o-')
    ax.set(xlabel='Retained visual layers K',ylabel='Full-correct to wrong (%)',xticks=ks)
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUTPUT/'correct_to_wrong_flip_rate_vs_k.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,subset,title in [(axes[0],ks,'All exit depths'),(axes[1],[k for k in ks if k>=16],'Detail: K >= 16')]:
        ax.boxplot([[r['answer_margin_drop'] for r in rows if r['cutoff']==k] for k in subset],
                   tick_labels=[str(k) for k in subset],showfliers=True)
        ax.axhline(0,color='black',linewidth=.7)
        ax.set(title=title,xlabel='Retained visual layers K',ylabel='Full margin - exit margin (nats)')
    fig.tight_layout();fig.savefig(OUTPUT/'margin_drop_distribution_vs_k.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,k in zip(axes.flat,ks):
        values=[r for r in rows if r['cutoff']==k]
        colors=['crimson' if r['full_correct'] and not r['candidate_correct'] else 'steelblue' for r in values]
        ax.scatter([r['question_mean_cosine_distance'] for r in values],
                   [r['answer_margin_drop'] for r in values],c=colors,s=13,alpha=.65)
        ax.axhline(0,color='gray',linewidth=.5);ax.set_title(f'K={k}');ax.set_xlabel('Mean Question cosine distance')
        ax.set_ylabel('Margin drop (nats)')
        if k==32:
            ax.text(.03,.9,'Exact hidden/logits;\ncosine roundoff only',transform=ax.transAxes,fontsize=9)
    fig.suptitle('Red: Full-correct -> wrong; blue: other paired samples. Axes vary by panel.')
    fig.tight_layout();fig.savefig(OUTPUT/'question_drift_vs_margin_drop.png',dpi=180);plt.close(fig)


def main():
    manifest=json.loads((OUTPUT/'manifest.json').read_text());verify_manifest(manifest)
    config=configuration();binding=digest(config)
    if json.loads((OUTPUT/'experiment_config.json').read_text())!=config:
        raise ValueError('Experiment binding changed before analysis')
    for stage in ('smoke','validate','run'):check_gate(stage,binding)
    datasets={stage:check_directory(OUTPUT/'checkpoints'/stage,
              manifest['validation_samples'] if stage=='validate' else
              [s for s in manifest['samples'] if s['question_id'] in manifest['smoke_question_ids']] if stage=='smoke' else manifest['samples'],binding)
              for stage in ('smoke','validate','run')}
    if [len(datasets[s]) for s in ('smoke','validate','run')]!=[24,12,240]:
        raise ValueError('Incomplete paired experiment; refuse full-cohort plots')
    rows=[r for sample in manifest['samples'] for r in datasets['run'][sample['question_id']]['rows']]
    write_csv(OUTPUT/'per_sample_results.csv',rows)
    atomic_json(OUTPUT/'per_sample_results.json',dict(binding=binding,completed=True,n=240,rows=rows,
                 details_location='checkpoints/run/<question_id>.json (per-token drift, candidate mappings and per-layer trace)'))
    summary=summarize_rows(rows)
    write_csv(OUTPUT/'summary_by_k.csv',summary);atomic_json(OUTPUT/'summary_by_k.json',summary)
    checks=[d['equivalence'] for sample in datasets['validate'].values() for d in sample['details']]
    validation=dict(binding=binding,smoke_images=24,validation_images=12,k32_exact_all=all(s['k32_exact'] for s in datasets['run'].values()),
                    smoke_cleanup_exact=all(s['native_forward_cleanup_exact'] for s in datasets['smoke'].values()),
                    smoke_k0_pixel_invariance=all(s['k0_pixel_invariance'] for s in datasets['smoke'].values()),
                    equivalence_all_passed=all(c['passed'] for c in checks),
                    max_logit_error=max(c['max_abs_logit_error'] for e in checks for c in e['candidate_checks']),
                    max_text_cosine_distance=max(c['max_text_cosine_distance'] for e in checks for c in e['candidate_checks']),
                    max_text_relative_rmse=max(c['max_text_relative_rmse'] for e in checks for c in e['candidate_checks']),
                    max_candidate_logp_error=max(c['candidate_logp_max_abs_error'] for c in checks),
                    max_margin_error=max(c['margin_abs_error'] for c in checks),
                    historical_max_margin_error=max(c['margin_abs_error'] for s in datasets['validate'].values() for c in s['historical_checks']),
                    tolerances=config['tolerances'])
    atomic_json(OUTPUT/'correctness_validation.json',validation)
    # Smoke and production must be deterministic despite different execution stage/order.
    comparison=[]
    fields=['answer_margin','correct_answer_log_probability','question_mean_cosine_distance']
    for q,s in datasets['smoke'].items():
        a=s['rows'];b=datasets['run'][q]['rows']
        comparison.append(dict(question_id=q,exact=all(x[f]==y[f] for x,y in zip(a,b) for f in fields)))
    if not all(c['exact'] for c in comparison):raise AssertionError('Smoke/production paired scores differ')
    atomic_json(OUTPUT/'smoke_production_consistency.json',comparison)
    all_measures=[]
    for sample in datasets['run'].values():
        for x in sample['full_candidates']:all_measures.append(('native_full',32,x['measurement']))
        for d in sample['details']:
            for x in d['candidates']:all_measures.append(('physical',d['cutoff'],x['measurement']))
    compute=[]
    for mode,k in [('native_full',32)]+[('physical',k) for k in CUTOFFS]:
        m=[x for mo,ko,x in all_measures if mo==mode and ko==k]
        compute.append(dict(mode=mode,cutoff=k,forwards=len(m),mean_forward_seconds=float(np.mean([x['forward_seconds'] for x in m])),
                            median_forward_seconds=float(np.median([x['forward_seconds'] for x in m])),
                            peak_allocated_mib=max(x['gpu_memory']['peak_allocated_mib'] for x in m),
                            peak_reserved_mib=max(x['gpu_memory']['peak_reserved_mib'] for x in m)))
    write_csv(OUTPUT/'compute_diagnostics.csv',compute)
    sequences={str(k):sorted({(r['original_sequence_length'],r['post_exit_sequence_length']) for r in rows if r['cutoff']==k}) for k in CUTOFFS}
    sample_diagnostics=[]
    for sample in manifest['samples']:
        values=[r for r in rows if r['question_id']==sample['question_id']]
        drifts=[r['question_mean_cosine_distance'] for r in values]
        sample_diagnostics.append(dict(question_id=sample['question_id'],nonincreasing_drift=all(b<=a+1e-12 for a,b in zip(drifts,drifts[1:])),
                                       correct_by_k={str(r['cutoff']):r['candidate_correct'] for r in values},
                                       first_k_with_correct_stable_at_all_later_tested_k=next((r['cutoff'] for i,r in enumerate(values) if all(x['candidate_correct'] for x in values[i:])),None)))
    atomic_json(OUTPUT/'sample_diagnostics.json',dict(layer_lengths=sequences,samples=sample_diagnostics,
                 caution='First stable tested K is retrospective on gold labels and finite K grid; not a deployable router or proof of monotonic correctness.'))
    annotated=[dict(question=s['question'],**r) for s in manifest['samples'] for r in rows
               if r['question_id']==s['question_id'] and r['cutoff'] in (16,20,24)]
    write_csv(OUTPUT/'degradation_cases.csv',sorted(annotated,key=lambda r:(r['cutoff'],-r['answer_margin_drop'],r['question_id'])))
    drift_analysis=[]
    for k in CUTOFFS[:-1]:
        values=[r for r in rows if r['cutoff']==k]
        q25,q75=np.percentile([r['question_mean_cosine_distance'] for r in values],[25,75])
        flips=[r for r in values if r['full_correct'] and not r['candidate_correct']]
        low_flip=[r for r in flips if r['question_mean_cosine_distance']<=q25]
        high_stable=[r for r in values if r['full_correct'] and r['candidate_correct'] and r['question_mean_cosine_distance']>=q75]
        drift_analysis.append(dict(cutoff=k,drift_q25=float(q25),drift_q75=float(q75),
                                   low_drift_correct_to_wrong_question_ids=[r['question_id'] for r in low_flip],
                                   high_drift_retained_correct_question_ids=[r['question_id'] for r in high_stable],
                                   minimum_drift_wrong_flip=min((r['question_mean_cosine_distance'] for r in flips),default=None)))
    atomic_json(OUTPUT/'drift_analysis.json',dict(definition='Low/high drift are within-K bottom/top quartiles, descriptive only; no outcome-based cohort selection.',
                                               nonincreasing_drift_images=sum(s['nonincreasing_drift'] for s in sample_diagnostics),
                                               by_k=drift_analysis))
    plots(rows,summary)
    print(json.dumps(dict(summary=summary,validation=validation,compute=compute),indent=2))


if __name__=='__main__':main()

"""Strict offline paired summaries/figures for the complete Phase 3.3 cohort."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from directed_suffix import MODES,CUTOFFS
from directed_suffix_probe import OUTPUT,PREVIOUS,FOCUS,config_and_manifest,existing_results
from prepare_fixed_k_data import digest
from fixed_k_exit import atomic_json
from summarize_fixed_k import summarize_rows,write_csv

LABELS=dict(question_only='Question only',answer_only='Answer only',question_answer='Question + Answer',global_='Global')
LABELS['global']=LABELS.pop('global_')
COLORS=dict(zip(MODES,['tab:blue','tab:orange','tab:green','tab:red']))


def summaries(rows):
    results=[]
    for mode in ['full']+MODES:
        selected=[r for r in rows if r['mode']==mode]
        for summary in summarize_rows(selected,[32] if mode=='full' else CUTOFFS):
            selected_k=[r for r in selected if r['cutoff']==summary['cutoff']]
            results.append(dict(mode=mode,**summary,
                                gold_logp_mean=float(np.mean([r['correct_answer_log_probability'] for r in selected_k])),
                                gold_logp_median=float(np.median([r['correct_answer_log_probability'] for r in selected_k]))))
    return results


def comparisons(rows):
    index={(r['question_id'],r['mode'],r['cutoff']):r for r in rows}
    ids=sorted({r['question_id'] for r in rows}); paired=[];summary=[]
    for k in CUTOFFS:
        for q in ids:
            modes={m:index[q,m,k] for m in MODES}
            qrow,arow,qa,g=(modes[m] for m in MODES)
            paired.append(dict(question_id=q,cutoff=k,full_correct=g['full_correct'],
                               q_correct=qrow['candidate_correct'],a_correct=arow['candidate_correct'],
                               qa_correct=qa['candidate_correct'],global_correct=g['candidate_correct'],
                               q_margin=qrow['answer_margin'],a_margin=arow['answer_margin'],qa_margin=qa['answer_margin'],global_margin=g['answer_margin'],
                               nonadditivity_drop_residual=qa['answer_margin_drop']-qrow['answer_margin_drop']-arow['answer_margin_drop'],
                               additional_global_margin_drop=g['answer_margin_drop']-qa['answer_margin_drop'],
                               global_vs_qa_question_drift=g['question_mean_cosine_distance']-qa['question_mean_cosine_distance']))
        selected=[r for r in paired if r['cutoff']==k]
        lost=[r for r in selected if r['full_correct'] and not r['global_correct']]
        summary.append(dict(cutoff=k,global_lost_full_correct_ids=[r['question_id'] for r in lost],
                            q_preserves_global_lost_ids=[r['question_id'] for r in lost if r['q_correct']],
                            a_preserves_global_lost_ids=[r['question_id'] for r in lost if r['a_correct']],
                            qa_preserves_global_lost_ids=[r['question_id'] for r in lost if r['qa_correct']],
                            both_singles_correct_but_qa_wrong_ids=[r['question_id'] for r in selected if r['full_correct'] and r['q_correct'] and r['a_correct'] and not r['qa_correct']],
                            qa_correct_global_wrong_ids=[r['question_id'] for r in selected if r['qa_correct'] and not r['global_correct']],
                            qa_wrong_global_correct_ids=[r['question_id'] for r in selected if not r['qa_correct'] and r['global_correct']],
                            nonadditivity_mean=float(np.mean([r['nonadditivity_drop_residual'] for r in selected])),
                            nonadditivity_max_abs=max(abs(r['nonadditivity_drop_residual']) for r in selected),
                            additional_global_margin_drop_mean=float(np.mean([r['additional_global_margin_drop'] for r in selected])),
                            additional_global_margin_drop_p95=float(np.percentile([r['additional_global_margin_drop'] for r in selected],95)),
                            additional_global_margin_drop_max=max(r['additional_global_margin_drop'] for r in selected)))
    return paired,summary


def plots(rows,summary):
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for mode in MODES:
        values=[s for s in summary if s['mode']==mode]
        axes[0].plot(CUTOFFS,[100*s['accuracy'] for s in values],'o-',label=LABELS[mode],color=COLORS[mode])
        axes[1].plot(CUTOFFS,[s['correct_to_wrong'] for s in values],'o-',label=LABELS[mode],color=COLORS[mode])
    baseline=next(s for s in summary if s['mode']=='full')
    axes[0].axhline(100*baseline['accuracy'],ls='--',color='black',label='Native Full')
    axes[0].set_ylabel('Candidate accuracy (%)');axes[1].set_ylabel('Full-correct -> wrong (count)')
    for ax in axes:ax.set_xlabel('Suffix starts at layer K');ax.set_xticks(CUTOFFS);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUTPUT/'accuracy_and_flips.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,k in zip(axes,CUTOFFS):
        values=[[r['answer_margin_drop'] for r in rows if r['mode']==m and r['cutoff']==k] for m in MODES]
        box=ax.boxplot(values,tick_labels=['Q','A','Q+A','Global'],showfliers=True,patch_artist=True)
        for patch,mode in zip(box['boxes'],MODES):patch.set_facecolor(COLORS[mode]);patch.set_alpha(.4)
        ax.axhline(0,color='black',lw=.6);ax.set_title(f'K={k}');ax.set_ylabel('Full - intervention margin (nats)')
    fig.suptitle('All 240 paired images; outliers retained; y scales vary')
    fig.tight_layout();fig.savefig(OUTPUT/'margin_drop_distribution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for row,k in enumerate([16,24]):
        for ax,mode in zip(axes[row],MODES):
            values=[r for r in rows if r['mode']==mode and r['cutoff']==k]
            for flipped in (False,True):
                group=[r for r in values if (r['full_correct'] and not r['candidate_correct'])==flipped]
                ax.scatter([r['question_mean_cosine_distance'] for r in group],[r['answer_margin'] for r in group],
                           c='crimson' if flipped else 'steelblue',s=55 if flipped else 13,
                           marker='*' if flipped else 'o',alpha=1 if flipped else .65,zorder=3 if flipped else 2)
            ax.axhline(0,color='gray',lw=.6);ax.set_title(f'{LABELS[mode]}, K={k}')
            ax.set_xlabel('Mean Question cosine distance');ax.set_ylabel('Answer margin: gold - wrong (nats)')
            if mode=='answer_only':ax.text(.03,.92,'Question states exactly Full',transform=ax.transAxes,fontsize=8)
    fig.suptitle('Red stars: Full-correct -> wrong (ties included). Independent axes; A-only cosine is roundoff.')
    fig.tight_layout();fig.savefig(OUTPUT/'question_drift_answer_margin.png',dpi=180);plt.close(fig)


def main():
    config,manifest=config_and_manifest();binding=digest(config)
    if json.loads((OUTPUT/'experiment_config.json').read_text())!=config:raise ValueError('Analysis binding changed')
    stages={}
    for stage in ('smoke','run'):
        samples=[s for s in manifest['samples'] if s['question_id'] in config['smoke_question_ids']] if stage=='smoke' else manifest['samples']
        gate=json.loads((OUTPUT/f'{stage}_complete.json').read_text())
        stages[stage]=existing_results(stage,samples,binding)
        if gate['binding']!=binding or not gate['passed'] or len(stages[stage])!=len(samples):raise ValueError('Incomplete cohort/gate')
    rows=[r for sample in manifest['samples'] for r in stages['run'][sample['question_id']]['rows']]
    if len(rows)!=3120 or len({(r['question_id'],r['mode'],r['cutoff']) for r in rows})!=3120:raise ValueError('Wrong paired result cardinality')
    for stage in stages.values():
        for data in stage.values():
            n=len(data['prompt_ids'])+1
            mapping=data['query_mapping']
            if set(mapping['question_queries']) & set(mapping['answer_prediction_queries']):
                raise AssertionError('Overlapping query groups')
            if mapping['answer_prediction_queries']!=[n-2] or len(mapping['visual_keys'])!=576:
                raise AssertionError('Invalid P-1/visual key mapping')
            for detail in data['details']:
                if detail['mode']=='full':continue
                for candidate in detail['candidates']:
                    records=candidate['hook_audit']
                    if [r['layer'] for r in records]!=list(range(32)):
                        raise AssertionError('Incomplete layer mask audit')
                    if any(r['applied']!=(r['layer']>=detail['cutoff']) or
                           r['cloned_mask']!=r['applied'] or not r['positions_unchanged'] or
                           r['mask_shape']!=[1,1,n,n] for r in records):
                        raise AssertionError('Invalid suffix/mask clone/positions/sequence length')
    write_csv(OUTPUT/'per_sample_results.csv',rows)
    atomic_json(OUTPUT/'per_sample_results.json',dict(binding=binding,n=240,completed=True,rows=rows,details='checkpoints/run/<question_id>.json'))
    summary=summaries(rows);write_csv(OUTPUT/'summary_by_condition.csv',summary);atomic_json(OUTPUT/'summary_by_condition.json',summary)
    compute=[]
    for mode,k in [('full',32)]+[(m,k) for m in MODES for k in CUTOFFS]:
        measures=[c['measurement'] for s in stages['run'].values() for d in s['details']
                  if d['mode']==mode and d['cutoff']==k for c in d['candidates']]
        compute.append(dict(mode=mode,cutoff=k,forwards=len(measures),
                            mean_forward_seconds=float(np.mean([m['forward_seconds'] for m in measures])),
                            median_forward_seconds=float(np.median([m['forward_seconds'] for m in measures])),
                            **{name:max(m['gpu_memory'][name] for m in measures)
                               for name in ('allocated_mib','reserved_mib','peak_allocated_mib','peak_reserved_mib')}))
    write_csv(OUTPUT/'compute_diagnostics.csv',compute)
    atomic_json(OUTPUT/'sequence_lengths.json',dict(
        sequence_lengths=sorted({r['sequence_length'] for r in rows}),
        visual_tokens=576,physical_deletion=False,
        note='All 32 layers retain the same full sequence; layer mask audits are in checkpoints. No acceleration experiment.'))
    paired,compare=comparisons(rows);write_csv(OUTPUT/'paired_path_comparisons.csv',paired);atomic_json(OUTPUT/'path_comparison_summary.json',compare)
    focus=[dict(question=s['question'],**r) for s in manifest['samples'] for r in rows if r['question_id']==s['question_id'] and r['question_id'] in FOCUS]
    write_csv(OUTPUT/'historical_three_cases.csv',focus);atomic_json(OUTPUT/'historical_three_cases.json',focus)
    identical=[]
    fields=['answer_margin','correct_answer_log_probability','question_mean_cosine_distance']
    for q,data in stages['smoke'].items():
        other=stages['run'][q]
        exact=all(a[f]==b[f] for a,b in zip(data['rows'],other['rows']) for f in fields)
        if not exact:raise AssertionError('Smoke (attention on) vs production (off) scores differ')
        identical.append(dict(question_id=q,exact=True))
    audits=[c['attention_audit'] for sample in stages['smoke'].values() for d in sample['details'] if d['mode']!='full' for c in d['candidates']]
    historical=[dict(question_id=q,cutoff=d['cutoff'],**d['extra_checks']) for q,s in stages['run'].items() for d in s['details'] if d['mode']=='global']
    equivalences=[c for s in stages['smoke'].values() for d in s['details'] if d['mode']=='global' for c in d['extra_checks']['current_physical_equivalence']]
    validation=dict(binding=binding,cpu_tests=72,smoke_images=len(stages['smoke']),production_images=240,
                    attention_forwards_audited=len(audits),attention_layers_audited=sum(len(a) for a in audits),
                    all_target_attention_exact_zero=all(r['target_attention_max']==0 for a in audits for r in a if r['applied']),
                    max_attention_row_sum_error=max(r['row_sum_max_error'] for a in audits for r in a),
                    all_k32_valid_text_hidden_logits_exact=True,all_answer_only_question_hidden_exact_full=True,
                    all_qa_question_hidden_exact_question_only=True,post_cleanup_native_full_exact=all(s['checks']['post_cleanup_native_full_exact'] for s in stages['smoke'].values()),
                    smoke_production_exact=identical,global_history_candidate_logp_max_abs_error=max(d['candidate_logp_max_abs_error'] for d in historical),
                    global_history_margin_max_abs_error=max(d['margin_abs_error'] for d in historical),
                    global_history_classification_differences=[d for d in historical if d['global_vs_historical_classification_changed']],
                    gpu_global_physical_max_logit_error=max(c['max_abs_logit_error'] for c in equivalences),
                    gpu_global_physical_max_text_cosine=max(c['max_text_cosine_distance'] for c in equivalences),
                    gpu_global_physical_max_text_relative_rmse=max(c['max_text_relative_rmse'] for c in equivalences),tolerances=config['tolerances'])
    atomic_json(OUTPUT/'correctness_validation.json',validation)
    write_csv(OUTPUT/'global_vs_historical_physical.csv',[{k:v for k,v in d.items() if k!='current_physical_equivalence'} for d in historical])
    plots(rows,summary)
    print(json.dumps(dict(summary=summary,paths=compare,validation=validation),indent=2))


if __name__=='__main__':main()

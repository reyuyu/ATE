"""CPU-only aggregation of completed, hash-bound source intervention data."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from benchmarks.dataset_registry import freeze_json,sha256,digest
from paper_rq1.analyze import write_csv
from .protocol import ROOT,OUT,LAYERS,DEEP,SPECIAL,TYPES,MODES,GROUPS,FAMILY,conditions
from .run import verify,verify_targets,gate
from .statistics import ci,fixed_effects


def collect():
    config=verify();targets=verify_targets();stages={stage:gate(stage) for stage in ['smoke','full','ablation']}
    if list(OUT.glob('**/failures/*.json')):raise ValueError('Unresolved recorded sample failures; no complete claim allowed')
    manifest=json.loads((OUT/'sample_manifest.json').read_text())
    selected=json.loads((OUT/'selected_token_manifest.json').read_text())
    rows=[];native=[];numerical=[];raw_bindings=[];environments=[]
    for stage in stages:
        for rank in range(4):
            env=json.loads((OUT/f'{stage}/shard{rank}/environment.json').read_text())
            if env['code_commit']!=config['code_commit'] or env['transformers']!='4.49.0' or env['parameter_dtypes']!=['torch.float16']:
                raise ValueError('Worker environment not frozen version')
            environments.append(dict(stage=stage,rank=rank,**{k:env[k] for k in ['code_commit','torch','transformers','cuda_runtime','model_revision','cuda_visible_devices']}))
            if stage=='smoke':
                checkpoint=next((OUT/f'{stage}/shard{rank}/checkpoints').glob('*.json'))
                record=json.loads(checkpoint.read_text())
                if not all(record[k] for k in ['passed','native_generate_tokens_exact','replay_all_steps_exact']):raise ValueError('Smoke invalid')
                for r in record['checks']:
                    if 'exact' in r and not r['exact']:raise ValueError('Observer no-op differs')
                    if 'noop_exact' in r and not r['noop_exact']:raise ValueError('Source no-op differs')
                    numerical.extend(r.get('audits',[]))
    for name in ['code_commit','torch','transformers','cuda_runtime','model_revision']:
        if len({env[name] for env in environments})!=1:raise ValueError('Cross-worker environment differs')
    for sample in manifest['samples']:
        name=sample['image_identity'].replace(':','_')
        full=json.loads((OUT/f"full/shard{sample['rank']}/checkpoints/{name}.json").read_text())
        record=json.loads((OUT/f"ablation/shard{sample['rank']}/checkpoints/{name}.json").read_text())
        if record['binding']!=digest(config) or record['sample_sha256']!=digest(sample) or record['selected_sha256']!=targets['selected_sha256']:
            raise ValueError('Formal input binding differs')
        positions=selected['images'][sample['image_identity']]['selected']
        if record['positions']!=len(positions):raise ValueError('Missing target')
        if len(record['rows'])!=56*len(positions) or len(record['source_rows'])!=44*len(positions):raise ValueError('Incomplete source observations')
        for target in positions:
            sub=[r for r in record['rows'] if r['generation_step']==target['generation_step']]
            expected={(mode,layer) for layer in LAYERS for mode in ['Full',*conditions(layer)]}
            if len(sub)!=56 or {(r['mode'],r['layer']) for r in sub}!=expected:raise ValueError('Missing or duplicate layer/mode')
            src=[r for r in record['source_rows'] if r['generation_step']==target['generation_step']]
            expected_sources={(name,l) for l in LAYERS for name in GROUPS+(['RECENT','OLDER_HISTORY'] if l in SPECIAL else [])}
            if len(src)!=44 or {(r['source'],r['layer']) for r in src}!=expected_sources:raise ValueError('Missing/duplicate native source')
        for audit in record['audits']:
            if audit['parent_sha256_before']!=audit['parent_sha256_after'] or not audit['observer_logits_exact'] or not audit['replay_logits_exact']:
                raise ValueError('Contaminated cache or incorrect native logits')
            numerical.extend(audit['observer_numerical'])
        rows.extend(record['rows']);native.extend(record['source_rows'])
    for stage in ['smoke','full','ablation']:
        for rank in range(4):
            for path in sorted((OUT/f'{stage}/shard{rank}/checkpoints').glob('*.json')):
                record=json.loads(path.read_text());raw=Path(record['raw_path'])
                if sha256(raw)!=record['raw_sha256']:raise ValueError('Raw logits SHA256 changed')
                with np.load(raw) as arrays:
                    values=arrays['altered_logits' if stage=='ablation' else 'full_logits']
                    if not np.isfinite(values).all():raise ValueError('Raw nonfinite logits')
                    if stage=='ablation' and values.shape[0]!=46*record['positions']:raise ValueError('Missing altered raw logits')
                raw_bindings.append(dict(stage=stage,rank=rank,image_id=record['image_id'],path=str(raw),bytes=raw.stat().st_size,sha256=record['raw_sha256']))
    if len({r['image_sha256'] for r in rows})!=200 or sum(r['mode']!='Full' for r in rows)!=46*targets['positions']:
        raise ValueError('Not all independent200 images/conditions present')
    for audit in numerical:
        if not audit['partition_exact'] or not audit['denominator_unchanged']:raise ValueError('Invalid source partition/denominator')
        if any(not np.isfinite(v) or v>1 for k,v in audit.items() if k.endswith('_error_to_bound')):
            raise ValueError('Frozen numerical reconstruction bound failed')
    audit=dict(passed=True,status='complete',images=200,target_positions=targets['positions'],role_counts=targets['counts'],
        intervention_forwards=46*targets['positions'],per_token_rows=len(rows),native_source_rows=len(native),
        source_commit=config['code_commit'],target_sha256=targets['selected_sha256'],
        all_replay_steps_exact=True,native_observer_logits_exact=True,cache_target_hash_checks=targets['positions'],
        missing_duplicate_conditions=0,numerical_checks=len(numerical),raw_logit_files=len(raw_bindings),
        raw_logit_bytes=sum(r['bytes'] for r in raw_bindings),stages=stages,environments=environments,
        max_numerical_errors={k:max(r[k] for r in numerical if k in r) for k in numerical[0]
            if k.endswith('_abs_error') or k.endswith('_error_to_bound')},
        max_history_split_error_to_bound=max(r.get('history_split_max_error_to_bound',0) for r in numerical),
        no_nan_inf_oom=True,hooks_and_weight_versions_unchanged=True)
    write_csv(OUT/'source_ablation_per_token.csv',rows);write_csv(OUT/'native_source_contributions.csv',native)
    write_csv(OUT/'raw_logits_manifest.csv',raw_bindings)
    freeze_json(OUT/'numerical_validation.json',audit)
    return rows,native


def summarize(rows,native):
    buckets=defaultdict(list)
    for r in rows:
        if r['mode']!='Full':buckets[(r['mode'],r['layer'],r['word_type'])].append(r)
    summary=[]
    for (mode,layer,role),sub in sorted(buckets.items()):
        loss=np.array([r['loss'] for r in sub]);absolute=np.array([r['abs_delta_logp'] for r in sub])
        stats=ci(sub,loss);aci=ci(sub,absolute);fci=ci(sub,[r['top1_flip'] for r in sub])
        summary.append(dict(mode=mode,layer=layer,word_type=role,**stats,median_loss=float(np.median(loss)),
            p90_loss=float(np.quantile(loss,.9)),p95_loss=float(np.quantile(loss,.95)),mean_abs_delta_logp=float(absolute.mean()),
            abs_ci95_low=aci['ci95_low'],abs_ci95_high=aci['ci95_high'],top1_flip_rate=fci['mean'],
            flip_ci95_low=fci['ci95_low'],flip_ci95_high=fci['ci95_high'],top1_flips=sum(r['top1_flip'] for r in sub),
            mean_js=float(np.mean([r['js_divergence'] for r in sub])),
            mean_full_logp=float(np.mean([r['full_token_logp'] for r in sub])),mean_entropy=float(np.mean([r['full_token_entropy'] for r in sub])),
            mean_step=float(np.mean([r['generation_step'] for r in sub])),mean_relative_position=float(np.mean([r['relative_generation_position'] for r in sub]))))
    write_csv(OUT/'source_layer_summary.csv',summary)
    grouped=defaultdict(list)
    for r in native:grouped[(r['source'],r['layer'],r['word_type'])].append(r)
    source_summary=[]
    for (source,layer,role),sub in sorted(grouped.items()):
        record=dict(source=source,layer=layer,word_type=role,n_images=len({r['image_sha256'] for r in sub}),n_positions=len(sub))
        for key in ['mass','contribution_norm','relative_residual_ratio','cosine_with_full_attention',
                    'attention_residual_cosine','image_history_cosine','group_cancellation_ratio']:
            valid=[r for r in sub if r[key] is not None]
            record[key+'_n_valid']=len(valid)
            if valid:
                s=ci(valid,[r[key] for r in valid])
                record.update({key+'_mean':s['mean'],key+'_ci95_low':s['ci95_low'],key+'_ci95_high':s['ci95_high']})
            else:record.update({key+'_mean':None,key+'_ci95_low':None,key+'_ci95_high':None})
        source_summary.append(record)
    write_csv(OUT/'native_source_summary.csv',source_summary)
    return summary,source_summary


def contrasts(rows):
    lookup={(r['image_sha256'],r['generation_step'],r['mode'],r['layer']):r for r in rows}
    main=[];role_differences=[];deep=[];recent=[];controlled=[];designs={}

    def contrast(tag,sub,values,**meta):
        r=dict(contrast=tag,**meta,**ci(sub,values,FAMILY));main.append(r);return r

    for layer in SPECIAL:
        for mode in ['History','Image','Prompt']:
            sub=[r for r in rows if r['word_type']=='Visual' and r['mode']==mode and r['layer']==layer]
            contrast('A_Visual_'+mode+'_vs_zero',sub,[r['loss'] for r in sub],analysis='raw',layer=layer,mode=mode)
        h=[r for r in rows if r['word_type']=='Visual' and r['mode']=='History' and r['layer']==layer]
        for mode in ['Image','Prompt']:
            contrast('A_Visual_History_minus_'+mode,h,
                [r['loss']-lookup[(r['image_sha256'],r['generation_step'],mode,layer)]['loss'] for r in h],
                analysis='paired',layer=layer,mode='History-'+mode)
        for mode in ['Image','Prompt','History']:
            sub=[r for r in rows if r['mode']==mode and r['layer']==layer]
            function={r['image_sha256']:r for r in sub if r['word_type']=='Function'}
            visual=[r for r in sub if r['word_type']=='Visual' and r['image_sha256'] in function]
            delta=[r['loss']-function[r['image_sha256']]['loss'] for r in visual]
            item=contrast('B_Visual_minus_Function',visual,delta,analysis='raw',layer=layer,mode=mode)
            role_differences.append(item)
            meta,effects=fixed_effects(sub);designs[mode+str(layer)]=meta
            controlled.extend(dict(layer=layer,mode=mode,**r) for r in effects)
            effect=next((r for r in effects if r['variable']=='Visual_vs_Function'),None)
            if effect is not None:
                item=dict(contrast='B_Visual_minus_Function',analysis='adjusted',layer=layer,mode=mode,
                    mean=effect['effect'],**{k:v for k,v in effect.items() if k not in ['effect','variable']})
                main.append(item);role_differences.append(item)
    for layer in DEEP:
        sub=[r for r in rows if r['mode']=='Whole' and r['layer']==layer and r['word_type']=='Visual']
        item=contrast('C_Visual_Whole_vs_zero',sub,[r['loss'] for r in sub],analysis='paired',layer=layer,mode='Whole')
        deep.append(dict(item,mean_abs_delta_logp=float(np.mean([r['abs_delta_logp'] for r in sub])),
            top1_flip_rate=float(np.mean([r['top1_flip'] for r in sub]))))
        if layer>25:
            delta=[r['loss']-lookup[(r['image_sha256'],r['generation_step'],'Whole',layer-1)]['loss'] for r in sub]
            deep.append(contrast('C_Visual_Whole_adjacent_difference',sub,delta,analysis='paired',layer=layer,mode=f'L{layer}-L{layer-1}'))
    for layer in SPECIAL:
        for role in TYPES:
            sub=[r for r in rows if r['mode']=='Recent' and r['layer']==layer and r['word_type']==role]
            older=[lookup[(r['image_sha256'],r['generation_step'],'Older',layer)] for r in sub]
            item=contrast('D_Recent_minus_Older_'+role,sub,[a['loss']-b['loss'] for a,b in zip(sub,older)],analysis='paired',layer=layer,mode='Recent-Older',word_type=role)
            recent.append(dict(item,mean_recent_loss=float(np.mean([r['loss'] for r in sub])),
                mean_older_loss=float(np.mean([r['loss'] for r in older])),
                recent_abs_change=float(np.mean([r['abs_delta_logp'] for r in sub])),older_abs_change=float(np.mean([r['abs_delta_logp'] for r in older])),
                recent_flip_rate=float(np.mean([r['top1_flip'] for r in sub])),older_flip_rate=float(np.mean([r['top1_flip'] for r in older])),
                empty_older_targets=sum(r['generation_step']==1 for r in sub)))
    if len(main)!=FAMILY:
        if all(d['status']=='estimated' for d in designs.values()):raise ValueError('Frozen41 contrasts changed')
        # Rank failure is reported rather than generating a fake adjusted estimate.
    write_csv(OUT/'planned_primary_contrasts.csv',main)
    write_csv(OUT/'visual_function_paired_comparison.csv',role_differences)
    if controlled:write_csv(OUT/'controlled_effect_analysis.csv',controlled)
    else:
        with (OUT/'controlled_effect_analysis.csv').open('x',newline='') as stream:
            csv.writer(stream).writerow(['layer','mode','variable','effect','ci95_low','ci95_high'])
    freeze_json(OUT/'controlled_model_audit.json',designs)
    write_csv(OUT/'deep_layer_peak_analysis.csv',deep);write_csv(OUT/'recent_vs_older_history.csv',recent)
    freeze_json(OUT/'deep_continuity_assessment.json',dict(
        measured_layers=DEEP,criterion='Every one of seven corrected Visual Whole lower bounds>0; otherwise not proven continuous. No equivalence/insensitivity inference from CI covering0.',
        every_deep_corrected_lower_bound_positive=all(r['familywise95_low']>0 for r in deep if r['contrast']=='C_Visual_Whole_vs_zero'),
        no_unmeasured_layer_inference=True,planned_contrasts_available=len(main),family=FAMILY))
    return main,role_differences,deep,recent


def label_audit():
    chosen=json.loads((OUT/'selected_token_manifest.json').read_text())['images']
    frequencies={role:Counter() for role in TYPES};words=Counter();multi=0
    for value in chosen.values():
        for r in value['annotation']['words']:
            words[r['word_type']]+=1;multi+=r['token_length']>1
        for r in value['selected']:
            if r['token_length']!=1 or r['generation_step']<1 or not r['exact_complete_single_token']:raise ValueError('Selected invalid lexical position')
            frequencies[r['word_type']][r['word'].lower()]+=1
    sample=sorted(chosen,key=lambda k:digest(['late-source-label-audit',chosen[k]['image_sha256']]))[:20]
    review=[]
    for image_id in sample:
        for r in chosen[image_id]['selected']:
            review.append(dict(image_id=image_id,caption=chosen[image_id]['caption'],**r))
    write_csv(OUT/'classification_audit_sample.csv',review)
    freeze_json(OUT/'classification_audit.json',dict(selected_manifest_sha256=sha256(OUT/'selected_token_manifest.json'),
        words_by_role=dict(words),multi_token_words_excluded=multi,selected_word_frequencies={r:dict(v) for r,v in frequencies.items()},
        input_hash_review_images=20,role_label_changes=0,limitation='Inherited conservative offline lexicon; context audit is not gold POS or visual grounding. No labels changed after outcomes.'))


def main():
    rows,native=collect();summary,source_summary=summarize(rows,native);contrast_result=contrasts(rows);label_audit()
    from .figures import figures
    figures(rows,native,summary,source_summary,*contrast_result)
    from .report import report
    report(rows,native,summary,source_summary,*contrast_result)
    freeze_json(OUT/'analysis_completion.json',dict(status='complete_collected_analysis' if len(contrast_result[0])==FAMILY else 'partial_statistics_rank_failure',
        source_commit=verify()['code_commit'],images=200,target_positions=verify_targets()['positions'],
        analysis_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['analyze.py','statistics.py','figures.py','report.py']},
        files_sha256={str(p.relative_to(OUT)):sha256(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ['.tmp','.lock'] and p.name!='analysis_completion.json'}))
    print('All independent200 images/conditions validated, analyzed and plotted.',flush=True)


if __name__=='__main__':main()

"""Independent whole-cohort scoring; no prompt/method selection in this stage."""
import csv
from collections import Counter,defaultdict
from full_common import *
from statistics_local import paired,summaries


def main():
    p=check()
    rows=json.loads((OUT/'sample_manifest.json').read_text())
    index={r['key']:r for r in rows}
    records={}
    bindings={}
    for shard in [0,1]:
        for key,b in read_store(OUT/f'bindings{shard}.sqlite'):
            bindings[key]=b
        for key,r in read_store(OUT/f'worker{shard}.sqlite'):
            assert key not in records
            assert index[r['key']]['shard']==shard
            assert r['protocol_sha256']==sha(OUT/'p2_full_protocol.json')
            records[key]=r
    for key,r in read_store(OUT/'reused_validation.sqlite'):
        assert key not in records and index[r['key']]['reuse_validation']
        records[key]=r
    aligned=0
    for shard in [0,1]:
        if (OUT/f'aligned{shard}.sqlite').exists():
            for key,r in read_store(OUT/f'aligned{shard}.sqlite'):
                assert key in records and records[key]['source']=='AUDITED_PRIOR_VALIDATION'
                assert index[r['key']]['shard']==shard and r['gpu_uuid']==UUIDS[shard]
                assert r['input_binding_sha256']==records[key]['input_binding_sha256']
                records[key]=r
                aligned+=1
    assert aligned==json.loads((OUT/'hardware_alignment_plan.json').read_text())['extra_generations']
    old_bindings={}
    for shard in [0,1]:
        old_bindings.update(dict(read_store(OLD/f'bindings{shard}.sqlite')))
    assert len(records)==len(rows)*2==12788
    image_devices=defaultdict(set)
    for r in records.values():
        if r.get('state','COMPLETE')=='COMPLETE':
            image_devices[r['image_sha256']].add(r['gpu_uuid'])
    assert all(len(v)==1 for v in image_devices.values()),'Canonical same-image hardware mismatch'
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(p['model'],local_files_only=True)
    pairs=defaultdict(dict)
    outputs=[]
    for key,r in records.items():
        row=index[r['key']]
        if r.get('state','COMPLETE')!='COMPLETE':
            continue
        assert row['gold']==r['gold'] and row['image_sha256']==r['image_sha256']
        new,text,ids=render(processor,row,p['suffix_exact_from_prior'])
        binding=old_bindings[row['key']+'|P2'] if row['reuse_validation'] else bindings[row['key']]
        assert binding['prompt']==text and binding['input_ids']==ids
        assert digest(binding)==r['input_binding_sha256']
        assert r['prompt_sha256']==new['prompt_sha256']
        assert len(r['token_ids'])==r['generated_tokens']<=row['max_new_tokens']
        assert processor.tokenizer.decode(r['token_ids'],skip_special_tokens=True).strip()==r['text'].strip()
        rescore=scored(row,r['text'],r['truncated'])
        assert rescore['score']==r['score']
        assert r['structural']['status']=='PASS'
        pairs[row['key']][r['condition']]=r
        outputs.append(dict(benchmark=row['benchmark'],key=row['key'],question_id=row['question_id'],
            image_sha256=row['image_sha256'],circular_group=row.get('circular_group'),prior_stage=row['prior_stage'],
            question=row['question'],gold=row['gold'],condition=r['condition'],text=r['text'],token_ids=r['token_ids'],
            score=r['score'],parsed=r['parsed'],diagnostic_score=r.get('diagnostic_score'),parse_failed=r['parse_failed'],
            eos=r['eos'],empty=r['empty'],truncated=r['truncated'],generated_tokens=r['generated_tokens'],
            input_binding_sha256=r['input_binding_sha256'],prompt_sha256=r['prompt_sha256'],
            source=r['source'],gpu_uuid=r['gpu_uuid'],protocol_sha256=r['protocol_sha256']))
    table(OUT/'p2_per_question.csv',outputs)
    raw=[]
    for key,pair in pairs.items():
        if set(pair)!={'Native','SVH'}:
            continue
        a,b=pair['Native'],pair['SVH']
        row=index[key]
        assert a['input_binding_sha256']==b['input_binding_sha256']
        raw.append(dict(key=key,benchmark=row['benchmark'],image_sha256=row['image_sha256'],
            circular_group=row.get('circular_group'),prior_stage=row['prior_stage'],reuse_validation=row['reuse_validation'],
            native_score=a['score'],svh_score=b['score'],
            native_diagnostic=a.get('diagnostic_score',a['score']),svh_diagnostic=b.get('diagnostic_score',b['score']),
            native_text=a['text'],svh_text=b['text'],gold=row['gold'],question=row['question'],
            first_token_flip=a['token_ids'][:1]!=b['token_ids'][:1]))
    all_expected_groups=defaultdict(set)
    for row in rows:
        if row['benchmark']=='MMBench':
            all_expected_groups[row['circular_group']].add(row['key'])
    mm=defaultdict(list)
    for r in raw:
        if r['benchmark']=='MMBench':
            mm[r['circular_group']].append(r)
    circles=[]
    for group,g in mm.items():
        if {r['key'] for r in g}!=all_expected_groups[group]:
            continue
        assert len({r['image_sha256'] for r in g})==1
        circles.append(dict(circular_group=group,benchmark='MMBench',image_sha256=g[0]['image_sha256'],
            prior_stage=g[0]['prior_stage'],reuse_validation=all(r['reuse_validation'] for r in g),
            n_rows=len(g),native_score=float(all(r['native_score'] for r in g)),
            svh_score=float(all(r['svh_score'] for r in g)),
            native_rotation_correct=[r['native_score'] for r in g],svh_rotation_correct=[r['svh_score'] for r in g],
            native_answers=[r['native_text'] for r in g],svh_answers=[r['svh_text'] for r in g],gold=[r['gold'] for r in g],
            same_native_svh_rows=sum(r['native_text']==r['svh_text'] for r in g)))
    table(OUT/'p2_mmbench_per_circular_group.csv',circles)
    def included(r,subset):
        if subset=='All':return True
        if subset=='PriorValidation':return r['reuse_validation']
        if subset=='UnselectedImages':return r['prior_stage']!='Discovery'
        if subset=='UntouchedImages':return r['prior_stage']=='Untouched'
        raise ValueError(subset)
    scores=[]
    sqa_scores=[]
    mmb_scores=[]
    for task,pool,unit in [('ScienceQA-IMG',[r for r in raw if r['benchmark']=='ScienceQA-IMG'],'question'),
                           ('MMBench',circles,'Circular_group'),
                           ('MMBench',[r for r in raw if r['benchmark']=='MMBench'],'rotation_row')]:
        for subset in ['All','PriorValidation','UnselectedImages','UntouchedImages']:
            group=[r for r in pool if included(r,subset)]
            metrics=['strict']+(['conservative_diagnostic'] if task=='ScienceQA-IMG' else [])
            for metric in metrics:
                stats=paired(group,'native_diagnostic' if metric!='strict' else 'native_score',
                    'svh_diagnostic' if metric!='strict' else 'svh_score')
                out=dict(benchmark=task,unit=unit,subset=subset,metric=metric,**stats,
                    both_correct=sum(r['native_score']==1 and r['svh_score']==1 for r in group),
                    both_wrong=sum(r['native_score']==0 and r['svh_score']==0 for r in group),
                    interpretation='Exploratory paired image-cluster interval; not independent external confirmation')
                scores.append(out)
                (sqa_scores if task=='ScienceQA-IMG' else mmb_scores).append(out)
    table(OUT/'p2_scienceqa_scores.csv',sqa_scores)
    table(OUT/'p2_mmbench_circular_scores.csv',mmb_scores)
    table(OUT/'p2_paired_comparisons.csv',scores)
    table(OUT/'p2_negative_and_rescue_cases.csv',[r for r in raw if r['native_score']!=r['svh_score']])
    latency=[]
    letters=[]
    for task in TASKS:
        for cond in ['Native','SVH']:
            group=[r for r in records.values() if r['benchmark']==task and r['condition']==cond and r.get('state','COMPLETE')=='COMPLETE']
            for field in ['generated_tokens','e2e_ms','ttft_ms','decode_ms_per_token','vision_ms','prefill_ms','peak_allocated']:
                latency.append(dict(benchmark=task,condition=cond,metric=field,cohort='P2_full_natural_EOS',n=len(group),
                    **summaries([r[field] for r in group]),
                    eos_rate=sum(r['eos'] for r in group)/len(group),truncated=sum(r['truncated'] for r in group),
                    empty=sum(r['empty'] for r in group),parse_failed=sum(r['parse_failed'] for r in group),
                    reused_count=sum(r['source']=='AUDITED_PRIOR_VALIDATION' for r in group),
                    interpretation='Per-request timings incl reused Validation; not controlled operator-speed benchmark'))
            for letter,n in sorted(Counter(r['parsed'] or 'UNPARSED' for r in group).items()):
                letters.append(dict(benchmark=task,condition=cond,letter=letter,n=n))
    # Reuse only matching frozen P2 fixed32 timing probes, not P0 or other task records.
    fixed_rows=[]
    for shard in [0,1]:
        with (OLD/f'performance_raw{shard}.csv').open() as stream:
            for r in csv.DictReader(stream):
                if r['benchmark'] not in TASKS or r['prompt']!='P2' or r['mode']!='fixed32':
                    continue
                assert r['input_binding_sha256']==digest(old_bindings[r['key']+'|P2'])
                assert int(r['generated_tokens'])==32
                fixed_rows.append(dict(r,source='AUDITED_PRIOR_EXACT_P2_FIXED32_PROBE',source_path=str(OLD/f'performance_raw{shard}.csv')))
    table(OUT/'p2_fixed32_performance_raw.csv',fixed_rows)
    for task in TASKS:
        for cond in ['Native','SVH']:
            g=[r for r in fixed_rows if r['benchmark']==task and r['condition']==cond]
            for field in ['generated_tokens','e2e_ms','ttft_ms','decode_ms_per_token']:
                latency.append(dict(benchmark=task,condition=cond,metric=field,cohort='AUDITED_PRIOR_P2_FIXED32_2_INPUTS_3_REPEATS',
                    n=len(g),**summaries([float(r[field]) for r in g]),source='previous completed prompt experiment, not fresh full-cohort timing',
                    interpretation='Same model/code/input/budget; tiny fixed-token subset, not population speed guarantee'))
    table(OUT/'p2_output_length_and_latency.csv',latency)
    table(OUT/'p2_letter_distribution.csv',letters)
    validation=dict(status='COMPLETE_P2_FULL_QUALITY' if len(raw)==6394 and len(circles)==1176 else 'PARTIAL',
        planned_predictions=12788,terminal_predictions=len(records),complete_predictions=len(outputs),
        complete_paired_questions=len(raw),scienceqa_pairs=sum(r['benchmark']=='ScienceQA-IMG' for r in raw),
        mmbench_complete_groups=len(circles),mmbench_paired_rows=sum(r['benchmark']=='MMBench' for r in raw),
        failures=[r for r in records.values() if r.get('state','COMPLETE')!='COMPLETE'],
        reused_validation_predictions=sum(r['source']=='AUDITED_PRIOR_VALIDATION' for r in records.values()),
        new_predictions=sum(r['source'] in ['NEW_GENERATION','REALIGNED_VALIDATION_NEW_GENERATION'] for r in records.values() if r.get('state','COMPLETE')=='COMPLETE'),
        same_image_methods_on_same_gpu=True,hardware_realigned_predictions=aligned,
        all_records_independently_rescored=True,all_generated_token_ids_decoded=True,all_prompt_bindings_verified=True,
        sources_unchanged=bool(check()),old_dense_physical_gate='FAIL_RETAINED',author_official=False,
        cpu_audit=json.loads((OUT/'independent_p2_validation_audit.json').read_text())['status'])
    write(OUT/'p2_completion.json',validation)
    print(json.dumps([r for r in scores if r['subset']=='All' and r['metric']=='strict']),flush=True)


if __name__=='__main__':
    main()

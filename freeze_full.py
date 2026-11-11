"""Gold-blind prompt reconstruction and independent historical Validation audit."""
import csv
import gzip
import subprocess
from collections import Counter, defaultdict
from full_common import *


def main():
    import torch
    from transformers import AutoProcessor
    assert not torch.cuda.is_initialized()
    disk()
    previous = json.loads((OLD/'protocol.json').read_text())
    for path, expected in previous['frozen_source_sha256'].items():
        assert sha(path) == expected, path
    prior_manifest = json.loads((OLD/'sample_manifest.json').read_text())
    assert sha(OLD/'sample_manifest.json') == previous['manifest_sha256']
    suffix = previous['prompts']['P2_choices']
    original = json.loads((FIRST/'protocol.json').read_text())
    assert sha(original['inventory_path']) == original['inventory_sha256']
    with gzip.open(original['inventory_path'], 'rt') as stream:
        rows = [r for r in map(json.loads, stream) if r['benchmark'] in TASKS]
    assert Counter(r['benchmark'] for r in rows) == {'ScienceQA-IMG': 2017, 'MMBench': 4377}
    index = {r['key']: r for r in rows}
    assert len(index) == len(rows)
    sqa = json.loads(SQA_PATH.read_text())
    csv.field_size_limit(2**30)
    with MMB_PATH.open() as stream:
        mmb = {str(r['index']): r for r in csv.DictReader(stream, delimiter='\t')}
    groups = defaultdict(list)
    for r in rows:
        if r['benchmark'] == 'ScienceQA-IMG':
            raw = sqa[str(r['question_id'])]
            opts = {chr(65+i): v for i, v in enumerate(raw['choices'])}
            assert r['options'] == opts and list(r['options']) == list(opts)
            assert r['gold'] == raw['answer'] and r['question'] == raw['question']
            assert r['body'] == body(raw['question'], opts, raw['hint'])
            assert r['max_new_tokens'] == 128
        else:
            raw = mmb[str(r['question_id'])]
            opts = {k: raw[k] for k in 'ABCD' if raw.get(k, '').strip() and raw[k].lower() not in ('nan','none')}
            hint = raw.get('hint', '')
            hint = '' if hint.lower() in ('nan', 'none') else hint
            assert r['question'] == raw['question'] and r['options'] == opts
            assert list(r['options']) == list(opts) and r['gold'] == raw['answer']
            assert r['body'] == body(raw['question'], opts, hint)
            assert r['circular_group'] == int(raw['index']) % 1000000
            groups[r['circular_group']].append(r)
    assert len(groups) == 1176 and all(len({r['image_sha256'] for r in g}) == 1 for g in groups.values())
    # Rebuild from allowed inputs only. Gold/lecture/solution never enter this expression.
    processor = AutoProcessor.from_pretrained(previous['model'], local_files_only=True)
    saved_bindings = {}
    for shard in [0, 1]:
        saved_bindings.update(dict(read_store(OLD/f'bindings{shard}.sqlite')))
    all_old_p2 = {}
    for shard in [0, 1]:
        for identifier, rec in read_store(OLD/f'worker{shard}.sqlite'):
            if rec['prompt'] == 'P2':
                all_old_p2[rec['key'], rec['condition']] = rec
    discovery = {r['image_sha256'] for r in prior_manifest if r['stage'] == 'Discovery'}
    validation = {r['image_sha256'] for r in prior_manifest if r['stage'] == 'Validation'}
    assert not discovery & validation
    validation_keys = {r['key'] for r in prior_manifest if r['stage']=='Validation' and r['benchmark'] in TASKS}
    audited = []
    reuse = Store(OUT/'reused_validation.sqlite')
    pairs = defaultdict(dict)
    for key in sorted(validation_keys):
        row = index[key]
        new, text, ids = render(processor, row, suffix)
        binding = saved_bindings[key+'|P2']
        assert text == binding['prompt'] and ids == binding['input_ids']
        for cond in ['Native', 'SVH']:
            r = all_old_p2[key, cond]
            assert r['stage']=='Validation' and r['source']=='NEW_GENERATION'
            assert r['protocol_sha256'] == sha(OLD/'protocol.json')
            assert r['input_binding_sha256'] == digest(binding)
            assert r['prompt_sha256'] == new['prompt_sha256']
            assert r['gold'] == row['gold'] and r['image_sha256'] == row['image_sha256']
            decoded = processor.tokenizer.decode(r['token_ids'], skip_special_tokens=True).strip()
            assert decoded == r['text'].strip()
            independent = scored(row, decoded, r['truncated'])
            assert independent['score'] == r['score']
            if row['benchmark']=='ScienceQA-IMG':
                assert decoded in row['options'] and float(decoded == chr(65+row['gold'])) == r['score']
            else:
                assert decoded in row['options'] and float(decoded == row['gold']) == r['score']
            assert r['structural']['status']=='PASS'
            pairs[key][cond] = r
            reuse.put(key+'|'+cond, dict(r, source='AUDITED_PRIOR_VALIDATION', prior_record_id='Validation|'+key+'|P2|'+cond,
                                       prior_record_sha256=digest(r)))
        audited.append(dict(key=key, benchmark=row['benchmark'], input_binding_sha256=digest(binding),
            native_tokens=pairs[key]['Native']['token_ids'], svh_tokens=pairs[key]['SVH']['token_ids'],
            native_score=pairs[key]['Native']['score'], svh_score=pairs[key]['SVH']['score']))
    reuse.conn.close()
    audit_summary=[]
    for task in TASKS:
        subset=[k for k in pairs if index[k]['benchmark']==task]
        for cond in ['Native','SVH']:
            if task=='ScienceQA-IMG':
                values=[pairs[k][cond]['score'] for k in subset]
            else:
                gg=defaultdict(list)
                for k in subset:
                    gg[index[k]['circular_group']].append(k)
                values=[float(all(pairs[k][cond]['score'] for k in group)) for group in gg.values()]
                assert len(values)==150 and all(len(g)==len(groups[index[g[0]]['circular_group']]) for g in gg.values())
            audit_summary.append(dict(benchmark=task,condition=cond,n=len(values),correct=sum(values),accuracy=100*sum(values)/len(values)))
    generation_source=BASE/'vlm_research/qwen3vl_overnight/engine.py'
    assert 'gold' not in generation_source.read_text().lower()
    run_source=(PRIOR/'run_prompts.py').read_text()
    assert "for condition in ['Native', 'SVH']" in run_source or "for condition in ['Native','SVH']" in run_source
    audit=dict(status='PASS_INDEPENDENT_P2_VALIDATION_AUDIT', scores=audit_summary,
        exact_prompt_reconstruction=True, no_gold_lecture_solution_render_fields=True, raw_choice_rotation_mapping=True,
        generated_token_decode_matches=True, separate_native_svh_calls_verified=True,
        discovery_validation_image_disjoint=True, validation_rows=len(pairs),
        identical_native_svh_token_sequences=sum(p['Native']['token_ids']==p['SVH']['token_ids'] for p in pairs.values()),
        identical_native_svh_scores=sum(p['Native']['score']==p['SVH']['score'] for p in pairs.values()),
        old_strict_numerical_status='FAIL_RETAINED', source_files={str(p):sha(p) for p in [SQA_PATH,MMB_PATH,generation_source,PRIOR/'run_prompts.py']})
    write(OUT/'independent_p2_validation_audit.json', audit)
    table(OUT/'independent_p2_validation_rows.csv',audited)
    # Freeze cluster sharding and evaluation subsets before any new answer is read.
    loads=[0,0]
    by_image=defaultdict(list)
    for row in rows:
        row['prior_stage'] = 'Discovery' if row['image_sha256'] in discovery else 'Validation' if row['image_sha256'] in validation else 'Untouched'
        row['reuse_validation'] = row['key'] in validation_keys
        by_image[row['image_sha256']].append(row)
    for image, group in sorted(by_image.items(),key=lambda x:(-sum(r['image_tokens']**2 for r in x[1]),x[0])):
        shard=min(range(2),key=lambda s:loads[s])
        for row in group:
            row['shard']=shard
        loads[shard]+=sum(r['image_tokens']**2 for r in group)
    for group in groups.values():
        assert len({r['shard'] for r in group})==1
    write(OUT/'sample_manifest.json',rows)
    sources=dict(previous['frozen_source_sha256'])
    sources.update({str(p):sha(p) for p in ROOT.glob('*.py')})
    protocol=dict(name='ATE_HF_QWEN_P2_SHORT_ANSWER', base_commit='d073f887d3e13634df2400b45cf710a81287f5b4',
        prior_protocol_sha256=sha(OLD/'protocol.json'), original_protocol_sha256=sha(FIRST/'protocol.json'),
        model=previous['model'], revision=previous['revision'], model_files=original['model_files'],
        sources=sources, generation=previous['generation'], caps=previous['caps'], suffix_exact_from_prior=suffix,
        algorithm=previous['algorithm'], numerical_status='FAIL_RETAINED', author_official=False,
        manifest_sha256=sha(OUT/'sample_manifest.json'), counts={'ScienceQA-IMG':2017,'MMBench_rows':4377,'MMBench_groups':1176},
        planned_predictions=12788, reused_predictions=2*len(pairs), new_predictions=12788-2*len(pairs),
        subset_definitions={'All':'entire cohort; exploratory prompt-selected cohort',
                            'PriorValidation':'exact previous Validation keys',
                            'UnselectedImages':'all images absent from previous Discovery across tasks',
                            'UntouchedImages':'all images absent from both previous Discovery and Validation'},
        seed=SEED, bootstrap_replicates=20000, image_cluster_pairing=True, gpu_uuids=UUIDS,
        disk_budget_mib=300,disk_min_free_mib=512, original_main_disk_policy_unchanged=True,
        performance='Natural EOS per request; fixed32 reuse only exact audited prior P2 probes, explicitly historical timing cohort',
        no_algorithm_changes=True, original_results_read_only=True)
    write(OUT/'p2_full_protocol.json',protocol)
    write(OUT/'cpu_ready.json',dict(status='PASS',cuda_initialized=torch.cuda.is_initialized(),audit=audit_summary))
    print(json.dumps(dict(audit=audit_summary,new=protocol['new_predictions'],reused=protocol['reused_predictions'])),flush=True)


if __name__=='__main__':
    main()

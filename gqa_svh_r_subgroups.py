"""CPU-only Query/Non-Query supplement; no generation or selection."""
import csv
import gzip
import json

import qwen_svh_r_full as run
from vlm_research.qwen3vl_overnight.metrics import paired


def audit_reuse():
    prior = json.loads((run.DISCOVERY/'completion.json').read_text())
    source = run.DISCOVERY/'full_records.jsonl.gz'
    expected = prior['artifacts']['full_records.jsonl.gz']['sha256']
    assert run.sha(source) == expected
    with gzip.open(source, 'rt') as stream:
        originals = {x['key']: x['record'] for x in map(json.loads, stream)}
    n = 0
    for key, record in run.read_store(run.OUT/'reused.sqlite'):
        assert record['original_record_sha256'] == run.digest(originals[key])
        assert record['prefill_layers'] == record['decode_layers'] == run.LAYERS
        n += 1
    assert n == run.check()['exact_reused_records'] == 292
    run.write(run.OUT/'reuse_validation.json', dict(status='PASS_CPU_EXACT_ARCHIVED_RECORD_REUSE',
        reused=n, archived_source=str(source), source_sha256=expected, script_sha256=run.sha(__file__),
        new_gpu_numerical_validation='NOT_RUN', historical_numerical_status='FAIL_RETAINED'))
    print('Exact archived-record reuse verified:', n)


def main():
    audit_reuse()
    source = run.P2/'gqa_per_question_types.csv'
    with source.open() as stream:
        types = {r['question_id']: r for r in csv.DictReader(stream)}
    assert len(types) == 12578
    with (run.OUT/'per_question.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['benchmark'] == 'GQA']
    assert len(rows) == 12578
    for row in rows:
        old = types[row['question_id']]
        for field in ['question', 'gold', 'image_sha256', 'prompt_sha256', 'input_binding_sha256']:
            assert old[field] == row[field], ('OFFICIAL_TYPE_BINDING', row['key'], field)
        row['structural'] = old['structural']
        assert row['structural'] != 'UNKNOWN'
    result = []
    groups = [('Query', [r for r in rows if r['structural'] == 'query']),
              ('Non-Query', [r for r in rows if r['structural'] != 'query'])]
    assert sum(len(pool) for _,pool in groups) == 12578
    for label, pool in groups:
        mapping = {r['key']: {'cluster': r['image_sha256']} for r in pool}
        records = {name: [dict(key=r['key'], score=float(r[name+'_score']),
            scoring={'score': float(r[name+'_score'])}) for r in pool] for name in ['Native', 'SVH', run.NAME]}
        for name, selected in records.items():
            result.append(dict(group=label, condition=name, n=len(pool),
                image_clusters=len({r['image_sha256'] for r in pool}),
                accuracy=100*sum(r['score'] for r in selected)/len(selected)))
        for baseline in ['Native', 'SVH']:
            a, b = records[baseline], records[run.NAME]
            stats = paired('GQA', a, b, mapping, repeats=20000, seed=run.SEED)
            result.append(dict(group=label, condition=run.NAME, baseline=baseline, n=len(pool),
                harm=sum(y['score'] < x['score'] for x,y in zip(a,b)),
                rescue=sum(y['score'] > x['score'] for x,y in zip(a,b)), **stats))
    run.table(run.OUT/'gqa_query_nonquery.csv', result)
    run.write(run.OUT/'gqa_subgroup_validation.json', dict(status='PASS_CPU_OFFICIAL_TYPE_BINDING',
        source=str(source), source_sha256=run.sha(source), script_sha256=run.sha(__file__),
        questions=12578, protocol_sha256=run.sha(run.OUT/'protocol.json'),
        no_new_gpu_work=True, statistical_scope='Exploratory descriptive subgroups, not independent confirmation'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

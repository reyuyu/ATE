"""Final independent count/score audit and hashes after every log writer has exited."""
import csv
from collections import defaultdict
from full_common import *


def csv_rows(name):
    with (OUT/name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    check()
    records=csv_rows('p2_per_question.csv')
    assert len(records)==12788
    assert len({(r['key'],r['condition']) for r in records})==12788
    pairs=defaultdict(dict)
    for r in records:
        pairs[r['key']][r['condition']]=r
    assert len(pairs)==6394 and all(set(p)=={'Native','SVH'} for p in pairs.values())
    for pair in pairs.values():
        a,b=pair['Native'],pair['SVH']
        assert a['input_binding_sha256']==b['input_binding_sha256'] and a['gpu_uuid']==b['gpu_uuid']
    sums=[]
    for task in TASKS:
        for cond in ['Native','SVH']:
            group=[r for r in records if r['benchmark']==task and r['condition']==cond]
            assert len(group)==(2017 if task=='ScienceQA-IMG' else 4377)
            if task=='ScienceQA-IMG':
                values=[float(r['score']) for r in group]
            else:
                circular=defaultdict(list)
                for r in group:
                    circular[r['circular_group']].append(float(r['score']))
                assert len(circular)==1176
                values=[float(all(v)) for v in circular.values()]
            sums.append(dict(benchmark=task,condition=cond,n=len(values),correct=sum(values),accuracy=100*sum(values)/len(values)))
    summaries=csv_rows('p2_paired_comparisons.csv')
    for r in sums:
        expected=next(x for x in summaries if x['benchmark']==r['benchmark'] and x['subset']=='All' and x['metric']=='strict'
            and x['unit']==('question' if r['benchmark']=='ScienceQA-IMG' else 'Circular_group'))
        assert abs(r['accuracy']-float(expected['native_acc' if r['condition']=='Native' else 'svh_acc']))<1e-10
    gqa=csv_rows('gqa_per_question_types.csv')
    assert len(gqa)==len({r['question_id'] for r in gqa})==12578
    delta=100*sum(float(r['svh_score'])-float(r['native_score']) for r in gqa)/len(gqa)
    distribution=csv_rows('gqa_type_distribution.csv')
    for axis in ['structural','semantic','detailed']:
        assert sum(int(r['n']) for r in distribution if r['axis']==axis)==12578
        assert abs(sum(float(r['contribution_pp']) for r in distribution if r['axis']==axis)-delta)<1e-10
    release=json.loads((OUT/'gpu_release_audit.json').read_text())
    assert release['status']=='PASS_OWN_PROCESSES_RELEASED'
    assert all(not Path('/proc/'+str(r['pid'])).exists() for r in release['workers'])
    v=json.loads((OUT/'validation.json').read_text())
    assert v['status']=='COMPLETE_BOTH_TASKS_INTEGRITY'
    v['analysis_sources']={str(f):sha(f) for f in ROOT.glob('*.py')}
    v['final_independent_score_counts']=sums
    v['final_seal_after_all_workers_exited']=True
    write(OUT/'validation.json',v)
    write(OUT/'final_integrity.json',dict(status='PASS_COUNTS_BINDINGS_SCORES_AND_CONTRIBUTION_CONSERVATION',
        p2_predictions=12788,gqa_pairs=12578,p2_sums=sums,gqa_delta_pp=delta,old_strict_numeric_status='FAIL_RETAINED'))
    hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*'))
        if f.is_file() and '.git' not in f.parts and f.name!='artifact_sha256.json' and '__pycache__' not in f.parts}
    write(OUT/'artifact_sha256.json',hashes)
    assert all(sha(ROOT/path)==expected for path,expected in hashes.items())
    print('FINAL_INTEGRITY_AND_HASHES_PASS',len(hashes),json.dumps(sums),flush=True)


if __name__=='__main__':
    main()

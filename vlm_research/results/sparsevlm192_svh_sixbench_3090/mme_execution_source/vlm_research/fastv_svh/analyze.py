"""Exact checkpoint verification; never promote missing/partial scores to full."""
from collections import Counter
import csv
import json
import sqlite3

import numpy as np

from .common import *


def write_csv(path, values, fields=None):
    fields = fields or list(dict.fromkeys(k for row in values for k in row))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for value in values:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in value.items()})


def main():
    protocol = verify()
    samples = {s['key']:s for s in rows(OUT / 'sample_manifest.jsonl.gz')}
    assignment = read(OUT / 'partition.json')['assignment']
    records = {}
    for path in sorted(OUT.glob('worker*/results.sqlite')):
        db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True)
        assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        for key, raw, h in db.execute('SELECT key,payload,sha256 FROM results'):
            r = json.loads(raw)
            assert key not in records and key in samples
            assert digest(r) == h and r['sample_sha256'] == digest(samples[key])
            assert r['protocol_sha256'] == digest(protocol)
            assert r['rank'] == assignment[key] == int(path.parent.name.removeprefix('worker'))
            assert r['image_sha256'] == samples[key]['image_sha256']
            assert r['input_binding']['prompt'] == samples[key]['prompt']
            assert r['generation']['kv']['persistent_original_visual_tokens'] == 0
            records[key] = r
        db.close()
    output = [records[k] for k in sorted(records)]
    if output:
        from .inputs import score
        from ..information_horizon_alignment import score as scorer
        scorer.AUTHOR = Path('/data/lk/ATE-resources/Information-Horizon')
        for r in output:
            assert score(samples[r['key']], r['generation']['text']) == r['scoring']
        write_csv(OUT / 'per_question.csv', [dict(key=r['key'], benchmark=r['benchmark'],
            image_sha256=r['image_sha256'], gold=r['gold'], text=r['generation']['text'],
            token_ids=r['generation']['token_ids'], score=r['scoring']['score'],
            parsed=r['scoring']['parsed'], eos=r['generation']['eos'],
            empty=r['generation']['empty'], truncated=r['generation']['truncated'],
            selected_visual_indices_sha256=r['generation']['selector']['selected_visual_indices_sha256']) for r in output])
        with gzip.GzipFile(filename=str(OUT / 'per_question.jsonl.gz'), mode='wb', mtime=0) as f:
            for r in output:
                f.write((json.dumps(r, ensure_ascii=False)+'\n').encode())
    summary, details = [], []
    for benchmark, expected in COUNTS.items():
        subset = [r for r in output if r['benchmark']==benchmark]
        value, ci, metric = None, None, 'Accuracy (%)'
        complete = len(subset)==expected
        if benchmark=='TextVQA':
            metric='VQA Soft Accuracy (%)'
        elif benchmark=='MME':
            metric='MME Accuracy+Accuracy+ points'
        elif benchmark=='MMBench':
            metric='Circular Accuracy (%)'
        if complete:
            from ..information_horizon_alignment.analyze import mme_score, circular_rows, paired_boot
            ss = [samples[r['key']] for r in subset]
            values = [r['scoring']['score'] for r in subset]
            if benchmark=='MME':
                totals, tasks = mme_score(ss, values)
                value = totals['Total']
                details.extend(dict(benchmark=benchmark, **t) for t in tasks)
            else:
                if benchmark=='MMBench':
                    ss, values = circular_rows(ss, values)
                    assert len(values)==1176
                value=100*float(np.mean(values))
                boot=paired_boot(ss, np.zeros(len(values)), values, seed=20261110, draws=20000)
                ci=[boot['ci_low'], boot['ci_high']]
            if benchmark=='POPE':
                for strategy in ('random','popular','adversarial'):
                    group=[r for r in subset if samples[r['key']]['strategy']==strategy]
                    tp=sum(r['gold']=='yes' and r['scoring']['parsed']=='yes' for r in group)
                    tn=sum(r['gold']=='no' and r['scoring']['parsed']=='no' for r in group)
                    fp=sum(r['gold']=='no' and r['scoring']['parsed']=='yes' for r in group)
                    fn=sum(r['gold']=='yes' and r['scoring']['parsed']=='no' for r in group)
                    precision=tp/(tp+fp) if tp+fp else 0
                    recall=tp/(tp+fn) if tp+fn else 0
                    details.append(dict(benchmark=benchmark, strategy=strategy, questions=len(group),
                        accuracy=100*(tp+tn)/len(group), precision=precision, recall=recall,
                        f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                        tp=tp,tn=tn,fp=fp,fn=fn))
        summary.append(dict(benchmark=benchmark, completed=len(subset), expected=expected,
            status='COMPLETE' if complete else 'NOT_RUN' if not subset else 'PARTIAL',
            metric=metric, fusion=value, own_score_ci95=ci, native=None, fastv_standalone=None, svh=None,
            delta_svh=None, paired_ci95=None, paired_status='MISSING_BOUND_ROW_LEVEL_CONTROLS',
            eos=sum(r['generation']['eos'] for r in subset),
            truncated=sum(r['generation']['truncated'] for r in subset),
            empty=sum(r['generation']['empty'] for r in subset)))
    write_csv(OUT / 'score_summary.csv', summary)
    write_csv(OUT / 'task_details.csv', details, fields=list(dict.fromkeys(k for r in details for k in r)) or ['status'])
    gate = read(OUT / 'gpu_smoke.json') if (OUT / 'gpu_smoke.json').exists() else None
    complete = len(output)==35256 and gate is not None and gate['passed']
    atomic(OUT / 'completion.json', dict(status='COMPLETE_GENERATION_UNPAIRED_CONTROLS' if complete else 'PARTIAL_BLOCKED',
        completed=len(output), requested=35256, gpu_smoke='PASS' if gate else 'NOT_RUN',
        cpu_tests='PASS', benchmarks=summary, data_acceptance=read(OUT / 'data_acceptance.json')['status'],
        no_partial_score_reported_as_full=True, paired_controls_complete=False,
        no_efficiency_claim=True, no_github_push=True))
    print('VERIFIED_FUSION_PREDICTIONS', len(output), '/ 35256')


if __name__ == '__main__':
    main()

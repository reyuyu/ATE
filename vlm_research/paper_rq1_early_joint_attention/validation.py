import argparse
import json
from pathlib import Path
import time
import unittest
from benchmarks.dataset_registry import freeze_json,sha256
from .protocol import OUT,ROOT


def cpu():
    path=OUT/f'analysis_tests_{time.time_ns()}.txt'
    with path.open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_early_joint_analysis'))
    value=dict(passed=result.wasSuccessful(),tests=result.testsRun,log=str(path.relative_to(OUT)),log_sha256=sha256(path),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'test_early_joint_analysis.py',
            *sorted((ROOT/'paper_rq1_early_joint_attention').glob('*.py'))]})
    freeze_json(OUT/f'analysis_validation_attempt_{time.time_ns()}.json',value)
    if not value['passed']:raise ValueError('Analysis tests failed')
    if (OUT/'analysis_cpu_validation.json').exists():
        previous=json.loads((OUT/'analysis_cpu_validation.json').read_text())
        if previous['source_sha256']!=value['source_sha256']:
            (OUT/'analysis_cpu_validation.json').rename(OUT/f'analysis_acceptance_previous_{time.time_ns()}.json')
        else:
            print('Analysis CPU revalidated',result.testsRun,flush=True);return
    freeze_json(OUT/'analysis_cpu_validation.json',value)
    print('Analysis CPU passed',result.testsRun,flush=True)


def tables():
    import numpy as np
    from .analyze import csv_read
    per=csv_read(OUT/'per_question_generation.csv')
    checks=0
    for filename in ['joint_ablation_accuracy_summary.csv','taskwise_accuracy_summary.csv']:
        for r in csv_read(OUT/filename):
            if not int(r['n_questions']):continue
            selected=[s for s in per if s['condition']==r['condition'] and (r['task']=='Overall' or s['task']==r['task'])]
            if len(selected)!=int(r['n_questions']):raise ValueError('Summary denominator mismatch')
            for field in ['correct_to_wrong','wrong_to_correct','answer_changed','text_changed','first_token_changed']:
                expected=sum(s[field]=='True' or s[field]=='1' for s in selected)
                if expected!=int(r[field]):raise ValueError('Summary count mismatch: '+field)
                checks+=1
            for field,raw in [('accuracy_pp','score'),('full_accuracy_pp','full_score'),('delta_accuracy_pp','delta_score')]:
                expected=np.mean([float(s[raw]) for s in selected])*100
                if not np.isclose(expected,float(r[field]),atol=1e-12,rtol=0):raise ValueError('Summary score mismatch')
                checks+=1
            if int(r['wrong_to_correct'])-int(r['correct_to_wrong'])!=sum(int(s['delta_score']) for s in selected):raise ValueError('Flip accounting mismatch')
    freeze_json(OUT/'table_consistency_audit.json',dict(passed=True,checks=checks,rows=len(per),
        source_csv_sha256={p.name:sha256(p) for p in [OUT/'per_question_generation.csv',OUT/'joint_ablation_accuracy_summary.csv',OUT/'taskwise_accuracy_summary.csv']}))
    print('Table audit passed',checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cpu','tables']);a=p.parse_args()
    cpu() if a.stage=='cpu' else tables()

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.dataset_registry import freeze_json, sha256
from rq2_visual_full import analyze as a
from rq2_visual_full.run import split, PRIMARY, OUT, ROOT


class Tests(unittest.TestCase):
    def test_fixed_roles_strict_bound_and_four_comparison_family(self):
        tasks=list(a.previous.old.TASKS)
        discovery=[dict(question_id=str(i),image_sha256=str(i),task=tasks[i%6]) for i in range(240)]
        fresh=[dict(question_id='v'+str(i),image_sha256='v'+str(i),task=tasks[i]) for i in range(6)]
        shared=dict(question_id='shared',image_sha256='0',task=tasks[0])
        ss=discovery+fresh+[shared];flags=split(ss,discovery)
        records=[]
        for i,s in enumerate(ss):
            conditions={}
            for key in a.KEYS:
                conditions[key]=dict(generation=dict(score=int(i%7!=0),eos=True,truncated=False,text='x',length=2),
                                     metric=dict(top1_flip=False,logp_loss=0.,js=0.))
            records.append(conditions)
        calls=[]
        def fake(values,identities,families,**kwargs):
            calls.append((len(identities),families,kwargs))
            return [dict(mean=float(values[:,j].mean()),ci95_low=-1.,ci95_high=1.,
                family95_low=-2.,family95_high=2.,family=f,clusters=len(set(identities)),
                repetitions=kwargs['repetitions'],degenerate_bootstrap=False) for j,f in enumerate(families)]
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(a,'OUT',Path(temp)),patch.object(a,'read',return_value={'statistics':{'repetitions':1000}}),patch.object(a,'bootstrap',side_effect=fake):
                summary,paired,tables=a.summarize(ss,flags,records)
            from rq2_visual_full import presentation
            with patch.object(presentation,'OUT',Path(temp)),patch.object(presentation,'read',return_value={'code_commit':'synthetic_test'}):
                presentation.report(summary,paired,tables,[dict(started_epoch=0.,finished_epoch=60.,seconds=60.)])
                self.assertIn('未通过', (Path(temp)/'report_zh.md').read_text())
        self.assertEqual([n for n,_,_ in calls],[240,7,6,247])
        self.assertTrue(all(f==[1,4,4,4,4,1]+[1]*5 for _,f,_ in calls))
        main=next(r for r in paired if r['role']=='primary_NI')
        self.assertEqual(main['population'],PRIMARY)
        self.assertFalse(main['lower_strictly_above_minus1'])
        self.assertEqual(sum(r['confirmatory_NI_eligible'] for r in paired),1)
        self.assertEqual(sum(r['role']=='secondary_confirmatory' for r in paired),4)
        self.assertEqual(len(summary),20);self.assertEqual(len(tables),120)

    def test_clustered_bootstrap_preserves_question_weights(self):
        from paper_rq1_early_joint_attention.statistics import bootstrap
        result=bootstrap(np.array([[1.],[1.],[-1.]]),['same','same','other'],[4],repetitions=1000,seed=4)[0]
        self.assertAlmostEqual(result['mean'],1/3)
        self.assertEqual(result['clusters'],2)
        self.assertLessEqual(result['family95_low'],result['ci95_low'])
        self.assertGreaterEqual(result['family95_high'],result['ci95_high'])


if __name__=='__main__':
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(stream.getvalue(),flush=True)
    assert result.wasSuccessful()
    names=['rq2_visual_full/analyze.py','rq2_visual_full/presentation.py','test_rq2_visual_full_analysis.py']
    freeze_json(OUT/'postprocessing_cpu_validation.json',dict(passed=True,tests=result.testsRun,
        source={n:sha256(ROOT/n) for n in names},output=stream.getvalue()))

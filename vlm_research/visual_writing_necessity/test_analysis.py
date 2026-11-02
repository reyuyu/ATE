import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import analyze
from .analyze import paired_bootstrap


class ClusterStatisticsTests(unittest.TestCase):
    def test_constant_and_zero_columns(self):
        values=np.array([[0.,2.],[0.,2.],[0.,2.]])
        draws=paired_bootstrap(values,['a','a','b'],200,19)
        self.assertTrue(np.array_equal(draws[:,0],np.zeros(200)))
        self.assertTrue(np.array_equal(draws[:,1],np.full(200,2.)))

    def test_clusters_and_question_weighting(self):
        values=np.array([[0.],[0.],[9.]])
        draws=paired_bootstrap(values,['a','a','b'],2000,31)[:,0]
        self.assertEqual(set(draws),{0.,3.,9.})
        self.assertGreater(np.sum(draws==3.),800)
        self.assertLess(np.sum(draws==3.),1200)

    def test_pairing_deterministic_sign_and_no_clipping(self):
        values=np.array([[1.,-1.],[4.,-4.],[-3.,3.],[2.,-2.]])
        x=paired_bootstrap(values,['a','a','b','c'],200,37)
        y=paired_bootstrap(values,['a','a','b','c'],200,37)
        self.assertTrue(np.array_equal(x,y))
        self.assertTrue(np.array_equal(x[:,0],-x[:,1]))
        self.assertLess(x[:,0].min(),0.)

    def test_flat_curve_zero_K80_and_empty_negative_csv(self):
        base=dict(delta_gold_nll=0.,delta_accuracy_pp=0.,delta_gold_nll_ci_low=0.,delta_gold_nll_ci_high=0.,
            delta_gold_nll_corrected_low=0.,delta_gold_nll_corrected_high=0.,delta_accuracy_ci_low=0.,
            delta_accuracy_ci_high=0.,delta_accuracy_corrected_low=0.,delta_accuracy_corrected_high=0.)
        overall=[dict(base,condition=c,subset='overall') for c in analyze.CONDITIONS]
        tasks=[dict(r,subset=t) for t in analyze.TASKS for r in overall]
        with tempfile.TemporaryDirectory() as directory,patch.object(analyze,'OUT',Path(directory)):
            self.assertIsNone(analyze.figures(overall,tasks))
            self.assertEqual(len(list(Path(directory).glob('*.pdf'))),5)
            analyze.dump_csv('negative_cases.csv',[],fields=['question_id','delta_gold_nll'])
            self.assertEqual((Path(directory)/'negative_cases.csv').read_text(),'question_id,delta_gold_nll\n')


if __name__=='__main__':unittest.main()

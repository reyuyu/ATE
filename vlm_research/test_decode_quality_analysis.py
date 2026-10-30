"""Postprocessing tests, separate from the immutable executing collector."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from paper_rq1_decode_layerwise_quality.analyze import csv_write,csv_read,choose_cases
from paper_rq1_decode_layerwise_quality.statistics import cluster_deltas,bootstrap,interval_summary


class AnalysisTests(unittest.TestCase):
    def test_csv_roundtrip_quotes_newlines(self):
        rows=[{'text':'a, b\n"c"','score':0.5},{'text':'中文'}]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.csv';csv_write(p,rows);actual=csv_read(p)
        self.assertEqual(actual[0]['text'],rows[0]['text']);self.assertEqual(actual[1]['text'],'中文')
    def test_exact_duplicate_cluster_scaling(self):
        a=bootstrap([0,1],['a','b'],1000,seed=3)
        b=bootstrap([0,0,1,1],['a','a','b','b'],1000,seed=3)
        np.testing.assert_array_equal(a['ordinary'],b['ordinary'])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):cluster_deltas([float('nan')],['a'])
    def test_delta_mean_not_absolute_mean(self):
        value=bootstrap([-1,1],['a','b'],1000);self.assertEqual(value['mean'][0],0)
    def test_family_result_not_nominal(self):
        result=interval_summary(0,[-.01,.01],[-.03,.03],.02)
        self.assertTrue(result['ordinary_noninferior']);self.assertFalse(result['family_noninferior'])
    def test_case_selection_keeps_negative_tails(self):
        rows=[dict(dataset='coco_caption',text_changed='True',question_id=str(i),layer=str(i),delta_CIDEr=str(c),delta_SPICE=str(s),truncated='False')
            for i,(c,s) in enumerate([(-5,-.8),(5,.3),(0,0),(-1,-.2)])]
        cases=choose_cases(rows);self.assertEqual(len(cases),4)
        self.assertEqual(cases[0][1]['delta_CIDEr'],'-5');self.assertEqual(cases[1][1]['delta_CIDEr'],'5')


if __name__=='__main__':unittest.main()

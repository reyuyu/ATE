"""Small CPU checks for partial-status reporting and paired statistical units."""
import unittest

import numpy as np

from vlm_research.information_horizon_alignment.analyze import circular_rows, paired_boot, summarize
from vlm_research.information_horizon_alignment.common import ORDER


class PostprocessingTests(unittest.TestCase):
    def test_cluster_question_weighting(self):
        rows=[dict(image_sha256=h) for h in ('a','a','b')]
        result=paired_boot(rows,np.zeros(3),np.array([.1,.1,-.2]),draws=2000)
        self.assertAlmostEqual(result['delta_pp'],0)
        self.assertAlmostEqual(result['ci_low'],-20)
        self.assertAlmostEqual(result['ci_high'],10)

    def test_circular_requires_all_rotations(self):
        rows=[dict(circular_group=g,image_sha256=h) for g,h in ((1,'a'),(1,'a'),(2,'b'))]
        collapsed,scores=circular_rows(rows,[1,0,1])
        self.assertEqual(len(collapsed),2)
        self.assertEqual(scores.tolist(),[0,1])

    def test_partial_scores_are_not_full_scores(self):
        rows=[dict(key=b+':0',benchmark=b) for b in ORDER]
        rows.append(dict(key='TextVQA:1',benchmark='TextVQA'))
        records={('TextVQA:0','Author-Native'):dict(result=dict(score=1,eos=True,truncated=False,
                    empty=False,repeated_3gram=False))}
        coverage,summary,partial,_,paired=summarize(rows,records)
        result=next(r for r in summary if r['benchmark']=='TextVQA' and r['stage']=='A')
        self.assertEqual(result['status'],'PARTIAL')
        self.assertEqual(result['value'],'')
        self.assertEqual(partial[0]['completed'],1)
        self.assertTrue(all(p['status']=='NOT_RUN' for p in paired))


if __name__=='__main__':
    unittest.main()

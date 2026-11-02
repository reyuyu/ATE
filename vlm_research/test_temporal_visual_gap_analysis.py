import unittest
import numpy as np
from temporal_visual_gap_analysis import bounds, paired, first_difference, cluster_summary


class Tests(unittest.TestCase):
    def test_paired_constant_and_identity(self):
        indices=np.random.default_rng(7).integers(0,4,(1000,4))
        self.assertEqual(paired([1,2,3,4],[1,2,3,4],indices),(0.,0.,0.))
        self.assertEqual(paired([2,3,4,5],[1,2,3,4],indices),(1.,1.,1.))

    def test_cluster_not_token_resampling(self):
        rows=[dict(image_sha256='a',x=1.)]*10+[dict(image_sha256='b',x=0.)]
        result=cluster_summary(rows,'x',repetitions=1000)
        self.assertEqual(result['images'],2)
        self.assertEqual(result['tokens'],11)
        self.assertAlmostEqual(result['mean'],10/11)
        self.assertEqual((result['ci_low'],result['ci_high']),(0.,1.))

    def test_first_divergence_and_length(self):
        self.assertEqual(first_difference([1,2],[1,2]),-1)
        self.assertEqual(first_difference([1,2],[1,3]),1)
        self.assertEqual(first_difference([1,2],[1,2,3]),2)


if __name__=='__main__': unittest.main()

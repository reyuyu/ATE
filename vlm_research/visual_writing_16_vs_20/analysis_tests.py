"""Small CPU checks for the predeclared image-cluster statistical contract."""
import unittest
import numpy as np
from visual_writing_necessity.analyze import paired_bootstrap
from .common import noninferiority


class AnalysisTests(unittest.TestCase):
    def test_whole_cluster_question_weighted_sampling(self):
        draws=paired_bootstrap([0.,0.,1.],['image-A','image-A','image-B'],1000,17)[:,0]
        self.assertEqual(set(np.unique(draws)),{0.,1/3,1.})

    def test_shared_draws_preserve_three_paired_comparisons(self):
        scores=np.array([[1,0,1],[0,1,0],[1,1,0],[1,0,0]])
        differences=np.stack([scores[:,1]-scores[:,0],scores[:,2]-scores[:,0],scores[:,2]-scores[:,1]],axis=1)*100
        draws=paired_bootstrap(differences,['a','a','b','c'],1000,23)
        np.testing.assert_allclose(draws[:,1]-draws[:,0],draws[:,2],atol=3e-14)

    def test_family_interval_is_wider_and_threshold_is_strict(self):
        values=np.arange(-10,11,dtype=float)
        regular=np.percentile(values,[2.5,97.5]);family=np.percentile(values,[1.25,98.75])
        self.assertLess(family[0],regular[0]);self.assertGreater(family[1],regular[1])
        self.assertFalse(noninferiority(-1,1));self.assertTrue(noninferiority(-.99,1))
        self.assertFalse(noninferiority(-2,2));self.assertTrue(noninferiority(-1.99,2))


if __name__=='__main__':unittest.main(verbosity=2)

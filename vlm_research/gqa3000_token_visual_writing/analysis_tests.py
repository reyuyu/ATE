"""Statistical unit tests using synthetic outcomes, before result aggregation."""
import unittest
import numpy as np
from .analyze import cluster_bootstrap,interval,seed_events,DERIVED

class ClusterTests(unittest.TestCase):
    def test_whole_clusters_question_weighted_not_cluster_mean(self):
        # Image A has two correct questions, image B has one wrong question.
        # Bootstrap means are exactly 0,2/3,1; never 1/2.
        x=np.array([[1.,2.],[1.,2.],[0.,0.]])
        draws=cluster_bootstrap(x,['a','a','b'],repetitions=2000,seed=7)
        self.assertEqual(set(draws[:,0]),{0.,2/3,1.})
        self.assertTrue(np.array_equal(draws[:,1],2*draws[:,0]))
    def test_paired_constants_exact_and_reproducible(self):
        x=np.array([[0.,3.],[1.,4.],[2.,5.],[7.,10.]])
        a=cluster_bootstrap(x,['a','b','b','c'],repetitions=512,seed=19)
        b=cluster_bootstrap(x,['a','b','b','c'],repetitions=512,seed=19)
        self.assertTrue(np.array_equal(a,b));self.assertTrue(np.allclose(a[:,1]-a[:,0],3,rtol=0,atol=1e-14))
    def test_three_seeds_are_averaged_before_question_resampling(self):
        seeds=np.array([[1.,0.,0.],[0.,1.,1.]])
        means=seeds.mean(1,keepdims=True);a=cluster_bootstrap(means,['a','b'],repetitions=512,seed=4)
        self.assertTrue(set(a[:,0])<={1/3,1/2,2/3})
    def test_harm_events_pair_matching_seeds_not_products_of_means(self):
        from unittest.mock import patch
        cols={'l1':0,'l2':1,'r1':2,'r2':3}
        values=np.array([[1.,0.,1.,0.]])
        with patch.dict(DERIVED,{'left':['l1','l2'],'right':['r1','r2']}):
            harm,rescue=seed_events('left','right',[0],values,cols)
        self.assertEqual(harm,0);self.assertEqual(rescue,0)

if __name__=='__main__':unittest.main(verbosity=2)

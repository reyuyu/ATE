import unittest
import numpy as np
from .analyze import bootstrap,holm,cluster_arrays
class StatisticalTests(unittest.TestCase):
 def test_question_weighted_variable_cluster_sizes(self):
  x=np.array([[0.],[0.],[1.]]);images=['a','a','b'];actual=bootstrap(x,images,31,9);rng=np.random.default_rng(9);ids=rng.integers(0,2,size=(31,2));expected=np.array([sum(i==1 for i in row)/sum(2 if i==0 else 1 for i in row) for row in ids]);np.testing.assert_array_equal(actual[:,0],expected)
 def test_paired_correlation_preserved(self):
  a=np.array([[.2,.4],[.5,1.],[.1,.2]]);b=bootstrap(a,['a','b','b'],100,4);np.testing.assert_allclose(b[:,1],b[:,0]*2,rtol=0,atol=0)
 def test_known_holm_family(self):np.testing.assert_allclose(holm(np.array([.001,.04,.02])),[.003,.04,.04])
 def test_sha_cluster_not_question_sampling(self):
  s,n=cluster_arrays(np.array([[1.],[0.],[.6]]),['a','a','b']);np.testing.assert_array_equal(n,[2,1]);np.testing.assert_allclose(s[:,0],[1.,.6])
if __name__=='__main__':unittest.main()

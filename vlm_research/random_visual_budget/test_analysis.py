import unittest
import numpy as np

from .analyze import cluster_intervals, bootstrap_weights, first_difference, condition_names, flags


class AnalysisTests(unittest.TestCase):
    def test_shared_endpoints(self):
        self.assertEqual(condition_names(0),['Only Text']*3)
        self.assertEqual(condition_names(1),['Full']*3)

    def test_zero_paired_change(self):
        actual=cluster_intervals(np.zeros(5),['a','a','b','c','c'])
        np.testing.assert_array_equal(actual,np.zeros((2,1)))

    def test_duplicate_questions_resample_as_image_cluster(self):
        assignment,weights,denom=bootstrap_weights(('a','a','b'))
        np.testing.assert_array_equal(assignment,[0,0,1])
        np.testing.assert_array_equal(denom,2*weights[:,0]+weights[:,1])
        self.assertEqual(weights.shape,(20000,2))
        actual=cluster_intervals(np.ones(3)*.125,['a','a','b'])
        np.testing.assert_array_equal(actual,np.full((2,1),.125))

    def test_seed_mean_not_independent_observations(self):
        correct=np.array([[1,0,0],[0,1,1],[1,1,0]],float)
        expected=correct.mean(1)
        interval=cluster_intervals(expected,['a','a','b'])
        self.assertEqual(interval.shape,(2,1))
        self.assertAlmostEqual(expected.mean(),5/9)

    def test_sequence_and_flags(self):
        self.assertEqual(first_difference([1,2],[1,2]),-1)
        self.assertEqual(first_difference([1,2],[1,3]),1)
        self.assertEqual(first_difference([1,2],[1,2,3]),2)
        self.assertTrue(flags('USER: hi',[1,1,1,1])['contains_role_markup'])
        self.assertEqual(flags('a',[1,1,1,1])['repeated_token_trigrams'],1)


if __name__=='__main__':unittest.main()

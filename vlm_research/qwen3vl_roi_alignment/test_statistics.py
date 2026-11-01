import unittest
import numpy as np

from .analyze import bootstrap,rho,boot_values


class StatisticsTests(unittest.TestCase):
    def test_spearman_and_undefined_constants(self):
        self.assertAlmostEqual(float(rho(np.array([1,2,2,4]),np.array([1,2,2,4]))),1)
        self.assertAlmostEqual(float(rho(np.array([1,2,3]),np.array([3,2,1]))),-1)
        self.assertTrue(np.isnan(rho(np.ones(3),np.arange(3))))

    def test_duplicate_image_questions_always_resampled_together(self):
        samples=[{'image_sha256':k} for k in ('same','same','other')]
        weights,indices,clusters=bootstrap(samples,100,123)
        self.assertEqual(clusters,2)
        np.testing.assert_array_equal(weights[:,0],weights[:,1])
        np.testing.assert_allclose(weights.sum(1),1)
        for row in indices:
            self.assertEqual(int((row==0).sum()),int((row==1).sum()))
        values=boot_values(np.array([1.,2.,4.]),indices)
        np.testing.assert_allclose(np.nanmean(values,axis=1),weights@np.array([1.,2.,4.]))


if __name__=='__main__':unittest.main()

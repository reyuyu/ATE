import unittest
import numpy as np
from .analyze import boot_mean


class TestStatistics(unittest.TestCase):
    def test_whole_images_and_question_weights(self):
        samples=[{'image_sha256':'a'},{'image_sha256':'a'},{'image_sha256':'b'}]
        mean,low,high,draws,clusters=boot_mean(np.array([0.,0.,10.]),samples,1000)
        self.assertEqual(clusters,2)
        self.assertAlmostEqual(float(mean[0]),10/3)
        self.assertEqual(set(np.unique(draws)),{0.,10/3,10.})

    def test_paired_contrast_and_constant(self):
        samples=[{'image_sha256':str(i)} for i in range(10)]
        x=np.arange(10.)
        matrix=np.stack([x,x+2,np.full(10,2)],axis=1)
        mean,low,high,draws,_=boot_mean(matrix,samples,1000)
        self.assertTrue(np.allclose(draws[:,1]-draws[:,0],2))
        self.assertEqual(low[2],2.);self.assertEqual(high[2],2.)


if __name__=='__main__':unittest.main()

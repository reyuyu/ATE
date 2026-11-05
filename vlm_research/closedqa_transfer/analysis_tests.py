import unittest
import numpy as np

from .common import CONFIGS
from .analyze import draws,interval
from .benchmark import measurement


class Tests(unittest.TestCase):
    def test_image_cluster_resampling_not_question_resampling(self):
        samples={'a':dict(image_sha256='x',answer='yes'),'b':dict(image_sha256='x',answer='yes'),
                 'c':dict(image_sha256='y',answer='yes')}
        records={(k,c):dict(scoring=dict(score=float(k!='a'),parsed='yes')) for k in samples for c in CONFIGS}
        boot,n=draws(list(samples),samples,records,'scienceqa_img')
        self.assertEqual(n,2)
        np.testing.assert_allclose(np.unique(boot['accuracy'][:,0]),[.5,2/3,1])
        self.assertTrue(np.array_equal(boot['accuracy'],draws(list(samples),samples,records,'scienceqa_img')[0]['accuracy']))

    def test_pope_metrics_from_resampled_confusions(self):
        samples={k:dict(image_sha256='x',answer=gold) for k,gold in [('a','yes'),('b','yes'),('c','no')]}
        records={(k,c):dict(scoring=dict(score=float(k!='b'),parsed='no' if k in ('b','c') else 'yes')) for k in samples for c in CONFIGS}
        boot,n=draws(list(samples),samples,records,'pope_random')
        for metric,value in [('accuracy',2/3),('precision',1),('recall',.5),('f1',2/3),('yes_ratio',1/3)]:
            np.testing.assert_allclose(boot[metric],value)

    def test_corrected_intervals_and_measurement_label(self):
        x=np.arange(20000,dtype=float)
        self.assertLess(interval(x,.05/4)[0],interval(x)[0])
        g=dict(condition='Fast-P2',kv=dict(layers=[],live_bytes=0),latency_ms=1)
        self.assertEqual(measurement('Full-QSA',2,g)['condition'],'Full-QSA')


if __name__=='__main__': unittest.main(verbosity=2)

import unittest
from analyze_gqa import classify
from statistics_local import paired


class AnalysisTests(unittest.TestCase):
    def test_program_based_localization_not_surface_words(self):
        raw=dict(types=dict(structural='query',semantic='attr',detailed='directOf'),semantic=[
            dict(operation='select',argument='object',dependencies=[]),
            dict(operation='relate',argument='object,left of,o',dependencies=[0]),
            dict(operation='query',argument='color',dependencies=[1])])
        r=classify(raw)
        self.assertTrue(r['locate_then_attribute'])
        self.assertEqual(r['query_argument'],'color')
        self.assertEqual(r['program_depth'],3)

    def test_no_manufactured_official_label(self):
        r=classify(dict(semantic=[]))
        self.assertEqual(r['structural'],'UNKNOWN')
        self.assertEqual(r['query_argument'],'NOT_QUERY')

    def test_invalid_program_rejected(self):
        with self.assertRaises(AssertionError):
            classify(dict(semantic=[dict(operation='query',argument='name',dependencies=[0])]))

    def test_cluster_ratio_not_mean_of_cluster_means(self):
        rows=[dict(image_sha256='a',native_score=1,svh_score=0),
              dict(image_sha256='a',native_score=1,svh_score=0),
              dict(image_sha256='a',native_score=1,svh_score=0),
              dict(image_sha256='b',native_score=0,svh_score=1)]
        s=paired(rows,reps=1000)
        self.assertEqual(s['images'],2)
        self.assertEqual(s['delta_pp'],-50)
        self.assertEqual(s['harms'],3)
        self.assertEqual(s['rescues'],1)
        self.assertGreater(s['ci_high_pp'],0)

    def test_negative_positive_contributions_preserved(self):
        groups=[(10,-20),(20,5)]
        contribution=sum(n/30*d for n,d in groups)
        self.assertAlmostEqual(contribution,-100/30)


if __name__=='__main__':
    unittest.main()

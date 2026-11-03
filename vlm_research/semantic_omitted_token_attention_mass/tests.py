import unittest
from .common import *
class Metrics(unittest.TestCase):
    def test_uniform(self):
        p,x,e=normalized(np.array([.5]),np.array([1152]));self.assertEqual(p[0],.5);self.assertEqual(x[0],0);self.assertEqual(e[0],1)
    def test_excess_sign(self):
        p,x,e=normalized(np.array([.25,.75]),np.array([1152,1152]));self.assertLess(x[0],0);self.assertGreater(x[1],0);self.assertAlmostEqual(e[0],1/3);self.assertEqual(e[1],3)
    def test_zero_denominator(self):
        for m,n in [(1.,1152),(.5,576),(1.0001,1152)]:self.assertTrue(np.isnan(normalized([m],[n])[2][0]))
    def test_cases(self):
        v=cases();self.assertEqual(len(v),16);self.assertEqual(sum(x['kind']=='harm' for x in v),12);self.assertEqual(sum(x['kind']=='rescue' for x in v),4);self.assertEqual([sum(x['rank']==i for x in v) for i in [0,1]],[8,8])
    def test_partition(self):
        for v in cases():
            R=v['spec']['relay_positions'];m=v['sample']['input_mapping'];self.assertEqual(R,list(range(m['visual_positions'][-1]+1,m['p_minus_one'])));self.assertEqual(flatten(v['groups']),R);self.assertTrue(all(g['scope'] in ['Question','Instruction','Assistant/template'] for g in v['groups']));self.assertEqual([t['position'] for t in v['tokens']],R)
    def test_rank(self):
        rows=[dict(mass_mean=.5,first_position=1,group_id=1,head_max=.9),dict(mass_mean=.6,first_position=2,group_id=2,head_max=.7)];self.assertEqual(sorted(rows,key=rankkey)[0]['group_id'],2)
    def test_layer_bands(self):
        self.assertEqual(sorted(sum([BANDS[b] for b in ['L0-3','L4-7','L8-11','L12-15','L16-19','L20-30','L31']],[])),list(range(32)))
if __name__=='__main__':unittest.main()

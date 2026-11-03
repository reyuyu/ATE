import unittest
from .common import *
class Audit(unittest.TestCase):
    def test_cpu(self):
        v=read(OUT/'cpu_validation.json');self.assertTrue(v['passed']);self.assertTrue(v['image_disjoint']);self.assertFalse(v['cuda_initialized']);self.assertEqual(v['pilot120'],120)
    def test_sets(self):
        for q,v in plan()['cases'].items():
            S,N,D=map(set,[v['S'],v['N'],v['D']]);w=v['writers'];self.assertEqual(set(w['Noun+Article']),N|D);self.assertEqual(set(w['Semantic+Noun+Article']),S|N|D)
            for n,x in w.items():
                if x is None:continue
                self.assertTrue(intact(x,v['spec']['groups']));self.assertTrue(set(x)<=set(v['spec']['relay_positions']))
                if 'Swap' in n:self.assertEqual(len(x),len(S))
            for n,c in v['controls'].items():
                b=S if n.startswith('S+') else S|N if n.startswith('SN+') else N
                if c['feasible']:self.assertEqual(len(set(w[n])-b),len(D-b));self.assertEqual(len(c['added']),c['added_tokens'])
    def test_disjoint(self):
        p=plan();self.assertEqual(len(p['cases']),132);self.assertFalse(set(p['pilot_ids'])&set(p['historical_ids']))
    def test_scorer(self):
        from paper_rq1.scorers import gqa_prediction
        self.assertEqual(gqa_prediction('White'),'white');self.assertNotEqual(gqa_prediction('Panda'),'panda bears')
    def test_parser_binding(self):
        e=read(OUT/'parser_environment.json');self.assertEqual(e['spacy'],'3.8.7');self.assertEqual(e['model'],'3.8.0');self.assertTrue(e['require_cpu'])
if __name__=='__main__':unittest.main()

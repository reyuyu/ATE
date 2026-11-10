import unittest
from types import SimpleNamespace

import torch

from .run import BASE,ROOT,GATE,KEY,samples,sha
from ..qwen3vl_svh_l28_repair.run import compare_records


class Tests(unittest.TestCase):
    def test_exact_scope(self):
        rows=samples()
        self.assertEqual(len(rows),14)
        self.assertEqual(rows[0]['key'],KEY)
        self.assertEqual(len({r['key'] for r in rows}),14)
        self.assertEqual(GATE['mean_abs'],.02)
        self.assertFalse(torch.cuda.is_initialized())

    def test_prefix_projection_math(self):
        x=torch.randn(1,13,16,dtype=torch.float64)
        w=torch.randn(8,16,dtype=torch.float64)
        y=torch.nn.functional.linear(x,w)
        compact=torch.nn.functional.linear(x[:,:-1],w)
        self.assertTrue(torch.allclose(y[:,:-1],compact,atol=1e-12,rtol=1e-12))

    def test_earliest_difference(self):
        a,b={},{}
        for layer,name in [(0,'layer_output'),(1,'layer_input'),(1,'input_norm'),(1,'q_input'),(1,'q_proj')]:
            value=torch.zeros(1,2,4)
            a[(0,layer,name)]=dict(value=value.clone(),positions=[0,1],axis=1,keys=None)
            b[(0,layer,name)]=dict(value=value.clone(),positions=[0,1],axis=1,keys=None)
        b[(0,1,'q_proj')]['value'][0,0,0]=.00390625
        rows=compare_records(SimpleNamespace(records=a),SimpleNamespace(records=b))
        first=next(r for r in rows if not r['equal'])
        self.assertEqual((first['layer'],first['operator']),(1,'q_proj'))

    def test_historical_engines_unchanged(self):
        for path in ['qwen3vl_svh_qk_repair/engine.py','qwen3vl_svh_numerical_repair/engine.py',
                     'qwen3vl_svh_l28_repair/engine.py','qwen3vl_c3v2_physical/engine.py']:
            self.assertEqual(sha(ROOT/'vlm_research'/path),sha(BASE/'vlm_research'/path))


if __name__=='__main__':
    unittest.main()

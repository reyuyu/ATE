import unittest

import torch

from .run import GATE, TARGETS, samples
from ..qwen3vl_svh_l28_repair.engine import restore_query_rows


class Tests(unittest.TestCase):
    def test_scope(self):
        targets, regressions = samples()
        self.assertEqual([r['key'] for r in targets], TARGETS)
        self.assertEqual(len(regressions), 7)
        self.assertEqual(GATE['mean_abs'], .02)
        self.assertFalse(torch.cuda.is_initialized())

    def test_qk_zero_rows_preserve_real_math(self):
        q = torch.randn(1, 2, 3, 4, dtype=torch.float64)
        kt = torch.randn(1, 2, 4, 6, dtype=torch.float64)
        before = q.clone()
        expanded, ix = restore_query_rows(q, [0, 3, 5], 6)
        self.assertTrue(torch.equal(q, before))
        self.assertTrue(torch.allclose(q @ kt, (expanded @ kt).index_select(-2, ix), atol=1e-12, rtol=1e-12))

    def test_mask_canonicalization(self):
        mask = torch.tensor([[0., -torch.inf], [0., 0.]])
        x = torch.tensor([[1., 2.], [3., 4.]])
        y = torch.tensor([[1., 99.], [3., 4.]])
        self.assertTrue(torch.equal(torch.where(mask > -1e20, x, 0.), torch.where(mask > -1e20, y, 0.)))


if __name__ == '__main__':
    unittest.main()

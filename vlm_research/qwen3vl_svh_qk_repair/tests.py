import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .run import GATE, TARGETS, samples
from ..qwen3vl_svh_l28_repair.engine import restore_query_rows
from .engine import QKAlignedExecution, AVAlignedExecution, qm


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

    def aligned_context(self, fail=False):
        execution = object.__new__(QKAlignedExecution)
        execution.physical, execution.step, execution.n = True, 0, 6
        execution.text = torch.tensor([0,3,5])
        execution.qk_row_alignment = []
        @contextmanager
        def parent_context(obj):
            yield obj
        def eager(module,q,k,v,attention_mask=None,**kwargs):
            weights = torch.matmul(q,k.transpose(-2,-1))
            if fail:
                raise RuntimeError('injected failure')
            return torch.matmul(weights,v), weights
        return execution,parent_context,eager

    def test_actual_qk_wrapper_and_cleanup(self):
        execution,parent,eager = self.aligned_context()
        q = torch.randn(1,2,3,4,dtype=torch.bfloat16)
        k = torch.randn(1,2,6,4,dtype=torch.bfloat16)
        v = torch.randn_like(k)
        before = [x.clone() for x in (q,k,v)]
        native_matmul = torch.matmul
        with patch.object(AVAlignedExecution,'context',parent),patch.object(qm,'eager_attention_forward',eager):
            with execution.context():
                out, weights = qm.eager_attention_forward(SimpleNamespace(layer_idx=22),q,k,v)
            self.assertIs(qm.eager_attention_forward,eager)
        self.assertIs(torch.matmul,native_matmul)
        self.assertEqual(out.shape,(1,2,3,4))
        self.assertEqual(weights.shape,(1,2,3,6))
        self.assertEqual(len(execution.qk_row_alignment),1)
        self.assertEqual(execution.qk_row_alignment[0]['extra_QK_flops'],2*2*3*4*6)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip((q,k,v),before)))

    def test_exception_cleanup(self):
        execution,parent,eager = self.aligned_context(fail=True)
        native_matmul = torch.matmul
        q = torch.zeros(1,2,3,4,dtype=torch.bfloat16)
        k = torch.zeros(1,2,6,4,dtype=torch.bfloat16)
        with patch.object(AVAlignedExecution,'context',parent),patch.object(qm,'eager_attention_forward',eager):
            with self.assertRaisesRegex(RuntimeError,'injected failure'):
                with execution.context():
                    qm.eager_attention_forward(SimpleNamespace(layer_idx=22),q,k,k)
            self.assertIs(qm.eager_attention_forward,eager)
        self.assertIs(torch.matmul,native_matmul)


if __name__ == '__main__':
    unittest.main()

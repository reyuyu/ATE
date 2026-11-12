import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from .runtime import row_attention,vision_runtime


class Tests(unittest.TestCase):
    def test_cpu_parity(self):
        m=SimpleNamespace(num_key_value_groups=1,training=False)
        q=torch.randn(1,2,17,8,dtype=torch.bfloat16)
        k=torch.randn_like(q)
        v=torch.randn_like(q)
        before=[x.clone() for x in (q,k,v)]
        a=qm.eager_attention_forward(m,q,k,v,None,scaling=.2)
        b=row_attention(m,q,k,v,None,scaling=.2,rows=3)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(before,(q,k,v))))
        self.assertFalse(torch.cuda.is_initialized())

    def test_exception_cleanup(self):
        before=qm.eager_attention_forward
        with self.assertRaises(RuntimeError):
            with vision_runtime():
                raise RuntimeError('injected')
        self.assertIs(qm.eager_attention_forward,before)

    def test_text_untouched(self):
        sentinel=object()
        with patch.object(qm,'eager_attention_forward',return_value=sentinel) as original:
            with vision_runtime():
                self.assertIs(qm.eager_attention_forward(SimpleNamespace(),None,None,None,None),sentinel)
            original.assert_called_once()


if __name__=='__main__':
    unittest.main()

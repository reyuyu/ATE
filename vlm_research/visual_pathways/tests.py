import unittest
import torch
from .common import assign
from .engine import masked, SourceMask, PrefillIntervention, metrics


class Test(unittest.TestCase):
    def test_mask_scope_and_causal(self):
        original = torch.zeros(1,1,9,9).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),torch.finfo(torch.float32).min)
        changed,count = masked(original,[1,6,8],[2,3,4])
        assert count == 6
        expected = original.clone()
        expected[:,:,6,[2,3,4]] = -torch.inf
        expected[:,:,8,[2,3,4]] = -torch.inf
        self.assertTrue(torch.equal(changed,expected))
        self.assertTrue(torch.equal(original[0,0,6],torch.tensor([0.,0,0,0,0,0,0,torch.finfo(torch.float32).min,torch.finfo(torch.float32).min])))

    def test_cluster_shards(self):
        rows=[dict(image_sha256=str(i//2),id=i) for i in range(18)]
        bins=assign(rows,3)
        self.assertEqual([len(b) for b in bins],[6]*3)
        sets=[{r['image_sha256'] for r in b} for b in bins]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_metrics(self):
        full=torch.tensor([1.,3.,2.]);changed=torch.tensor([3.,1.,2.])
        m=metrics(full,changed)
        self.assertAlmostEqual(m['signed_logp_loss'],2.)
        self.assertEqual(m['top1_flip'],1)
        self.assertEqual(metrics(full,full)['signed_logp_loss'],0.)

    def test_exception_cleanup(self):
        module=torch.nn.Linear(2,2)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with SourceMask(module,[2],[0]):
                raise RuntimeError('fixture')
        self.assertFalse(module._forward_hooks or module._forward_pre_hooks)

    def test_only_p1_output_zero(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__();self.self_attn=torch.nn.Module();self.self_attn.o_proj=torch.nn.Identity()
        layers=[Layer() for _ in range(8)]
        value=torch.ones(1,9,3)
        with PrefillIntervention(layers,9,[2,3],skip=True) as hook:
            results=[l.self_attn.o_proj(value) for l in layers]
        self.assertEqual(hook.events,[2,3,4,5])
        for i,out in enumerate(results):
            self.assertTrue(torch.equal(out[:,:-1],value[:,:-1]))
            self.assertEqual(float(out[:,-1].sum()),0. if i in (2,3,4,5) else 3.)
        self.assertFalse(any(l.self_attn.o_proj._forward_hooks for l in layers))


if __name__=='__main__':
    unittest.main()

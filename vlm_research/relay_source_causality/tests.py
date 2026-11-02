import unittest
import torch
from .engine import source_edges,apply_mask

class SourceTests(unittest.TestCase):
    def setUp(self):
        self.layout=dict(prompt_length=9,I=[2,3,4],R=[5,6,7],A=[8],other=[0,1]);self.bos=0
        self.mask=torch.zeros(1,1,9,9).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min).half()
    def test_image_scope(self):
        changed,blocked=apply_mask(self.mask,self.layout,0,'R-Image-OFF')
        self.assertEqual(int(blocked.sum()),9)
        for q in range(9):
            for k in range(9):self.assertEqual(bool(blocked[q,k]),q in [5,6,7] and k in [2,3,4])
        self.assertTrue(torch.equal(changed[0,0][~blocked],self.mask[0,0][~blocked]))
    def test_text_partition_keeps_bos_self_image(self):
        _,blocked=apply_mask(self.mask,self.layout,0,'R-Text-OFF')
        self.assertEqual([torch.where(blocked[q])[0].tolist() for q in [5,6,7]],[[1],[1,5],[1,5,6]])
        self.assertFalse(blocked[:,0].any());self.assertFalse(blocked.diagonal().any());self.assertFalse(blocked[:,2:5].any())
    def test_real_softmax_zero_normalization(self):
        for c in ['R-Image-OFF','R-Text-OFF']:
            changed,blocked=apply_mask(self.mask,self.layout,0,c)
            probabilities=torch.softmax(torch.randn(1,4,9,9)+changed.float(),dim=-1)
            self.assertEqual(float(probabilities[0,:,blocked].max()),0.)
            self.assertTrue(torch.allclose(probabilities.sum(-1),torch.ones(1,4,9)))
    def test_noop_exact_and_no_future(self):
        for c in ['Full','R-Image-OFF','R-Text-OFF']:
            changed,blocked=apply_mask(self.mask,self.layout,0,c,noop=True)
            self.assertTrue(torch.equal(changed,self.mask));self.assertFalse(blocked.triu(0).any());self.assertFalse(blocked[8].any())
    def test_reject_noncausal_and_unknown(self):
        with self.assertRaises(AssertionError):apply_mask(torch.zeros_like(self.mask),self.layout,0,'R-Image-OFF')
        with self.assertRaises(ValueError):source_edges(self.layout,0,'unknown','cpu')

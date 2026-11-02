import unittest,torch
from .common import CONDITIONS,writer_layers
from .engine import blocked_edges,apply_mask

class WindowTests(unittest.TestCase):
    def setUp(self):
        self.layout=dict(prompt_length=9,I=[2,3,4],R=[5,6,7],A=[8],other=[0,1])
        self.mask=torch.zeros(1,1,9,9,dtype=torch.float16).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)
    def test_18_conditions_partition(self):
        self.assertEqual(len(set(CONDITIONS)),18)
        for k in range(8):
            self.assertEqual(set(writer_layers(f'Only-W{k}'))|set(writer_layers(f'Drop-W{k}')),set(range(32)))
            self.assertFalse(set(writer_layers(f'Only-W{k}'))&set(writer_layers(f'Drop-W{k}')))
    def test_every_condition_layer_scope(self):
        for c in CONDITIONS:
            for layer in range(32):
                b=blocked_edges(self.layout,layer,c,0,'cpu')
                for q in range(9):
                    for k in range(9):
                        self.assertEqual(bool(b[q,k]),k in [2,3,4] and (q==8 or q in [5,6,7] and layer not in writer_layers(c)))
    def test_decode_all_layers_image_only(self):
        for c in CONDITIONS:
            for p in [9,10,39]:
                for layer in range(32):
                    old=torch.zeros(1,1,1,p+1,dtype=torch.float16);a,b=apply_mask(old,self.layout,layer,c,p)
                    self.assertEqual(torch.where(b[0])[0].tolist(),[2,3,4])
                    self.assertTrue(torch.equal(a[0,0][~b],old[0,0][~b]))
    def test_probability_zero_and_rows(self):
        for c in CONDITIONS:
            a,b=apply_mask(self.mask,self.layout,8,c,0)
            w=torch.softmax(torch.randn(1,4,9,9)+a.float(),-1)
            self.assertEqual(float(w[0,:,b].max()),0.)
            self.assertTrue(torch.allclose(w.sum(-1),torch.ones(1,4,9)))
            self.assertTrue(torch.equal(a[0,0][~b],self.mask[0,0][~b]))
    def test_bos_self_image_queries_native(self):
        for c in CONDITIONS:
            a,b=apply_mask(self.mask,self.layout,0,c,0)
            self.assertFalse(b[:,0].any());self.assertFalse(b.diagonal().any());self.assertFalse(b[[0,1,2,3,4]].any())
            self.assertFalse(b.triu().any())
    def test_noop_and_native_strict(self):
        for c in CONDITIONS+['Native']:
            a,_=apply_mask(self.mask,self.layout,0,c,0,noop=True)
            self.assertTrue(torch.equal(a,self.mask))
        self.assertFalse(blocked_edges(self.layout,0,'Native',0,'cpu').any())
    def test_reject_unknown_or_invisible(self):
        with self.assertRaises(ValueError):writer_layers('Only-W8')
        with self.assertRaises(AssertionError):apply_mask(self.mask.masked_fill(self.mask==0,-torch.inf),self.layout,0,'No-Writer',0)

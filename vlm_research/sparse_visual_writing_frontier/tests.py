import unittest,torch
from .common import NEW,BASELINES,KEEP,SAME_COUNT,layer_mode
from .engine import blocked_edges,apply_mask

class FrontierTests(unittest.TestCase):
    def setUp(self):
        self.layout=dict(prompt_length=9,I=[2,3,4],R=[5,6,7],A=[8],other=[0,1])
        self.mask=torch.zeros(1,1,9,9,dtype=torch.float16).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)
    def test_frozen_sets(self):
        self.assertEqual([len(KEEP[c]) for c in NEW],[16,12,12,8,8,8]);self.assertEqual(len(NEW)*240,1440)
        for c in NEW:self.assertTrue(set(range(8,12))<=set(KEEP[c]))
        self.assertEqual(KEEP['Keep-0-3+8-11'],[0,1,2,3,8,9,10,11])
    def test_all_masks_exact_scope(self):
        for c in NEW+BASELINES:
            for layer in range(32):
                b=blocked_edges(self.layout,layer,c,0,'cpu')
                for q in range(9):
                    for k in range(9):self.assertEqual(bool(b[q,k]),k in [2,3,4] and (q==8 or q in [5,6,7] and layer not in KEEP[c]))
    def test_historical_implementation_identical(self):
        from visual_write_window.engine import apply_mask as old_apply
        for c in NEW:
            for k in range(32):
                a,b=apply_mask(self.mask,self.layout,k,c,0);x,y=old_apply(self.mask,self.layout,k,layer_mode(c,k),0)
                self.assertTrue(torch.equal(a,x) and torch.equal(b,y))
        for c in BASELINES:
            for k in range(32):
                a,b=apply_mask(self.mask,self.layout,k,c,0);x,y=old_apply(self.mask,self.layout,k,c,0)
                self.assertTrue(torch.equal(a,x) and torch.equal(b,y))
    def test_decode_uniform_image_mask(self):
        for c in NEW:
            for past in [9,10,39]:
                for k in range(32):
                    a,b=apply_mask(torch.zeros(1,1,1,past+1,dtype=torch.float16),self.layout,k,c,past)
                    self.assertEqual(torch.where(b[0])[0].tolist(),[2,3,4]);self.assertEqual(int(torch.isneginf(a).sum()),3)
    def test_actual_softmax_probability_zero(self):
        for c in NEW:
            for k in [0,4,8,12,31]:
                a,b=apply_mask(self.mask,self.layout,k,c,0);w=torch.softmax(torch.randn(1,4,9,9)+a.float(),-1)
                self.assertEqual(float(w[0,:,b].max()),0.);self.assertTrue(torch.allclose(w.sum(-1),torch.ones(1,4,9)))
    def test_bos_self_image_queries_and_noop(self):
        for c in NEW:
            a,b=apply_mask(self.mask,self.layout,0,c,0)
            self.assertFalse(b[:,0].any() or b.diagonal().any() or b[[0,1,2,3,4]].any())
            self.assertTrue(torch.equal(a[0,0][~b],self.mask[0,0][~b]))
            n,_=apply_mask(self.mask,self.layout,0,c,0,True);self.assertTrue(torch.equal(n,self.mask))
    def test_same_count_comparisons_complete(self):
        self.assertEqual(len(SAME_COUNT),4)
        for a,b in SAME_COUNT:self.assertEqual(len(KEEP[a]),len(KEEP[b]));self.assertNotEqual(KEEP[a],KEEP[b])
    def test_reject_unknown_condition(self):
        with self.assertRaises(KeyError):layer_mode('Keep-0-10',0)
    def test_quality_point_boundaries(self):
        from .analyze import quality_flags
        self.assertEqual(quality_flags(200,202),dict(point_minus1_candidate=True,point_minus2_candidate=True))
        self.assertEqual(quality_flags(198,202),dict(point_minus1_candidate=False,point_minus2_candidate=True))
        self.assertEqual(quality_flags(197,202),dict(point_minus1_candidate=False,point_minus2_candidate=False))
    def test_pareto_ties_and_dominance(self):
        from .analyze import frontier_flags
        rows=[dict(retained_layers=n,correct=c) for n,c in [(0,153),(4,188),(8,187),(8,200),(8,200),(12,199),(16,205),(32,202)]]
        self.assertEqual(frontier_flags(rows),[True,True,False,True,True,False,True,False])

import copy,unittest,torch
from .common import NEW,BASELINES,KEEP,MATCH,selected,writers
from .engine import blocked_edges,apply_mask

class JointTests(unittest.TestCase):
    def setUp(self):
        self.layout=dict(prompt_length=9,I=[2,3,4],R=[5,6,7],A=[8],other=[0,1])
        self.spec=dict(relay_positions=[5,6,7],selection=dict(writers={'Semantic-k':[6]},k=1))
        self.mask=torch.zeros(1,1,9,9,dtype=torch.float16).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)
    def test_exact_frozen_layer_sets(self):
        self.assertEqual([KEEP[c] for c in NEW],[list(range(16)),list(range(12)),list(range(4,12))])
        self.assertEqual(len(NEW)*240,720)
        for c in NEW:self.assertEqual(KEEP[c],KEEP[MATCH[c]])
    def test_joint_direction_and_scope_all_layers(self):
        for c in NEW:
            for layer in range(32):
                b=blocked_edges(self.layout,layer,c,self.spec,0,'cpu')
                for q in range(9):
                    for k in range(9):self.assertEqual(bool(b[q,k]),k in [2,3,4] and (q==8 or q in [5,6,7] and (layer not in KEEP[c] or q!=6)))
    def test_joint_is_union_of_single_forbidden_edges(self):
        for c in NEW:
            for layer in range(32):
                b=blocked_edges(self.layout,layer,c,self.spec,0,'cpu')
                se=blocked_edges(self.layout,layer,'Semantic-k',self.spec,0,'cpu')
                la=blocked_edges(self.layout,layer,MATCH[c],self.spec,0,'cpu')
                self.assertTrue(torch.equal(b,se|la))
    def test_frozen_single_masks_bitwise_equal(self):
        from semantic_visual_writers.engine import apply_mask as sem
        from sparse_visual_writing_frontier.engine import apply_mask as layer
        for c in BASELINES:
            for k in range(32):
                a,b=apply_mask(self.mask,self.layout,k,c,self.spec,0)
                x,y=sem(self.mask,self.layout,[6],0) if c=='Semantic-k' else layer(self.mask,self.layout,k,c,0)
                self.assertTrue(torch.equal(a,x) and torch.equal(b,y))
    def test_decode_all32_and_final_possible_query(self):
        for c in NEW:
            for past in [9,10,39]:
                for k in range(32):
                    a,b=apply_mask(torch.zeros(1,1,1,past+1,dtype=torch.float16),self.layout,k,c,self.spec,past)
                    self.assertEqual(torch.where(b[0])[0].tolist(),[2,3,4]);self.assertEqual(int(torch.isneginf(a).sum()),3)
    def test_bos_self_image_other_sources_unchanged(self):
        for c in NEW:
            for k in [0,7,15,31]:
                a,b=apply_mask(self.mask,self.layout,k,c,self.spec,0)
                self.assertFalse(b[:,0].any() or b.diagonal().any() or b[[0,1,2,3,4]].any())
                self.assertTrue(torch.equal(a[0,0][~b],self.mask[0,0][~b]))
    def test_noop_and_native_bitwise_original(self):
        for c in NEW+['Native']:
            for k in [0,4,12,31]:
                a,_=apply_mask(self.mask,self.layout,k,c,self.spec,0,noop=c!='Native')
                self.assertTrue(torch.equal(a,self.mask))
    def test_actual_forbidden_softmax_zero(self):
        for c in NEW:
            for k in [0,7,15,31]:
                a,b=apply_mask(self.mask,self.layout,k,c,self.spec,0);w=torch.softmax(torch.randn(1,4,9,9)+a.float(),-1)
                self.assertEqual(float(w[0,:,b].max()),0.);self.assertTrue(torch.allclose(w.sum(-1),torch.ones(1,4,9)))
    def test_allowed_connection_count(self):
        for c in NEW:
            allowed=0
            for layer in range(32):
                b=blocked_edges(self.layout,layer,c,self.spec,0,'cpu');allowed+=3*3-int(b[5:8,2:5].sum())
            self.assertEqual(allowed,len(KEEP[c])*1*3)
    def test_selection_not_mutated_and_invalid_rejected(self):
        before=copy.deepcopy(self.spec)
        for c in NEW:
            for k in range(32):apply_mask(self.mask,self.layout,k,c,self.spec,0)
        self.assertEqual(self.spec,before)
        bad=copy.deepcopy(self.spec);bad['selection']['writers']['Semantic-k']=[8]
        with self.assertRaises(AssertionError):apply_mask(self.mask,self.layout,4,'Joint-12',bad,0)
    def test_interaction_sign_and_unique_harm(self):
        from .analyze import interaction,unique_harm
        self.assertEqual(interaction(0,1,1,1),-1);self.assertTrue(unique_harm(0,1,1,1))
        self.assertEqual(interaction(1,1,1,1),0);self.assertFalse(unique_harm(0,1,1,0))
        self.assertEqual(interaction(1,0,0,0),1)
    def test_predeclared_point_quality_thresholds(self):
        from .analyze import quality_flags
        self.assertEqual(quality_flags(200,202),dict(point_minus1_candidate=True,point_minus2_candidate=True))
        self.assertEqual(quality_flags(198,202),dict(point_minus1_candidate=False,point_minus2_candidate=True))
        self.assertEqual(quality_flags(197,202),dict(point_minus1_candidate=False,point_minus2_candidate=False))

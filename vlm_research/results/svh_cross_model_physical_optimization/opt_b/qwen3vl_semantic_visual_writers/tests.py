"""CPU scope, whole-word, image-layout and cleanup tests."""
import inspect
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from .engine import roles, blocked_edges, apply_mask, WriterMask
from ..semantic_visual_writers.groups import select, intact, subsets, RANDOM_SEEDS


class Tests(unittest.TestCase):
    def setUp(self):
        self.layout = dict(input_ids=[1,2,9,9,9,10,11,12,13],input_length=9,
            image_positions=[2,3,4],p_minus_1=8,question_positions=[6,7])
        self.mapping = roles(self.layout,10)
        self.old = torch.zeros(1,1,9,9,dtype=torch.bfloat16).masked_fill(
            torch.ones(9,9,dtype=torch.bool).triu(1),torch.finfo(torch.bfloat16).min)

    def test_vision_end_is_relay(self):
        self.assertEqual(self.mapping,dict(I=[2,3,4],R=[5,6,7],A=[8]))

    def test_variable_image_count(self):
        for count in [1,5,128,800]:
            layout = dict(input_ids=[1]+[9]*count+[10,11,12,13],input_length=count+5,
                image_positions=list(range(1,count+1)),p_minus_1=count+4,question_positions=[count+2])
            self.assertEqual(len(roles(layout,10)['I']),count)

    def test_wrong_boundary(self):
        with self.assertRaises(AssertionError):
            roles(self.layout,99)

    def test_all_only_answer_mask(self):
        b=blocked_edges(self.mapping,[5,6,7],9,0,'cpu')
        self.assertEqual(int(b.sum()),3)
        self.assertTrue(b[8,2:5].all())

    def test_no_includes_closing_marker(self):
        b=blocked_edges(self.mapping,[],9,0,'cpu')
        self.assertTrue(b[5,2:5].all())
        self.assertEqual(int(b.sum()),12)

    def test_mask_direction(self):
        b=blocked_edges(self.mapping,[6],9,0,'cpu')
        for q in range(9):
            for k in range(9):
                self.assertEqual(bool(b[q,k]),q in [5,7,8] and k in [2,3,4])

    def test_preserve_causal_bos_self(self):
        b=blocked_edges(self.mapping,[6],9,0,'cpu')
        changed=apply_mask(self.old,b,0)
        self.assertTrue(torch.equal(changed[0,0][~b],self.old[0,0][~b]))
        self.assertTrue((changed[0,0,:,0]==0).all())
        self.assertTrue((changed[0,0].diagonal()==0).all())

    def test_softmax_zero(self):
        b=blocked_edges(self.mapping,[],9,0,'cpu')
        changed=apply_mask(self.old,b,0)
        probabilities=torch.softmax(torch.randn(1,4,9,9)+changed.float(),-1)
        self.assertEqual(float(probabilities[0,:,b].max()),0.)
        self.assertTrue(torch.allclose(probabilities.sum(-1),torch.ones(1,4,9)))

    def test_decode_all_writers_always_block(self):
        for writers in [[],[5],[5,6,7]]:
            b=blocked_edges(self.mapping,writers,9,11,'cpu')
            changed=apply_mask(None,b,11)
            self.assertEqual(torch.where(b[0])[0].tolist(),[2,3,4])
            self.assertTrue(torch.isneginf(changed[0,0,0,2:5]).all())

    def test_noop_preserves_object_and_none(self):
        b=blocked_edges(self.mapping,[],9,0,'cpu')
        self.assertIs(apply_mask(self.old,b,0,True),self.old)
        self.assertIsNone(apply_mask(None,b,0,True))

    def test_reject_invalid_writer(self):
        with self.assertRaises(AssertionError):
            blocked_edges(self.mapping,[8],9,0,'cpu')

    def test_reject_future_unmasked(self):
        b=blocked_edges(self.mapping,[],9,0,'cpu')
        with self.assertRaises(AssertionError):
            apply_mask(torch.zeros_like(self.old),b,0)

    def test_whole_word_and_budget(self):
        groups=[dict(positions=[2,3],core=True),dict(positions=[4],core=True),
            dict(positions=[5],core=False),dict(positions=[6,7],core=False)]
        s=select(groups,[2,3,4,5],list(range(2,8)),'What color?')
        self.assertEqual(s['k'],2)
        for c,positions in s['writers'].items():
            self.assertTrue(intact(positions,groups))
            if c in ['Semantic-k','Question-Last-k','R-Last-k'] or c.startswith('Random-'):
                self.assertEqual(len(positions),2)
        self.assertEqual(sorted(s['writers']['Semantic-k']+s['writers']['Drop-Semantic']),list(range(2,8)))

    def test_scaffold_deficit_no_drop(self):
        groups=[dict(positions=[i],core=True) for i in range(2,6)]+[dict(positions=[6],core=False)]
        s=select(groups,list(range(2,7)),list(range(2,7)),'How many?')
        self.assertEqual(s['scaffold_budget_deficit'],s['k']-1)

    def test_selector_has_no_gold_or_image(self):
        self.assertEqual(list(inspect.signature(select).parameters),['groups','question_positions','relay','question'])
        self.assertEqual(RANDOM_SEEDS,[20261103,20261107,20261111])

    def test_subset_unique_words(self):
        groups=[dict(positions=[1,2]),dict(positions=[3])]
        for count,chosen in subsets(groups).items():
            self.assertEqual(sum(len(g['positions']) for g in chosen),count)
            self.assertEqual(len(chosen),len({id(g) for g in chosen}))

    def test_cleanup_exception(self):
        ctl=WriterMask.__new__(WriterMask)
        module=nn.Identity()
        ctl.handles=[module.register_forward_hook(lambda *args:None)]
        ctl.__exit__(RuntimeError,RuntimeError('forced'),None)
        self.assertFalse(module._forward_hooks)

    def test_layout_comparison_uses_mrope_not_just_count(self):
        from .run import layout_comparison
        a=dict(layout=dict(input_ids=[1,9,10],image_positions=[1],image_grid_thw=[1,2,8]),
            mrope=dict(position_ids_sha256='a',delta=-1))
        b=dict(layout=dict(input_ids=[1,9,10],image_positions=[1],image_grid_thw=[1,4,4]),
            mrope=dict(position_ids_sha256='b',delta=-2))
        result=layout_comparison(a,b)
        self.assertFalse(result['same_layout'])
        self.assertIn('image_grid_thw',result['differing_fields'])
        self.assertIn('mrope',result['differing_fields'])

    def test_first_logp_loss_sign_and_shift(self):
        import numpy as np
        from .analyze import first_metrics
        a=np.array([1.,0.,-1.])
        self.assertEqual(first_metrics(a,a),dict(flip=0,signed_loss=0.))
        self.assertAlmostEqual(first_metrics(a,a+10)['signed_loss'],0.,places=12)
        self.assertGreater(first_metrics(a,np.array([0.,2.,-1.]))['signed_loss'],0.)

    def test_full_vocabulary_distribution_metrics(self):
        import numpy as np
        from .analyze import distribution
        a=np.array([1.,0.,-1.])
        same=distribution(a,a)
        self.assertTrue(same['first_logits_exact'])
        self.assertEqual(same['first_distribution_js'],0.)
        self.assertEqual(same['first_distribution_tv'],0.)
        changed=distribution(a,np.array([-1.,0.,1.]))
        self.assertGreater(changed['first_distribution_js'],0.)
        self.assertGreater(changed['first_distribution_tv'],0.)


if __name__=='__main__':
    unittest.main(verbosity=2)

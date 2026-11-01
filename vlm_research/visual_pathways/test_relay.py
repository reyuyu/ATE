import unittest
import torch
from .engine import SourceMask,PrefillIntervention


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__();self.o_proj=torch.nn.Identity()

    def forward(self,hidden_states,attention_mask):
        logits=hidden_states@hidden_states.transpose(-1,-2)
        weights=(logits[:,None]+attention_mask).softmax(-1)
        return self.o_proj((weights@hidden_states[:,None])[:,0]),weights


class TestRelay(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.value=torch.randn(1,9,4)
        self.mask=torch.zeros(1,1,9,9).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),-torch.inf)

    def test_p1_softmax_source_and_other_queries(self):
        a=Attention();plain,full=a(self.value,self.mask)
        with SourceMask(a,[8],[2,3,4]) as edit:
            out,weights=a(hidden_states=self.value,attention_mask=self.mask)
        self.assertTrue(torch.equal(plain[:,:8],out[:,:8]))
        self.assertTrue(torch.equal(full[:,:,:8],weights[:,:,:8]))
        self.assertEqual(float(weights[:,:,8,[2,3,4]].max()),0.)
        self.assertEqual(edit.audit['changed_entries'],3)
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(1,1,9)))

    def test_relay_mask_keeps_p1_direct_and_visual_queries(self):
        layers=[torch.nn.Module() for _ in range(6)]
        for layer in layers:
            layer.self_attn=Attention()
        _,full=layers[0].self_attn(self.value,self.mask)
        with PrefillIntervention(layers,9,[2,3,4],relay_off=True) as controller:
            _,weights=layers[0].self_attn(hidden_states=self.value,attention_mask=self.mask)
        self.assertEqual(float(weights[:,:,5:8,2:5].max()),0.)
        self.assertTrue(torch.equal(weights[:,:,8],full[:,:,8]))
        self.assertTrue(torch.equal(weights[:,:,2:5],full[:,:,2:5]))
        self.assertFalse(layers[0].self_attn._forward_pre_hooks)

    def test_all_noop_exact(self):
        module=Attention();plain,full=module(self.value,self.mask)
        with SourceMask(module,[0,1,5,6,7],[2,3,4],noop=True):
            out,weights=module(hidden_states=self.value,attention_mask=self.mask)
        self.assertTrue(torch.equal(out,plain) and torch.equal(weights,full))


if __name__=='__main__':
    unittest.main()

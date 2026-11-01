from types import SimpleNamespace
import unittest

import torch

from ..qwen3vl_image_source.tests import Tests as SourceMaskTests, ToyAttention
from .engine import CONDITIONS, PrefillOnly


class FakeCache:
    length=0
    def get_seq_length(self, _):
        return self.length


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(36)])
        for layer in self.layers:
            layer.self_attn=ToyAttention()

    def forward(self, inputs_embeds, cache_position, past_key_values, visual_pos_masks=None, deepstack_visual_embeds=None):
        n=inputs_embeds.shape[1]
        mask=torch.zeros(1,1,n,past_key_values.length+n)
        if n>1:
            mask.masked_fill_(torch.ones(n,n).triu(1).bool(),float('-inf'))
            for layer in self.layers:
                layer.self_attn(hidden_states=inputs_embeds,attention_mask=mask)
        past_key_values.length+=n
        return inputs_embeds


class GenerationMaskTests(unittest.TestCase):
    def test_fixed_condition_layers(self):
        self.assertEqual(len(CONDITIONS['Keep L26']),35)
        self.assertNotIn(26,CONDITIONS['Keep L26'])
        self.assertEqual(CONDITIONS['All OFF'],list(range(36)))

    def test_masks_removed_before_decode_and_exception(self):
        lm=ToyLM()
        model=SimpleNamespace(model=SimpleNamespace(language_model=lm))
        layout=dict(input_length=8,p_minus_1=7,image_positions=[2,3,4])
        visual=torch.tensor([[False,False,True,True,True,False,False,False]])
        cache=FakeCache()
        with PrefillOnly(model,layout,CONDITIONS['Keep L26'],cache) as control:
            lm(inputs_embeds=torch.randn(1,8,4),cache_position=torch.arange(8),past_key_values=cache,
               visual_pos_masks=visual,deepstack_visual_embeds=[1,2,3])
            self.assertEqual(len(control.interventions),35)
            self.assertTrue(all(not x.handles and x.calls==1 for x in control.interventions.values()))
            lm(inputs_embeds=torch.randn(1,1,4),cache_position=torch.tensor([8]),past_key_values=cache)
            self.assertEqual(len(control.calls),2)
        self.assertFalse(lm._forward_hooks)
        self.assertFalse(lm._forward_pre_hooks)
        cache=FakeCache()
        try:
            with PrefillOnly(model,layout,CONDITIONS['All OFF'],cache) as control:
                control.begin(lm,(),dict(inputs_embeds=torch.randn(1,8,4),cache_position=torch.arange(8),
                    past_key_values=cache,visual_pos_masks=visual,deepstack_visual_embeds=[1,2,3]))
                raise ValueError('active masks cleanup')
        except ValueError:
            pass
        self.assertTrue(all(not layer.self_attn._forward_hooks and not layer.self_attn._forward_pre_hooks for layer in lm.layers))


if __name__=='__main__':
    unittest.main()

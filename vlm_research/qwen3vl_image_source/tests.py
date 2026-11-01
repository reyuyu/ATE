import types
import unittest

import torch

from .engine import ImageOff, image_mask, metrics


class ToyAttention(torch.nn.Module):
    def forward(self, hidden_states, attention_mask):
        weights = (hidden_states @ hidden_states.transpose(-1, -2)).unsqueeze(1) + attention_mask
        probabilities = weights.softmax(-1)
        return probabilities @ hidden_states.unsqueeze(1), probabilities


class Tests(unittest.TestCase):
    def setUp(self):
        self.n, self.q, self.visual = 8, 7, [2, 3, 4]
        self.mask = torch.zeros(1, 1, 8, 8).masked_fill(torch.ones(8, 8).triu(1).bool(), float('-inf'))

    def test_only_query_image_changes(self):
        changed = image_mask(self.mask, self.n, self.q, self.visual)
        self.assertTrue(torch.equal(self.mask[:, :, :-1], changed[:, :, :-1]))
        self.assertEqual(int((changed != self.mask).sum()), 3)
        self.assertTrue(torch.isneginf(changed[:, :, self.q, self.visual]).all())
        self.assertEqual(float(changed[0, 0, self.q, self.q]), 0)

    def test_native_softmax_and_cleanup(self):
        module = ToyAttention()
        model = types.SimpleNamespace(model=types.SimpleNamespace(language_model=types.SimpleNamespace(
            layers=[types.SimpleNamespace(self_attn=module)])))
        layout = dict(input_length=self.n, p_minus_1=self.q, image_positions=self.visual)
        x = torch.randn(1, self.n, 3)
        full = module(hidden_states=x, attention_mask=self.mask)
        with ImageOff(model, 0, layout, noop=True):
            noop = module(hidden_states=x, attention_mask=self.mask)
        self.assertTrue(torch.equal(full[0], noop[0]))
        with ImageOff(model, 0, layout) as hook:
            changed = module(hidden_states=x, attention_mask=self.mask)
            self.assertEqual(hook.calls, 1)
            self.assertEqual(hook.audit['image_probability_max'], 0)
        self.assertTrue(torch.equal(full[1][:, :, :-1], changed[1][:, :, :-1]))
        self.assertTrue(torch.equal(self.mask[0, 0].isneginf(), torch.ones(8, 8).triu(1).bool()))
        try:
            with ImageOff(model, 0, layout):
                raise ValueError('fixture')
        except ValueError:
            pass
        self.assertFalse(module._forward_hooks)
        self.assertFalse(module._forward_pre_hooks)

    def test_signed_absolute_not_absolute_of_mean(self):
        full = torch.tensor([2., 0., -1.])
        better = metrics(full, torch.tensor([3., 0., -1.]))
        worse = metrics(full, torch.tensor([0., 2., -1.]))
        self.assertLess(better['signed_logp_loss'], 0)
        self.assertEqual(better['absolute_logp_change'], abs(better['signed_logp_loss']))
        self.assertEqual(worse['top1_flip'], 1)
        self.assertEqual(metrics(full, full)['signed_logp_loss'], 0)

    def test_invalid_visual_positions(self):
        for visual in ([self.q], [], [2, 2], [8]):
            with self.assertRaises(AssertionError):
                image_mask(self.mask, self.n, self.q, visual)


if __name__ == '__main__':
    unittest.main()

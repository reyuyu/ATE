import math
import unittest

import torch

from information_flow_window9.masks import block, edges, intervention, probability_change, window


class Attention(torch.nn.Module):
    def forward(self, x, attention_mask=None, **kwargs):
        return x, torch.softmax(attention_mask.expand(-1, 2, -1, -1), -1)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()


class Tests(unittest.TestCase):
    def setUp(self):
        self.mapping = {'visual_positions': [1, 2], 'question_positions': [3, 4], 'p_minus_1': 6}
        self.mask = torch.zeros(4, 1, 7, 7).masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), -torch.inf)

    def test_windows(self):
        self.assertEqual(window(0), tuple(range(5)))
        self.assertEqual(window(4), tuple(range(9)))
        self.assertEqual(window(16), tuple(range(12, 21)))
        self.assertEqual(window(31), tuple(range(27, 32)))
        self.assertEqual(len(window(27)), 9)
        with self.assertRaises(ValueError):
            window(32)

    def test_directions(self):
        self.assertEqual(edges('image_to_last', self.mapping), ([6], [1, 2]))
        self.assertEqual(edges('question_to_last', self.mapping), ([6], [3, 4]))
        self.assertEqual(edges('image_to_question', self.mapping), ([3, 4], [1, 2]))

    def test_only_target_cartesian_entries(self):
        for path in ('image_to_last', 'question_to_last', 'image_to_question'):
            q, k = edges(path, self.mapping)
            original = self.mask.clone()
            changed = block(self.mask, [0, 2], q, k, validate=True)
            self.assertTrue(torch.equal(self.mask, original))
            for b in range(4):
                for i in range(7):
                    for j in range(7):
                        target = b in [0, 2] and i in q and j in k
                        self.assertEqual(changed[b, 0, i, j], -torch.inf if target else self.mask[b, 0, i, j])

    def test_question_mask_preserves_last_access(self):
        q, k = edges('image_to_question', self.mapping)
        changed = block(self.mask, [0], q, k, validate=True)
        self.assertTrue(torch.equal(changed[:, :, 6], self.mask[:, :, 6]))

    def test_restore_normal_and_exception(self):
        layers = torch.nn.ModuleList([Layer() for _ in range(32)])
        for fail in (False, True):
            try:
                with intervention(layers, 'image_to_question', self.mapping, [0, 4, 16, 31], True) as stats:
                    for layer in layers:
                        layer.self_attn(torch.zeros(1), attention_mask=self.mask, past_key_value=None)
                    self.assertEqual(stats['target_weights_checked'], 23)
                    if fail:
                        raise RuntimeError('test')
            except RuntimeError:
                if not fail:
                    raise
            self.assertEqual(sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in layers.modules()), 0)

    def test_formula_fixed_reference(self):
        self.assertAlmostEqual(probability_change(math.log(.4), math.log(.1)), -75)
        self.assertAlmostEqual(probability_change(math.log(.1), math.log(.4)), 300)
        self.assertEqual(probability_change(-1000, -1000), 0)
        with self.assertRaises(ValueError):
            probability_change(0, float('nan'))

    def test_reject_wrong_positions(self):
        with self.assertRaises(ValueError):
            edges('image_to_last', {**self.mapping, 'question_positions': [6]})


if __name__ == '__main__':
    unittest.main()

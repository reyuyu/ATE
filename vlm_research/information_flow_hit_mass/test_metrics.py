import unittest
import numpy as np
import torch

from .attention import FIELDS, add_shares, layer_rows, mass_heads, observe_mass, window_rows
from .analyze import hit_stats, spearman
from information_flow_window9.masks import first_token_hit
from roi_sink_common import sink_candidates


class Tests(unittest.TestCase):
    def test_mass_direction_and_empty_sink(self):
        mapping = {'visual_positions': list(range(1, 577)), 'question_positions': [577, 578], 'p_minus_1': 580}
        weights = torch.zeros(1, 2, 581, 581)
        weights[:, :, 580, 0] = .2
        weights[:, :, 580, 1] = .3
        weights[:, :, 580, 2] = .1
        weights[:, :, 577, 1] = .4
        weights[:, :, 578, 1] = .2
        result = mass_heads(weights, mapping, [0])
        np.testing.assert_allclose(result, [[.2, .4, .3, .3, .3]] * 2, rtol=1e-6)
        empty = mass_heads(weights, mapping, [])
        self.assertTrue(np.all(empty[:, [2, 4]] == 0))

    def test_mass_ratio_and_clipped_window(self):
        heads = np.ones((32, 32, 5)) * .1
        heads[:, :, 1] = .5
        heads[:, :, 3] = .25
        rows = layer_rows(heads)
        self.assertAlmostEqual(rows[0]['last_sink_over_image'], .2)
        self.assertAlmostEqual(rows[0]['question_sink_over_image'], .4)
        windows = window_rows(rows)
        self.assertEqual(windows[0]['window_layers'], 5)
        self.assertEqual(windows[16]['window_layers'], 9)
        self.assertEqual(windows[31]['window_start'], 27)
        self.assertAlmostEqual(windows[16]['last_sink_over_image'], .2)
        self.assertIsNone(add_shares({f: 0 for f in FIELDS})['last_sink_over_image'])

    def test_window_ratio_is_ratio_of_raw_means(self):
        rows = [{'layer': i, **{f: float(i + 1) for f in FIELDS}} for i in range(32)]
        rows[0]['last_sink_mass'] = 0
        value = window_rows(rows)[0]['last_sink_over_image']
        self.assertAlmostEqual(value, 14/15)

    def test_flip_accounting(self):
        rows = [{'full_first_token_hit': a, 'masked_first_token_hit': b, 'hit_to_miss': a and not b,
                 'miss_to_hit': not a and b} for a, b in [(True, True), (True, False), (False, True), (False, False)]]
        s = hit_stats(rows)
        self.assertEqual(s['hit_to_miss'], 1)
        self.assertEqual(s['miss_to_hit'], 1)
        self.assertEqual(s['first_token_hit_delta_pp'], 0)
        self.assertEqual(s['masked_first_token_hit_rate_pct'], 50)

    def test_rank_ties_and_constant(self):
        self.assertAlmostEqual(spearman([1, 2, 2, 3], [2, 4, 4, 6]), 1)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(spearman([None, 1], [2, 3]))

    def test_same_neutral_rule_no_outcomes(self):
        maps = np.ones((3, 32, 2, 576)) * 1e-4
        maps[:, :, :, 17] = .1
        self.assertEqual(sink_candidates(maps)['indices'], [17])
        self.assertEqual(sink_candidates(np.ones_like(maps))['indices'], [])

    def test_first_token_definition_unchanged(self):
        self.assertTrue(first_token_hit('Right', ['right']))
        self.assertFalse(first_token_hit('Lar', ['large']))

    def test_observer_cleanup_on_exception(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Identity()
        layers = [Layer() for _ in range(32)]
        with self.assertRaises(RuntimeError):
            with observe_mass(layers, {}, []):
                raise RuntimeError('injected')
        self.assertTrue(all(not layer.self_attn._forward_hooks for layer in layers))


if __name__ == '__main__':
    unittest.main()

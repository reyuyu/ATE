"""Analytic checks for the statistical interpretation, no model/GPU needed."""
import unittest
import numpy as np

from attention_probe import key_masks, summarize_rows
from compare_attention import conditional_spatial_distances


class AttentionStatisticsTest(unittest.TestCase):
    def test_uniform_causal_with_padding_and_future_visual_keys(self):
        # Two padded and two future image IDs must not inflate the baseline.
        ids = [99, 1, 99, 99, 2, 99, 99]
        accessible, visual, nonvisual = key_masks(ids, [0, 1, 1, 1, 1, 1, 1], 4, 99)
        rows = np.array([[0, .25, .25, .25, .25, 0, 0]])
        stat = summarize_rows(rows, accessible, visual, nonvisual, 0)[0]
        self.assertEqual(stat["visual_keys"], 2)
        self.assertEqual(stat["nonvisual_keys"], 2)
        self.assertEqual(stat["visual_attention_mass"], .5)
        self.assertEqual(stat["uniform_visual_baseline"], .5)
        self.assertEqual(stat["visual_attention_density_vs_uniform"], 1.)
        self.assertAlmostEqual(stat["visual_nonvisual_density_ratio"], 1.)
        self.assertAlmostEqual(stat["visual_entropy_nats"], np.log(2))
        self.assertEqual(stat["top10_visual_conditional_share"], .5)

    def test_high_visual_mass_can_have_low_per_token_density(self):
        ids = [99] * 9 + [1]
        a, v, n = key_masks(ids, [1] * 10, 9, 99)
        rows = np.array([[.8 / 9] * 9 + [.2]])
        stat = summarize_rows(rows, a, v, n, 2)[0]
        self.assertAlmostEqual(stat["visual_attention_mass"], .8)
        self.assertAlmostEqual(stat["visual_attention_density_vs_uniform"], .8 / .9)
        self.assertAlmostEqual(stat["visual_nonvisual_density_ratio"], 4 / 9)
        self.assertAlmostEqual(stat["visual_entropy_normalized"], 1.)

    def test_concentrated_visual_and_zero_visual_mass(self):
        a, v, n = key_masks([99] * 576 + [1], [1] * 577, 576, 99)
        rows = np.zeros((2, 577))
        rows[0, 17], rows[0, -1] = .6, .4
        rows[1, -1] = 1
        stats = summarize_rows(rows, a, v, n, 0)
        self.assertEqual(stats[0]["top10_patch_count"], 58)
        self.assertEqual(stats[0]["visual_entropy_nats"], 0.)
        self.assertEqual(stats[0]["top10_visual_attention_mass"], .6)
        self.assertEqual(stats[0]["top10_visual_conditional_share"], 1.)
        self.assertFalse(stats[1]["visual_entropy_defined"])
        self.assertEqual(stats[1]["visual_nonvisual_density_ratio"], 0.)

    def test_invalid_probabilities_rejected(self):
        a, v, n = key_masks([99, 1, 1], [1, 1, 1], 1, 99)
        for row in [[.5, .4, .1], [.5, .6, 0], [np.nan, .5, 0], [-.1, 1.1, 0]]:
            with self.assertRaises(ValueError):
                summarize_rows([row], a, v, n, 0)

    def test_spatial_distance_ignores_mass_and_detects_disjoint_support(self):
        _, _, same = conditional_spatial_distances(np.array([.1, .2]), np.array([.2, .4]))
        self.assertAlmostEqual(same["conditional_spatial_total_variation"], 0.)
        self.assertAlmostEqual(same["conditional_spatial_js_divergence_nats"], 0.)
        _, _, different = conditional_spatial_distances(np.array([1., 0.]), np.array([0., 1.]))
        self.assertEqual(different["conditional_spatial_total_variation"], 1.)
        self.assertAlmostEqual(different["conditional_spatial_js_divergence_nats"], np.log(2))
        self.assertEqual(different["conditional_spatial_cosine_similarity"], 0.)


if __name__ == "__main__":
    unittest.main()

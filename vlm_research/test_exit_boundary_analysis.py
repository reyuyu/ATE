"""CPU numerical oracles for cached exit-boundary statistics; no model imports."""
import unittest
import numpy as np
from exit_boundary_analysis import ranks, row_correlation, attention_metrics, boundary_row, choose_best, flip_category, paired_summary


class BoundaryStatisticsTest(unittest.TestCase):
    def test_average_rank_ties_and_batch(self):
        np.testing.assert_array_equal(ranks([4, 1, 1, 9, 4]), [3.5, 1.5, 1.5, 5, 3.5])
        np.testing.assert_array_equal(ranks([[1, 1, 3], [9, 5, 5]]), [[1.5, 1.5, 3], [3, 1.5, 1.5]])
        self.assertAlmostEqual(row_correlation(ranks([1, 1, 3]), ranks([9, 9, 2]))[0], -1)

    def test_constant_correlation_undefined(self):
        self.assertTrue(np.isnan(row_correlation([1, 1, 1], [1, 2, 3])[0]))

    def test_uniform_entropy_area_normalization(self):
        a = np.full((32, 32, 2, 576), .25/576)
        result = attention_metrics(a, list(range(72)), list(range(72, 90)))
        np.testing.assert_allclose(result['roi_enrichment'], 1)
        np.testing.assert_allclose(result['total_visual_mass'], .25)
        np.testing.assert_allclose(result['normalized_entropy'], 1)
        np.testing.assert_allclose(result['sink_nonroi_conditional_share'], 18/576)

    def test_head_conditioning_before_mean_is_not_pooled_entropy(self):
        a = np.zeros((32, 32, 2, 576))
        for head in range(32):
            a[:, head, :, head] = .1 if head % 2 else .8
        result = attention_metrics(a, list(range(16)), [31])
        np.testing.assert_allclose(result['normalized_entropy'], 0)
        pooled = a[0, :, 0].mean(0); pooled /= pooled.sum()
        self.assertGreater(-(pooled*np.log(np.maximum(pooled, 1e-300))).sum()/np.log(576), .4)

    def test_off_by_one_and_gain_sign(self):
        shape = (32, 32, 2)
        m = {name: np.broadcast_to(np.arange(32)[:, None, None], shape).astype(float) for name in ['roi_enrichment', 'normalized_entropy', 'sink_nonroi_conditional_share']}
        m['total_visual_mass'] = np.full(shape, .1)
        r = boundary_row('qid', 'answer', 24, m, {24: .2, 32: -.1}, True)
        self.assertEqual((r['before_layer'], r['after_layer']), (23, 24))
        self.assertEqual((r['roi_before'], r['roi_after'], r['delta_roi']), (23, 24, 1))
        self.assertAlmostEqual(r['gain'], .3)
        self.assertEqual(r['flip_category'], 'wrong_to_correct')

    def test_zero_margin_and_best_sparse_grid_ties(self):
        self.assertEqual(flip_category(.1, 0), 'correct_to_wrong')
        self.assertEqual(flip_category(0, 1e-12), 'wrong_to_correct')
        score = {0: 8., 8: 7., 12: 1., 16: 1.+1e-15, 20: .5, 24: .6, 28: .9, 32: 2.}
        best, ties, _ = choose_best(score, [12, 16, 20, 24, 28])
        self.assertEqual(best, 12); self.assertEqual(ties, [12, 16])
        self.assertEqual(choose_best(score, list(score))[0], 0)

    def test_roi_drop_can_coexist_with_entropy_drop(self):
        a = np.full((32, 32, 2, 576), .25/576)
        a[24] = 0; a[24, :, :, 575] = .25
        r = attention_metrics(a, list(range(72)), [575])
        self.assertLess(r['roi_enrichment'][24].mean()-r['roi_enrichment'][23].mean(), 0)
        self.assertLess(r['normalized_entropy'][24].mean()-r['normalized_entropy'][23].mean(), 0)
        self.assertAlmostEqual(r['total_visual_mass'][24].mean(), r['total_visual_mass'][23].mean())

    def test_low_mass_flags_include_individual_heads(self):
        shape = (32, 32, 2)
        m = {name: np.ones(shape) for name in ['roi_enrichment', 'normalized_entropy', 'sink_nonroi_conditional_share']}
        m['total_visual_mass'] = np.full(shape, .1)
        m['total_visual_mass'][15, 0, 0] = 1e-6
        r = boundary_row('qid', 'question', 16, m, {16: 1., 32: 1.}, True)
        self.assertTrue(r['low_mass_flag']); self.assertEqual(r['endpoint_heads_below_1e_minus4'], 1)
        self.assertIsNone(boundary_row('qid', 'question', 16, m, {16: 1., 32: 1.}, True, np.zeros(32, bool)))

    def test_paired_bootstrap_reproducible_negative_oracle(self):
        shape = (32, 32, 2)
        m = {name: np.ones(shape) for name in ['roi_enrichment', 'normalized_entropy', 'sink_nonroi_conditional_share']}
        m['total_visual_mass'] = np.full(shape, .1)
        rows = []
        for i, gain in enumerate([-3, -1, 1, 3]):
            r = boundary_row(str(i), 'question', 16, m, {16: gain, 32: 0.}, True)
            r['delta_roi'] = -gain; r['delta_entropy'] = gain
            rows.append(r)
        a, b = paired_summary(rows, 'test'), paired_summary(rows, 'test')
        self.assertEqual(a, b)
        self.assertEqual(a['roi_down_gain_positive'], 2)
        self.assertAlmostEqual(a['delta_roi_gain_pearson'], -1)
        self.assertAlmostEqual(a['delta_roi_gain_spearman'], -1)
        self.assertAlmostEqual(a['delta_roi_group_mean_difference'], -4)
        self.assertAlmostEqual(a['delta_roi_gain_pearson_ci_high'], -1)
        self.assertGreater(a['delta_roi_gain_pearson_undefined_bootstrap'], 0)


if __name__ == '__main__':
    unittest.main()

import unittest
import torch

from .common import OPEN_LAYERS
from .engine import controller
from ..qwen3vl_semantic_visual_writers.engine import roles, blocked_edges, apply_mask


class Tests(unittest.TestCase):
    def test_exact_joint_boundary(self):
        self.assertEqual(OPEN_LAYERS, list(range(22)))
        ctl = controller(None, {})
        self.assertEqual(ctl.open_layers, set(range(22)))
        self.assertEqual(set(range(36))-ctl.open_layers, set(range(22, 36)))

    def test_all_layer_scopes(self):
        layout = dict(input_ids=[1, 9, 9, 10, 11, 12, 13], image_positions=[1, 2],
            input_length=7, p_minus_1=6, question_positions=[4, 5])
        mapping = roles(layout, 10)
        old = torch.zeros(1, 1, 7, 7, dtype=torch.bfloat16).masked_fill(
            torch.ones(7, 7, dtype=torch.bool).triu(1), torch.finfo(torch.bfloat16).min)
        for layer in range(36):
            writers = mapping['R'] if layer in OPEN_LAYERS else []
            b = blocked_edges(mapping, writers, 7, 0, 'cpu')
            self.assertEqual(torch.where(b.any(-1))[0].tolist(), [6] if layer < 22 else [3, 4, 5, 6])
            self.assertFalse(b[mapping['I']].any())
            mask = apply_mask(old, b, 0)
            self.assertTrue(torch.equal(mask[0, 0][~b], old[0, 0][~b]))
            self.assertEqual(float(torch.softmax(mask.float(), -1)[0, :, b].max()), 0.)
            self.assertIs(apply_mask(old, b, 0, noop=True), old)
            decode = blocked_edges(mapping, writers, 7, 10, 'cpu')
            self.assertEqual(torch.where(decode[0])[0].tolist(), [1, 2])

    def test_one_sided_noninferiority_boundary(self):
        from .analyze import decision
        self.assertEqual(decision(-.99, -1.4, .1), 'SUPPORTED')
        self.assertEqual(decision(-1., -1.2, .1), 'INSUFFICIENT')
        self.assertEqual(decision(-1.5, -2., -1.1), 'BELOW_MARGIN')

    def test_bootstrap_pair_cluster_estimator(self):
        import numpy as np
        from ..qwen3vl_visual_writing_necessity.analyze import paired_bootstrap
        x = np.array([1., 3., 10.])
        actual = paired_bootstrap(x, ['a', 'a', 'b'], repetitions=20, seed=20261104)
        w = np.random.default_rng(20261104).multinomial(2, [.5, .5], size=20)
        expected = (w @ np.array([4., 10.])) / (w @ np.array([2., 1.]))
        self.assertTrue(np.array_equal(actual[:, 0], expected))

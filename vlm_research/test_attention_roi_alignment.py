"""Small CPU checks for read-only regional attention and original frozen cohort."""
import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from attention_roi_alignment.run import NativeRows, PILOT, ROI, regional
from attention_roi_alignment.analyze import boolean, window_values
from roi_sink_common import sink_candidates


class AlignmentTests(unittest.TestCase):
    def test_uniform_roi_enrichment_one(self):
        result = regional(np.full((32, 2, 2, 576), .4/576), list(range(12)), [0, 20])
        np.testing.assert_allclose(result['p_roi_enrichment'], 1)
        np.testing.assert_allclose(result['p_image'], .4)
        np.testing.assert_allclose(result['p_sink_mass'], .4*2/576)
        np.testing.assert_allclose(result['p_sink_nonroi_mass'], .4/576)

    def test_head_ratios_before_head_mean(self):
        rows = np.full((32, 2, 2, 576), .2/576)
        rows[:, 0, :, 0] = .1
        rows[:, 1, :, 0] = .01
        result = regional(rows, [0], [])
        expected = np.mean([.1/(.1+.2*575/576), .01/(.01+.2*575/576)])
        np.testing.assert_allclose(result['p_roi_share'], expected)
        self.assertNotAlmostEqual(expected, .11/(.11+.4*575/576))

    def test_empty_roi_not_fake_zero_enrichment(self):
        result = regional(np.full((32, 2, 2, 576), 1/576), [], [])
        self.assertTrue(np.isnan(result['q_roi_enrichment']).all())
        self.assertTrue((result['p_sink_mass'] == 0).all())

    def test_frozen_geometry_exact_sixteen(self):
        ids = [s['question_id'] for s in json.loads(PILOT.read_text())['samples']['gqa_chooserel']]
        entries = {s['question_id']: s for s in json.loads((ROI/'manifest.json').read_text())['samples']}
        self.assertEqual(len(ids), 32)
        self.assertEqual(sum(not entries[q]['geometry_exclusion_reasons'] for q in ids), 16)

    def test_sink_neutral_persistence_rule(self):
        rows = np.full((3, 32, 2, 576), .1/575)
        rows[..., 7] = .9
        self.assertEqual(sink_candidates(rows)['indices'], [7])
        self.assertEqual(sink_candidates(rows)['holdout_indices'], [7])
        rows[:2, :25, :, 7] = .1/575
        self.assertEqual(sink_candidates(rows)['indices'], [])

    def test_nonfinite_rejected(self):
        rows = np.full((32, 2, 2, 576), np.nan)
        with self.assertRaises(ValueError):
            regional(rows, [], [])

    def test_window_start_means_three_layers(self):
        rows = [dict(layer=i, p_bos=i/32, q_bos=0., p_image=.5, q_image=.2) for i in range(32)]
        self.assertAlmostEqual(window_values(rows, 29)['p_bos'], 30/32)
        with self.assertRaises(ValueError):
            window_values(rows, 30)

    def test_csv_booleans_not_string_truthiness(self):
        self.assertFalse(boolean('False'))
        self.assertTrue(boolean('True'))
        with self.assertRaises(ValueError):
            boolean('')

    def test_native_observer_readonly_and_exception_cleanup(self):
        model = torch.nn.Module()
        model.language_model = torch.nn.Module()
        model.language_model.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Identity()
        model.language_model.model.layers = torch.nn.ModuleList([layer])
        model.language_model.model.norm = torch.nn.Identity()
        model.config = SimpleNamespace(image_token_index=32000, text_config=SimpleNamespace(num_attention_heads=2))
        inputs = {'input_ids': torch.tensor([[1]+[32000]*576+[12, 13]])}
        weights = torch.tril(torch.ones(579, 579))
        weights = (weights/weights.sum(-1, keepdim=True))[None, None].expand(1, 2, -1, -1).half()
        original = weights.clone()
        hidden = torch.zeros(1, 579, 4)
        with self.assertRaisesRegex(RuntimeError, 'forced cleanup'):
            with NativeRows(model, inputs, [577], 578, 1) as stream:
                output = stream.capture(0)(None, (), (hidden, weights))
                self.assertIs(output[0], hidden)
                self.assertIsNone(output[1])
                self.assertTrue(torch.equal(weights, original))
                self.assertEqual(stream.rows[0].shape, (2, 2, 576))
                self.assertAlmostEqual(stream.bos[0][1], float(weights[0, 0, 578, 0]))
                raise RuntimeError('forced cleanup')
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()

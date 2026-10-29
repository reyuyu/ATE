from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmark_interventions.hooks import ORIGINALS, original_modules, PrefillIntervention
from benchmarks.dataset_registry import sha256
from benchmarks.smoke import hook_footprint
from cross_dataset_c1_c2.hooks import AttentionPrefill
from knockout_probe import model_layers
from p1_sliding_window.hooks import KINDS, WindowTrace
from p1_sliding_window.run import measured, cosine, plans
from p1_sliding_window.analyze import ordered
from residual_only_pilot.hooks import ResidualOnly
from test_knockout import tiny_model, tiny_inputs


class SlidingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = tiny_model()
        cls.originals = original_modules({k: sha256(p) for k, p in ORIGINALS.items()})

    def test_noop_all_boundary_windows_exact(self):
        inputs = tiny_inputs()
        base, _ = measured(self.model, inputs, 11)
        for kind in KINDS:
            for start in (0, 2, 29):
                out, _ = measured(self.model, inputs, 11, kind, start, noop=True)
                np.testing.assert_array_equal(out['logits'], base['logits'])

    def test_only_target_query_and_requested_layers(self):
        inputs = tiny_inputs()
        for kind in KINDS:
            for start in (0, 29):
                _, edits = measured(self.model, inputs, 11, kind, start, strict_zero=self.originals['c2'].zero_query)
                self.assertEqual({r['layer'] for r in edits}, set(range(start, start+3)))
                self.assertTrue(all(r['query'] == 11 and r['other_tokens_bitwise_exact'] and r['target_zero'] for r in edits))

    def test_original_l2_l4_implementations_exact(self):
        inputs = tiny_inputs()
        ids = inputs['input_ids'][0].tolist()
        mapping = dict(prompt_ids=ids, prompt_length=12, p_minus_1=11,
                       question_positions=[6, 7, 8], visual_positions=[2, 3, 4, 5])
        for kind in KINDS:
            current, _ = measured(self.model, inputs, 11, kind, 2)
            reference = (AttentionPrefill(self.model, inputs, mapping, self.originals) if kind == 'A-Skip' else
                         PrefillIntervention(self.model, inputs, mapping, 'C2', self.originals, observe=False) if kind == 'M-Skip' else
                         ResidualOnly(self.model, ids, 'P1-Joint', self.originals['c2'].zero_query))
            with torch.inference_mode(), reference:
                expected = self.model(**inputs, use_cache=False).logits[0, 11].cpu().numpy()
            np.testing.assert_array_equal(expected, current['logits'])

    def test_joint_exact_target_residual(self):
        inputs = tiny_inputs()
        before = {}
        with ExitStack() as stack:
            for i in (2, 3, 4):
                def enter(module, args, i=i):
                    before[i] = args[0][:, 11].clone()
                def leave(module, args, out, i=i):
                    self.assertTrue(torch.equal(before[i], out[0][:, 11]))
                stack.callback(model_layers(self.model)[i].register_forward_pre_hook(enter).remove)
                stack.callback(model_layers(self.model)[i].register_forward_hook(leave).remove)
            measured(self.model, inputs, 11, 'Joint-Skip', 2)

    def test_single_token_candidate_causal_reuse(self):
        from attention_position_probe import append_answer_inputs
        inputs = tiny_inputs()
        for kind in ('Full',)+KINDS:
            for start in (0, 29):
                a, _ = measured(self.model, append_answer_inputs(inputs, [10]), 11, kind, start)
                b, _ = measured(self.model, append_answer_inputs(inputs, [20]), 11, kind, start)
                np.testing.assert_array_equal(a['logits'], b['logits'])

    def test_final_window_hidden_depth_and_cosine(self):
        inputs = tiny_inputs()
        full, _ = measured(self.model, inputs, 11)
        current, _ = measured(self.model, inputs, 11, 'Joint-Skip', 29)
        self.assertEqual(list(current['layer_indices']), [31])
        self.assertEqual(cosine(current['layer_states'][0], full['layer_states'][31]),
                         cosine(current['layer_states'][-1], full['layer_states'][31]))
        self.assertAlmostEqual(cosine(np.array([1., 0]), np.array([1., 0])), 0.)

    def test_exception_cleanup_and_cache_rejection(self):
        before = hook_footprint(self.model)
        inputs = tiny_inputs()
        for kind in KINDS:
            with self.assertRaisesRegex(RuntimeError, 'expected'):
                with WindowTrace(self.model, 11, kind, 2):
                    raise RuntimeError('expected')
            with self.assertRaises(ValueError):
                with WindowTrace(self.model, 11, kind, 2):
                    self.model(**inputs, use_cache=True)
        self.assertEqual(before, hook_footprint(self.model))

    def test_refuse_multi_token_shortcut(self):
        sample = {'candidates': ['a'], 'metadata': {'input_mapping': {'p_minus_1': 11}}}
        with patch('p1_sliding_window.run.candidate_inputs', return_value=[({}, [{}, {}])]):
            with self.assertRaises(ValueError):
                plans(None, {}, sample)

    def test_ranking_ties_deterministic(self):
        rows = [{'window_start': i, 'mean_absolute_change': float(i//2)} for i in (3, 2, 1, 0)]
        self.assertEqual([r['window_start'] for r in ordered(rows, True)], [2, 3, 0])
        self.assertEqual([r['window_start'] for r in ordered(rows, False)], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()

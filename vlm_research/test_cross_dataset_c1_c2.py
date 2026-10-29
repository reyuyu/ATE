import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from benchmark_interventions.hooks import ORIGINALS, original_modules, footprint
from benchmarks.dataset_registry import digest, sha256
from cross_dataset_c1_c2.hooks import AttentionPrefill
from cross_dataset_c1_c2.run import atomic_json, assert_prefix, decide, exact_historical
from cross_dataset_c1_c2.analyze import clustered_stats
from test_knockout import tiny_model, tiny_inputs


class C1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.originals = original_modules({key: sha256(path) for key, path in ORIGINALS.items()})
        cls.model = tiny_model()

    def inputs(self):
        inputs = tiny_inputs()
        mapping = dict(prompt_length=12, p_minus_1=11, question_positions=[6, 7, 8],
                       prompt_ids=inputs['input_ids'][0].tolist(), visual_positions=[2, 3, 4, 5])
        return inputs, mapping

    def test_noop_bitwise_and_no_hook_leaks(self):
        inputs, mapping = self.inputs()
        hooks = footprint(self.model)
        with torch.inference_mode():
            native = self.model(**inputs, use_cache=False).logits
            with AttentionPrefill(self.model, inputs, mapping, self.originals, noop=True):
                current = self.model(**inputs, use_cache=False).logits
        self.assertTrue(torch.equal(native, current))
        self.assertEqual(hooks, footprint(self.model))

    def test_exact_original_c1_and_locality(self):
        inputs, mapping = self.inputs()
        with torch.inference_mode():
            with AttentionPrefill(self.model, inputs, mapping, self.originals) as current:
                logits = self.model(**inputs, use_cache=False).logits
            layers = self.model.language_model.model.layers
            with self.originals['c2'].SinkStageTrace(layers, self.model.language_model.model.norm,
                    11, 0, [2, 3, 4, 5], [6, 7, 8], [0, 1, 6, 7, 8, 9, 10], [], 'C1') as old:
                expected = self.model(**inputs, use_cache=False, output_attentions=True).logits
            old.assert_complete()
        self.assertTrue(torch.equal(logits, expected))
        self.assertEqual([r['layer'] for r in current.records], [2, 3, 4])
        self.assertTrue(all(r['other_tokens_bitwise_exact'] and r['target_zero'] for r in current.records))

    def test_cache_generation_edits_prefill_only(self):
        inputs, mapping = self.inputs()
        hooks = footprint(self.model)
        with torch.inference_mode(), AttentionPrefill(self.model, inputs, mapping, self.originals) as trace:
            self.model.generate(**inputs, do_sample=False, use_cache=True, max_new_tokens=3, pad_token_id=2)
        self.assertEqual([r['layer'] for r in trace.records], [2, 3, 4])
        self.assertTrue(trace.completed)
        self.assertEqual(hooks, footprint(self.model))

    def test_exception_cleanup(self):
        inputs, mapping = self.inputs()
        hooks = footprint(self.model)
        with self.assertRaisesRegex(RuntimeError, 'expected'):
            with AttentionPrefill(self.model, inputs, mapping, self.originals):
                raise RuntimeError('expected')
        self.assertEqual(hooks, footprint(self.model))


class IntegrityTests(unittest.TestCase):
    def test_prefix_mismatch_rejected(self):
        a = {'audits': [{'kind': 'generation_prefill', 'prefix_sha256': {'0': 'a'}}]}
        assert_prefix(a, a)
        b = {'audits': [{'kind': 'generation_prefill', 'prefix_sha256': {0: 'changed'}}]}
        with self.assertRaises(AssertionError):
            assert_prefix(a, b)

    def test_raw_logit_baseline_mismatch_rejected_even_same_scores(self):
        row = dict(generated_text='yes', generated_token_ids=[7], generated_token_logp=[-1.], candidate_scores=[],
                   metrics={}, eos_observed=False, length_limit_reached=False, audits=[])
        passed, _ = exact_historical(row, row, {'Full_generation_0': np.zeros(8)}, {'Full_generation_0': np.ones(8)}, 'Full')
        self.assertFalse(passed)

    def test_atomic_checkpoint_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'a.json'
            atomic_json(path, {'n': 1})
            with self.assertRaises(FileExistsError):
                atomic_json(path, {'n': 2})
            self.assertEqual(json.loads(path.read_text()), {'n': 1})

    def test_cluster_bootstrap_groups_actual_images(self):
        stats = clustered_stats([[1.], [1.], [-1.]], ['same', 'same', 'different'], iterations=1000)[0]
        self.assertEqual(stats['images'], 2)
        self.assertEqual(stats['questions'], 3)
        self.assertAlmostEqual(stats['mean'], 1/3)
        self.assertLessEqual(stats['bonferroni8_familywise95ci'][0], stats['paired_image_cluster_95ci'][0])
        self.assertGreaterEqual(stats['bonferroni8_familywise95ci'][1], stats['paired_image_cluster_95ci'][1])

    def test_global_fallback_when_any_baseline_not_exact(self):
        for exact in (True, False):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = {'test': 'frozen'}
                for i in range(4):
                    directory = root/f'smoke/shard{i}'
                    directory.mkdir(parents=True)
                    (directory/'validation.json').write_text(json.dumps({'correctness_passed': True,
                        'config_sha256': digest(config), 'historical_baselines_exact': exact if i == 2 else True}))
                    (directory/'weight_audit.json').write_text(json.dumps({'passed': True}))
                with patch('cross_dataset_c1_c2.run.load_config', return_value=config):
                    decide(root)
                decision = json.loads((root/'decision.json').read_text())
                self.assertEqual(decision['reuse_history'], exact)
                self.assertEqual(decision['run_conditions'], ['C1'] if exact else ['Full', 'C1', 'C2'])


if __name__ == '__main__':
    unittest.main()

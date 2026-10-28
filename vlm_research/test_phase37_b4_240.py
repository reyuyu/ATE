"""Synthetic CPU checks, never research observations."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from test_knockout import tiny_model, tiny_inputs
from knockout_probe import model_layers
from phase3_7.pathways import observed_forward, footprint, state_hash, input_hash
from phase3_7_b4_240.observe import forward, group_masks
from phase3_7_b4_240.protocol import CONDITIONS, RULES, GROUPS, split_samples, atomic, digest, sha
from phase3_7_b4_240.run import checkpoint
from phase3_7_b4_240.statistics import exact_mcnemar, summarize, paired_intervals


def score_row(index, full=1., b4=1.):
    gain = b4-full
    row = dict(question_id=f'{index:04}', img_id=str(index), answer='right',
               cohort='confirmation224', full_margin=full, b4_margin=b4, gain=gain,
               gold_logp_change=gain, wrong_logp_change=0., right_minus_left_change=gain)
    for layer in (0, 1):
        for group, mass in zip(GROUPS, [.5, .2, .1, .2]):
            row[f'l{layer}_full_{group}_mass'] = mass
            row[f'l{layer}_b4_{group}_mass'] = mass
            row[f'l{layer}_local_{group}_redistribution'] = 0.
    return row


class BatchObserveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = tiny_model()

    def setUp(self):
        self.inputs = tiny_inputs()
        self.qpos = [6, 7, 8]
        self.apos = 10

    def run_forward(self, condition='C0', **kwargs):
        return forward(self.model, self.inputs, self.qpos, self.apos, [0], condition,
                       expected_visual=4, **kwargs)

    def test_only_original_b4_condition_and_layers(self):
        self.assertEqual(CONDITIONS, ['C0', 'C1'])
        self.assertEqual(RULES['layers'], [0, 1])
        value = self.run_forward('C1')
        self.assertEqual([r['applied'] for r in value['audit']['mask_records']], [i < 2 for i in range(32)])
        for r in value['audit']['local_redistribution']:
            self.assertEqual(r['query_indices'], [10])
            self.assertEqual(r['key_indices'], self.qpos)
            self.assertEqual(r['masked_mass_after'], 0.)
            self.assertTrue(r['non_target_queries_bitwise_exact'])

    def test_readonly_observer_exact_original_full_and_b4(self):
        for condition, original in [('C0', 'B0'), ('C1', 'B4')]:
            a = self.run_forward(condition)
            b = observed_forward(self.model, self.inputs, self.qpos, self.apos, original, expected_visual=4)
            self.assertTrue(torch.equal(a['text_logits'], b['text_logits']))
            for key in ['hidden', 'final_hidden', 'attention', 'mass']:
                np.testing.assert_array_equal(a['arrays'][key], b['arrays'][key])

    def test_noop_effective_text_logits_exact(self):
        a = self.run_forward()
        b = self.run_forward('C1', noop=True)
        self.assertTrue(torch.equal(a['text_logits'], b['text_logits']))
        np.testing.assert_array_equal(a['diagnostics']['mass'], b['diagnostics']['mass'])

    def test_question_all32_layers_and_final_exact(self):
        a = self.run_forward()
        b = self.run_forward('C1')
        self.assertEqual(a['diagnostics']['question_layer_sha256'], b['diagnostics']['question_layer_sha256'])
        self.assertEqual(len(a['diagnostics']['question_layer_sha256']), 32)
        np.testing.assert_array_equal(a['arrays']['final_hidden'][:-1], b['arrays']['final_hidden'][:-1])

    def test_four_groups_partition_not_combining_bos_with_other_text(self):
        masks = group_masks(self.inputs, self.qpos, [0], 99, 4)
        self.assertEqual([torch.where(m)[0].tolist() for m in masks],
                         [[2, 3, 4, 5], [6, 7, 8], [0], [1, 9, 10, 11]])
        value = self.run_forward('C1')
        mass = value['diagnostics']['mass']
        np.testing.assert_allclose(mass.sum(-1), 1, atol=1e-6)
        self.assertTrue((mass[:, :, 1] == 0).all())
        self.assertTrue((mass[:, :, 2] > 0).all())

    def test_local_redistribution_and_global_comparison_are_distinct(self):
        a = self.run_forward()
        b = self.run_forward('C1')
        before = b['diagnostics']['local_native_mass']
        after = b['diagnostics']['mass']
        np.testing.assert_array_equal(before[0], a['diagnostics']['mass'][0])
        np.testing.assert_allclose((after-before).sum(-1), 0, atol=1e-6)
        self.assertTrue((after[:, :, [0, 2, 3]] >= before[:, :, [0, 2, 3]]).all())

    def test_shifted_real_input_mapping(self):
        for key in ['input_ids', 'attention_mask']:
            prefix = torch.full((1, 1), 31 if key == 'input_ids' else 1, dtype=self.inputs[key].dtype)
            self.inputs[key] = torch.cat([prefix, self.inputs[key]], 1)
        value = forward(self.model, self.inputs, [7, 8, 9], 11, [1], 'C1', expected_visual=4)
        self.assertEqual(value['audit']['local_redistribution'][0]['query_indices'], [11])
        self.assertEqual(value['audit']['local_redistribution'][0]['key_indices'], [7, 8, 9])

    def test_invalid_condition_bos_and_overlap_fail_closed(self):
        with self.assertRaises(ValueError):
            self.run_forward('B2')
        for bos in [[], [6], [2]]:
            with self.assertRaises(ValueError):
                group_masks(self.inputs, self.qpos, bos, 99, 4)

    def test_normal_and_exception_cleanup_input_and_weight_bytes(self):
        before = footprint(self.model)
        weights = state_hash(self.model)
        input_stamp = input_hash(self.inputs)
        self.run_forward('C1')
        def fail(*args):
            raise RuntimeError('injected observer failure')
        handle = model_layers(self.model)[12].register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                self.run_forward('C1')
        finally:
            handle.remove()
        self.assertEqual(before, footprint(self.model))
        self.assertEqual(weights, state_hash(self.model))
        self.assertEqual(input_stamp, input_hash(self.inputs))


class BatchStatisticsTest(unittest.TestCase):
    def test_split_exact_complement_and_leading_zero_ids(self):
        data = [dict(question_id=f'{i:06}', img_id=str(i)) for i in range(240)]
        actual = split_samples(list(reversed(data)), [str(i) for i in range(16)])
        self.assertEqual(actual[0]['question_id'], '000000')
        self.assertEqual(sum(s['cohort'] == 'confirmation224' for s in actual), 224)
        self.assertEqual(sum(s['cohort'] == 'discovery16' for s in actual), 16)

    def test_duplicate_or_missing_frozen_sample_rejected(self):
        data = [dict(question_id=str(i), img_id=str(i)) for i in range(240)]
        for bad in [data[:-1], data[:-1] + [data[0]]]:
            with self.assertRaises(ValueError):
                split_samples(bad, [str(i) for i in range(16)])
        with self.assertRaises(ValueError):
            split_samples(data, [str(i) for i in range(15)])

    def test_exact_mcnemar_known_values_and_symmetry(self):
        self.assertEqual(exact_mcnemar(0, 0), 1)
        self.assertEqual(exact_mcnemar(0, 5), .0625)
        self.assertEqual(exact_mcnemar(1, 9), 22/1024)
        self.assertEqual(exact_mcnemar(9, 1), exact_mcnemar(1, 9))
        self.assertEqual(exact_mcnemar(2, 2), 1)
        with self.assertRaises(ValueError):
            exact_mcnemar(-1, 0)

    def test_paired_bootstrap_preserves_exact_zero_differences(self):
        rows = [score_row(i, (-1)**i, (-1)**i) for i in range(8)]
        ci = paired_intervals(rows, 'same', 200)
        self.assertEqual(ci['accuracy_difference'], [0, 0])
        self.assertEqual(ci['mean_gain'], [0, 0])
        self.assertEqual(ci, paired_intervals(rows, 'same', 200))

    def test_positive_margin_gain_not_equivalent_accuracy_improvement(self):
        rows = [score_row(0, -2., -1.), score_row(1, 2., 3.)]
        result = summarize(rows, 'positive', 100)
        self.assertEqual(result['mean_gain'], 1)
        self.assertEqual(result['accuracy_difference'], 0)
        self.assertEqual(result['recovered'], 0)

    def test_ties_wrong_harmed_and_recovered_orientation(self):
        rows = [score_row(0, 1., 0.), score_row(1, 0., 1.), score_row(2, -1., 0.)]
        result = summarize(rows, 'ties', 100)
        self.assertEqual(result['harmed'], 1)
        self.assertEqual(result['recovered'], 1)
        self.assertEqual(result['full_ties'], 1)
        self.assertEqual(result['b4_ties'], 2)
        self.assertEqual(result['accuracy_difference'], 0)

    def test_image_not_head_is_sampling_unit_and_empty_subset_honest(self):
        with self.assertRaises(ValueError):
            summarize([score_row(0), score_row(0)], 'duplicate', 100)
        self.assertEqual(summarize([], 'empty')['status'], 'empty')


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_hash_binding_identity_and_artifact_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'observation.npz'
            artifact.write_bytes(b'synthetic')
            sample = dict(question_id='001', img_id='007', cohort='confirmation224')
            body = dict(passed=True, binding='config', sample_sha256=digest(sample), conditions=CONDITIONS,
                        row=dict(sample), artifacts={'observation.npz': sha(artifact)})
            path = root / 'checkpoint.json'
            atomic(path, dict(body=body, sha256=digest(body)))
            self.assertEqual(checkpoint(path, 'config', sample, root), body)
            with self.assertRaises(ValueError):
                checkpoint(path, 'another-config', sample, root)
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                checkpoint(path, 'config', sample, root)

    def test_incomplete_condition_and_foreign_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = dict(question_id='001', img_id='007', cohort='confirmation224')
            body = dict(passed=True, binding='config', sample_sha256=digest(sample), conditions=['C0'],
                        row=dict(sample), artifacts={})
            path = root / 'checkpoint.json'
            atomic(path, dict(body=body, sha256=digest(body)))
            with self.assertRaises(ValueError):
                checkpoint(path, 'config', sample, root)


if __name__ == '__main__':
    unittest.main()

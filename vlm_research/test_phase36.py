"""CPU-only Phase3.6 protocol, restart, statistics and wrapper regression."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from phase3_6 import protocol as p
from phase3_6.analyze import aggregate, correlation, interval, paired_accuracy, rank


class Phase36ProtocolTests(unittest.TestCase):
    def test_exact_frozen_grid(self):
        self.assertEqual(len(p.CONDITIONS), 9)
        self.assertEqual(len(p.RESEARCH), 6)
        actual = {(c['source'], c['start'], c['strength']) for c in p.CONDITIONS if c['name'] in p.RESEARCH}
        self.assertEqual(actual, {(23,24,.5),(23,24,1),(24,24,.5),(24,24,1),(23,25,1),(24,25,1)})

    def test_shards_are_sorted_complete_disjoint_and_keep_leading_zero(self):
        rows = [{'question_id': f'{i:05d}'} for i in reversed(range(240))]
        shards = p.split(rows)
        self.assertEqual([len(s) for s in shards], [60]*4)
        self.assertEqual(sum(shards, []), [f'{i:05d}' for i in range(240)])

    def test_reject_missing_duplicate_cohort(self):
        rows = [{'question_id': str(i)} for i in range(240)]
        with self.assertRaises(ValueError): p.split(rows[:-1])
        rows[-1] = rows[0]
        with self.assertRaises(ValueError): p.split(rows)

    def fixture(self, root):
        sample = dict(question_id='001', image_sha256='frozen')
        artifact = root / 'artifact'
        artifact.write_bytes(b'actual-bytes')
        body = dict(binding='binding', sample_sha256=p.digest(sample), question_id='001', passed=True,
                    rows=[dict(condition=c['name']) for c in p.CONDITIONS], artifacts={'artifact': p.sha(artifact)})
        path = root / 'checkpoint.json'
        p.atomic(path, dict(body=body, sha256=p.digest(body)))
        return sample, path, body

    def test_valid_resume_and_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sample, path, body = self.fixture(root)
            self.assertEqual(p.checkpoint(path, 'binding', sample, root), body)
            (root / 'artifact').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'artifact'): p.checkpoint(path, 'binding', sample, root)

    def test_foreign_binding_and_changed_sample_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sample, path, body = self.fixture(root)
            with self.assertRaises(ValueError): p.checkpoint(path, 'other-shard-run', sample, root)
            with self.assertRaises(ValueError): p.checkpoint(path, 'binding', dict(sample, image_sha256='changed'), root)

    def test_missing_or_duplicate_condition_rejected_even_with_valid_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sample, path, body = self.fixture(root)
            body['rows'][-1] = body['rows'][0]
            p.atomic(path, dict(body=body, sha256=p.digest(body)))
            with self.assertRaisesRegex(ValueError, 'condition'): p.checkpoint(path, 'binding', sample, root)

    def test_checkpoint_envelope_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sample, path, body = self.fixture(root)
            p.atomic(path, dict(body=body, sha256='bad'))
            with self.assertRaises(ValueError): p.checkpoint(path, 'binding', sample, root)
            body['artifacts'] = {'../outside': 'bad'}
            p.atomic(path, dict(body=body, sha256=p.digest(body)))
            with self.assertRaises(ValueError): p.checkpoint(path, 'binding', sample, root)

    def test_no_aggregation_before_four_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(samples=[dict(question_id='001')], shards=[['001']]*4)
            with patch('phase3_6.analyze.protocol', return_value=(config, 'binding')):
                with self.assertRaises(FileNotFoundError): aggregate(root)
            self.assertFalse((root / 'per_sample.csv').exists())
            self.assertFalse((root / 'report.md').exists())

    def test_rank_ties_and_spearman(self):
        np.testing.assert_equal(rank([4,1,1,8]), [2,.5,.5,3])
        self.assertAlmostEqual(correlation(rank([1,2,2,4]), rank([4,3,3,1])), -1)
        self.assertIsNone(correlation([1,1,1],[2,3,4]))

    def test_ties_are_wrong_in_paired_statistics(self):
        result = paired_accuracy([1,1,-1,0], [0,-1,1,0])
        self.assertEqual(result['full_correct_to_wrong'], 2)
        self.assertEqual(result['full_wrong_to_correct'], 1)
        self.assertEqual(result['accuracy_difference'], -.25)
        self.assertEqual(result['exact_mcnemar_two_sided_p'], 1.)

    def test_image_paired_bootstrap_is_reproducible_and_zero_exact(self):
        a = interval([0,0,0,0])
        self.assertEqual((a['low'],a['high']), (0,0))
        self.assertEqual(interval([1,-1,2]), interval([1,-1,2]))


class Phase36WrapperTests(unittest.TestCase):
    def test_effective_text_capture_and_exception_cleanup(self):
        import torch
        from types import SimpleNamespace
        from phase3_6.run import evaluated
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.config = SimpleNamespace(image_token_index=7)
            def forward(self, **inputs):
                return SimpleNamespace(logits=torch.arange(24.).reshape(1,4,6))
        model = Model()
        inputs = dict(input_ids=torch.tensor([[7,7,3,4]]), attention_mask=torch.ones(1,4,dtype=torch.long))
        def fake_forward(m,x,*args,**kwargs):
            output = m(**x)
            return dict(prediction_logits=output.logits[0,2], question_hidden=torch.zeros(1))
        with patch('roi_sink_common.forward', side_effect=fake_forward):
            out = evaluated(model, inputs, [2], 2)
        self.assertEqual(list(out['text_logits'].shape), [2,6])
        self.assertEqual(len(model._forward_hooks), 0)
        with patch('roi_sink_common.forward', side_effect=RuntimeError('deliberate')):
            with self.assertRaisesRegex(RuntimeError, 'deliberate'): evaluated(model, inputs, [2], 2)
        self.assertEqual(len(model._forward_hooks), 0)

    def test_input_pollution_is_detected(self):
        import torch
        from types import SimpleNamespace
        from phase3_6.run import evaluated
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.config = SimpleNamespace(image_token_index=7)
            def forward(self, **inputs):
                return SimpleNamespace(logits=torch.zeros(1,3,8))
        model = Model()
        inputs = dict(input_ids=torch.tensor([[7,2,3]]), attention_mask=torch.ones(1,3,dtype=torch.long))
        def polluted(m,x,*args,**kwargs):
            m(**x); x['input_ids'][0,1] = 4
            return dict(prediction_logits=torch.zeros(8), question_hidden=torch.zeros(1))
        with patch('roi_sink_common.forward', side_effect=polluted):
            with self.assertRaisesRegex(AssertionError, 'contamination'): evaluated(model, inputs, [1], 1)
        self.assertEqual(len(model._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()

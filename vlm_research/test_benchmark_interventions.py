import copy
import hashlib
import unittest

import torch

from benchmark_interventions.hooks import ORIGINALS, original_modules, PrefillIntervention, footprint
from benchmark_interventions.run import check_prefix


class InterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.originals = original_modules({k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in ORIGINALS.items()})

    def test_original_mlp_only_one_query(self):
        x = torch.randn(1, 17, 8)
        before = x.clone()
        y, audit = self.originals['c2'].zero_query(x, 11)
        self.assertTrue(torch.equal(x, before))
        self.assertEqual(int(torch.count_nonzero(y[:, 11])), 0)
        self.assertTrue(torch.equal(x[:, :11], y[:, :11]))
        self.assertTrue(torch.equal(x[:, 12:], y[:, 12:]))
        self.assertTrue(audit['other_tokens_bitwise_exact'])

    def test_original_c2_reject_invalid_query(self):
        with self.assertRaises(ValueError):
            self.originals['c2'].zero_query(torch.ones(1, 3, 4), 3)

    def test_hash_change_rejected(self):
        with self.assertRaises(ValueError):
            original_modules(dict(b4='0'*64, c2='0'*64))

    def test_prefix_invariant(self):
        refs = {}
        check_prefix(refs, 'Full', dict(kind='gen', prefix_sha256={0: 'a'}))
        check_prefix(refs, 'B4', dict(kind='gen', prefix_sha256={0: 'a'}))
        with self.assertRaises(AssertionError):
            check_prefix(refs, 'C2', dict(kind='gen', prefix_sha256={0: 'changed'}))

    def test_only_three_conditions(self):
        with self.assertRaises(ValueError):
            PrefillIntervention(None, {}, {}, 'C1', self.originals)

    def test_duplicate_or_teacher_forced_question_rejected(self):
        x = {'input_ids': torch.tensor([[1, 2, 3, 4]])}
        for q in ([1, 1], [3], []):
            with self.assertRaises(ValueError):
                PrefillIntervention(None, x, {'p_minus_1': 3, 'question_positions': q}, 'B4', self.originals)

    def test_native_generation_cleanup_after_first_forward(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = torch.nn.Module()
                self.language_model.model = torch.nn.Module()
                self.language_model.model.layers = torch.nn.ModuleList([])

            def forward(self, x):
                return x+1

        model = Model()
        x = {'input_ids': torch.tensor([[1, 2, 3, 4]])}
        mapping = dict(p_minus_1=3, question_positions=[1], prompt_length=4, prompt_ids=[1, 2, 3, 4])
        before = footprint(model)
        with PrefillIntervention(model, x, mapping, 'Full', self.originals):
            self.assertTrue(torch.equal(model(torch.tensor([1])), torch.tensor([2])))
            self.assertEqual(footprint(model), before)
            model(torch.tensor([2]))
        self.assertEqual(footprint(model), before)


if __name__ == '__main__':
    unittest.main()

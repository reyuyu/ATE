"""CPU scope tests; no model loading or GPU allocation."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from .common import CONDITIONS, LAYERS, roles, target_rows
from .engine import RoleBypass
from ..qwen3vl_early_windows.experiment import hook_counts


class Tests(unittest.TestCase):
    def setUp(self):
        self.layout = dict(input_ids=[1, 2, 9, 9, 9, 10, 11, 12, 13, 14],
                           image_positions=[2, 3, 4], p_minus_1=9, question_positions=[6, 7])
        self.mapping = roles(self.layout, 10)

    def test_roles(self):
        self.assertEqual(self.mapping, {'I': [2, 3, 4], 'R': [6, 7, 8], 'A': [9]})

    def test_variable_visual_length(self):
        for count in (1, 5, 128):
            n = count + 6
            layout = dict(input_ids=[1] + [9]*count + [10, 4, 5, 6, 7],
                image_positions=list(range(1, count+1)), p_minus_1=n-1, question_positions=[count+2])
            self.assertEqual(len(roles(layout, 10)['I']), count)

    def test_reject_wrong_boundary(self):
        with self.assertRaises(AssertionError):
            roles(self.layout, 999)

    def test_disjoint_roles(self):
        combined = sum(self.mapping.values(), [])
        self.assertEqual(len(combined), len(set(combined)))

    def test_prefill_scope(self):
        for name in CONDITIONS:
            rows = target_rows(name, self.mapping, True)
            expected = {'Full': [], 'I-Bypass': [2, 3, 4], 'R-Bypass': [6, 7, 8],
                        'A-Bypass': [9], 'I+A-Bypass': [2, 3, 4, 9]}[name]
            self.assertEqual(rows, expected)

    def test_decode_scope(self):
        for name in CONDITIONS:
            self.assertEqual(target_rows(name, self.mapping, False),
                             [0] if name in ('A-Bypass', 'I+A-Bypass') else [])

    def test_fixed_layers(self):
        self.assertEqual(LAYERS, [1, 2, 3, 4, 5, 6])

    def model(self):
        model = nn.Module()
        model.model = nn.Module()
        model.model.language_model = nn.Module()
        model.model.language_model.layers = nn.ModuleList()
        for _ in range(36):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.o_proj = nn.Identity()
            model.model.language_model.layers.append(layer)
        return model

    def test_exact_edit_and_noop(self):
        model = self.model()
        for condition in CONDITIONS:
            for noop in (False, True):
                ctl = RoleBypass(model, condition, self.mapping, 10, noop)
                ctl.query_length, ctl.calls = 10, [{'phase': 'prefill'}]
                original = torch.arange(40).reshape(1, 10, 4).float() + 1
                result = ctl.edit(1, None, None, original)
                rows = target_rows(condition, self.mapping, True)
                expected = original.clone()
                if not noop:
                    expected[:, rows] = 0
                self.assertTrue(torch.equal(result, expected))
                self.assertTrue(torch.equal(original, torch.arange(40).reshape(1, 10, 4).float()+1))

    def test_hook_cleanup_exception(self):
        model = self.model()
        before = hook_counts(model)
        with self.assertRaisesRegex(RuntimeError, 'forced'):
            with RoleBypass(model, 'I+A-Bypass', self.mapping, 10):
                raise RuntimeError('forced')
        self.assertEqual(hook_counts(model), before)

    def test_hook_layers_only(self):
        model = self.model()
        with RoleBypass(model, 'R-Bypass', self.mapping, 10):
            self.assertEqual([i for i, layer in enumerate(model.model.language_model.layers)
                if layer.self_attn.o_proj._forward_hooks], LAYERS)

    def test_prefill_decode_positions(self):
        ctl = RoleBypass(self.model(), 'A-Bypass', self.mapping, 10)
        ctl.begin(None, (), {'input_ids': torch.ones(1, 10), 'cache_position': torch.arange(10)})
        ctl.begin(None, (), {'input_ids': torch.ones(1, 1), 'cache_position': torch.tensor([10])})
        self.assertEqual(ctl.calls[-1]['phase'], 'decode')

    def test_reject_shared_prefix_cache(self):
        ctl = RoleBypass(self.model(), 'A-Bypass', self.mapping, 10)
        with self.assertRaises(AssertionError):
            ctl.begin(None, (), {'input_ids': torch.ones(1, 1), 'cache_position': torch.tensor([10])})

    def test_bootstrap_zero(self):
        from .analyze import paired_bootstrap
        self.assertEqual(float(paired_bootstrap([0, 0], ['a', 'b'], 500).max()), 0)

    def test_bootstrap_cluster_pairing(self):
        from .analyze import paired_bootstrap
        import numpy as np
        # Duplicate questions in one image stay together; both clusters average zero.
        draws = paired_bootstrap([1, -1, 0], ['a', 'a', 'b'], 500)
        self.assertTrue(np.equal(draws, 0).all())

    def test_mcnemar(self):
        from .analyze import exact_mcnemar
        self.assertEqual(exact_mcnemar(0, 0), 1)
        self.assertEqual(exact_mcnemar(4, 0), .125)


if __name__ == '__main__':
    unittest.main(verbosity=2)

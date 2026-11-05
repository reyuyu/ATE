import unittest
from dataclasses import replace

import torch

from .common import LABELS, policy
from .engine import AttributionWriter
from ..middle10_recovery.engine import RecoveryWriter
from ..c3v2_middle_reaccess import tests as fixtures


class Tests(fixtures.Tests):
    def execute_dense(self, label, reference=False):
        c = (RecoveryWriter(self.a, replace(self.layout, cutoff=21), 'E3', 4, True, dense=True) if reference
             else AttributionWriter(self.a, self.layout, label, 4, True))
        calls, handles = [], []
        for i, layer in enumerate(self.a.layers):
            def count(module, args, output, i=i):
                calls.append((i, args[0].shape[1]))
            handles.append(layer.mlp.register_forward_hook(count))
        try:
            with torch.inference_mode(), c.context():
                first = c.forward(inputs_embeds=self.h).last_hidden_state.clone()
                pointers = [tuple(v.data_ptr() for v in p) for p in c.cache.entries]
                second = c.forward(inputs_embeds=torch.ones(1, 1, 32)).last_hidden_state.clone()
                self.assertEqual(pointers, [tuple(v.data_ptr() for v in p) for p in c.cache.entries])
        finally:
            for h in handles:
                h.remove()
        return first, second, c, calls

    def test_A_matches_frozen_E3_dense_exact(self):
        a, b = self.execute_dense('A'), self.execute_dense('A', reference=True)
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))

    def test_B_only_receiver_permissions(self):
        a, b = AttributionWriter(self.a, self.layout, 'A'), AttributionWriter(self.a, self.layout, 'B')
        for i in range(32):
            change = a.prefill_mask(i) != b.prefill_mask(i)
            expected = torch.zeros_like(change)
            if i > 20:
                for q in self.layout.relay:
                    expected[0, 0, q, list(self.layout.image)] = True
            self.assertTrue(torch.equal(change, expected))

    def test_prefill_D_E_native_exact(self):
        with torch.inference_mode():
            native = self.decoder(inputs_embeds=self.h, use_cache=False).last_hidden_state
        for label in ('D', 'E'):
            self.assertTrue(torch.equal(self.execute_dense(label)[0], native))

    def test_D_E_identical_prefill_and_only_decode_skip_differs(self):
        d, e = self.execute_dense('D'), self.execute_dense('E')
        self.assertTrue(torch.equal(d[0], e[0]))
        self.assertEqual(policy('D')['decode_attention_skip'], (2, 3, 4, 5))
        self.assertEqual(policy('E')['decode_attention_skip'], ())

    def test_no_physical_exit_full_mlp_and_dense_cache(self):
        for label in LABELS:
            first, second, c, calls = self.execute_dense(label)
            self.assertEqual(first.shape[1], 14)
            self.assertTrue(torch.isfinite(first).all() and torch.isfinite(second).all())
            self.assertEqual(c.cache.used, [15]*32)
            for i in range(32):
                self.assertEqual([n for l, n in calls if l == i], [14, 1])
            self.assertTrue(all(v['blocked_probability_max'] == 0 for v in c.events))

    def test_no_other_prefill_query_changed_when_restoring_p1(self):
        for a, b in (('A', 'C'), ('B', 'D')):
            x, y = self.execute_dense(a), self.execute_dense(b)
            for hx, hy in zip(x[2].hidden, y[2].hidden):
                self.assertTrue(torch.equal(hx[:, :-1], hy[:, :-1]))

    def test_cleanup_normal_exception_and_input_unchanged(self):
        methods = [b.self_attn.forward for b in self.a.layers]
        before = self.h.clone()
        for label in LABELS:
            self.execute_dense(label)
            self.assertEqual(methods, [b.self_attn.forward for b in self.a.layers])
            c = AttributionWriter(self.a, self.layout, label)
            with self.assertRaises(ValueError), c.context():
                raise ValueError('intentional cleanup check')
            self.assertEqual(methods, [b.self_attn.forward for b in self.a.layers])
        self.assertTrue(torch.equal(before, self.h))


if __name__ == '__main__':
    unittest.main(verbosity=2)

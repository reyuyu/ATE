import unittest
from dataclasses import replace

import torch

from .common import CUTOFF, KEEP, NEW, PREFILL_KEEP, WRITER_LAST
from .engine import RecoveryWriter
from ..ate_fulljoint_v2.engine import LAYERS
from ..c3v2_middle_reaccess import tests as frozen_fixture


class Tests(frozen_fixture.Tests):
    def execute_new(self, label, dense=False):
        layout = replace(self.layout, cutoff=CUTOFF[label])
        ctl = RecoveryWriter(self.a, layout, label, 4, True, dense)
        calls, handles = [], []
        for i, layer in enumerate(self.a.layers):
            for op in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'mlp'):
                target = layer.mlp if op == 'mlp' else getattr(layer.self_attn, op)
                def capture(m, a, o, i=i, op=op):
                    calls.append((i, op, a[0].shape[1]))
                handles.append(target.register_forward_hook(capture))
        try:
            with torch.inference_mode(), ctl.context():
                first = ctl.forward(inputs_embeds=self.h).last_hidden_state.clone()
                pointers = [None if p is None else tuple(v.data_ptr() for v in p)
                            for p in ctl.cache.entries]
                second = ctl.forward(inputs_embeds=torch.ones(1, 1, 32)).last_hidden_state.clone()
                self.assertEqual(pointers, [None if p is None else tuple(v.data_ptr() for v in p)
                                            for p in ctl.cache.entries])
        finally:
            for handle in handles:
                handle.remove()
        return first, second, ctl, calls

    def test_recovery_dense_equivalence(self):
        for label in NEW:
            with self.subTest(label=label):
                a, b = self.execute_new(label), self.execute_new(label, dense=True)
                torch.testing.assert_close(a[0], b[0].index_select(1, a[2].text_index),
                                           atol=3e-7, rtol=3e-6)
                torch.testing.assert_close(a[1], b[1], atol=3e-7, rtol=3e-6)
                self.assertTrue(torch.isfinite(a[0]).all() and torch.isfinite(a[1]).all())

    def test_recovery_mask_scope(self):
        baseline = self.execute(KEEP['C3+Middle10'], 21)[2]
        causal = torch.triu(torch.ones(14, 14, dtype=torch.bool), diagonal=1)
        for label in NEW:
            c = RecoveryWriter(self.a, replace(self.layout, cutoff=21), label, 4)
            for layer in range(21):
                mask = c.prefill_mask(layer)[0, 0]
                self.assertTrue((mask[causal] < -1e20).all())
                self.assertTrue((mask == 0).any(-1).all())
                changed = mask != baseline.prefill_mask(layer)[0, 0]
                expected = torch.zeros_like(changed)
                if label in ('E1', 'E3', 'E4') and layer == 20:
                    for q in self.layout.relay:
                        expected[q, list(self.layout.image)] = True
                if label in ('E2', 'E3', 'E4') and layer in (0, 1, 6, 7, 8, 9, 10):
                    expected[-1, list(self.layout.image)] = True
                if label == 'E5' and layer == 20:
                    expected[-1, list(self.layout.image)] = True
                self.assertTrue(torch.equal(changed, expected), (label, layer))

    def test_no_extra_persistent_image_kv(self):
        for label in NEW:
            _, _, c, _ = self.execute_new(label)
            info = c.cache.audit()
            self.assertEqual(info['retained_image_layers'], list(KEEP[label]))
            for i in LAYERS:
                self.assertIsNone(c.cache.entries[i])
            self.assertEqual(info['historical_cache_copies'], 0)
            self.assertEqual(info['append_concatenations'], 0)

    def test_e4_only_prefill_p1_attention_restored(self):
        e3, e4 = self.execute_new('E3'), self.execute_new('E4')
        for i in LAYERS:
            for op in ('q_proj', 'o_proj'):
                self.assertEqual([n for l, o, n in e3[3] if l == i and o == op], [13])
                self.assertEqual([n for l, o, n in e4[3] if l == i and o == op], [14])
            self.assertEqual([n for l, o, n in e4[3] if l == i and o == 'k_proj'], [14])
            self.assertIsNone(e4[2].cache.entries[i])
        for before, after in zip(e3[2].hidden, e4[2].hidden):
            self.assertTrue(torch.equal(before[:, :-1], after[:, :-1]))
        early = [e for e in e4[2].visual_events if e['prefill'] and e['layer'] in LAYERS]
        self.assertEqual(len(early), 4)
        self.assertTrue(all(e['p1_computed'] and e['answer_image_mass'] == 0 for e in early))

    def test_all_native_mlp_and_hidden_exit_shapes(self):
        for label in NEW:
            _, _, c, calls = self.execute_new(label)
            for i in range(32):
                n = 14 if i < 21 else 10
                self.assertEqual([s for l, op, s in calls if l == i and op == 'mlp'], [n, 1])
            self.assertTrue(all(e['blocked_probability_max'] == 0 for e in c.events))
            for event in c.visual_events:
                i = event['layer']
                if event['image_keys_present'] and event['p1_computed']:
                    allow = i in (PREFILL_KEEP[label] if event['prefill'] else KEEP[label])
                    self.assertEqual(event['answer_image_mass'] > 0, allow)
                if event['prefill']:
                    self.assertEqual(event['relay_image_mass'] > 0, i <= WRITER_LAST[label])

    def test_e5_exit_not_mask_change_relative_m9(self):
        m9 = self.execute(KEEP['C3+Middle9'], 20)
        e5 = self.execute_new('E5')
        torch.testing.assert_close(m9[0], e5[0], atol=3e-7, rtol=3e-6)
        torch.testing.assert_close(m9[1], e5[1], atol=3e-7, rtol=3e-6)

    def test_recovery_cleanup_exception(self):
        original = [l.self_attn.forward for l in self.a.layers]
        decoder = self.decoder.forward
        for label in NEW:
            c = RecoveryWriter(self.a, replace(self.layout, cutoff=21), label, 4)
            with self.assertRaises(ValueError):
                with c.context():
                    raise ValueError('intentional exception')
            self.assertEqual(original, [l.self_attn.forward for l in self.a.layers])
            self.assertEqual(decoder, self.decoder.forward)


if __name__ == '__main__':
    unittest.main(verbosity=2)

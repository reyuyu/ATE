import unittest
from unittest.mock import patch

import torch

from .common import KEEP
from .engine import RegionWriter, RegionReference
from . import engine
from ..c3v2_middle_reaccess import tests as fixtures


class Tests(fixtures.Tests):
    def region(self, label):
        c = RegionWriter(self.a, self.layout, label, 4, True)
        calls, handles = [], []
        for i, layer in enumerate(self.a.layers):
            def capture(module, args, output, i=i):
                calls.append((i, args[0].shape[1]))
            handles.append(layer.mlp.register_forward_hook(capture))
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

    def test_all_prefill_hidden_native_exact(self):
        with torch.inference_mode():
            full = self.decoder(inputs_embeds=self.h, use_cache=False).last_hidden_state
        results = [self.region(k) for k in KEEP]
        self.assertTrue(all(torch.equal(x[0], full) for x in results))
        for h in zip(*(x[2].hidden for x in results)):
            self.assertTrue(all(torch.equal(h[0], x) for x in h[1:]))

    def test_actual_probabilities_only_decode_image_blocked(self):
        for label in KEEP:
            _, _, c, _ = self.region(label)
            self.assertEqual(len(c.visual_events), 64)
            for e in c.visual_events:
                allowed = e['prefill'] or e['layer'] in KEEP[label]
                self.assertEqual(e['image_keys_permitted'], allowed)
                if allowed:
                    self.assertGreater(e['answer_image_mass'], 0)
                else:
                    self.assertEqual(e['answer_image_mass'], 0)
                if e['prefill']:
                    self.assertGreater(e['relay_image_mass'], 0)
            self.assertTrue(all(e['blocked_probability_max'] == 0 for e in c.events))

    def test_prefill_mask_native_causal(self):
        for label in KEEP:
            c = RegionWriter(self.a, self.layout, label)
            for i in range(32):
                m = c.prefill_mask(i)[0, 0]
                lower = torch.tril(torch.ones_like(m, dtype=torch.bool))
                self.assertTrue((m[lower] == 0).all())
                self.assertTrue((m[~lower] < -1e20).all())
            blocked = torch.isneginf(c.decode_mask[0, 0, 0, :self.layout.length])
            self.assertEqual(blocked.nonzero().flatten().tolist(), list(self.layout.image))

    def test_full_mlp_dense_positions_and_independent_cache(self):
        previous = None
        for label in KEEP:
            first, second, c, calls = self.region(label)
            self.assertTrue(torch.isfinite(first).all() and torch.isfinite(second).all())
            self.assertEqual(c.cache.used, [15]*32)
            self.assertEqual(c.layout.cutoff, 32)
            self.assertFalse(c.physical)
            for i in range(32):
                self.assertEqual([n for l, n in calls if l == i], [14, 1])
            if previous is not None:
                self.assertTrue(all(a[0].data_ptr() != b[0].data_ptr()
                                    for a, b in zip(previous.entries, c.cache.entries)))
            previous = c.cache

    def test_normal_exception_cleanup_and_inputs(self):
        original = [l.self_attn.forward for l in self.a.layers]
        decoder = self.decoder.forward
        before = self.h.clone()
        for label in KEEP:
            self.region(label)
            with self.assertRaises(ValueError), RegionWriter(self.a, self.layout, label).context():
                raise ValueError('intentional cleanup')
            hooks = [(len(l.self_attn._forward_pre_hooks), len(l.self_attn.o_proj._forward_hooks))
                     for l in self.a.layers]
            with self.assertRaises(ValueError), RegionReference(self.a, self.layout, label).context():
                raise ValueError('intentional cleanup')
            self.assertEqual(original, [l.self_attn.forward for l in self.a.layers])
            self.assertEqual(decoder, self.decoder.forward)
            self.assertEqual(hooks, [(len(l.self_attn._forward_pre_hooks), len(l.self_attn.o_proj._forward_hooks))
                                     for l in self.a.layers])
        self.assertTrue(torch.equal(before, self.h))

    def test_frozen_loop_factories_restore_and_export_real_positions(self):
        original = engine.frozen.AttributionWriter, engine.frozen.NativeMaskReference

        def loop(a, layout, label, **kwargs):
            c = engine.frozen.AttributionWriter(a, layout, label, audit=True)
            reference = engine.frozen.NativeMaskReference(a, layout, label)
            self.assertEqual(c.keep, frozenset(KEEP['Early']))
            self.assertEqual(reference.policy['decode_image_layers'], KEEP['Early'])
            with torch.inference_mode(), c.context():
                c.forward(inputs_embeds=self.h)
                c.forward(inputs_embeds=torch.ones(1, 1, 32))
            return {}, [], [], []

        with patch.object(engine.frozen, 'generate', loop):
            result = engine.generate(self.a, self.layout, 'Early', audit=True)
        self.assertEqual(result[0]['condition'], 'Early')
        self.assertEqual([r['position_ids'] for r in result[0]['rotary_audit']],
                         [list(range(14)), [14]])
        self.assertEqual(original, (engine.frozen.AttributionWriter, engine.frozen.NativeMaskReference))
        with patch.object(engine.frozen, 'generate', side_effect=ValueError('intentional')):
            with self.assertRaises(ValueError):
                engine.generate(self.a, self.layout, 'Late')
        self.assertEqual(original, (engine.frozen.AttributionWriter, engine.frozen.NativeMaskReference))


if __name__ == '__main__':
    unittest.main(verbosity=2)

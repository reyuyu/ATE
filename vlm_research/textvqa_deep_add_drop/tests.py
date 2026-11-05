import unittest

import torch

from .common import KEEP, LABELS, select_samples
from .engine import LayerWriter, LayerReference
from ..c3v2_middle_reaccess import tests as fixtures


class Tests(fixtures.Tests):
    def execute_layer(self, label):
        c = LayerWriter(self.a, self.layout, label, 4, True)
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

    def test_all22_native_prefill_and_exact_decode_image_scope(self):
        with torch.inference_mode():
            native = self.decoder(inputs_embeds=self.h, use_cache=False).last_hidden_state
        for label in LABELS:
            first, second, c, calls = self.execute_layer(label)
            self.assertTrue(torch.equal(first, native))
            self.assertTrue(torch.isfinite(first).all() and torch.isfinite(second).all())
            self.assertEqual(c.cache.used, [15]*32)
            self.assertFalse(c.physical)
            self.assertEqual(len(c.visual_events), 64)
            for i in range(32):
                self.assertEqual([n for l, n in calls if l == i], [14, 1])
                mask = c.prefill_mask(i)[0, 0]
                lower = torch.tril(torch.ones_like(mask, dtype=torch.bool))
                self.assertTrue((mask[lower] == 0).all())
                self.assertTrue((mask[~lower] < -1e20).all())
            for e in c.visual_events:
                allowed = e['prefill'] or e['layer'] in KEEP[label]
                self.assertEqual(e['image_keys_permitted'], allowed)
                if allowed:
                    self.assertGreater(e['answer_image_mass'], 0)
                else:
                    self.assertEqual(e['answer_image_mass'], 0)
            self.assertTrue(all(e['blocked_probability_max'] == 0 for e in c.events))

    def test_frozen_add_drop_sets(self):
        for i in range(21, 32):
            self.assertEqual(set(KEEP[f'Add-L{i}'])-set(range(11, 21)), {i})
            self.assertEqual(set(range(11, 32))-set(KEEP[f'Drop-L{i}']), {i})
            self.assertEqual(len(KEEP[f'Add-L{i}']), 11)
            self.assertEqual(len(KEEP[f'Drop-L{i}']), 20)

    def test_cleanup_exception_and_independent_state(self):
        methods = [l.self_attn.forward for l in self.a.layers]
        before = self.h.clone()
        previous = None
        for label in LABELS:
            _, _, c, _ = self.execute_layer(label)
            if previous is not None:
                self.assertTrue(all(a[0].data_ptr() != b[0].data_ptr()
                                    for a, b in zip(previous.entries, c.cache.entries)))
            previous = c.cache
            for control in (LayerWriter(self.a, self.layout, label), LayerReference(self.a, self.layout, label)):
                with self.assertRaises(ValueError), control.context():
                    raise ValueError('intentional cleanup')
            self.assertEqual(methods, [l.self_attn.forward for l in self.a.layers])
            self.assertTrue(all(not l.self_attn._forward_pre_hooks and not l.self_attn.o_proj._forward_hooks
                                for l in self.a.layers))
        self.assertTrue(torch.equal(before, self.h))

    def test_selection_clusters_and_not_scores(self):
        samples = {str(i): dict(image_sha256=f'{i//3:064x}', answer='unused') for i in range(501)}
        a, order, end = select_samples(samples)
        b, _, _ = select_samples({k: dict(v, score=i) for i, (k, v) in enumerate(reversed(list(samples.items())))})
        self.assertEqual(set(a), set(b))
        self.assertEqual(len(a), 201)
        chosen = set(order[:end])
        self.assertTrue(all((k in a) == (s['image_sha256'] in chosen) for k, s in samples.items()))


if __name__ == '__main__':
    unittest.main(verbosity=2)

import unittest

import torch

from .common import KEEP, FIRST_LAYER
from .engine import writer
from ..c3v2_middle_reaccess import tests as fixtures


class Tests(fixtures.Tests):
    def test_new_region_scope_and_native_prefill(self):
        self.assertEqual(KEEP, tuple(range(FIRST_LAYER, 25)))
        self.assertEqual(len(KEEP), 25-FIRST_LAYER)
        c = writer(self.a, self.layout, True)
        with torch.inference_mode():
            native = self.decoder(inputs_embeds=self.h, use_cache=False).last_hidden_state
            with c.context():
                first = c.forward(inputs_embeds=self.h).last_hidden_state.clone()
                pointers = [tuple(v.data_ptr() for v in p) for p in c.cache.entries]
                second = c.forward(inputs_embeds=torch.ones(1, 1, 32)).last_hidden_state
                self.assertEqual(pointers, [tuple(v.data_ptr() for v in p) for p in c.cache.entries])
        self.assertTrue(torch.equal(first, native))
        self.assertTrue(torch.isfinite(second).all())
        self.assertEqual(len(c.visual_events), 64)
        for event in c.visual_events:
            allowed = event['prefill'] or event['layer'] in KEEP
            self.assertEqual(event['image_keys_permitted'], allowed)
            if allowed:
                self.assertGreater(event['answer_image_mass'], 0)
            else:
                self.assertEqual(event['answer_image_mass'], 0)
        self.assertTrue(all(e['blocked_probability_max'] == 0 for e in c.events))
        self.assertEqual(c.cache.used, [15]*32)
        self.assertFalse(c.physical)
        self.assertFalse(c.policy['prefill_p1_attention_skip'])
        self.assertFalse(c.policy['decode_attention_skip'])

    def test_new_region_cleanup_exception(self):
        original = [b.self_attn.forward for b in self.a.layers]
        with self.assertRaises(ValueError), writer(self.a, self.layout).context():
            raise ValueError('intentional')
        self.assertEqual(original, [b.self_attn.forward for b in self.a.layers])


if __name__ == '__main__':
    unittest.main(verbosity=2)

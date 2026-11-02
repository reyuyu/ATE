import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.smoke import hook_footprint
from paper_rq1.engine import clone_cache, cache_fingerprint
from test_rq2_source_knockout import Tiny, inputs
from .common import gate, CONDITIONS
from .engine import forward, native_counterfactual, temporal_generate


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return ' '.join(map(str, tokens))


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261102)
        self.model = Tiny()
        cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
            num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, attention_dropout=0.)
        cfg._attn_implementation = 'eager'; cfg.image_token_index = 60
        self.model.config = cfg
        self.model.language_model = LlamaForCausalLM(cfg).eval()
        self.data, self.tokenizer = inputs(), Tokenizer()

    def test_gates(self):
        self.assertEqual([gate('Periodic25', 'a', t) for t in range(1, 9)], [True, False, False, False]*2)
        self.assertEqual([gate('Periodic50', 'a', t) for t in range(1, 5)], [True, False]*2)
        self.assertTrue(all(gate('Full', 'a', t) for t in range(1, 96)))
        self.assertFalse(any(gate('None', 'a', t) for t in range(1, 96)))
        for c in CONDITIONS:
            self.assertEqual([gate(c, 'a', t) for t in range(1, 96)], [gate(c, 'a', t) for t in range(1, 96)])
        with self.assertRaises(ValueError): gate('Full', 'a', 0)
        with self.assertRaises(ValueError): gate('new', 'a', 1)

    def test_cache_no_alias(self):
        _, cache, _, _ = forward(self.model, self.data, [], None)
        before = cache_fingerprint(cache)
        cloned = clone_cache(cache)
        for original, copied in zip(cache.key_cache+cache.value_cache, cloned.key_cache+cloned.value_cache):
            self.assertNotEqual(original.data_ptr(), copied.data_ptr())
        forward(self.model, self.data, [7], cloned, blocked=True)
        self.assertEqual(before, cache_fingerprint(cache))

    def test_noop_and_native_counterfactual(self):
        base = temporal_generate(self.model, self.tokenizer, self.data, 'a', 'Full', limit=4, audit=True)
        noop = temporal_generate(self.model, self.tokenizer, self.data, 'a', 'None', limit=4, audit=True, noop=True)
        self.assertEqual(base[0]['token_ids'], noop[0]['token_ids'])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(base[3], noop[3])))
        result, rows, full, off, _ = native_counterfactual(self.model, self.tokenizer, self.data, 4, audit=True, noop=True)
        self.assertEqual(result['token_ids'], base[0]['token_ids'])
        self.assertTrue((full == off).all())
        self.assertTrue(all(r['signed_logp_loss'] == 0 for r in rows))

    def test_scope_actual_zero_and_cleanup(self):
        foot = hook_footprint(self.model)
        ids = self.data['input_ids'].clone()
        result, rows, full, off, audit = native_counterfactual(self.model, self.tokenizer, self.data, 4, audit=True)
        self.assertTrue(all(x['image_probability_max'] == 0 for r in audit for x in r['ImageOFF']))
        self.assertTrue(all(x['image_probability_max'] > 0 for r in audit for x in r['Full']))
        self.assertEqual(foot, hook_footprint(self.model))
        self.assertTrue(torch.equal(ids, self.data['input_ids']))
        with patch.object(self.model, 'forward', side_effect=RuntimeError('intentional')):
            with self.assertRaises(RuntimeError): forward(self.model, self.data, [], None, blocked=True)
        self.assertEqual(foot, hook_footprint(self.model))

    def test_every_step_gates_and_prefill_exact(self):
        base = temporal_generate(self.model, self.tokenizer, self.data, 'a', 'Full', limit=6, audit=True)
        for c in CONDITIONS:
            g, first, records, _, debug = temporal_generate(self.model, self.tokenizer, self.data, 'a', c, limit=6, audit=True)
            self.assertTrue(torch.equal(base[4]['full_logits'], debug['full_logits']))
            self.assertEqual(base[0]['token_ids'][0], g['token_ids'][0])
            for row in records:
                self.assertEqual(row['on'], gate(c, 'a', row['t']))
                self.assertTrue(all(p['blocked'] == (not row['on']) for p in row['probabilities']))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

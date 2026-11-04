import types
import unittest

import torch

from .engine import FastHandoff, InplaceKV, Layout


class Tests(unittest.TestCase):
    def layout(self):
        return Layout(9, (2, 3, 4), (0, 1, 5, 6, 7, 8), (5, 6, 7), (6,), 1)

    def test_compact_cache_transient_and_capacity(self):
        cache = InplaceKV(1, self.layout(), 'Fast-P2', 4, 'cpu')
        k = torch.arange(72.).reshape(1, 2, 9, 4)
        a, b = cache.update(k, -k, 0)
        self.assertIs(a, k)
        self.assertTrue(torch.equal(cache.entries[0][0][..., :6, :], k[..., [0, 1, 5, 6, 7, 8], :]))
        pointer = cache.entries[0][0].data_ptr()
        cache.prefill = False
        cache.logical_length = 9
        for _ in range(3):
            a, b = cache.update(torch.ones(1, 2, 1, 4), torch.zeros(1, 2, 1, 4), 0)
            cache.logical_length += 1
            self.assertEqual(cache.entries[0][0].data_ptr(), pointer)
        self.assertEqual(a.shape[-2], 9)
        audit = cache.audit()
        self.assertEqual(audit['append_concatenations'], 0)
        self.assertEqual(audit['layers'][0]['original_positions'], [0, 1, 5, 6, 7, 8, 9, 10, 11])
        with self.assertRaisesRegex(RuntimeError, 'capacity'):
            cache.update(torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4), 0)

    def test_dense_and_trimmed_capacities(self):
        for condition, count in [('D1 Dense', 9), ('Fast-P1', 9), ('Fast-P2', 6)]:
            cache = InplaceKV(1, self.layout(), condition, 3, 'cpu')
            cache.update(torch.ones(1, 2, 9, 4), torch.ones(1, 2, 9, 4), 0)
            cache.logical_length = 9
            self.assertEqual(cache.used[0], count)
            self.assertEqual(cache.audit()['layers'][0]['capacity_tokens'], count + 2)

    def adapter(self):
        import transformers
        if transformers.__version__ == '4.57.6':
            from transformers import Qwen3VLTextConfig, Qwen3VLTextModel
            torch.manual_seed(4)
            cfg = Qwen3VLTextConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=100,
                rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]})
            cfg._attn_implementation = 'eager'
            decoder = Qwen3VLTextModel(cfg).eval()
            model = types.SimpleNamespace(model=types.SimpleNamespace(language_model=decoder))
            return types.SimpleNamespace(kind='qwen', model=model, layers=decoder.layers,
                                         inputs={'input_ids': torch.arange(9)[None]})
        from transformers import LlamaConfig, LlamaModel
        torch.manual_seed(4)
        cfg = LlamaConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, vocab_size=100,
                          attn_implementation='eager')
        decoder = LlamaModel(cfg).eval()
        model = types.SimpleNamespace(language_model=types.SimpleNamespace(model=decoder))
        return types.SimpleNamespace(kind='llava', model=model, layers=decoder.layers,
                                     inputs={'input_ids': torch.arange(9)[None]})

    def decoder(self, adapter):
        return (adapter.model.model.language_model if adapter.kind == 'qwen'
                else adapter.model.language_model.model)

    def positions(self, adapter, length, start=0):
        p = torch.arange(start, start + length)[None]
        return p[None].expand(3, 1, length) if adapter.kind == 'qwen' else p

    @torch.inference_mode()
    def test_native_noop_exact(self):
        adapter = self.adapter()
        decoder = self.decoder(adapter)
        embed = torch.randn(1, 9, 32)
        expected = decoder(inputs_embeds=embed, attention_mask=torch.ones(1, 9), use_cache=True,
                           position_ids=self.positions(adapter, 9))
        control = FastHandoff(adapter, self.layout(), 'Native-Noop', audit=True)
        original = decoder.forward
        with control.context():
            actual = decoder(inputs_embeds=embed, cache_position=torch.arange(9), use_cache=True,
                             position_ids=self.positions(adapter, 9))
        self.assertTrue(torch.equal(expected.last_hidden_state, actual.last_hidden_state))
        self.assertEqual(decoder.forward, original)
        self.assertEqual(len(control.audit_rows), 3)
        self.assertTrue(all(not l.self_attn._forward_hooks for l in adapter.layers))

    @torch.inference_mode()
    def test_prefill_and_decode_functional_equivalence(self):
        adapter = self.adapter()
        embed = torch.randn(1, 9, 32)
        decode = torch.randn(1, 1, 32)
        outputs = []
        for condition in ('D1 Dense', 'Fast-P1', 'Fast-P2'):
            control = FastHandoff(adapter, self.layout(), condition, audit=True)
            with control.context():
                prefill = control.forward(inputs_embeds=embed, cache_position=torch.arange(9),
                                          position_ids=self.positions(adapter, 9))
                after = control.forward(inputs_embeds=decode, cache_position=torch.tensor([9]),
                                        position_ids=self.positions(adapter, 1, 9))
            h = prefill.last_hidden_state
            outputs.append((h[:, [0, 1, 5, 6, 7, 8]] if condition == 'D1 Dense' else h,
                            after.last_hidden_state))
            self.assertTrue(all(row['blocked_probability_max'] == 0 for row in control.audit_rows))
        for actual in outputs[1:]:
            for x, y in zip(outputs[0], actual):
                self.assertTrue(torch.allclose(x, y, atol=2e-6, rtol=2e-5))

    def test_exception_cleanup(self):
        adapter = self.adapter()
        control = FastHandoff(adapter, self.layout(), 'Fast-P2', audit=True)
        decoder = self.decoder(adapter)
        original = decoder.forward
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with control.context():
                raise RuntimeError('injected')
        self.assertEqual(decoder.forward, original)
        self.assertTrue(all(not l.self_attn._forward_hooks for l in adapter.layers))

    def test_precomputed_masks(self):
        adapter = self.adapter()
        control = FastHandoff(adapter, self.layout(), 'D1 Dense')
        early, late = control.prefill_early[0, 0], control.prefill_late[0, 0]
        self.assertTrue(torch.isneginf(early[8, 2:5]).all())
        self.assertTrue(torch.isneginf(late[5:9, 2:5]).all())
        self.assertTrue((early.diag() == 0).all())
        self.assertTrue((late == 0).any(-1).all())
        self.assertTrue((early[torch.ones(9, 9, dtype=torch.bool).triu(1)] < 0).all())


if __name__ == '__main__':
    unittest.main()

"""CPU-only gates, run before CUDA is exposed."""
import unittest
from types import SimpleNamespace
import torch
from .engine import restore_columns
from ..qwen3vl_c3v2_physical.engine import CompactKV, Execution
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel


class Tests(unittest.TestCase):
    def test_restore_and_zero_probability(self):
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn_like(k)
        mask = torch.zeros(1, 1, 3, 3).masked_fill(torch.ones(3, 3).triu(1).bool(), -torch.inf)
        a, b, m, ix = restore_columns(k, v, mask, [0, 4, 6], 7)
        self.assertTrue(torch.equal(a[..., ix, :], k))
        self.assertTrue(torch.equal(b[..., ix, :], v))
        self.assertTrue(torch.equal(m[..., ix], mask))
        dead = [1, 2, 3, 5]
        self.assertEqual(a[..., dead, :].count_nonzero(), 0)
        p = torch.softmax(torch.randn(1, 2, 3, 7) + m, -1)
        self.assertEqual(p[..., dead].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(p).all())

    def test_invalid_mapping(self):
        k = torch.ones(1, 2, 3, 4)
        for positions in ([0, 0, 2], [-1, 1, 2], [0, 1, 4]):
            with self.assertRaises(ValueError):
                restore_columns(k, k, torch.zeros(1, 1, 3, 3), positions, 4)

    def test_compact_kv_positions_and_storage(self):
        cache = CompactKV(7, torch.tensor([0, 4, 6]), 6)
        k = torch.arange(28.).reshape(1, 1, 7, 4)
        first, _ = cache.update(k, k + 1, 0)
        self.assertTrue(torch.equal(first, k))
        saved = cache.entries[0][0].data_ptr()
        self.assertTrue(torch.equal(cache.entries[0][0][..., :3, :], k[..., [0, 4, 6], :]))
        cache.prefill = False
        second, _ = cache.update(torch.full((1, 1, 1, 4), 99.), torch.ones(1, 1, 1, 4), 0)
        self.assertEqual(second.shape[-2], 4)
        self.assertEqual(saved, cache.entries[0][0].data_ptr())
        self.assertEqual(float(second[0, 0, -1, 0]), 99.)

    def test_independent_cache(self):
        a = CompactKV(3, torch.tensor([0, 2]), 4)
        b = CompactKV(3, torch.tensor([0, 2]), 4)
        k = torch.ones(1, 1, 3, 4)
        a.update(k, k, 0)
        b.update(k, k, 0)
        self.assertNotEqual(a.entries[0][0].data_ptr(), b.entries[0][0].data_ptr())

    def test_native_noop_cleanup_on_exception(self):
        config = Qwen3VLTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=36,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                  vocab_size=64, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
        model = Qwen3VLTextModel(config)
        wrapper = SimpleNamespace(model=SimpleNamespace(language_model=model), parameters=model.parameters)
        layout = dict(input_length=5, image_positions=[1, 2])
        functions = [l.self_attn.forward.__func__ for l in model.layers]
        native_forward = model.forward.__func__
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with Execution(wrapper, layout, "N", noop=True).context():
                raise RuntimeError("intentional")
        self.assertEqual(native_forward, model.forward.__func__)
        self.assertEqual(functions, [l.self_attn.forward.__func__ for l in model.layers])

    def test_gate_frozen_and_signed_failures(self):
        from .run import GATE, compare
        self.assertEqual(GATE, dict(mean_abs=.02, max_abs=.5, top1_equal=True, complete_tokens_equal=True))
        a = dict(token_ids=[0])
        rows = compare(a, [torch.tensor([1., 0.])], a, [torch.tensor([1., .1])])
        self.assertFalse(rows[0]["pass"])


if __name__ == "__main__":
    unittest.main()

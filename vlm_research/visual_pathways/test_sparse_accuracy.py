"""Small CPU regression suite for persistent image-source masks."""
import unittest
import torch
from .sparse_accuracy import GenerationMask, conditions, KEEP, LOW
from .engine import masked
from .common import assign


class MaskTests(unittest.TestCase):
    def test_layer_sets(self):
        for model, n in [('qwen', 36), ('internvl', 28)]:
            c = conditions(model)
            self.assertEqual(len(c), 6)
            self.assertEqual(c['Full'], [])
            self.assertEqual(len(c['Only Text']), n)
            for name, keep in KEEP[model].items():
                self.assertEqual(set(range(n))-set(c[name]), set(keep))
            self.assertEqual(c['Low-8 OFF'], LOW[model])

    def test_prefill_local_mask(self):
        m = torch.full((1, 1, 7, 7), -torch.inf)
        m[0, 0] = torch.triu(m[0, 0], diagonal=1)
        out, count = masked(m, [6], [1, 2, 3])
        self.assertEqual(count, 3)
        self.assertTrue(torch.equal(out[:, :, :6], m[:, :, :6]))
        self.assertEqual(int((out != m).sum()), 3)
        self.assertTrue(torch.isfinite(out[0, 0, 6, [0, 4, 5, 6]]).all())

    def test_decode_mask(self):
        m = torch.zeros(1, 1, 1, 11)
        out, count = masked(m, [0], [1, 2, 3])
        self.assertEqual(count, 3)
        probs = out.softmax(-1)
        self.assertEqual(float(probs[..., [1, 2, 3]].max()), 0)
        self.assertEqual(float(probs.sum()), 1)

    def test_cluster_assignment(self):
        rows = [dict(image_sha256=str(i//3), question_id=str(i)) for i in range(30)]
        bins = assign(rows, 3)
        owners = {}
        for j, group in enumerate(bins):
            for row in group:
                self.assertEqual(owners.setdefault(row['image_sha256'], j), j)
        self.assertEqual(sum(map(len, bins)), 30)

    def test_persistent_hooks_and_cleanup(self):
        class Attention(torch.nn.Module):
            def forward(self, hidden_states, attention_mask):
                probs = attention_mask.softmax(-1).expand(1, 2, -1, -1)
                return hidden_states, probs
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()
            def forward(self, x, mask):
                return self.self_attn(hidden_states=x, attention_mask=mask)[0]
        class LM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(), Layer()])
            def forward(self, inputs_embeds, attention_mask, cache_position, past_key_values):
                x = inputs_embeds
                for layer in self.layers:
                    x = layer(x, attention_mask)
                return x
        class Cache:
            length = 0
            def get_seq_length(self, i):
                return self.length
        lm, cache = LM(), Cache()
        with GenerationMask(lm, lm.layers, 5, [1, 2], [0], cache) as edit:
            lm(inputs_embeds=torch.ones(1, 5, 4), attention_mask=torch.zeros(1, 1, 5, 5),
               cache_position=torch.arange(5), past_key_values=cache)
            cache.length = 5
            lm(inputs_embeds=torch.ones(1, 1, 4), attention_mask=torch.zeros(1, 1, 1, 6),
               cache_position=torch.tensor([5]), past_key_values=cache)
            self.assertEqual(edit.audits[0]['prefill_calls'], 1)
            self.assertEqual(edit.audits[0]['decode_calls'], 1)
            self.assertEqual(edit.audits[0]['masked_probability_max'], 0)
        self.assertFalse(lm._forward_pre_hooks)
        self.assertFalse(lm.layers[0].self_attn._forward_hooks)
        try:
            with GenerationMask(lm, lm.layers, 5, [1, 2], [0], cache):
                raise RuntimeError('fixture')
        except RuntimeError:
            pass
        self.assertFalse(lm.layers[0].self_attn._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()

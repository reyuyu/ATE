import unittest
from unittest.mock import patch
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import test_early_joint_attention as previous_tests
from paper_rq1_early_joint_attention.engine import generate
from decode_five_vs_six import NEW


class SixLayerTests(unittest.TestCase):
    def test_exact_layers_and_temporal_scope(self):
        helper = previous_tests.Tests()
        outputs, trace, model = helper.scoped(NEW['mode'], NEW['layers'])
        self.assertEqual(trace.edits, 6)
        for k in range(32):
            self.assertEqual(outputs[0][k].sum(), 24)
            self.assertEqual(outputs[1][k].sum(), 0 if k < 6 else 6)
        for block in model.layers:
            self.assertFalse(block.self_attn._forward_pre_hooks)
            self.assertFalse(block.self_attn.o_proj._forward_hooks)

    def test_real_cpu_six_cache_noop_and_first_token(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24,
            num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=128, attn_implementation='eager', eos_token_id=2)
        model = LlamaForCausalLM(config).eval()
        prompt = dict(input_ids=torch.tensor([[1, 5, 6, 7]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
        with patch('paper_rq1_early_joint_attention.engine.model_layers', lambda m: m.model.layers):
            full, logits = generate(model, prompt, dict(layers=[], mode='Full'), 4, collect=True)
            noop, no_logits = generate(model, prompt, NEW, 4, noop=True, collect=True)
            changed, trace = generate(model, prompt, NEW, 4, collect=True, audit=True)
            self.assertEqual(full['token_ids'], noop['token_ids'])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(logits, no_logits)))
            self.assertTrue(torch.equal(logits[0], trace[0]))
            self.assertEqual(full['token_ids'][0], changed['token_ids'][0])
            self.assertEqual(changed['edits'], 6 * changed['decode_steps'])


if __name__ == '__main__':
    unittest.main()

from contextlib import ExitStack
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import test_early_joint_attention as existing
from paper_rq1_early_joint_attention.engine import JointAttention, generate
from first_answer_attention.engine import supplement_generate
from four_layer_attention.run import CONDITIONS, NEW
from paper_rq1_early_joint_attention.statistics import bootstrap


class Tests(unittest.TestCase):
    def test_exact_29_windows_and_reuse(self):
        self.assertEqual(len(CONDITIONS), 29)
        self.assertEqual(len(NEW), 28)
        self.assertNotIn('W02', NEW)
        for start, c in enumerate(CONDITIONS.values()):
            self.assertEqual(c, dict(layers=list(range(start, start + 4)), decode=True))
        self.assertEqual(CONDITIONS['W28']['layers'], [28, 29, 30, 31])

    def test_all29_scope_and_cleanup(self):
        helper = existing.Tests()
        for c in CONDITIONS.values():
            model = helper.model()
            with patch('paper_rq1_early_joint_attention.engine.model_layers', lambda m: m.layers):
                with ExitStack() as stack:
                    stack.enter_context(JointAttention(model, c['layers'], 'P-1-only', 4, audit=True))
                    stack.enter_context(JointAttention(model, c['layers'], 'Decode-only', 4, audit=True))
                    for k in range(32):
                        p, d = helper.call(model, k), helper.call(model, k, True)
                        self.assertTrue(torch.equal(p[:, :-1], torch.full((1, 3, 3), 2.)))
                        self.assertEqual(float(p[:, -1].sum()), 0 if k in c['layers'] else 6)
                        self.assertEqual(float(d.sum()), 0 if k in c['layers'] else 6)
            for layer in model.layers:
                self.assertFalse(layer.self_attn._forward_pre_hooks)
                self.assertFalse(layer.self_attn.o_proj._forward_hooks)

    def test_family29_correction_includes_reused(self):
        x = np.tile(np.array([1., 0., -1., 0.])[:, None], (1, 29))
        result = bootstrap(x, list('abcd'), [29] * 29, repetitions=10000, seed=1)
        self.assertEqual(len(result), 29)
        for r in result:
            self.assertEqual(r['family'], 29)
            self.assertLessEqual(r['family95_low'], r['ci95_low'])
            self.assertGreaterEqual(r['family95_high'], r['ci95_high'])

    def test_real_cpu_early_mid_late_cache(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24, num_hidden_layers=32,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
            attn_implementation='eager', eos_token_id=2)
        model = LlamaForCausalLM(config).eval()
        prompt = dict(input_ids=torch.tensor([[1, 5, 6, 7]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
        with patch('paper_rq1_early_joint_attention.engine.model_layers', lambda m: m.model.layers):
            full, logits = generate(model, prompt, dict(layers=[], mode='Full'), limit=4, collect=True)
            for key in ['W00', 'W02', 'W14', 'W28']:
                g, first, trace = supplement_generate(model, prompt, CONDITIONS[key], noop=True, limit=4, collect=True)
                self.assertEqual(g['token_ids'], full['token_ids'])
                self.assertTrue(torch.equal(first, logits[0]))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(logits, trace)))
                changed, first, _ = supplement_generate(model, prompt, CONDITIONS[key], limit=4, audit=True)
                self.assertEqual(changed['edits'], 4 * changed['length'])
                self.assertEqual(changed['token_ids'][0], int(first.argmax()))
            recovered, _ = generate(model, prompt, dict(layers=[], mode='Full'), limit=4)
            self.assertEqual(recovered['token_ids'], full['token_ids'])
        self.assertFalse(model._forward_hooks)

    def test_bad_gpu_is_not_an_infinite_wait(self):
        from four_layer_attention.run import wait_for_idle
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'wrong'}):
            with self.assertRaises(ValueError): wait_for_idle(0)

    def test_summary_window_indices_and_reuse(self):
        from four_layer_attention.analyze import summarize
        g = dict(token_ids=[5, 2], text='yes', eos=True, truncated=False, length=2)
        rows = summarize(np.ones((4, 30)), [[dict(g) for _ in range(30)] for _ in range(4)],
            'test', np.array(list('abcd')))
        self.assertEqual(len(rows), 30)
        self.assertEqual((rows[1]['start_layer'], rows[1]['end_layer']), (0, 3))
        self.assertEqual((rows[29]['start_layer'], rows[29]['end_layer']), (28, 31))
        self.assertEqual(rows[3]['condition'], 'W02')
        self.assertEqual(rows[3]['source'], 'reused-E')
        self.assertTrue(all(r['delta_pp'] == 0 for r in rows))


if __name__ == '__main__': unittest.main()

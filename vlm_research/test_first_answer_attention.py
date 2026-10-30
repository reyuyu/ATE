from contextlib import ExitStack
import unittest
from unittest.mock import patch
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import test_early_joint_attention as existing
from paper_rq1_early_joint_attention.engine import JointAttention, generate
from first_answer_attention.engine import CONDITIONS, supplement_generate, first_metrics


class Tests(unittest.TestCase):
    def test_exact_five_conditions(self):
        from first_answer_attention.run import TASKS, cohort
        smoke_samples = [next(s for s in cohort() if s['task'] == t) for t in list(TASKS)[:2]]
        self.assertEqual(len(smoke_samples), 2)
        self.assertEqual(list(CONDITIONS), list('ABCDE'))
        self.assertEqual(CONDITIONS['D'], dict(layers=list(range(6)), decode=True))
        self.assertEqual(CONDITIONS['E'], dict(layers=[2, 3, 4, 5], decode=True))

    def test_composed_temporal_scope_and_cleanup(self):
        helper = existing.Tests()
        for c in CONDITIONS.values():
            model = helper.model()
            with patch('paper_rq1_early_joint_attention.engine.model_layers', lambda m: m.layers):
                with ExitStack() as stack:
                    stack.enter_context(JointAttention(model, c['layers'], 'P-1-only', 4, audit=True))
                    if c['decode']:
                        stack.enter_context(JointAttention(model, c['layers'], 'Decode-only', 4, audit=True))
                    for k in range(32):
                        prefill, decode = helper.call(model, k), helper.call(model, k, True)
                        self.assertTrue(torch.equal(prefill[:, :-1], torch.full((1, 3, 3), 2.)))
                        self.assertEqual(float(prefill[:, -1].sum()), 0 if k in c['layers'] else 6)
                        self.assertEqual(float(decode.sum()), 0 if k in c['layers'] and c['decode'] else 6)
            for layer in model.layers:
                self.assertFalse(layer.self_attn._forward_pre_hooks)
                self.assertFalse(layer.self_attn.o_proj._forward_hooks)

    def test_first_metrics(self):
        full, changed = torch.tensor([2., 0., 1.]), torch.tensor([0., 2., 1.])
        m = first_metrics(full, changed)
        self.assertTrue(m['first_top1_flip'])
        self.assertAlmostEqual(m['first_logp_loss'], 2., places=6)
        self.assertEqual(first_metrics(full, full)['first_logp_loss'], 0.)
        with self.assertRaises(AssertionError): first_metrics(full, changed * float('nan'))

    def test_real_cpu_cache_and_observer(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24, num_hidden_layers=32,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
            attn_implementation='eager', eos_token_id=2)
        model = LlamaForCausalLM(config).eval()
        prompt = dict(input_ids=torch.tensor([[1, 5, 6, 7]]), attention_mask=torch.ones(1, 4, dtype=torch.long))
        with patch('paper_rq1_early_joint_attention.engine.model_layers', lambda m: m.model.layers):
            full, logits = generate(model, prompt, dict(layers=[], mode='Full'), limit=4, collect=True)
            outputs = {}
            for key, c in CONDITIONS.items():
                noop, first, trace = supplement_generate(model, prompt, c, noop=True, limit=4, collect=True)
                self.assertEqual(noop['token_ids'], full['token_ids'])
                self.assertTrue(torch.equal(first, logits[0]))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(trace, logits)))
                changed, first, _ = supplement_generate(model, prompt, c, limit=4, audit=True)
                self.assertEqual(changed['token_ids'][0], int(first.argmax()))
                outputs[key] = first
            self.assertTrue(torch.equal(outputs['A'], outputs['C']))
            self.assertTrue(torch.equal(outputs['B'], outputs['D']))
            again, _ = generate(model, prompt, dict(layers=[], mode='Full'), limit=4)
            self.assertEqual(full['token_ids'], again['token_ids'])
        self.assertFalse(model._forward_hooks)
        for layer in model.model.layers:
            self.assertFalse(layer.self_attn._forward_pre_hooks)
            self.assertFalse(layer.self_attn.o_proj._forward_hooks)

    def test_wrapper_exception_cleanup(self):
        helper = existing.Tests()
        model = helper.model()
        model.register_forward_hook = lambda callback: type('H', (), {'remove': lambda self: None})()
        prompt = {'input_ids': torch.ones(1, 4, dtype=torch.long)}
        with patch('paper_rq1_early_joint_attention.engine.model_layers', lambda m: m.layers):
            with patch('first_answer_attention.engine.generate', side_effect=RuntimeError('injected')):
                with self.assertRaises(RuntimeError): supplement_generate(model, prompt, CONDITIONS['D'])
        for layer in model.layers:
            self.assertFalse(layer.self_attn._forward_pre_hooks)
            self.assertFalse(layer.self_attn.o_proj._forward_hooks)


if __name__ == '__main__': unittest.main()

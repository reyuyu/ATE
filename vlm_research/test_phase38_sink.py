"""CPU tests of the actual Transformers 4.49 LLaMA residual computation."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from phase38_sink_hooks import CONDITIONS, SinkStageTrace, bos_mask, zero_query, vector_stats


class Phase38Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3801)
        c = LlamaConfig(vocab_size=40, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
                        attention_dropout=0.)
        c._attn_implementation = 'eager'
        self.model = LlamaForCausalLM(c).eval()
        self.ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9]])

    def trace(self, condition):
        return SinkStageTrace(self.model.model.layers, self.model.model.norm, 6, 0, [1, 2], [3, 4], [0, 3, 4, 5], [1], condition)

    def run_model(self):
        return self.model(self.ids, attention_mask=torch.ones_like(self.ids), use_cache=False, output_hidden_states=True)

    def test_observation_exact_native_and_cleanup(self):
        with torch.inference_mode():
            native = self.run_model()
            with self.trace('C0') as t:
                observed = self.run_model(); t.assert_complete()
            self.assertTrue(torch.equal(native.logits, observed.logits))
            self.assertTrue(torch.equal(native.logits, self.run_model().logits))
        self.assertEqual(len(t.metrics), 8)
        self.assertEqual(t.edits, [])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_attention_bypass_exact_residual_mlp_still_runs(self):
        with torch.inference_mode(), self.trace('C1') as t:
            self.run_model(); t.assert_complete()
        self.assertEqual([e['layer'] for e in t.edits], [2, 3, 4])
        for i in (2, 3, 4):
            self.assertTrue(torch.equal(t.mid[i], t.inputs[i]))
            self.assertTrue(torch.count_nonzero(t.mlp_updates[i]))
            np.testing.assert_array_equal(t.hidden[i], (t.inputs[i]+t.mlp_updates[i]).numpy())

    def test_mlp_bypass_retains_attention_update(self):
        with torch.inference_mode(), self.trace('C2') as t:
            self.run_model(); t.assert_complete()
        for i in (2, 3, 4):
            self.assertTrue(torch.count_nonzero(t.attention_updates[i]))
            self.assertFalse(torch.count_nonzero(t.mlp_updates[i]))
            np.testing.assert_array_equal(t.hidden[i], t.mid[i].numpy())
        self.assertEqual([e['kind'] for e in t.edits], ['mlp_before_residual']*3)

    def test_bos_only_softmax_and_non_target_causal_invariants(self):
        with torch.inference_mode(), self.trace('C0') as native:
            baseline = self.run_model()
        for name in list(CONDITIONS)[1:]:
            with self.subTest(name=name), torch.inference_mode(), self.trace(name) as t:
                result = self.run_model(); t.assert_complete()
            self.assertEqual(t.prefix_hashes, native.prefix_hashes)
            np.testing.assert_array_equal(t.hidden[1], native.hidden[1])
            self.assertTrue(torch.equal(baseline.logits[:, :6], result.logits[:, :6]))
            if name == 'C3':
                self.assertEqual(len(t.edits), 6)
                for i in (2, 3, 4):
                    self.assertEqual(t.metrics[i]['bos_mass'], 1.)
                    self.assertEqual(t.metrics[i]['image_mass'], 0.)
            if name == 'C4':self.assertEqual([e['layer'] for e in t.edits], [5, 6, 7])

    def test_mask_preserves_every_other_query_and_original(self):
        original = torch.triu(torch.full((1, 1, 9, 9), torch.finfo(torch.float16).min, dtype=torch.float16), diagonal=1)
        before = original.clone()
        out, audit = bos_mask(original, 7, 0)
        self.assertTrue(torch.equal(original, before))
        self.assertTrue(torch.equal(out[:, :, :7], original[:, :, :7]))
        self.assertTrue(torch.equal(out[:, :, 8:], original[:, :, 8:]))
        self.assertEqual(torch.where(out[0, 0, 7] == 0)[0].tolist(), [0])
        self.assertTrue(audit['other_queries_bitwise_exact'])

    def test_invalid_masks_and_outputs(self):
        for mask in [None, torch.zeros(3, 3), torch.zeros(1, 1, 3, 3), torch.full((1, 1, 3, 3), float('nan'))]:
            with self.subTest(mask=mask), self.assertRaises(ValueError):bos_mask(mask, 2, 0)
        with self.assertRaises(ValueError):zero_query(torch.ones(2, 3, 4), 1)
        with self.assertRaises(AssertionError):zero_query(torch.full((1, 3, 4), float('nan')), 1)

    def test_zero_query_is_not_inplace_and_preserves_future_token(self):
        x = torch.randn(1, 9, 8); before = x.clone()
        y, audit = zero_query(x, 4)
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(x[:, :4], y[:, :4]))
        self.assertTrue(torch.equal(x[:, 5:], y[:, 5:]))
        self.assertFalse(torch.count_nonzero(y[:, 4]))
        self.assertTrue(audit['target_zero'])

    def test_exception_cleanup_and_other_instance_isolation(self):
        other = copy.deepcopy(self.model)
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, 'deliberate'), self.trace('C3'):
            raise RuntimeError('deliberate')
        with torch.inference_mode():native = self.run_model()
        with torch.inference_mode(), self.trace('C1'):
            observed = other(self.ids, attention_mask=torch.ones_like(self.ids), use_cache=False)
        self.assertTrue(torch.equal(native.logits, observed.logits))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.model.state_dict().items()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_failure_during_forward_removes_all_hooks(self):
        def fail(m, args, out):raise RuntimeError('deliberate forward failure')
        h = self.model.model.layers[3].mlp.register_forward_hook(fail)
        with self.assertRaisesRegex(RuntimeError, 'deliberate forward failure'), torch.inference_mode(), self.trace('C1'):
            self.run_model()
        h.remove()
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_update_norm_direction_uses_actual_residual(self):
        r = vector_stats(torch.tensor([3., 4.]), torch.tensor([6., 8.]))
        self.assertEqual(r['norm'], 5.)
        self.assertEqual(r['relative_norm'], .5)
        self.assertEqual(r['cosine'], 1.)
        self.assertIsNone(vector_stats(torch.zeros(2), torch.ones(2))['cosine'])

    def test_mass_partition_and_headwise_sink_denominator(self):
        with torch.inference_mode(), self.trace('C0') as t:
            self.run_model(); t.assert_complete()
        for i,r in t.metrics.items():
            self.assertAlmostEqual(sum(r[k] for k in ['bos_mass','image_mass','question_mass','other_text_mass']),1.,places=6)
            row=t.rows[i].astype(np.float64)
            expected=np.mean(row[:,2]/row[:,[1,2]].sum(-1))
            self.assertAlmostEqual(r['sink_inside_image'],expected,places=3)

    def test_drift_and_summary_tie_rule(self):
        from phase38_sink_stage import drift
        from summarize_phase38_sink import stats
        self.assertEqual(drift([1.,2.],[1.,2.]),dict(l2=0.,relative_l2=0.,cosine_distance=0.))
        row=dict(gain=-.1,gold_logp_gain=-.2,correct=False,correct_to_wrong=True,wrong_to_correct=False)
        s=stats([row])
        self.assertEqual(s['accuracy'],0.)
        self.assertEqual(s['correct_to_wrong'],1)
        self.assertEqual(s['negative_gain'],1)

    def test_grid_and_selection_frozen(self):
        from phase38_sink_stage import choose_samples, ROOT
        self.assertEqual(list(CONDITIONS), ['C0','C1','C2','C3','C4'])
        a, rule = choose_samples(); b, _ = choose_samples()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 16)
        self.assertEqual(sum(s['answer']=='left' for s in a), 8)
        self.assertEqual(sum(s['answer']=='right' for s in a), 8)
        self.assertNotIn('15133706', [s['question_id'] for s in a])
        right = [s for s in a if s['answer']=='right']
        self.assertTrue(any(s['selection_historical_full_margin']>0 for s in right))
        self.assertTrue(any(s['selection_historical_full_margin']<=0 for s in right))


if __name__ == '__main__':unittest.main()

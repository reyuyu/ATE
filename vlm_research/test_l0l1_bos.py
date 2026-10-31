import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.smoke import hook_footprint
from first_answer_attention.engine import first_metrics, supplement_generate
from l0l1_bos.analyze import contrasts
from l0l1_bos.capture import prefill_capture, vector_change, head_mass
from l0l1_bos.protocol import CONDITION, cases, old
from l0l1_bos.run import first_metrics_from_history, smoke_samples


class Tests(unittest.TestCase):
    def test_only_new_condition(self):
        self.assertEqual(CONDITION, dict(layers=[0, 1], decode=True))

    def test_real_task_mapping_smoke_selection(self):
        samples = smoke_samples()
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]['task'], list(old.TASKS)[0])
        self.assertEqual(samples[1]['task'], list(old.TASKS)[-1])

    def test_cases_outcome_blind(self):
        samples = old.cohort()
        chosen = [(r['task'], r['question_id']) for r in cases(samples)]
        changed = [{**s, 'answer': 'irrelevant', 'score': -1} for s in samples]
        self.assertEqual(chosen, [(r['task'], r['question_id']) for r in cases(changed)])
        self.assertEqual(len(chosen), 12)
        for task in old.TASKS:
            self.assertEqual(sum(t == task for t, q in chosen), 2)

    def test_reused_logp_formula(self):
        full, off = torch.tensor([1., 3., 0.]), torch.tensor([2., 1., -1.])
        reference = first_metrics(full, full)
        reference['full_first_logits_sha256'] = 'test'
        computed = first_metrics_from_history(reference, off)
        for key, value in first_metrics(full, off).items():
            self.assertEqual(computed[key], value)

    def test_interaction_sign_and_pairing(self):
        additive = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(float(contrasts(additive)[:, 4].mean()), 0.)
        synergistic = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
        self.assertEqual(float(contrasts(synergistic)[:, 4].mean()), .25)
        self.assertEqual(float(contrasts(synergistic)[:, 3].mean()), .5)

    def test_mass_disjoint_and_vector_metrics(self):
        keys = dict(bos=[0], image=[1, 2], prompt=[3, 4])
        row = np.array([[.5, .1, .2, .1, .1], [.1, .3, .3, .2, .1]])
        mass = head_mass(row, keys)
        np.testing.assert_allclose(sum(mass.values()), 1.)
        self.assertTrue(vector_change(row, row)['exact'])
        self.assertEqual(vector_change(row, row)['l2_change'], 0.)
        self.assertAlmostEqual(vector_change(row, 2*row)['relative_l2_change'], 1.)

    def model(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        config = LlamaConfig(vocab_size=40, hidden_size=16, intermediate_size=24, num_hidden_layers=32,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
            attn_implementation='eager', eos_token_id=2)
        return LlamaForCausalLM(config).eval()

    def test_capture_actual_prefill_and_kv(self):
        model = self.model()
        prompt = dict(input_ids=torch.tensor([[1, 4, 5, 6, 7, 8]]), attention_mask=torch.ones(1, 6, dtype=torch.long))
        keys = dict(bos=[0], image=[1, 2], prompt=[3, 4, 5])
        fp = hook_footprint(model)
        with patch('l0l1_bos.capture.model_layers', lambda m: m.model.layers), patch('paper_rq1_early_joint_attention.engine.model_layers', lambda m: m.model.layers):
            with torch.inference_mode():
                native = model(**prompt, use_cache=True).logits[0, -1].float()
            full, a, stats = prefill_capture(model, prompt, keys)
            off, b, changed = prefill_capture(model, prompt, keys, off=True)
            g, first, trace = supplement_generate(model, prompt, CONDITION, collect=True, limit=4)
            self.assertTrue(torch.equal(native, full))
            self.assertTrue(torch.equal(off, first))
            self.assertEqual(int(off.argmax()), g['token_ids'][0])
            self.assertEqual(a['attention_row_0'].shape, (2, 6))
            np.testing.assert_array_equal(a['attention_row_0'], b['attention_row_0'])
            self.assertEqual(changed[0]['attention_output_norm'], 0.)
            self.assertEqual(changed[1]['attention_output_norm'], 0.)
            self.assertFalse(np.array_equal(a['hidden_in_2'], b['hidden_in_2']))
            for layer in range(2, 6):
                for name in ('key', 'value'):
                    np.testing.assert_array_equal(a[f'bos_{name}_{layer}'], b[f'bos_{name}_{layer}'])
            self.assertEqual(g['edits'], 2*g['length'])
        self.assertEqual(hook_footprint(model), fp)

    def test_capture_exception_cleanup(self):
        model = self.model()
        prompt = dict(input_ids=torch.tensor([[1, 4, 5]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
        keys = dict(bos=[0], image=[1], prompt=[2])
        fp = hook_footprint(model)
        with patch('l0l1_bos.capture.model_layers', lambda m: m.model.layers), patch('paper_rq1_early_joint_attention.engine.model_layers', lambda m: m.model.layers):
            with patch.object(model.model.layers[1].self_attn.q_proj, 'forward', side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    prefill_capture(model, prompt, keys, off=True)
        self.assertEqual(hook_footprint(model), fp)


if __name__ == '__main__':
    unittest.main()

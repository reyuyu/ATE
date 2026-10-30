import unittest
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
import transformers.models.llama.modeling_llama as llama
from paper_rq1_token_role_decode.protocol import MODEL_PATH, LAYERS, TYPES
from paper_rq1_token_role_decode.labels import classify, align, select
from paper_rq1_token_role_decode.engine import decompose, DecodeIntervention, step, owned_clone, cache_fingerprint, step_inputs, metrics
from benchmarks.smoke import hook_footprint


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = LlamaConfig(hidden_size=16, intermediate_size=24, num_attention_heads=2, num_key_value_heads=2,
            num_hidden_layers=32, vocab_size=64, max_position_embeddings=1024, attention_dropout=0.)
        config._attn_implementation = 'eager'
        self.language_model = LlamaForCausalLM(config).half().eval()

    def forward(self, **kwargs):
        return self.language_model(**kwargs)


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_fixed_roles_priority_and_ambiguity(self):
        self.assertEqual(classify('on')[0], 'Relation')
        self.assertEqual(classify('in')[0], 'Relation')
        self.assertEqual(classify('the')[0], 'Function')
        self.assertEqual(classify('red')[0], 'Visual')
        self.assertEqual(classify('orange')[0], 'Ambiguous')
        self.assertEqual(classify('floribund')[0], 'Unknown')

    def test_actual_tokenizer_strict_alignment(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
        ids = tokenizer.encode('A man is holding a red ball.', add_special_tokens=False) + [tokenizer.eos_token_id]
        result = align(tokenizer, ids)
        self.assertTrue(result['roundtrip_exact'])
        self.assertEqual(result['tokens'][-1]['category'], 'EOS')
        for w in result['words']:
            if w['main_eligible']:
                self.assertEqual(w['token_length'], 1)
                self.assertGreaterEqual(w['token_steps'][0], 1)
                self.assertEqual(result['text'][w['char_start']:w['char_end']], w['word'])
        self.assertTrue(any(w['word'] == 'red' and w['main_eligible'] for w in result['words']))

    def test_multitoken_word_not_independent_tail(self):
        class Fake:
            all_special_ids = [2]
            eos_token_id = 2
            def decode(self, ids, **kwargs): return 'walking' if len(ids) > 1 else 'ing'
            def __call__(self, text, **kwargs): return dict(input_ids=[5, 6], offset_mapping=[(0, 4), (4, 7)])
            def convert_ids_to_tokens(self, token): return str(token)
        result = align(Fake(), [5, 6, 2])
        self.assertEqual(len(result['words']), 1)
        self.assertEqual(result['words'][0]['token_length'], 2)
        self.assertFalse(result['words'][0]['main_eligible'])

    def test_roundtrip_failure_is_audited_not_guessed(self):
        class Fake:
            all_special_ids = [2]
            eos_token_id = 2
            def decode(self, ids, **kwargs): return 'the'
            def __call__(self, text, **kwargs): return dict(input_ids=[9], offset_mapping=[(0, 3)])
            def convert_ids_to_tokens(self, token): return str(token)
        result = align(Fake(), [7, 2])
        self.assertFalse(result['roundtrip_exact']); self.assertEqual(result['words'], [])
        self.assertEqual(result['tokens'][0]['category'], 'Unaligned')

    def test_selection_hash_anchor_no_confidence_input(self):
        words = [dict(word_type=role, word=role, main_eligible=True, token_steps=[t], token_ids=[t+10], token_length=1)
            for role in TYPES for t in [2, 9, 16]]
        annotation = dict(words=words)
        a = select('image', annotation, 20)
        for w in words: w['confidence'] = np.random.random()
        b = select('image', annotation, 20)
        self.assertEqual([r['generation_step'] for r in a], [r['generation_step'] for r in b])
        self.assertEqual(len(a), 3)
        self.assertEqual(len({r['generation_step'] for r in a}), 1)
        # Real tokens cannot have multiple roles; the artificial shared positions
        # exercise only deterministic anchor choice, not the manifest validator.

    def test_projection_head_slices_bias_and_rounding_bound(self):
        p = torch.randn(2, 581).softmax(-1).half(); v = torch.randn(2, 581, 4).half()
        native_head = (p.unsqueeze(-2) @ v).squeeze(-2).reshape(1, 1, 8)
        projection = torch.nn.Linear(8, 8, bias=True).half()
        native = projection(native_head)
        keys = list(range(3, 579))
        contribution, audit = decompose(projection, p, v, keys, native_head, native, True)
        expected_head = (p[:, keys].float().unsqueeze(-2) @ v[:, keys].float()).squeeze(-2)
        expected = sum(torch.nn.functional.linear(expected_head[h][None, None], projection.weight.float()[:, h*4:(h+1)*4]) for h in range(2))
        torch.testing.assert_close(contribution, expected)
        self.assertLessEqual(audit['projection_max_error_to_bound'], 1)
        self.assertTrue(audit['bias_present'])
        before = contribution.clone()
        with torch.no_grad(): projection.bias.add_(1)
        contribution, _ = decompose(projection, p, v, keys, native_head, projection(native_head), True)
        self.assertTrue(torch.equal(before, contribution))

    def test_image_subtraction_no_renormalization(self):
        model = Toy(); projection = model.language_model.model.layers[3].self_attn.o_proj
        p = torch.randn(2, 581).softmax(-1).half(); v = torch.randn(2, 581, 8).half()
        head = (p.unsqueeze(-2) @ v).squeeze(-2).reshape(1, 1, 16)
        keys = list(range(2, 578)); native = projection(head)
        contribution, _ = decompose(projection, p, v, keys, head, native, True)
        weights_before = p.clone(); values_before = v.clone(); hooks = hook_footprint(model)
        with DecodeIntervention(model, 'Image', 3, keys, audit=True) as trace:
            trace.pv = p, v
            altered = projection(head)
        self.assertTrue(torch.equal(altered, (native.float()-contribution).half()))
        self.assertTrue(torch.equal(p, weights_before)); self.assertTrue(torch.equal(v, values_before))
        self.assertEqual(hook_footprint(model), hooks)

    def test_cache_clone_independent_and_time_mapping(self):
        cache = DynamicCache()
        for i in range(32): cache.update(torch.randn(1, 2, 7, 8).half(), torch.randn(1, 2, 7, 8).half(), i)
        before = cache_fingerprint(cache); cloned = owned_clone(cache)
        cloned.key_cache[0].zero_()
        self.assertEqual(before, cache_fingerprint(cache))
        prompt = dict(input_ids=torch.ones(1, 7, dtype=torch.long), attention_mask=torch.ones(1, 7, dtype=torch.long))
        current = step_inputs(prompt, [11, 12, 13])
        self.assertEqual(current['input_ids'].tolist(), [[13]])
        self.assertEqual(current['position_ids'].item(), 9)
        self.assertEqual(current['cache_position'].item(), 9)
        self.assertEqual(current['attention_mask'].shape[-1], 10)

    def test_noop_native_exact_and_hooks_cleanup(self):
        model = Toy(); prompt = dict(input_ids=torch.tensor([[1, 4, 5]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
        _, history, _ = step(model, prompt, None)
        inputs = step_inputs(prompt, [6]); footprint = hook_footprint(model)
        full, _, _ = step(model, inputs, history)
        noop, _, trace = step(model, inputs, history, 'All32', noop=True)
        self.assertTrue(torch.equal(full, noop)); self.assertEqual(len(trace['edits']), 32)
        _, _, trace = step(model, inputs, history, 'Whole', 14)
        self.assertEqual([e['layer'] for e in trace['edits']], [14])
        self.assertEqual(hook_footprint(model), footprint)
        original = llama.eager_attention_forward
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with DecodeIntervention(model, 'Image', 3, list(range(576))):
                raise RuntimeError('fixture')
        self.assertIs(llama.eager_attention_forward, original)
        self.assertEqual(hook_footprint(model), footprint)

    def test_fp32_full_vocabulary_metrics_direction(self):
        full = torch.tensor([2., 1., 0.]); altered = torch.tensor([3., 1., 0.])
        r = metrics(full, altered)
        self.assertLess(r['loss'], 0); self.assertGreater(r['relative_probability_change'], 0)
        self.assertFalse(r['top1_flip']); self.assertTrue(np.isfinite(r['js_divergence']))
        same = metrics(full, full); self.assertEqual(same['loss'], 0); self.assertEqual(same['js_divergence'], 0)

    def test_reject_prefill_intervention(self):
        model = Toy(); layer = model.language_model.model.layers[0]
        with self.assertRaisesRegex(AssertionError, 'one-token'):
            with DecodeIntervention(model, 'Whole', 0):
                layer.self_attn.o_proj(torch.randn(1, 2, 16).half())
        self.assertFalse(layer.self_attn.o_proj._forward_hooks)

    def test_frozen_layers_no_search(self):
        self.assertEqual(LAYERS, [0, 3, 5, 9, 14, 15, 16, 26, 31])


if __name__ == '__main__': unittest.main()

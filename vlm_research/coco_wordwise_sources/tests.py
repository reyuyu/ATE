import unittest
import torch
from transformers import AutoTokenizer
import transformers.models.llama.modeling_llama as llama
from benchmarks.smoke import hook_footprint
from test_late_attention_sources import Toy, prompt
from .common import MODEL_PATH
from .engine import Observation, groups_at, labels


class Tests(unittest.TestCase):
    def test_five_groups_consumed_query_and_prefill(self):
        p = prompt()
        for t in [0, 1, 2, 8]:
            groups, _ = groups_at(p, t, 12)
            self.assertEqual(len(groups['SELF']), int(t > 0))
            self.assertEqual(len(groups['GENERATED_OLDER']), max(0, t-1))
            if t:
                self.assertEqual(groups['SELF'], [p['input_ids'].shape[1]+t-1])
            self.assertEqual(sorted(v for g in groups.values() for v in g), list(range(p['input_ids'].shape[1]+t)))

    def test_readonly_all_layers_two_decode_steps(self):
        model = Toy()
        p = prompt()
        n = p['input_ids'].shape[1]
        def trajectory():
            cache, logits = None, []
            for t in range(3):
                args = dict(p) if t == 0 else dict(input_ids=torch.tensor([[8]]), attention_mask=torch.ones(1, n+t, dtype=torch.long))
                args['cache_position'] = torch.arange(n) if t == 0 else torch.tensor([n+t-1])
                with torch.inference_mode():
                    result = model(**args, past_key_values=cache, use_cache=True, return_dict=True)
                cache = result.past_key_values
                logits.append(result.logits[0, -1].clone())
            return logits
        native = trajectory()
        footprint = hook_footprint(model)
        original = llama.eager_attention_forward
        with Observation(model, p, 12) as observer:
            observed = trajectory()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(native, observed)))
        self.assertEqual(len(observer.mass), 3)
        self.assertEqual(observer.mass[0].shape, (32, 2, 5))
        self.assertEqual(len(observer.contribution_rows), 36)
        self.assertEqual(len(observer.audits), 6)
        self.assertEqual(footprint, hook_footprint(model))
        self.assertIs(original, llama.eager_attention_forward)

    def test_exception_cleanup(self):
        model = Toy()
        original = llama.eager_attention_forward
        footprint = hook_footprint(model)
        with self.assertRaises(RuntimeError):
            with Observation(model, prompt(), 12):
                raise RuntimeError('intentional')
        self.assertIs(original, llama.eager_attention_forward)
        self.assertEqual(footprint, hook_footprint(model))

    def test_exception_inside_decode_restores_eager(self):
        model = Toy()
        p = prompt()
        n = p['input_ids'].shape[1]
        original = llama.eager_attention_forward
        footprint = hook_footprint(model)
        def fail(module, args, output):
            if output.shape[1] == 1:
                raise RuntimeError('intentional real decode failure')
        handle = model.language_model.model.layers[15].self_attn.o_proj.register_forward_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, 'intentional real decode'):
                with Observation(model, p, 12), torch.inference_mode():
                    result = model(**p, cache_position=torch.arange(n), use_cache=True, return_dict=True)
                    model(input_ids=torch.tensor([[8]]), attention_mask=torch.ones(1, n+1, dtype=torch.long),
                        cache_position=torch.tensor([n]), past_key_values=result.past_key_values, use_cache=True, return_dict=True)
        finally:
            handle.remove()
        self.assertIs(original, llama.eager_attention_forward)
        self.assertEqual(footprint, hook_footprint(model))

    def test_image_cluster_and_fixed_effect_statistics(self):
        import numpy as np
        from .statistics import ci, adjusted
        summary = ci({'a': 1., 'b': 3.})
        self.assertEqual(summary['n_images'], 2)
        self.assertEqual(summary['mean'], 2.)
        rng = np.random.default_rng(91)
        rows = []
        for image in range(12):
            for t, role in enumerate(['Function', 'Visual', 'Relation']*3):
                rows.append(dict(image_sha256=str(image), role=role, t=t, relative_t=t/(image+12),
                    logp=float(rng.normal()), entropy=float(rng.uniform()), word_token_length=int(rng.integers(1, 4)),
                    outcome=image+2*(role == 'Visual')-(role == 'Relation')+.5*t))
        audit, results = adjusted(rows, 'outcome')
        self.assertEqual(audit['status'], 'estimated')
        self.assertAlmostEqual(next(r['effect'] for r in results if r['variable'] == 'Visual_vs_Function'), 2.)
        self.assertAlmostEqual(next(r['effect'] for r in results if r['variable'] == 'Relation_vs_Function'), -1.)

    def test_multi_token_words_retained(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
        ids = tokenizer('A person is skateboarding near an umbrella.', add_special_tokens=False)['input_ids']+[2]
        annotation, rows = labels(tokenizer, ids)
        self.assertEqual(len(rows), len(ids))
        self.assertEqual(rows[-1]['role'], 'EOS')
        multi = [w for w in annotation['words'] if w['token_length'] > 1]
        self.assertTrue(multi)
        for word in multi:
            self.assertEqual(sum(rows[t]['word_first'] for t in word['token_steps']), 1)
            self.assertEqual([rows[t]['subword_index'] for t in word['token_steps']], list(range(word['token_length'])))


if __name__ == '__main__':
    unittest.main()

"""Necessary CPU checks: query scope, cache independence, time mapping and selection."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from paper_rq1.engine import AttentionStep, cache_fingerprint, clone_cache, native_keys, sensitivity, step_inputs
from paper_rq1.protocol import COUNTS, select_windows
from paper_rq1.data import one_per_image
from paper_rq1.analyze import paired_bootstrap, zero_effect_harm_bound
from paper_rq1.scorers import gqa_prediction, short_score
from paper_rq1.refine import PhotoAliases, flickr_id


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Identity()
        self.mlp = torch.nn.Identity()

    def forward(self, hidden):
        updated = hidden+self.self_attn.o_proj(hidden*.1)
        return updated+self.mlp(updated*.2)


def fake_model():
    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.model = torch.nn.Module()
    model.language_model.model.layers = torch.nn.ModuleList([Layer() for _ in range(32)])
    return model


class RQ1Tests(unittest.TestCase):
    def test_published_cross_source_aliases_are_transitive(self):
        aliases = PhotoAliases()
        aliases.link(['vg:123', 'coco2014:456', 'flickr:789'])
        aliases.link(['openimages:abc', 'flickr:789'])
        self.assertEqual(aliases.find('vg:123'), aliases.find('openimages:abc'))
        self.assertNotEqual(aliases.find('vg:124'), aliases.find('openimages:abc'))
        self.assertEqual(flickr_id('https://farm6.staticflickr.com/2/789_ab_z.jpg'), 'flickr:789')
        self.assertIsNone(flickr_id('https://example.org/789_ab_z.jpg'))

    def test_zero_change_bootstrap_not_automatic_low_harm(self):
        self.assertGreater(zero_effect_harm_bound(150), .01)
        self.assertGreater(zero_effect_harm_bound(800), .01)
        self.assertLess(zero_effect_harm_bound(1000), .01)
    def test_paired_bootstrap_keeps_identical_conditions_identical(self):
        matrix = np.array([[1., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
        result = paired_bootstrap(matrix, repetitions=1000)
        np.testing.assert_array_equal(result[:, 0], result[:, 1])
        self.assertEqual(result.shape, (1000, 3))

    def test_gqa_published_conversion_not_candidate_ranking(self):
        self.assertEqual(gqa_prediction('Dog.'), 'dog')
        self.assertEqual(short_score(dict(dataset='gqa_balanced', answer='dog'), 'Dog.'), 1.)
        self.assertEqual(short_score(dict(dataset='gqa_balanced', answer='dog'), 'A dog.'), 0.)

    def test_exact_target_totals(self):
        self.assertEqual(sum(x[0] for x in COUNTS.values()), 2000)
        self.assertEqual(sum(x[1] for x in COUNTS.values()), 5700)

    def test_selection_eight_and_reference_coverage(self):
        ranking = {s: float(s) for s in range(30)}
        chosen = select_windows(ranking)
        self.assertEqual(chosen, [0, 1, 2, 9, 20, 27, 28, 29])
        self.assertEqual(len(select_windows({s: 0. for s in range(30)})), 8)
        with self.assertRaises(ValueError):
            select_windows({s: float(s) for s in range(29)})

    def test_duplicate_question_image_not_resampled(self):
        rows = [dict(question_id=str(i), image_identity='openimages:x') for i in range(5)]
        self.assertEqual(len(one_per_image(rows, 3)), 1)
        self.assertEqual(one_per_image(rows, 3, {'openimages:x'}), [])

    def test_cache_deep_clone_and_parent_unchanged(self):
        original = DynamicCache()
        key = torch.randn(1, 2, 7, 4)
        value = torch.randn_like(key)
        original.update(key, value, 0)
        fingerprint = cache_fingerprint(original)
        child = clone_cache(original)
        child.key_cache[0][..., 0, :] = 7.
        child.update(key[:, :, :1], value[:, :, :1], 0)
        self.assertEqual(original.get_seq_length(), 7)
        self.assertEqual(child.get_seq_length(), 8)
        self.assertEqual(cache_fingerprint(original), fingerprint)
        self.assertNotEqual(child.key_cache[0].data_ptr(), original.key_cache[0].data_ptr())

    def test_true_prefill_and_decode_positions(self):
        prompt = dict(input_ids=torch.tensor([[1, 32000, 12, 13]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                      pixel_values=torch.randn(1, 3, 4, 4))
        self.assertIs(step_inputs(prompt, [])['pixel_values'], prompt['pixel_values'])
        for t in (1, 4, 8, 16):
            inputs = step_inputs(prompt, [42]*t)
            self.assertEqual(inputs['input_ids'].shape, (1, 1))
            self.assertEqual(inputs['position_ids'].item(), 4+t-1)
            self.assertEqual(inputs['cache_position'].item(), 4+t-1)
            self.assertEqual(inputs['attention_mask'].shape[-1], 4+t)
            self.assertNotIn('pixel_values', inputs)

    def test_native_mass_groups_disjoint(self):
        prompt = dict(input_ids=torch.tensor([[1]+[32000]*576+[12, 13]]))
        keys = native_keys(prompt, [577], 4)
        self.assertEqual(len(keys['image']), 576)
        self.assertEqual(keys['history'], [579, 580, 581, 582])
        flattened = [i for group in keys.values() for i in group]
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_only_target_query_and_three_layers_mlp_still_called(self):
        model = fake_model()
        original = torch.randn(1, 5, 4)
        native = original.clone()
        for layer in model.language_model.model.layers:
            native = layer(native)
        handles, calls = [], []
        for i, layer in enumerate(model.language_model.model.layers):
            handles.append(layer.mlp.register_forward_hook(lambda m, a, o, i=i: calls.append(i)))
        with AttentionStep(model, query=4, start=2) as trace:
            changed = original.clone()
            for layer in model.language_model.model.layers:
                changed = layer(changed)
        self.assertTrue(torch.equal(native[:, :4], changed[:, :4]))
        self.assertFalse(torch.equal(native[:, 4], changed[:, 4]))
        self.assertEqual([e['layer'] for e in trace.edits], [2, 3, 4])
        self.assertEqual(calls, list(range(32)))
        for h in handles:
            h.remove()
        for m in model.modules():
            self.assertFalse(m._forward_hooks)

    def test_noop_and_exception_cleanup(self):
        model = fake_model()
        x = torch.randn(1, 1, 4)
        with self.assertRaisesRegex(RuntimeError, 'forced'):
            with AttentionStep(model, query=0, start=29, noop=True):
                for layer in model.language_model.model.layers:
                    expected = layer.self_attn.o_proj(x)
                    self.assertTrue(torch.equal(expected, x))
                raise RuntimeError('forced')
        for m in model.modules():
            self.assertFalse(m._forward_hooks)

    def test_js_logp_math_not_accuracy(self):
        x = torch.tensor([3., 1., -2.])
        zero = sensitivity(x, x)
        self.assertAlmostEqual(zero['js'], 0.)
        self.assertAlmostEqual(zero['full_token_delta_logp'], 0.)
        self.assertFalse(zero['top1_changed'])
        opposite = sensitivity(x, x.flip(-1))
        self.assertTrue(opposite['top1_changed'])
        self.assertLessEqual(opposite['js'], np.log(2.))
        self.assertLess(opposite['full_token_delta_logp'], 0.)


if __name__ == '__main__':
    unittest.main()

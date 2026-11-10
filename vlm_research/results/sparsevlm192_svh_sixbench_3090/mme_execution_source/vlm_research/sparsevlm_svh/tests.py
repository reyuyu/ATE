"""Tiny CPU regressions; no frozen model loading or CUDA context."""
from contextlib import ExitStack
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from ..fastv_svh.tests import FusionTests, fixture
from ..ate_fulljoint_v2.engine import AnswerWriter, LAYERS
from .engine import Fusion
from .official import kernels, prune_reference
from .checkpoint_audit import audit_sample, score_immutable
from .common import digest


def make(prune=True, dense=False, capture=True):
    a, layout, h = fixture(prefix=5)
    a.inputs['pixel_values'] = torch.zeros((1, 3, 4, 4))
    return a, layout, h, Fusion(a, layout, 4, prune=prune, dense=dense,
        audit=True, capture=capture, verify_reference=True)


class SparseTests(unittest.TestCase):
    def test_official_scorer_does_not_mutate_frozen_sample(self):
        sample = dict(key='MMBench:regression', benchmark='MMBench', answer='A',
            options={'A': 'Team A and Team B.', 'B': 'Team A and Team C.'})
        before = deepcopy(sample)
        result = score_immutable(sample, 'A and B')
        self.assertEqual(sample, before)
        self.assertEqual(result['parsed'], 'Z')

    def test_legacy_binding_requires_exact_official_replay(self):
        from ..fastv_svh.inputs import score
        sample = dict(key='MMBench:regression', benchmark='MMBench', answer='A',
            options={'A': 'Team A and Team B.', 'B': 'Team A and Team C.'})
        replay = deepcopy(sample)
        record = dict(generation={'text': 'A and B'}, scoring=score(replay, 'A and B'),
            sample_sha256=digest(replay))
        with self.assertRaises(AssertionError):
            audit_sample(record, sample)
        receipt = audit_sample(record, sample, allow_legacy_scoring_mutation=True)
        self.assertEqual(receipt['changed_fields'], ['options'])
        self.assertEqual(sample['options']['A'], 'Team A and Team B.')
        record['sample_sha256'] = '0' * 64
        with self.assertRaises(AssertionError):
            audit_sample(record, sample, allow_legacy_scoring_mutation=True)

    def test_legacy_binding_does_not_accept_unrelated_input_change(self):
        sample = dict(key='GQA:regression', benchmark='GQA', answer='yes')
        record = dict(sample_sha256='0' * 64, generation={'text': 'yes'},
                      scoring={'parsed': 'yes', 'score': 1.0})
        with self.assertRaises(AssertionError):
            audit_sample(record, sample, allow_legacy_scoring_mutation=True)

    def test_official_source_ast_and_three_stage_recycling(self):
        ns, prune, provenance = kernels()
        self.assertTrue(all(p['AST_unmodified'] for p in provenance))
        torch.manual_seed(20261110)
        left, visual, right, dimension = 3, 576, 7, 24
        h = torch.randn(1, left + visual + right, dimension)
        raters = (torch.tensor([0, 0]), torch.tensor([1, right - 1]))
        for layer, expected in ((2, 309), (6, 204), (15, 113)):
            n = h.shape[1]
            weights = torch.randn(1, 2, n, n).softmax(-1)
            torch.manual_seed(12)
            actual, count, _ = prune(h, weights, left, visual, left + visual,
                raters, layer, torch.arange(n)[None])
            torch.manual_seed(12)
            expected_h = prune_reference(h, weights, left, visual, raters, layer)
            self.assertTrue(torch.equal(actual, expected_h))
            self.assertEqual(int(count), expected)
            self.assertTrue(torch.equal(actual[:, :left], h[:, :left]))
            self.assertTrue(torch.equal(actual[:, left + expected:], h[:, left + visual:]))
            h, visual = actual, expected

    def test_no_pruning_is_strict_svh_including_decode(self):
        a, layout, h, ctl = make(prune=False)
        baseline = AnswerWriter(a, layout, 'C3-V2', layout.relay, 4, True)
        decode = torch.randn(1, 1, h.shape[-1])
        with torch.inference_mode():
            with baseline.context():
                reference = a.model.language_model.model(inputs_embeds=h).last_hidden_state.clone()
                reference_decode = a.model.language_model.model(inputs_embeds=decode).last_hidden_state.clone()
            with ctl.context():
                actual = a.model.language_model.model(inputs_embeds=h).last_hidden_state.clone()
                actual_decode = a.model.language_model.model(inputs_embeds=decode).last_hidden_state.clone()
        self.assertTrue(torch.equal(reference, actual))
        self.assertTrue(torch.equal(reference_decode, actual_decode))

    def test_physical_shapes_positions_probabilities_mlp_and_decode_skip(self):
        a, layout, h, ctl = make()
        calls = {name: [0] * 32 for name in ('mlp', 'input_layernorm', 'post_attention_layernorm')}
        projections = []
        with torch.inference_mode(), ExitStack() as stack:
            for i, layer in enumerate(a.layers):
                for name in calls:
                    handle = getattr(layer, name).register_forward_hook(
                        lambda m, args, out, name=name, i=i: calls[name].__setitem__(i, calls[name][i] + 1))
                    stack.callback(handle.remove)
                def check(m, args, kw):
                    expected = a.model.language_model.model.rotary_emb(args[0], kw['position_ids'])
                    self.assertTrue(torch.equal(kw['cache_position'], kw['position_ids'][0]))
                    self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, kw['position_embeddings'])))
                handle = layer.register_forward_pre_hook(check, with_kwargs=True)
                stack.callback(handle.remove)
                for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    handle = getattr(layer.self_attn, name).register_forward_hook(
                        lambda m, args, out, i=i, name=name: projections.append((i, name)))
                    stack.callback(handle.remove)
            with ctl.context():
                a.model.language_model.model(inputs_embeds=h)
                before = len(projections)
                a.model.language_model.model(inputs_embeds=torch.randn(1, 1, h.shape[-1]))
        self.assertTrue(all(v == [2] * 32 for v in calls.values()))
        self.assertFalse(any(i in LAYERS for i, name in projections[before:]))
        self.assertEqual([r['visual_rows'] for r in ctl.structure], [576]*3 + [309]*4 + [204]*9 + [113]*4 + [0]*12)
        self.assertTrue(all(r['visual_rows'] == r['physically_allocated_visual_rows'] for r in ctl.structure))
        self.assertTrue(all(e['blocked_probability_max'] == 0 for e in ctl.events))
        self.assertTrue(all(s['official_score_and_topk_exact'] and s['independent_recycling_reference_exact']
                            for s in ctl.selection))
        self.assertEqual(ctl.cache.audit()['persistent_original_visual_tokens'], 0)

    def test_dense_functional_reference(self):
        a, layout, h, physical = make()
        dense = Fusion(a, layout, 4, dense=True, audit=True, capture=True, verify_reference=True)
        with torch.inference_mode():
            with physical.context():
                p = a.model.language_model.model(inputs_embeds=h).last_hidden_state.clone()
            with dense.context():
                d = a.model.language_model.model(inputs_embeds=h).last_hidden_state.clone()
        self.assertTrue(torch.allclose(p, d, atol=1e-5, rtol=1e-5))
        self.assertEqual([s['kept_stage_indices'] for s in physical.selection],
                         [s['kept_stage_indices'] for s in dense.selection])
        self.assertEqual(dense.structure[19]['physically_allocated_visual_rows'], 576)
        self.assertEqual(dense.structure[20]['physically_allocated_visual_rows'], 0)

    def test_exception_cleanup_and_no_cross_request_alias(self):
        a, layout, h, ctl = make()
        originals = [l.self_attn.forward for l in a.layers]
        decoder = a.model.language_model.model.forward
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'injected'):
            with patch.object(a.layers[6].mlp, 'forward', side_effect=RuntimeError('injected')):
                with ctl.context():
                    a.model.language_model.model(inputs_embeds=h)
        self.assertEqual(originals, [l.self_attn.forward for l in a.layers])
        self.assertEqual(decoder, a.model.language_model.model.forward)
        self.assertIsNone(ctl.original_attention)
        first = Fusion(a, layout, 4)
        second = Fusion(a, layout, 4)
        with torch.inference_mode():
            with first.context():
                x = a.model.language_model.model(inputs_embeds=h).last_hidden_state.clone()
            with second.context():
                y = a.model.language_model.model(inputs_embeds=h).last_hidden_state.clone()
        self.assertTrue(torch.equal(x, y))
        self.assertTrue(all(p[0].data_ptr() != q[0].data_ptr() for p, q in
                            zip(first.cache.entries, second.cache.entries) if p is not None))


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    unittest.main(verbosity=2)

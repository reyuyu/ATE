"""CPU logic gates use tiny randomly initialized Llama, never the frozen weights."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .engine import Fusion, LAYERS, TextOnlyKV, OFFICIAL, selector
from ..ate_fulljoint_v2.engine import AnswerWriter
from ..physical_handoff_fastpath.engine import Layout


def fixture(prefix=2):
    torch.manual_seed(20261110)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=32, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=2048)
    cfg._attn_implementation = 'eager'
    lm = LlamaForCausalLM(cfg).eval()
    n = prefix + 576 + 5
    image = tuple(range(prefix, prefix + 576))
    text = tuple(i for i in range(n) if i not in image)
    layout = Layout(n, image, text, tuple(range(image[-1]+1, n-1)), (n-3,), 20)
    wrapper = SimpleNamespace(language_model=lm)
    a = SimpleNamespace(kind='llava', model=wrapper, layers=lm.model.layers,
                        inputs={'input_ids': torch.zeros((1, n), dtype=torch.long)})
    h = torch.randn(1, n, cfg.hidden_size)
    return a, layout, h


class FusionTests(unittest.TestCase):
    def test_partition_keeps_sha_and_circular_groups_on_one_rank(self):
        from .preflight import partition
        samples = [dict(key='a', benchmark='MMBench', circular_group='g', image_sha256='sha1', max_new_tokens=1024),
                   dict(key='b', benchmark='MMBench', circular_group='g', image_sha256='sha2', max_new_tokens=1024),
                   dict(key='c', benchmark='GQA', image_sha256='sha2', max_new_tokens=32),
                   dict(key='d', benchmark='MME', image_sha256='sha3', max_new_tokens=128),
                   dict(key='e', benchmark='MME', image_sha256='sha3', max_new_tokens=128)]
        result = partition(samples)
        self.assertEqual(result, partition(list(reversed(samples))))
        ranks = result['assignment']
        self.assertEqual(ranks['a'], ranks['b'])
        self.assertEqual(ranks['b'], ranks['c'])
        self.assertEqual(ranks['d'], ranks['e'])
        self.assertEqual(sum(result['question_counts']), 5)

    def test_gpu_precheck_rejects_occupied_cards_even_with_free_memory(self):
        from .common import idle_uuids
        gpu = '0, GPU-test, NVIDIA GeForce RTX 3090, 10, 24000, 0'
        self.assertEqual(idle_uuids(dict(gpus=gpu, processes='')), ['GPU-test'])
        self.assertEqual(idle_uuids(dict(gpus=gpu, processes='GPU-test, 7, python, 1')), [])

    def test_deferred_schedule_is_explicit_and_does_not_drop_manifest_rows(self):
        from . import run
        original = [dict(benchmark='GQA') for _ in range(32882)] + [dict(benchmark='MME') for _ in range(2374)]
        acceptance = dict(status='BLOCKED_MME_REFERENCE_BYTES', blockers=[dict(key='MME:image:0')])
        p = dict(all_six_ready=False, data_acceptance_sha256='test')
        with patch.object(run, 'read', return_value=acceptance), patch.object(run, 'sha', return_value='test'):
            with self.assertRaisesRegex(AssertionError, 'implicit'):
                run.accepted_samples(p, original)
            p['execution_schedule'] = dict(deferred_benchmarks=['MME'],
                ready_benchmarks=[b for b in run.COUNTS if b!='MME'], ready_questions=32882,
                requested_questions=35256, deferred_questions=2374)
            self.assertEqual(len(run.accepted_samples(p, original)), 32882)
            self.assertEqual(len(original), 35256)
            acceptance['blockers'] = [dict(key='GQA:bad')]
            with self.assertRaises(AssertionError):
                run.accepted_samples(p, original)

    def test_scoring_gqa_is_exact_not_substring(self):
        from .inputs import score
        sample = dict(benchmark='GQA', answer='red')
        self.assertEqual(score(sample, 'Red')['score'], 1.)
        self.assertEqual(score(sample, 'red...')['score'], 1.)
        self.assertEqual(score(sample, 'red .')['score'], 0.)
        self.assertEqual(score(sample, 'red car')['score'], 0.)

    def test_official_source_and_selector(self):
        select, provenance = selector(str(OFFICIAL))
        self.assertEqual(provenance['selection_AST_changes'], 0)
        weights = torch.arange(590.).reshape(1, 1, 1, 590).repeat(1, 4, 1, 1)
        keep, top, _ = select((None, weights), 4, 576, 2/3, 590, torch.device('cpu'))
        self.assertEqual(top.numel(), 192)
        self.assertEqual(keep.numel(), 206)
        self.assertEqual(sorted(top.tolist()), list(range(388, 580)))
        self.assertEqual([v for v in keep.tolist() if not 4 <= v < 580], list(range(4))+list(range(580, 590)))

    def test_disabled_pruning_is_frozen_svh_exact_including_decode(self):
        a, layout, h = fixture()
        frozen = AnswerWriter(a, layout, 'C3-V2', layout.relay, 4, True)
        new = Fusion(a, layout, 4, prune=False, audit=True)
        with frozen.context():
            expected = a.model.language_model.model(inputs_embeds=h).last_hidden_state
            next_h = torch.randn(1, 1, 32)
            expected_decode = a.model.language_model.model(inputs_embeds=next_h).last_hidden_state
        with new.context():
            actual = a.model.language_model.model(inputs_embeds=h).last_hidden_state
            actual_decode = a.model.language_model.model(inputs_embeds=next_h).last_hidden_state
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_decode, actual_decode))
        old = frozen.cache.audit()
        now = new.cache.audit()
        self.assertEqual(old['live_bytes'], now['live_bytes'])
        for p, q in zip(frozen.cache.entries, new.cache.entries):
            if p is not None:
                self.assertTrue(torch.equal(p[0][..., :len(layout.text)+1, :], q[0][..., :len(layout.text)+1, :]))
                self.assertTrue(torch.equal(p[1][..., :len(layout.text)+1, :], q[1][..., :len(layout.text)+1, :]))

    def test_physical_rows_cache_selector_masks_and_all_mlps(self):
        a, layout, h = fixture(prefix=5)
        ctl = Fusion(a, layout, 4, audit=True, verify_selector=True)
        counts = {'mlp': [0]*32, 'norm': [0]*32, 'post_norm': [0]*32}
        calls = {'q': [0]*32, 'k': [0]*32, 'v': [0]*32, 'o': [0]*32}
        residual, observed = {}, []
        with ExitStack() as stack:
            for i, layer in enumerate(a.layers):
                for name, module in [('mlp', layer.mlp), ('norm', layer.input_layernorm),
                                     ('post_norm', layer.post_attention_layernorm)]:
                    handle = module.register_forward_hook(lambda m, args, out, i=i, name=name:
                                                           counts[name].__setitem__(i, counts[name][i]+1))
                    stack.callback(handle.remove)
                for name in calls:
                    module = getattr(layer.self_attn, name+'_proj')
                    handle = module.register_forward_hook(lambda m, args, out, i=i, name=name:
                                                           calls[name].__setitem__(i, calls[name][i]+1))
                    stack.callback(handle.remove)
                def before(m, args, kw, i=i):
                    pos = kw['position_ids']
                    observed.append((i, args[0].shape[1], pos[0].tolist()))
                    self.assertTrue(torch.equal(kw['cache_position'], pos[0]))
                    expected = a.model.language_model.model.rotary_emb(args[0], pos)
                    self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, kw['position_embeddings'])))
                    if i in LAYERS:
                        residual[i] = args[0][:, -1].clone()
                handle = layer.register_forward_pre_hook(before, with_kwargs=True)
                stack.callback(handle.remove)
                def post(m, args, i=i):
                    if i in LAYERS:
                        self.assertTrue(torch.equal(args[0][:, -1], residual.pop(i)))
                handle = layer.post_attention_layernorm.register_forward_pre_hook(post)
                stack.callback(handle.remove)
            with ctl.context():
                out = a.model.language_model.model(inputs_embeds=h)
                self.assertEqual(out.last_hidden_state.shape[1], len(layout.text))
                before_decode = {k: v.copy() for k, v in calls.items()}
                a.model.language_model.model(inputs_embeds=torch.randn(1, 1, 32))
                for i in LAYERS:
                    for name in calls:
                        self.assertEqual(calls[name][i], before_decode[name][i])
        self.assertTrue(all(v == [2]*32 for v in counts.values()))
        self.assertTrue(ctl.selection['reference_top192_exact'])
        self.assertTrue(ctl.selection['reference_probability_exact'])
        self.assertTrue(ctl.selection['extra_matrix_flops'] > 0)
        for row in ctl.structure:
            self.assertEqual(row['visual_rows'], 576 if row['layer'] < 3 else 192 if row['layer'] < 20 else 0)
        for i, size, pos in observed[:32]:
            self.assertEqual([p for p in pos if p not in layout.image], list(layout.text))
        self.assertTrue(all(r['blocked_probability_max'] == 0 for r in ctl.events))
        self.assertTrue(all(r['row_sum_error'] < .005 for r in ctl.events))
        audit = ctl.cache.audit()
        self.assertEqual(audit['persistent_original_visual_tokens'], 0)
        self.assertEqual(audit['append_concatenations'], 0)
        self.assertEqual(audit['historical_cache_copies'], 0)
        self.assertEqual(sum(r['layer'] in LAYERS and r['used_tokens']==0 for r in audit['layers']), 4)
        self.assertTrue(all(r['used_tokens'] == len(layout.text)+1 for r in audit['layers'] if r['layer'] not in LAYERS))

    def test_exception_restores_original_functions(self):
        a, layout, h = fixture()
        original = [m.self_attn.forward for m in a.layers]
        decoder_forward = a.model.language_model.model.forward
        ctl = Fusion(a, layout)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with patch.object(a.layers[4].mlp, 'forward', side_effect=RuntimeError('injected')):
                with ctl.context():
                    a.model.language_model.model(inputs_embeds=h)
        self.assertEqual(a.model.language_model.model.forward, decoder_forward)
        self.assertEqual([m.self_attn.forward for m in a.layers], original)
        self.assertIsNone(ctl.original_attention)
        self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in a.model.language_model.modules()))

    def test_fresh_request_determinism_and_no_cache_aliasing(self):
        a, layout, h = fixture()
        first, second = Fusion(a, layout, 3), Fusion(a, layout, 3)
        with first.context():
            x = a.model.language_model.model(inputs_embeds=h).last_hidden_state.clone()
        with second.context():
            y = a.model.language_model.model(inputs_embeds=h).last_hidden_state
        self.assertTrue(torch.equal(x, y))
        self.assertEqual(first.selection['selected_visual_indices_sha256'], second.selection['selected_visual_indices_sha256'])
        self.assertTrue(all(p[0].data_ptr() != q[0].data_ptr()
                            for p, q in zip(first.cache.entries, second.cache.entries) if p is not None))

    def test_capacity_is_checked(self):
        _, layout, _ = fixture()
        cache = TextOnlyKV(layout, 32, 1)
        cache.text_index = torch.tensor(layout.text)
        keys = torch.randn(1, 2, layout.length, 8)
        cache.update(keys, keys.clone(), 0)
        cache.prefill = False
        with self.assertRaisesRegex(AssertionError, 'capacity'):
            cache.update(keys[..., :1, :], keys[..., :1, :], 0)

    def test_reject_incompatible_layout(self):
        a, layout, _ = fixture()
        layout.cutoff = 24
        with self.assertRaises(AssertionError):
            Fusion(a, layout)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    unittest.main(verbosity=2)

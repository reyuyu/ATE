import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from paper_rq1.engine import clone_cache, cache_fingerprint
from .engine import roles, target_positions, RoleBypass
from .common import LAYERS, selected


def prompt(extra=6):
    return dict(input_ids=torch.tensor([[1, 31, 32000]+[32000]*575+[42]*extra]),
                attention_mask=torch.ones((1, 578+extra), dtype=torch.long))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Identity()

    def forward(self, hidden_states, **kwargs):
        return self.o_proj(hidden_states)


def blocks():
    return [type('Block', (), {'self_attn': Attention()})() for _ in range(32)]


class Tests(unittest.TestCase):
    def test_dynamic_roles(self):
        for length in [4, 7, 21]:
            p = prompt(length)
            # The layout must reject inconsistent real masks rather than guess.
            p['attention_mask'] = torch.ones_like(p['input_ids'])
            r = roles(p)
            self.assertEqual(r['I'], list(range(2, 578)))
            self.assertEqual(r['R'], list(range(578, p['input_ids'].shape[1]-1)))
            self.assertEqual(r['A'], [p['input_ids'].shape[1]-1])
            self.assertEqual(r['other'], [0, 1])

    def test_reject_bad_layout(self):
        p = prompt(); p['attention_mask'] = torch.ones_like(p['input_ids'])
        p['input_ids'][0, 5] = 0
        with self.assertRaises(ValueError): roles(p)
        p = prompt(); p['attention_mask'] = torch.ones_like(p['input_ids']); p['attention_mask'][0, -1] = 0
        with self.assertRaises(ValueError): roles(p)

    def test_roles_and_decode(self):
        p = prompt(); p['attention_mask'] = torch.ones_like(p['input_ids']); r = roles(p)
        for role in ['I-Bypass', 'R-Bypass', 'A-Bypass']:
            self.assertEqual(target_positions(role, r, False), r[role[0]])
            self.assertEqual(target_positions(role, r, True), [0] if role == 'A-Bypass' else [])

    def test_exact_scope_and_layers(self):
        p = prompt(); p['attention_mask'] = torch.ones_like(p['input_ids']); r = roles(p)
        native = torch.randn((1, r['prompt_length'], 8))
        for role in ['Full', 'I-Bypass', 'R-Bypass', 'A-Bypass']:
            b = blocks()
            with patch('vlm_research.query_role_attention.engine.model_layers', return_value=b):
                with RoleBypass(None, r, role) as trace:
                    for k in range(32):
                        out = b[k].self_attn(native, cache_position=torch.arange(r['prompt_length']), past_key_value=None)
                        expected = native.clone()
                        if k in LAYERS: expected[:, target_positions(role, r, False)] = 0
                        self.assertTrue(torch.equal(out, expected))
                self.assertFalse(trace.handles)
                self.assertTrue(all(not x.self_attn._forward_pre_hooks and not x.self_attn.o_proj._forward_hooks for x in b))

    def test_noop_and_exception_cleanup(self):
        p = prompt(); p['attention_mask'] = torch.ones_like(p['input_ids']); r = roles(p)
        b = blocks(); native = torch.randn((1, r['prompt_length'], 3))
        with patch('vlm_research.query_role_attention.engine.model_layers', return_value=b):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                with RoleBypass(None, r, 'I-Bypass', noop=True):
                    out = b[2].self_attn(native, cache_position=torch.arange(r['prompt_length']), past_key_value=None)
                    self.assertTrue(torch.equal(native, out))
                    raise RuntimeError('intentional')
        self.assertTrue(all(not x.self_attn._forward_pre_hooks and not x.self_attn.o_proj._forward_hooks for x in b))

    def test_real_decode_and_cache_alignment(self):
        p = prompt(); p['attention_mask'] = torch.ones_like(p['input_ids']); r = roles(p)
        native = torch.randn((1, 1, 4)); b = blocks()
        cache = type('Cache', (), {'get_seq_length': lambda self, k: r['prompt_length']})()
        with patch('vlm_research.query_role_attention.engine.model_layers', return_value=b):
            with RoleBypass(None, r, 'A-Bypass'):
                out = b[2].self_attn(native, cache_position=torch.tensor([r['prompt_length']]), past_key_value=cache)
                self.assertEqual(int(torch.count_nonzero(out)), 0)
            with self.assertRaises(AssertionError):
                with RoleBypass(None, r, 'A-Bypass'):
                    b[2].self_attn(native, cache_position=torch.tensor([r['prompt_length']+1]), past_key_value=cache)

    def test_independent_cache_clone(self):
        cache = DynamicCache()
        cache.update(torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4), 0)
        before = cache_fingerprint(cache); other = clone_cache(cache)
        other.key_cache[0].zero_()
        self.assertEqual(cache_fingerprint(cache), before)
        self.assertNotEqual(cache_fingerprint(other), before)

    def test_input_only_selection(self):
        tasks = ['ChooseAttr', 'ChooseCat', 'ChooseRel', 'CompareAttr', 'LogicalObj', 'QueryAttr']
        pool = [dict(task=t, question_id=f'{j}-{i}', image_sha256=f'image-{j}-{i}',
                     input_mapping=dict(prompt_ids=[1, i]), answer='x', correct=True)
                for j, t in enumerate(tasks) for i in range(500)]
        first = [s['question_id'] for s in selected(pool)]
        changed = copy.deepcopy(pool)
        for row in changed: row.update(answer='other', correct=False)
        self.assertEqual(first, [s['question_id'] for s in selected(changed)])
        self.assertEqual(len(first), 240)

    def test_paired_cluster_bootstrap(self):
        import numpy as np
        from .analyze import paired_bootstrap
        samples = [dict(image_sha256='a'),dict(image_sha256='a'),dict(image_sha256='b')]
        scores = np.array([[1.,0.],[1.,0.],[0.,1.]])
        draws = paired_bootstrap(samples,scores,repetitions=2000)
        self.assertTrue(np.allclose(draws.sum(1),1.))
        self.assertEqual(set(draws[:,0]), {0.,2/3,1.})

    def test_first_token_signed_loss(self):
        from first_answer_attention.engine import first_metrics
        full = torch.tensor([2.,1.,0.]);changed = torch.tensor([3.,1.,0.])
        values = first_metrics(full,changed)
        self.assertLess(values['first_logp_loss'],0.)

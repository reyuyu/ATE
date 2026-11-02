import inspect
import unittest

import torch
from .engine import apply_mask, blocked_edges, WriterMask
from .groups import select, intact, subsets


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.layout = dict(prompt_length=9, I=[2,3,4], R=[5,6,7], A=[8], other=[0,1])
        self.old = torch.zeros(1,1,9,9).half().masked_fill(
            torch.ones(9,9,dtype=torch.bool).triu(1), torch.finfo(torch.float16).min)

    def test_all_writers_only_blocks_answer(self):
        _, blocked = apply_mask(self.old, self.layout, [5,6,7], 0)
        self.assertEqual(int(blocked.sum()), 3)
        self.assertTrue(blocked[8,2:5].all())

    def test_no_writer_blocks_all_text_visual_edges(self):
        changed, blocked = apply_mask(self.old, self.layout, [], 0)
        self.assertEqual(int(blocked.sum()), 12)
        self.assertTrue(torch.equal(changed[0,0][~blocked], self.old[0,0][~blocked]))

    def test_sparse_query_key_direction(self):
        _, blocked = apply_mask(self.old, self.layout, [6], 0)
        for q in range(9):
            for k in range(9):
                self.assertEqual(bool(blocked[q,k]), q in [5,7,8] and k in [2,3,4])

    def test_decode_always_blocks_original_image(self):
        for writers in ([], [5], [5,6,7]):
            old = torch.zeros(1,1,1,12).half()
            changed, blocked = apply_mask(old, self.layout, writers, 11)
            self.assertEqual(torch.where(blocked[0])[0].tolist(), [2,3,4])
            self.assertTrue(torch.isneginf(changed[0,0,0,2:5]).all())

    def test_softmax_zero_keeps_bos_self_and_row_sum(self):
        changed, blocked = apply_mask(self.old, self.layout, [5], 0)
        weights = torch.softmax(torch.randn(1,4,9,9)+changed.float(), -1)
        self.assertEqual(float(weights[0,:,blocked].max()), 0)
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(1,4,9)))
        self.assertTrue((changed[0,0,:,0] == 0).all())
        self.assertTrue((changed[0,0].diagonal() == 0).all())

    def test_noop_exact(self):
        for writers in ([], [6], [5,6,7]):
            changed, _ = apply_mask(self.old, self.layout, writers, 0, noop=True)
            self.assertTrue(torch.equal(changed, self.old))

    def test_reject_illegal_or_outside_query(self):
        with self.assertRaises(AssertionError):
            apply_mask(torch.zeros_like(self.old), self.layout, [6], 0)
        with self.assertRaises(AssertionError):
            blocked_edges(self.layout, [8], 0, 'cpu')

    def test_hook_cleanup_exception(self):
        from torch import nn
        ctl = WriterMask.__new__(WriterMask)
        module = nn.Identity()
        ctl.handles = [module.register_forward_hook(lambda *args: None)]
        ctl.__exit__(RuntimeError, RuntimeError('forced'), None)
        self.assertFalse(module._forward_hooks)


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.groups = [dict(positions=[2,3], core=True), dict(positions=[4], core=True),
                       dict(positions=[5], core=False), dict(positions=[6,7], core=False)]

    def test_whole_subword_group(self):
        self.assertFalse(intact([2], self.groups))
        self.assertTrue(intact([2,3], self.groups))

    def test_exact_budget_and_suffix(self):
        spec = select(self.groups, [2,3,4,5], list(range(2,8)), 'What color?')
        k = spec['k']
        self.assertEqual(k, 2)
        for c, positions in spec['writers'].items():
            self.assertTrue(intact(positions, self.groups), c)
            if c in ('Semantic-k','Question-Last-k','R-Last-k') or c.startswith('Random-'):
                self.assertEqual(len(positions), k)

    def test_scaffold_deficit_is_explicit(self):
        groups = [dict(positions=[i],core=True) for i in range(2,6)] + [dict(positions=[6],core=False)]
        spec = select(groups, list(range(2,7)), list(range(2,7)), 'How many things?')
        self.assertEqual(spec['scaffold_actual_budget'], 1)
        self.assertEqual(spec['scaffold_budget_deficit'], spec['k']-1)

    def test_text_only_no_gold_input(self):
        self.assertEqual(list(inspect.signature(select).parameters), ['groups','question_positions','relay','question'])

    def test_random_repeat_and_same_text_pair(self):
        a = select(self.groups,[2,3,4,5],list(range(2,8)),'Same text?')
        b = select(self.groups,[2,3,4,5],list(range(2,8)),'Same text?')
        self.assertEqual(a, b)

    def test_drop_is_selected_core_complement(self):
        spec = select(self.groups,[2,3,4,5],list(range(2,8)),'Where?')
        self.assertEqual(sorted(spec['writers']['Drop-Semantic']+spec['writers']['Semantic-k']), list(range(2,8)))

    def test_subset_does_not_reuse_word(self):
        for budget, chosen in subsets(self.groups).items():
            self.assertEqual(sum(len(g['positions']) for g in chosen), budget)
            self.assertEqual(len(chosen), len({id(g) for g in chosen}))


if __name__ == '__main__':
    unittest.main(verbosity=2)

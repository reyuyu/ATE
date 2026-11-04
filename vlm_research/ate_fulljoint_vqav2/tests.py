import unittest
from contextlib import ExitStack

import torch

from ..fastp2_bypass_writer.engine import LAYERS
from .engine import JointWriter
from .common import conditional_writers, score, normalize


class Tests(unittest.TestCase):
    def setUp(self):
        from ..fastp2_bypass_writer.tests import Tests as Fixture
        Fixture.setUp(self)

    def execute(self, name):
        ctl = JointWriter(self.a, self.layout, name, [6, 7, 8], 4, True)
        calls, mlp, shapes = {}, [0] * 8, []
        with ExitStack() as stack:
            for i, layer in enumerate(self.a.layers):
                for op in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    def count(m, args, output, i=i, op=op):
                        calls.setdefault((i, op), []).append(args[0].shape[1])
                    handle = getattr(layer.self_attn, op).register_forward_hook(count)
                    stack.callback(handle.remove)
                def count_mlp(m, args, output, i=i):
                    mlp[i] += 1
                    shapes.append((i, args[0].shape[1]))
                handle = layer.mlp.register_forward_hook(count_mlp)
                stack.callback(handle.remove)
            with torch.inference_mode(), ctl.context():
                first = ctl.forward(inputs_embeds=self.h).last_hidden_state
                second = ctl.forward(inputs_embeds=torch.ones(1, 1, 32)).last_hidden_state
        return first, second, ctl, calls, mlp, shapes

    def test_latest_condition(self):
        for text, n, hit in (('What is here?', 3, True), ('Which one?', 3, True),
                             ('Is it red?', 14, False), ('Is it red?', 15, True)):
            writers, triggered = conditional_writers(text, list(range(n)), [100, 101, 102, 103])
            self.assertEqual(triggered, hit)
            self.assertEqual(writers, list(range(n)) + ([100, 101, 102, 103] if hit else []))

    def test_official_consensus(self):
        for n, expected in ((0, 0), (1, .3), (2, .6), (3, .9), (4, 1), (10, 1)):
            self.assertAlmostEqual(score('White', ['white'] * n + ['black'] * (10 - n)), expected)
        self.assertEqual(normalize('The two cats.'), '2 cats')

    def test_final_mask_reaches_partial_rows(self):
        ctl = JointWriter(self.a, self.layout, 'C4', [6, 7, 8], 4)
        self.assertTrue(torch.equal(ctl.active_mask,
                                   ctl.prefill_early.index_select(-2, ctl.active_index)))
        for q in (9, 10, 11, 12):
            self.assertTrue(torch.isneginf(ctl.active_mask[0, 0, ctl.active.index(q),
                                                       list(self.layout.image)]).all())
        self.assertTrue((ctl.active_mask == 0).any(-1).all())

    def test_physical_matches_zero_reference(self):
        a, b = self.execute('C4'), self.execute('C4-ZeroRef')
        for x, y in zip(a[:2], b[:2]):
            torch.testing.assert_close(x, y, atol=3e-7, rtol=3e-6)
        for i in LAYERS:
            for op in ('q_proj', 'o_proj'):
                self.assertEqual(a[3][i, op], [9])
            for op in ('k_proj', 'v_proj'):
                self.assertEqual(a[3][i, op], [14])
        self.assertEqual(a[4], [2] * 8)

    def test_full_dense_reference(self):
        a, b = self.execute('C4'), self.execute('C4-DenseRef')
        torch.testing.assert_close(a[0], b[0][:, list(self.layout.text)], atol=3e-7, rtol=3e-6)
        torch.testing.assert_close(a[1], b[1], atol=3e-7, rtol=3e-6)

    def test_exit_and_cache(self):
        a = self.execute('C4')
        self.assertIn((5, 14), a[5])
        self.assertIn((6, 10), a[5])
        cache = a[2].cache.audit()
        self.assertEqual(cache['historical_cache_copies'], 0)
        self.assertEqual(cache['append_concatenations'], 0)
        for row in cache['layers']:
            self.assertEqual(row['used_tokens'], 0 if row['layer'] in LAYERS else 11)

    def test_fresh_cache_and_cleanup(self):
        a, b = self.execute('C4'), self.execute('C4')
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
        self.assertNotEqual(a[2].cache.entries[0][0].data_ptr(), b[2].cache.entries[0][0].data_ptr())
        old = [layer.self_attn.forward for layer in self.a.layers]
        ctl = JointWriter(self.a, self.layout, 'C4', [6, 7, 8], 4)
        with self.assertRaises(RuntimeError):
            with ctl.context():
                raise RuntimeError('Intentional cleanup test')
        self.assertEqual(old, [layer.self_attn.forward for layer in self.a.layers])


if __name__ == '__main__':
    unittest.main(verbosity=2)

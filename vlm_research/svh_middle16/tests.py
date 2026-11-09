import unittest
from dataclasses import replace
from types import SimpleNamespace
import torch

from .common import MIDDLE16, OLD_REACCESS, Algorithm
from .engine import MiddleWriter, summarize_cache
from .prepare import partition
from ..physical_handoff_fastpath.engine import Layout


class CPUChecks(unittest.TestCase):
    def setUp(self):
        self.layout = Layout(9, (1, 2, 3), (0, 4, 5, 6, 7, 8), (4, 5, 6, 7), (5, 6), 24)
        self.layers = [torch.nn.Linear(1, 1, dtype=torch.float16) for _ in range(32)]
        decoder = SimpleNamespace()
        model = SimpleNamespace(language_model=SimpleNamespace(model=decoder))
        self.adapter = SimpleNamespace(kind='llava', model=model, layers=self.layers,
                                      inputs={'input_ids': torch.zeros((1, 9), dtype=torch.long)})
        self.adapter.model.parameters = lambda: iter([torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))])

    def test_fixed_algorithm(self):
        self.assertEqual(MIDDLE16.reaccess, tuple(range(8, 24)))
        self.assertEqual(MIDDLE16.exit_before, 24)
        self.assertEqual(OLD_REACCESS.exit_before, 25)
        self.assertEqual(len(MIDDLE16.reaccess), 16)

    def test_skip_overlap_rejected(self):
        with self.assertRaises(AssertionError):
            Algorithm('bad', (3, 8), 24)

    def test_exit_rejected(self):
        with self.assertRaises(AssertionError):
            Algorithm('bad', (24,), 24)

    def test_masks_and_writer_boundary(self):
        w = MiddleWriter(self.adapter, self.layout, MIDDLE16.reaccess, 4)
        for layer in range(32):
            mask = w.prefill_mask(layer)[0, 0]
            if layer >= 24:
                self.assertEqual(tuple(mask.shape), (6, 6))
                continue
            self.assertEqual(tuple(mask.shape), (9, 9))
            self.assertTrue(torch.isneginf(mask[8, [1, 2, 3]]).all() if layer not in MIDDLE16.reaccess
                            else (mask[8, [1, 2, 3]] == 0).all())
            blocked = mask[list(self.layout.relay)][:, list(self.layout.image)]
            self.assertTrue((blocked == 0).all() if layer < 20 else torch.isneginf(blocked).all())
            self.assertTrue((mask == 0).any(-1).all())
            future = mask[torch.triu(torch.ones_like(mask, dtype=torch.bool), 1)]
            self.assertTrue(((future == torch.finfo(mask.dtype).min) | torch.isneginf(future)).all())

    def test_cache_physical_layers_and_no_append_copy(self):
        w = MiddleWriter(self.adapter, self.layout, MIDDLE16.reaccess, 4)
        c = w.cache
        for layer in range(32):
            if layer in MIDDLE16.answer_skip:
                continue
            n = 9 if layer < 24 else 6
            k = torch.arange(n*2, dtype=torch.float16).reshape(1, 1, n, 2)
            c.update(k, k + 1, layer)
        pointers = [(p[0].data_ptr(), p[1].data_ptr()) if p else None for p in c.entries]
        c.prefill = False
        for layer in range(32):
            if layer not in MIDDLE16.answer_skip:
                c.update(torch.zeros((1, 1, 1, 2)), torch.zeros((1, 1, 1, 2)), layer)
        c.logical_length = 10
        summary = summarize_cache(c, self.layout, MIDDLE16, False)
        self.assertEqual(summary['retained_image_layers'], list(range(8, 24)))
        self.assertEqual(pointers, [(p[0].data_ptr(), p[1].data_ptr()) if p else None for p in c.entries])
        self.assertTrue(all(c.entries[i] is None for i in MIDDLE16.answer_skip))

    def test_image_and_circular_units_not_split(self):
        samples = [dict(key='a', benchmark='GQA', image_sha256='x'),
                   dict(key='b', benchmark='MMBench', image_sha256='x', circular_group=5),
                   dict(key='c', benchmark='MMBench', image_sha256='y', circular_group=5),
                   dict(key='d', benchmark='MME', image_sha256='z'),
                   dict(key='e', benchmark='MME', image_sha256='z')]
        first = partition(samples)
        self.assertEqual(first, partition(samples))
        a = first['assignment']
        self.assertEqual(a['a'], a['b'])
        self.assertEqual(a['b'], a['c'])
        self.assertEqual(a['d'], a['e'])


if __name__ == '__main__':
    unittest.main()

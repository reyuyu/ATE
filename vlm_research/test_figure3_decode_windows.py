import unittest
from collections import Counter
import numpy as np
import test_early_joint_attention as legacy
from paper_rq1_early_joint_attention.statistics import bootstrap
from figure3_decode_windows.run import CONDITIONS, cohort, rank_for


class Figure3Tests(unittest.TestCase):
    def test_exact_five_windows(self):
        self.assertEqual([c['layers'] for k, c in CONDITIONS.items() if k != 'Full'],
            [list(range(s, s + 6)) for s in [0, 6, 12, 18, 26]])
        for key, condition in CONDITIONS.items():
            if key == 'Full':
                continue
            outputs, trace, _ = legacy.Tests().scoped(condition['mode'], condition['layers'])
            self.assertEqual(trace.edits, 6)
            for k in range(32):
                self.assertEqual(outputs[0][k].sum(), 24)
                self.assertEqual(outputs[1][k].sum(), 0 if k in condition['layers'] else 6)

    def test_original3000_and_image_sharding(self):
        samples = cohort()
        self.assertEqual(len(samples), 3000)
        self.assertEqual(set(Counter(s['task'] for s in samples).values()), {500})
        seen = {}
        for sample in samples:
            rank = rank_for(sample)
            self.assertIn(rank, [0, 1, 2])
            self.assertEqual(seen.setdefault(sample['image_sha256'], rank), rank)

    def test_five_comparison_correction(self):
        stats = bootstrap(np.array([[1, 0, -1, 0, 1], [0, 0, 1, 0, 0], [-1, 0, 0, 0, -1]]),
            ['a', 'a', 'b'], [5] * 5, repetitions=1000)
        for row in stats:
            self.assertLessEqual(row['family95_low'], row['ci95_low'])
            self.assertGreaterEqual(row['family95_high'], row['ci95_high'])

    def test_length_groups(self):
        from figure3_decode_windows.analyze import length_group
        self.assertEqual(length_group(dict(token_ids=[5, 2], eos=True)), 'EOS-only tail')
        self.assertEqual(length_group(dict(token_ids=[5, 6, 2], eos=True)), 'Content tail')
        self.assertEqual(length_group(dict(token_ids=[2], eos=True)), 'Immediate EOS')
        self.assertEqual(length_group(dict(token_ids=[5] * 32, eos=False)), 'Full missing EOS')


if __name__ == '__main__':
    unittest.main()

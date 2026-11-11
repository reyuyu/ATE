import unittest
import qwen_svh_r_full as run


class FrozenConfiguration(unittest.TestCase):
    def test_layers(self):
        self.assertEqual(run.LAYERS, list(range(19, 36)))
        self.assertEqual(len(run.LAYERS), 17)
        self.assertFalse(set(run.LAYERS) & set(range(1, 7)))

    def test_counts(self):
        self.assertEqual(sum(run.COUNTS.values()), 35346)
        self.assertEqual(run.COUNTS['POPE'], 9000)

    def test_status(self):
        self.assertEqual(run.state({'status': 'SKIPPED_OOM'}), 'SKIPPED_OOM')
        self.assertEqual(run.state({'state': 'BLOCKED_OOM'}), 'BLOCKED_OOM')

    def test_compact_retains_failures(self):
        r = run.compact({'status': 'SKIPPED_OOM', 'error': 'OOM', 'input_binding': {}})
        self.assertEqual(r['state'], 'SKIPPED_OOM')
        self.assertNotIn('score', r)
        self.assertEqual(r['error'], 'OOM')

    def test_score_normalization(self):
        r = run.compact({'scoring': {'score': .6, 'parsed': 'cat'}})
        self.assertEqual(r['score'], .6)
        self.assertEqual(r['scoring']['parsed'], 'cat')


if __name__ == '__main__':
    unittest.main()

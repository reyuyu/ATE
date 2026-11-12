import unittest
from collections import defaultdict

from .freeze import choose_cases, choose_pretest, inventory, shards, GATE, scorers


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = inventory()

    def test_scope_and_regression(self):
        cases = choose_cases(self.rows)
        self.assertEqual(len(cases), 24)
        self.assertIn('GQA:20705853', [c['sample']['key'] for c in cases])
        self.assertEqual(GATE['mean_abs'], .02)

    def test_complete_groups(self):
        mapping, _ = shards(self.rows)
        groups = defaultdict(set)
        for r in self.rows:
            groups[('image', r['image_sha256'])].add(mapping[r['key']]['shard'])
            if r['benchmark'] == 'MME':
                groups[('MME', r['pair_id'])].add(mapping[r['key']]['shard'])
            if r['benchmark'] == 'MMBench':
                groups[('MMBench', r['circular_group'])].add(mapping[r['key']]['shard'])
        self.assertTrue(all(len(v) == 1 for v in groups.values()))
        chosen = choose_pretest(self.rows)
        self.assertEqual(len(chosen), 48)
        for task, field in [('MME', 'pair_id'), ('MMBench', 'circular_group')]:
            keys = {r[field] for r in chosen if r['benchmark'] == task}
            self.assertEqual({r['key'] for r in chosen if r['benchmark'] == task},
                {r['key'] for r in self.rows if r['benchmark'] == task and r[field] in keys})

    def test_gold_is_not_prompt(self):
        for r in self.rows:
            if r['benchmark'] == 'ScienceQA-IMG':
                self.assertIsInstance(r['gold'], int)
                self.assertEqual(len(r['options']), len(set(r['options'])))
                self.assertNotIn('lecture', r)
                self.assertNotIn('solution', r)
                self.assertIn('Options:', r['body'])
            if r['benchmark'] == 'TextVQA':
                self.assertEqual(len(r['gold']), 10)
                self.assertIn('Reference OCR token:', r['body'])

    def test_scoring_sources(self):
        s = scorers()
        s.adapters()
        row = next(r for r in self.rows if r['benchmark'] == 'ScienceQA-IMG')
        self.assertEqual(s.score(row, list(row['options'])[row['gold']])['score'], 1)
        self.assertEqual(s.score(dict(benchmark='POPE', gold='no'), 'No.')['score'], 1)


if __name__ == '__main__':
    unittest.main()

"""Fixed-set and cohort regressions; no model load or CUDA."""
import json
import unittest
from late_windows import CONDITIONS, OUT, prior


class Tests(unittest.TestCase):
    def test_exact_requested_sets(self):
        self.assertEqual(CONDITIONS, {'L24_L35': list(range(24, 36)), 'L20_L34': list(range(20, 35))})
        self.assertEqual([len(v) for v in CONDITIONS.values()], [12, 15])

    def test_only_declared_delta_from_previous(self):
        old = set(range(24, 35))
        self.assertEqual(set(CONDITIONS['L24_L35'])-old, {35})
        self.assertEqual(set(CONDITIONS['L20_L34'])-old, {20, 21, 22, 23})
        self.assertTrue(all(not set(v) & set(range(1, 7)) for v in CONDITIONS.values()))

    def test_same_original_discovery(self):
        rows = json.loads((prior.OUT / 'sample_manifest.json').read_text())
        self.assertEqual(len(rows), 585)
        self.assertEqual(len({r['key'] for r in rows}), 585)
        self.assertEqual(sum(r['analysis_task'] == 'TextVQA' for r in rows), 250)
        shards = {}
        for row in rows:
            self.assertEqual(row['split'], 'Discovery')
            self.assertEqual(shards.setdefault(row['image_sha256'], row['shard']), row['shard'])
        self.assertNotEqual(OUT, prior.OUT)

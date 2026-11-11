"""Exact fixed windows, original image clusters, and unchanged runtime."""
import json
import unittest
from shifted15 import BASE_OUT, CONDITIONS, EXPECTED, OUT, ROOT, runtime


class Tests(unittest.TestCase):
    def test_exact_requested_windows(self):
        self.assertEqual(CONDITIONS, {
            'L19_L33': list(range(19, 34)), 'L18_L32': list(range(18, 33)),
            'L17_L31': list(range(17, 32)), 'L16_L30': list(range(16, 31))})
        self.assertTrue(all(len(v) == 15 for v in CONDITIONS.values()))
        self.assertEqual(EXPECTED, 2340)
        self.assertTrue(all(not set(v) & set(range(1, 7)) for v in CONDITIONS.values()))

    def test_readonly_runtime_configuration(self):
        self.assertEqual(runtime.OUT, OUT)
        self.assertEqual(runtime.ROOT, ROOT)
        self.assertEqual(runtime.CONDITIONS, CONDITIONS)
        self.assertNotEqual(OUT, BASE_OUT)

    def test_original_cohort_and_image_shards(self):
        rows = json.loads((BASE_OUT/'sample_manifest.json').read_text())
        self.assertEqual(len(rows), 585)
        self.assertEqual(len({r['key'] for r in rows}), 585)
        self.assertEqual(sum(r['analysis_task'] == 'TextVQA' for r in rows), 250)
        shards = {}
        for row in rows:
            self.assertEqual(row['split'], 'Discovery')
            self.assertEqual(shards.setdefault(row['image_sha256'], row['shard']), row['shard'])

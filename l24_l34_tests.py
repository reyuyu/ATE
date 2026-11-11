"""Small CPU regressions for the one-candidate experiment."""
import json
import unittest
from l24_l34 import LAYERS, NAME, PREVIOUS


class Tests(unittest.TestCase):
    def test_exact_user_set(self):
        self.assertEqual(LAYERS, list(range(24, 35)))
        self.assertEqual(len(LAYERS), 11)
        self.assertNotIn(35, LAYERS)
        self.assertEqual(NAME, 'L24_L34')

    def test_answer_skip_unchanged(self):
        self.assertFalse(set(LAYERS) & set(range(1, 7)))

    def test_original_discovery_only(self):
        rows = [r for r in json.loads((PREVIOUS / 'sample_manifest.json').read_text()) if r['split'] == 'Discovery']
        self.assertEqual(len(rows), 585)
        self.assertEqual(len({r['key'] for r in rows}), 585)
        self.assertEqual(sum(r['analysis_task'] == 'TextVQA' for r in rows), 250)
        shards = {}
        for row in rows:
            self.assertEqual(shards.setdefault(row['image_sha256'], row['shard']), row['shard'])


if __name__ == '__main__':
    unittest.main()

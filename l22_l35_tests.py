"""Check the corrected layer boundary and exact unchanged cohort."""
import json
import unittest
from pathlib import Path
from l22_l35 import LAYERS, MODEL, NAME, ORIGINAL_OUT, OUT, ROOT, runtime


class Tests(unittest.TestCase):
    def test_actual_model_layer_boundary(self):
        count = json.loads((Path(MODEL)/'config.json').read_text())['text_config']['num_hidden_layers']
        self.assertEqual(count, 36)
        self.assertEqual(LAYERS, list(range(22, count)))
        self.assertEqual(len(LAYERS), 14)
        self.assertNotIn(36, LAYERS)

    def test_adapter_uses_frozen_worker_and_one_set(self):
        self.assertEqual(runtime.OUT, OUT)
        self.assertNotEqual(OUT, ORIGINAL_OUT)
        self.assertEqual(Path(runtime.__file__), ROOT/'l22_l35.py')
        self.assertEqual(runtime.NAME, NAME)
        self.assertEqual(runtime.LAYERS, LAYERS)
        self.assertEqual(set(LAYERS)-set(range(24, 36)), {22, 23})
        self.assertFalse(set(LAYERS) & set(range(1, 7)))

    def test_same_original_discovery_only(self):
        rows = json.loads((ORIGINAL_OUT/'sample_manifest.json').read_text())
        self.assertEqual(len(rows), 585)
        self.assertEqual(len({r['key'] for r in rows}), 585)
        self.assertEqual(sum(r['analysis_task'] == 'TextVQA' for r in rows), 250)
        shards = {}
        for row in rows:
            self.assertEqual(row['split'], 'Discovery')
            self.assertEqual(shards.setdefault(row['image_sha256'], row['shard']), row['shard'])

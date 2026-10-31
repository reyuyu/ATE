import json
import unittest
from benchmarks.dataset_registry import digest
from first_answer_mass.finalize import original_result_digest


class SerializationTests(unittest.TestCase):
    def test_integer_layer_hash_is_restored_without_data_changes(self):
        value=dict(rows=[{'image_mass':.25}],spatial_maps={k:[[k*.001]] for k in range(32)})
        decoded=json.loads(json.dumps(value))
        before=json.dumps(decoded,sort_keys=True)
        self.assertNotEqual(digest(value),digest(decoded))
        self.assertEqual(original_result_digest(decoded),digest(value))
        self.assertEqual(before,json.dumps(decoded,sort_keys=True))

    def test_data_corruption_still_fails_the_original_digest(self):
        value=dict(rows=[],spatial_maps={k:[[k*.001]] for k in range(32)})
        decoded=json.loads(json.dumps(value))
        decoded['spatial_maps']['15'][0][0]+=.001
        self.assertNotEqual(original_result_digest(decoded),digest(value))

    def test_nonspatial_checkpoints_and_manifests_unchanged(self):
        for value in [dict(rows=[],spatial_maps={}),dict(model='frozen',questions=3000)]:
            self.assertEqual(original_result_digest(value),digest(value))

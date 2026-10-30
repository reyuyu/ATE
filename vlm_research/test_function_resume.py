import json
import tempfile
import unittest
from pathlib import Path
from paper_rq1_single_layer_gqa6.run import atomic_json
from paper_rq1_function_attention_sparsify.resume_collection import archive_attempt


class ResumeTests(unittest.TestCase):
    def test_preserves_scientific_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for p in ['environment.json','resource_check.json','failure_1.json','checkpoints/image.json']:
                atomic_json(root/p,dict(identity=p))
            before=(root/'checkpoints/image.json').read_bytes();archive=archive_attempt(root)
            self.assertEqual(before,(root/'checkpoints/image.json').read_bytes())
            self.assertTrue((archive/'environment.json').exists());self.assertTrue((archive/'failure_1.json').exists())
            self.assertFalse((root/'resource_check.json').exists())

    def test_immutable_checkpoint_still_rejects_different_result(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sample.json';atomic_json(path,dict(result=1))
            with self.assertRaises(ValueError):atomic_json(path,dict(result=2))


if __name__=='__main__':unittest.main()

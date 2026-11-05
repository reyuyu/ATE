import unittest

from ..accuracy_autoresearch import tests as previous
from .runtime import *


class Tests(previous.Tests):
    def test_original_clock_is_not_reset(self):
        self.assertEqual(read(ROOT/'task_deadline.json'),read(PREVIOUS/'task_deadline.json'))

    def test_unchanged_execution_and_extra_source_binding(self):
        self.assertEqual(original.source_hashes(),read(PRIOR/'execution_freeze.json')['sources'])
        self.assertIn('vlm_research/accuracy_autoresearch_continued/runtime.py',sources())

    def test_historical_checkpoint_is_verified_not_copied(self):
        pool=read(PRIOR/'sample_manifest.json')
        k=read(PRIOR/'split_manifest.json')['coarse'][0]
        value=read_record(config('native'),k,pool[k])
        self.assertIsNotNone(value)
        self.assertEqual(value['generation']['score'],score(pool[k],value['generation']['text']))
        self.assertNotEqual(OUT,PRIOR)


if __name__=='__main__': unittest.main(verbosity=2)

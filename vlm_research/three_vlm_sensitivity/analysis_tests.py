"""CPU-only statistical fixtures; no synthetic experiment results or figures."""
import unittest
import base64
import hashlib
from unittest.mock import patch
import numpy as np

from .analyze import bootstrap
from .finish import gpu_snapshot
from .common import OUT, write


class StatisticalTests(unittest.TestCase):
    def test_constant(self):
        values = np.full((4, 3), 2.)
        result = bootstrap(values, ['a', 'a', 'b', 'c'], 2000)
        self.assertTrue(np.array_equal(result, np.full((2, 3), 2.)))

    def test_question_weighted_whole_clusters(self):
        values = np.array([[0.], [10.], [20.]])
        result = bootstrap(values, ['a', 'a', 'b'], 2000)
        # The two-cluster resampling outcomes are AA=5, AB=10, BB=20, not 12.5.
        self.assertTrue(np.array_equal(result[:, 0], np.array([5., 20.])))

    def test_repeatable(self):
        values = np.arange(18).reshape(6, 3)
        a = bootstrap(values, ['a', 'a', 'b', 'c', 'd', 'e'], 2000)
        b = bootstrap(values, ['a', 'a', 'b', 'c', 'd', 'e'], 2000)
        self.assertTrue(np.array_equal(a, b))

    def test_duplicates_not_independent(self):
        values = np.array([[1., -1.], [-1., 1.]])
        result = bootstrap(values, ['same_image', 'same_image'], 2000)
        self.assertTrue(np.array_equal(result, np.zeros((2, 2))))

    def test_snapshot_preserves_truncated_utf8(self):
        raw = b'process /tmp/\xe4\xb8'
        with patch('vlm_research.three_vlm_sensitivity.finish.subprocess.check_output', return_value=raw):
            snapshot = gpu_snapshot()
        self.assertEqual(base64.b64decode(snapshot['snapshot_raw_base64']), raw)
        self.assertEqual(snapshot['snapshot_raw_sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(snapshot['utf8_replacement_count'], 1)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StatisticalTests))
    write(OUT / 'analysis_cpu_tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        scope='CPU-only statistical fixtures; no experiment outputs fabricated'))
    raise SystemExit(0 if result.wasSuccessful() else 1)

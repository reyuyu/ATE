"""L22-L35 CPU checks with the previously corrected strict cache fixture."""
import io
import time
import unittest
from l22_l35 import OUT, sha, write
from cpu_cache_checks import CacheChecks, flatten


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(['l22_l35_tests', 'tests', 'wider_tests'])
    replaced = 'tests.Tests.test_independent_cache_mutation'
    kept = [t for t in flatten(suite) if t.id() != replaced]
    assert len(kept) == 15
    kept.extend(flatten(unittest.defaultTestLoader.loadTestsFromTestCase(CacheChecks)))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(kept))
    write(OUT/'cpu_validation.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL', count=result.testsRun,
        log=stream.getvalue(), time=time.time(), source_sha256=sha(__file__),
        historical_fixture_failure='Preserved in the L24-L34 result directory; undefined unused capacity is not floating-point output.',
        replacement='Exact live rows, all allocated bytes, independent pointers, unchanged source; initialized full-capacity torch.equal.',
        production_code_and_thresholds_changed=False))
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)

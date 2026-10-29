"""Record structured CPU acceptance against the exact execution source hashes."""
import io
import json
import unittest

from benchmarks.dataset_registry import digest, freeze_json
from .inventory import OUTPUT
from .run import source_hashes


def main():
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_paper_rq1'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    value = dict(passed=result.wasSuccessful(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), source_sha256=source_hashes())
    key = digest(value['source_sha256'])
    freeze_json(OUTPUT/'cpu_attempts'/f'{key}.json', value)
    log = OUTPUT/'cpu_attempts'/f'{key}.txt'
    if not log.exists():
        log.write_text(buffer.getvalue())
    latest = OUTPUT/'cpu_validation.json'
    if latest.exists() and json.loads(latest.read_text()) != value:
        if (OUTPUT/'execution_manifest.json').exists():
            raise ValueError('Do not replace CPU acceptance after execution freeze')
        previous = json.loads(latest.read_text())
        freeze_json(OUTPUT/'cpu_attempts'/f"{digest(previous['source_sha256'])}.json", previous)
        latest.write_text(json.dumps(value, indent=2)+'\n')
    else:
        freeze_json(latest, value)
    (OUTPUT/'cpu_tests_final.txt').write_text(buffer.getvalue())
    print(buffer.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()

import io
import json
import unittest

from benchmarks.dataset_registry import digest,freeze_json
from .protocol import OUTPUT
from .run import source_hashes


def main():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_gqa6_update_decomposition'))
    value=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),source_sha256=source_hashes())
    key=digest(value['source_sha256']);freeze_json(OUTPUT/'cpu_attempts'/f'{key}.json',value)
    path=OUTPUT/'cpu_attempts'/f'{key}.txt'
    if not path.exists():
        with path.open('x') as f:f.write(stream.getvalue())
    path=OUTPUT/'cpu_validation.json'
    if path.exists() and json.loads(path.read_text())!=value:
        if (OUTPUT/'execution_manifest.json').exists():raise ValueError('No acceptance edits after execution freeze')
        with path.open('w') as f:json.dump(value,f,indent=2)
    else:freeze_json(path,value)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()

"""Record CPU acceptance on exact sources before immutable GPU execution freeze."""
import io
import json
import unittest

from benchmarks.dataset_registry import digest, freeze_json
from .protocol import OUTPUT
from .run import source_hashes


def main():
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_paper_rq1_single_layer_gqa6'))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    value=dict(passed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),source_sha256=source_hashes())
    key=digest(value['source_sha256'])
    freeze_json(OUTPUT/'cpu_attempts'/f'{key}.json',value)
    log=OUTPUT/'cpu_attempts'/f'{key}.txt'
    if not log.exists():
        with log.open('x') as f:f.write(stream.getvalue())
    latest=OUTPUT/'cpu_validation.json'
    if latest.exists() and json.loads(latest.read_text())!=value:
        if (OUTPUT/'execution_manifest.json').exists():raise ValueError('No CPU acceptance overwrite after GPU freeze')
        with latest.open('w') as f:json.dump(value,f,indent=2)
    else:freeze_json(latest,value)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()

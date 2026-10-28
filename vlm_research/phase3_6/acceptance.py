"""Fixed-environment CPU acceptance with a runtime-only model path override."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import os
from pathlib import Path
import subprocess
import sys
import unittest

from .protocol import ROOT, atomic, digest, frozen_sources, git
from .run import packages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--audit-dir', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU acceptance requires CUDA_VISIBLE_DEVICES empty before startup')
    versions = packages()
    args.audit_dir.mkdir(parents=True, exist_ok=False)
    import vlm_common
    vlm_common.default_model = lambda: str(args.model_path.resolve())
    results = {}
    for name in ['inheritance', 'all']:
        suite = (unittest.defaultTestLoader.loadTestsFromName('test_attention_inheritance') if name == 'inheritance'
                 else unittest.defaultTestLoader.discover(str(ROOT), pattern='test*.py', top_level_dir=str(ROOT)))
        expected = suite.countTestCases()
        with (args.audit_dir / f'{name}.txt').open('x') as stream, redirect_stdout(stream), redirect_stderr(stream):
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        passed = result.wasSuccessful() and not result.skipped and result.testsRun == expected
        if name == 'inheritance': passed = passed and expected == 7
        else: passed = passed and expected >= 114
        results[name] = dict(passed=passed, tests=result.testsRun, expected=expected,
                             failures=len(result.failures), errors=len(result.errors), skips=len(result.skipped))
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    (args.audit_dir / 'pip_check.txt').write_text(check.stdout+check.stderr)
    (args.audit_dir / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    passed = all(r['passed'] for r in results.values()) and check.returncode == 0
    atomic(args.audit_dir / 'validation.json', dict(passed=passed, code_commit=git('rev-parse', 'HEAD'),
           source_binding=digest(frozen_sources()), packages=versions, tests=results,
           interpreter=sys.executable, gpu_forward=False))
    print(results, 'PASS' if passed else 'FAIL')
    if not passed: raise SystemExit(1)


if __name__ == '__main__': main()

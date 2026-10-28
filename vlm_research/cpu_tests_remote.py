"""Run the original CPU suite with only its local model path adapted."""
import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--suite', choices=['inheritance', 'all'], default='all')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not args.model_path.is_dir():
        parser.error('Model directory missing; restore the fixed snapshot first')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import vlm_common
    from attention_inheritance_remote import PINS
    versions = {name: importlib.metadata.version(name) for name in PINS}
    if versions != PINS:
        parser.error('Fixed requirements.txt versions are required; actual=' + repr(versions))
    original = vlm_common.default_model
    vlm_common.default_model = lambda: str(args.model_path.resolve())
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            if args.suite == 'inheritance':
                suite = unittest.defaultTestLoader.loadTestsFromName('test_attention_inheritance')
            else:
                suite = unittest.defaultTestLoader.discover(str(vlm_common.ROOT), pattern='test_*.py')
            result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    finally:
        vlm_common.default_model = original
        (args.output_dir / 'cpu_tests.txt').write_text(buffer.getvalue())
        print(buffer.getvalue(), end='')
    expected = 7 if args.suite == 'inheritance' else 101
    passed = result.wasSuccessful() and result.testsRun == expected and not result.skipped
    report = dict(passed=passed, suite=args.suite, tests_run=result.testsRun, expected_tests=expected,
                  skipped=result.skipped, failures=[(str(t), text) for t, text in result.failures],
                  errors=[(str(t), text) for t, text in result.errors],
                  model_path=str(args.model_path.resolve()), python=platform.python_version(),
                  interpreter=sys.executable, timestamp=datetime.now(timezone.utc).isoformat(),
                  packages=versions)
    (args.output_dir / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

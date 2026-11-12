"""CPU-only executable test record before the later GPU stage is frozen."""
import io
import unittest

import torch

from .freeze import OUT
from ..qwen3vl_overnight.common import write


def main():
    assert not torch.cuda.is_initialized()
    suite=unittest.defaultTestLoader.loadTestsFromNames([
        'vlm_research.qwen3vl_overnight.tests','vlm_research.qwen3vl_svhr_search.tests'])
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),log=log.getvalue(),
        cuda_initialized=torch.cuda.is_initialized(),scope='CPU logic only; no claim of physical GPU validation')
    write(OUT/'cpu_validation.json',record)
    print(log.getvalue())
    assert result.wasSuccessful() and not torch.cuda.is_initialized()


if __name__=='__main__':
    main()

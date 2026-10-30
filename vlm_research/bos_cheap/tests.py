import io
import time
import unittest
from benchmarks.dataset_registry import digest,freeze_json
from .protocol import OUT
from .run import sources


def main():
    from test_bos_cheap import Tests
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    text=stream.getvalue();print(text,flush=True);source=sources();key=digest(source);folder=OUT/'cpu_attempts';folder.mkdir(parents=True,exist_ok=True)
    p=folder/f'{key}_{time.time_ns()}'
    freeze_json(p.with_suffix('.json'),dict(passed=result.wasSuccessful(),tests=result.testsRun,source_sha256=source))
    p.with_suffix('.txt').write_text(text)
    if result.wasSuccessful():
        if (OUT/'cpu_validation.json').exists():(OUT/'cpu_validation.json').rename(folder/f'previous_{time.time_ns()}.json')
        freeze_json(OUT/'cpu_validation.json',dict(passed=True,tests=result.testsRun,source_sha256=source,attempt=str(p.with_suffix('.json').relative_to(OUT))))
    else:raise SystemExit(1)


if __name__=='__main__':main()

"""Post-collection compatibility for the frozen integer-key checkpoint hash."""
from pathlib import Path
import unittest
from benchmarks.dataset_registry import digest, freeze_json, sha256
from . import analyze
from .run import OUT, ROOT, read, sources


def original_result_digest(value):
    # JSON converts layer keys to strings. Recreate the writer's integer-key
    # representation before checking its original sorted-key digest; no values change.
    if isinstance(value, dict) and 'rows' in value and 'spatial_maps' in value:
        maps = value['spatial_maps']
        if maps:
            assert set(maps) == {str(k) for k in range(32)}
            value = dict(value, spatial_maps={int(k): v for k,v in maps.items()})
    return digest(value)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromName('test_first_answer_mass_finalize')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    assert read(OUT / 'execution_manifest.json')['source'] == sources()
    freeze_json(OUT / 'postprocess_compatibility.json', dict(
        reason='Frozen writer hashed integer layer keys; JSON roundtrip stringified them. Restore only those key types before comparing the original digest.',
        scope='Postprocessing only; frozen collection sources, checkpoint bytes, values and thresholds unchanged.',
        tests=result.testsRun, passed=True, source={name:sha256(ROOT/name) for name in
            ['first_answer_mass/finalize.py','test_first_answer_mass_finalize.py']}))
    analyze.digest = original_result_digest
    analyze.main()


if __name__=='__main__':
    main()

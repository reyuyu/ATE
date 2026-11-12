"""Small deterministic tests of paired, question-weighted cluster statistics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import analyze


class Tests(unittest.TestCase):
    def test_zero_paired_difference(self):
        actual = analyze.paired_bootstrap(np.zeros((5, 38)), ['a', 'a', 'b', 'c', 'd'], repetitions=100)
        self.assertTrue(np.array_equal(actual, np.zeros((100, 38))))

    def test_clustered_question_weighting(self):
        values = np.array([[1., -1.], [3., -3.], [10., -10.]])
        result = analyze.paired_bootstrap(values, ['a', 'a', 'b'], repetitions=100, seed=17)
        rng = np.random.default_rng(17)
        weights = rng.multinomial(2, [.5, .5], size=100)
        expected = (weights @ np.array([[4., -4.], [10., -10.]]))/(weights @ np.array([2., 1.]))[:, None]
        self.assertTrue(np.array_equal(result, expected))

    def test_paired_reproducibility_and_correction(self):
        values = np.arange(300, dtype=float).reshape(10, 30)
        a = analyze.paired_bootstrap(values, [str(i//2) for i in range(10)], repetitions=2000)
        b = analyze.paired_bootstrap(values, [str(i//2) for i in range(10)], repetitions=2000)
        self.assertTrue(np.array_equal(a, b))
        normal = np.quantile(a, [.025, .975], axis=0)
        corrected = np.quantile(a, [.025/36, 1-.025/36], axis=0)
        self.assertTrue((corrected[0] <= normal[0]).all() and (corrected[1] >= normal[1]).all())

    def test_summary_and_five_figure_outputs(self):
        # Synthetic fixtures stay in a temporary directory, never research results.
        samples = [dict(task=t, question_id=t+str(i), image_sha256=t+str(i))
            for t in ('ChooseAttr', 'ChooseCat', 'ChooseRel', 'CompareAttr', 'LogicalObj', 'QueryAttr') for i in range(2)]
        records = {}
        for s in samples:
            for j, c in enumerate(analyze.CONDITIONS):
                correct = int(j % 3 != 2)
                records[analyze.key(s), c] = dict(generation=dict(correct=correct, first_token_flip=0,
                    text='test', eos=True, truncated=False, empty=False, repeated_trigram=False, generated_tokens=2))
        original_out = analyze.OUT
        try:
            with tempfile.TemporaryDirectory(prefix='qwen-writer-analysis-unit-') as temp:
                analyze.OUT = Path(temp)
                reference = analyze.read(original_out / 'llava_reference_summary.json')
                analyze.seal(analyze.OUT / 'llava_reference_summary.json', reference)
                overall = analyze.summarize(samples, records, 'fixture', bootstrap=False)
                for r in overall:
                    r.update(ci_low=r['delta_accuracy_pp']-1, ci_high=r['delta_accuracy_pp']+1,
                        Bonferroni36_low=r['delta_accuracy_pp']-2, Bonferroni36_high=r['delta_accuracy_pp']+2)
                tasks = []
                for task in sorted({s['task'] for s in samples}):
                    tasks += analyze.summarize([s for s in samples if s['task'] == task], records, task, bootstrap=False)
                self.assertEqual(overall[0]['correct'], 12)
                self.assertEqual(overall[2]['correct_to_wrong'], 12)
                analyze.figures(overall, tasks)
                self.assertEqual(len(list(analyze.OUT.glob('*.png'))), 5)
                self.assertEqual(len(list(analyze.OUT.glob('*.pdf'))), 5)
                from PIL import Image
                for path in analyze.OUT.glob('*.png'):
                    pixels = np.asarray(Image.open(path).convert('RGB'))
                    self.assertGreater(float(pixels.std()), 5.)
        finally:
            analyze.OUT = original_out


if __name__ == '__main__':
    import sys
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    analyze.seal(analyze.OUT / 'analysis_tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        analysis_sha256=analyze.sha256(analyze.__file__), test_source_sha256=analyze.sha256(__file__)))
    assert result.wasSuccessful()

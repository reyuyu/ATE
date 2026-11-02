"""CPU-only regression of existing query-local hooks and output-blind sampling."""
import unittest
import torch

from .common import CONDITIONS, TASKS, OUT, select, key, write
from ..qwen3vl_early_windows.tests import Toy, Checks
from ..qwen3vl_early_windows.experiment import AttentionBypass, hook_counts
from ..internvl3_stage.engine import WindowHook


class BoundaryChecks(unittest.TestCase):
    def test_sampling_output_blind(self):
        rows = [dict(task=t, question_id=str(i), image_sha256=f'{t}_{i}', answer='left')
                for t in TASKS for i in range(45)]
        a = select(rows)
        b = select([dict(r, correct=False, answer='changed') for r in reversed(rows)])
        self.assertEqual([key(r) for r in a], [key(r) for r in b])
        self.assertEqual(len(a), 120)
        self.assertEqual(len({r['image_sha256'] for r in a}), 120)

    def test_all_frozen_windows(self):
        for name, conditions in CONDITIONS.items():
            for label, targets in conditions.items():
                model = Toy()
                x = torch.randn(1, 7, 9)
                baseline = hook_counts(model)
                constructor = (lambda noop: AttentionBypass(model, targets, 7, noop)) if name == 'qwen' else (
                    lambda noop: WindowHook(model.model.language_model.layers, targets, 7, noop))
                with constructor(False):
                    y = model(inputs_embeds=x, cache_position=torch.arange(7))
                    for i in range(36):
                        self.assertTrue(torch.equal(y[i][:, :-1], (x+1)[:, :-1]))
                        self.assertTrue(torch.equal(y[i][:, -1], torch.zeros_like(x[:, -1]) if i in targets else (x+1)[:, -1]))
                    for step in range(3):
                        z = model(inputs_embeds=x[:, :1], cache_position=torch.tensor([7+step]))
                        self.assertTrue(all(torch.count_nonzero(z[i]) == 0 for i in targets))
                with constructor(True):
                    y = model(inputs_embeds=x, cache_position=torch.arange(7))
                    self.assertTrue(all(torch.equal(v, x+1) for v in y))
                self.assertEqual(hook_counts(model), baseline)
                with self.assertRaises(RuntimeError):
                    with constructor(False):
                        raise RuntimeError('intentional cleanup check')
                self.assertEqual(hook_counts(model), baseline)


if __name__ == '__main__':
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Checks, BoundaryChecks)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(OUT / 'cpu_tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
          failures=len(result.failures), errors=len(result.errors), device='CPU only'))
    raise SystemExit(not result.wasSuccessful())

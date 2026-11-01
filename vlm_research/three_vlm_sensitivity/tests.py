import unittest
from contextlib import ExitStack
import torch

from ..internvl3_smoke.engine import P1OutputHook, hooks_count
from .metrics import metrics
from .common import OUT, write


class Tests(unittest.TestCase):
    def test_target_only(self):
        module = torch.nn.Linear(4, 4)
        x = torch.randn(1, 9, 4)
        native = module(x)
        before = hooks_count(module)
        with P1OutputHook(module, 8) as hook:
            off = module(x)
        self.assertEqual(hook.calls, 1)
        self.assertTrue(torch.equal(off[:, :8], native[:, :8]))
        self.assertEqual(int(torch.count_nonzero(off[:, 8])), 0)
        self.assertEqual(hooks_count(module), before)

    def test_noop(self):
        module = torch.nn.Linear(5, 5)
        x = torch.randn(1, 7, 5)
        with P1OutputHook(module, 6, noop=True):
            observed = module(x)
        self.assertTrue(torch.equal(module(x), observed))

    def test_exception_cleanup(self):
        module = torch.nn.Identity()
        try:
            with P1OutputHook(module, 3):
                raise RuntimeError('intentional')
        except RuntimeError:
            pass
        self.assertEqual(hooks_count(module), 0)

    def test_sign_and_reference(self):
        full = torch.tensor([3., 1., 0.])
        off = torch.tensor([0., 4., 1.])
        m = metrics(full, off)
        self.assertEqual(m['full_token_id'], 0)
        self.assertEqual(m['off_token_id'], 1)
        self.assertGreater(m['signed_loss'], 0)
        self.assertEqual(m['absolute_change'], abs(m['signed_loss']))
        self.assertEqual(m['top1_flip'], 1)

    def test_negative_loss_retained(self):
        m = metrics(torch.tensor([1., 0.]), torch.tensor([4., 0.]))
        self.assertLess(m['signed_loss'], 0)
        self.assertGreater(m['absolute_change'], 0)

    def test_mean_absolute_not_absolute_mean(self):
        x = torch.tensor([-2., 2.])
        self.assertEqual(float(x.abs().mean()), 2)
        self.assertEqual(float(x.mean().abs()), 0)

    def test_invalid_logits(self):
        with self.assertRaises(AssertionError):
            metrics(torch.tensor([float('nan'), 1.]), torch.tensor([0., 1.]))


if __name__ == '__main__':
    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    import transformers
    write(OUT / f'cpu_tests_{transformers.__version__}.json',
          dict(passed=result.wasSuccessful(), tests=result.testsRun, transformers=transformers.__version__))
    raise SystemExit(0 if result.wasSuccessful() else 1)

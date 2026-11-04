import unittest
from types import SimpleNamespace
import torch

from .stable import numerical_kernels, linear_fp32, linear_tiles
from .diagnose import compare_runs, error


class Tests(unittest.TestCase):
    def test_matched_linear_shape_and_weights(self):
        torch.manual_seed(7)
        linear = torch.nn.Linear(32, 40).half().eval()
        x = torch.randn(1, 17, 32).half()
        keep = torch.tensor([0, 1, 8, 9, 14, 16])
        before = linear.weight.detach().clone()
        with torch.inference_mode():
            a = linear_fp32(linear, x)[:, keep]
            b = linear_fp32(linear, x[:, keep])
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(before, linear.weight))
        self.assertEqual(b.dtype, torch.float16)

    def test_context_cleanup_exception(self):
        layers = [SimpleNamespace(modules=lambda: []) for _ in range(32)]
        head = torch.nn.Linear(3, 4)
        adapter = SimpleNamespace(kind='llava', layers=layers, model=SimpleNamespace(get_output_embeddings=lambda:head))
        original = head.forward
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with numerical_kernels(adapter, 'fp32_linears'):
                self.assertNotEqual(original, head.forward)
                raise RuntimeError('sentinel')
        self.assertEqual(original, head.forward)

    def test_canonical_tiles_keep_dtype_and_rows(self):
        torch.manual_seed(29)
        module=torch.nn.Linear(48,72).half()
        x=torch.randn(1,93,48).half()
        keep=torch.tensor([0,1,20,35,42,73,90,92])
        self.assertTrue(torch.equal(linear_tiles(module,x)[:,keep],linear_tiles(module,x[:,keep])))
        self.assertTrue(torch.equal(linear_tiles(module,x[:,:1]),module(x[:,:1])))

    def test_shared_prefix_only(self):
        a = dict(token_ids=[1, 2], text='a')
        b = dict(token_ids=[0, 2], text='b')
        p = compare_runs('llava', a, b, [torch.tensor([[1., 1.]]), torch.tensor([[500., 3.]])],
                         [torch.tensor([[1.01, 1.]]), torch.tensor([[-500., 3.]])], [], [])
        self.assertEqual(p['common_prefix_prediction_steps'], 1)
        self.assertTrue(p['numerical_pass'])
        self.assertFalse(p['greedy_pass'])
        self.assertAlmostEqual(p['max_abs_logits'], .01, places=6)

    def test_nonfinite_pattern(self):
        x = error(torch.tensor([0., -float('inf')]), torch.tensor([0., 1.]))
        self.assertFalse(x['finite_pattern_equal'])


if __name__ == '__main__':
    unittest.main()

"""CPU checks for the exact 38-condition layer schedule and inherited masking."""
import unittest
import torch

from .common import CONDITIONS, ORDER, writer_layers
from .engine import LayerWriterMask, blocked_edges, apply_mask
from ..qwen3vl_semantic_visual_writers.engine import roles


class Tests(unittest.TestCase):
    def setUp(self):
        self.layout = dict(input_ids=[1, 2, 9, 9, 9, 10, 11, 12, 13], input_length=9,
            image_positions=[2, 3, 4], p_minus_1=8, question_positions=[6, 7])
        self.mapping = roles(self.layout, 10)
        self.old = torch.zeros(1, 1, 9, 9, dtype=torch.bfloat16).masked_fill(
            torch.ones(9, 9, dtype=torch.bool).triu(1), torch.finfo(torch.bfloat16).min)

    def test_exact_matrix(self):
        self.assertEqual(len(CONDITIONS), 38)
        self.assertEqual(set(CONDITIONS), set(ORDER))
        self.assertEqual(writer_layers('All-Writer'), list(range(36)))
        self.assertEqual(writer_layers('No-Writer'), [])
        for drop in range(36):
            self.assertEqual(writer_layers(f'Drop-L{drop}'), [i for i in range(36) if i != drop])

    def test_only_target_layer_changes(self):
        for condition in CONDITIONS:
            for layer in range(36):
                writers = self.mapping['R'] if layer in writer_layers(condition) else []
                b = blocked_edges(self.mapping, writers, 9, 0, 'cpu')
                expected = torch.zeros(9, 9, dtype=torch.bool)
                expected[8, 2:5] = True
                if not writers:
                    expected[5:8, 2:5] = True
                self.assertTrue(torch.equal(b, expected))

    def test_all_decode_queries_always_image_off(self):
        for condition in CONDITIONS:
            for layer in range(36):
                writers = self.mapping['R'] if layer in writer_layers(condition) else []
                b = blocked_edges(self.mapping, writers, 9, 12, 'cpu')
                self.assertEqual(torch.where(b[0])[0].tolist(), [2, 3, 4])
                self.assertEqual(float(torch.softmax(apply_mask(None, b, 12).float(), -1)[0, 0, 0, 2:5].max()), 0.)

    def test_mask_causal_non_target_bos_self(self):
        b = blocked_edges(self.mapping, [], 9, 0, 'cpu')
        mask = apply_mask(self.old, b, 0)
        self.assertTrue(torch.equal(mask[0, 0][~b], self.old[0, 0][~b]))
        self.assertTrue((mask[0, 0, :, 0] == 0).all())
        self.assertTrue((mask[0, 0].diagonal() == 0).all())
        self.assertTrue(torch.isneginf(mask[0, 0][b]).all())
        logits = torch.randn(1, 2, 9, 9)
        full = torch.softmax(logits + self.old.float(), -1)
        changed = torch.softmax(logits + mask.float(), -1)
        self.assertTrue(torch.equal(full[:, :, :5], changed[:, :, :5]))
        self.assertEqual(float(changed[0, :, b].max()), 0.)

    def test_noop_exact_object(self):
        b = blocked_edges(self.mapping, [], 9, 0, 'cpu')
        self.assertIs(apply_mask(self.old, b, 0, True), self.old)
        self.assertIsNone(apply_mask(None, blocked_edges(self.mapping, [], 9, 12, 'cpu'), 12, True))

    def test_vision_end_and_dynamic_images(self):
        self.assertEqual(self.mapping['R'][0], 5)
        for count in [1, 16, 192, 900]:
            layout = dict(input_ids=[1]+[9]*count+[10, 11, 12], input_length=count+4,
                image_positions=list(range(1, count+1)), p_minus_1=count+3, question_positions=[count+2])
            self.assertEqual(roles(layout, 10)['R'], [count+1, count+2])

    def test_invalid_condition(self):
        with self.assertRaises(AssertionError):
            writer_layers('Keep-L26')

    def test_inherited_exception_cleanup(self):
        from torch import nn
        module = nn.Identity()
        ctl = LayerWriterMask.__new__(LayerWriterMask)
        ctl.handles = [module.register_forward_hook(lambda *args: None)]
        ctl.__exit__(RuntimeError, RuntimeError('forced'), None)
        self.assertFalse(module._forward_hooks)


if __name__ == '__main__':
    unittest.main(verbosity=2)

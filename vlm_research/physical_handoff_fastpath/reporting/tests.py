"""Independent arithmetic accounting checks; no real GPU model or dataset run."""
import types
import unittest

import torch
from transformers import CLIPVisionConfig, CLIPVisionModel

from ..ledger import Census


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, image_size=8, patch_size=4)
        cfg._attn_implementation = 'eager'
        self.vision = CLIPVisionModel(cfg).eval()
        self.head = torch.nn.Linear(32, 100)

    def get_image_features(self, pixels):
        return self.vision(pixels)

    def get_output_embeddings(self):
        return self.head

    def forward(self, pixels):
        return self.get_image_features(pixels)


class Tests(unittest.TestCase):
    @torch.inference_mode()
    def test_clip_bmm_supplement_matches_executed_profiler_flops(self):
        model = Toy()
        adapter = types.SimpleNamespace(kind='llava', model=model)
        census = Census(adapter)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
            with census.context():
                model(torch.zeros(1, 3, 8, 8))
        qkv = next(row for row in census.rows() if row['phase'] == 'vision' and row['component'] == 'qkv')
        layers = qkv['estimated_flops'] // (6 * 1 * 5 * 32 * 32)
        self.assertEqual(layers, 2)
        predicted = 4 * 1 * 4 * 5 * 5 * 8 * layers
        executed = sum(event.flops for event in prof.key_averages() if event.key == 'aten::bmm')
        self.assertEqual(predicted, executed)

    def test_census_restores_methods_after_exception(self):
        model = Toy()
        adapter = types.SimpleNamespace(kind='llava', model=model)
        original = torch.matmul
        vision = model.get_image_features
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with Census(adapter).context():
                raise RuntimeError('injected')
        self.assertIs(torch.matmul, original)
        self.assertEqual(model.get_image_features, vision)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))


if __name__ == '__main__':
    unittest.main()

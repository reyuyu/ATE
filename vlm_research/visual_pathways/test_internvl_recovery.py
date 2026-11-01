"""Guard the native last-position projection and unchanged no-cache contract."""
from types import SimpleNamespace
import unittest

import torch

from .engine import InternPrefill
from ..internvl3_smoke.engine import tensor_hash


class NativeProjectionTest(unittest.TestCase):
    def test_prefill_projects_only_last_position_without_cache(self):
        class LM(torch.nn.Module):
            def forward(self, inputs_embeds, attention_mask, **kwargs):
                self.kwargs = kwargs
                assert kwargs['logits_to_keep'] == 1
                assert kwargs['use_cache'] is False and kwargs['past_key_values'] is None
                return SimpleNamespace(logits=torch.arange(7.).reshape(1,1,7),past_key_values=None)
        model = torch.nn.Module()
        model.language_model = LM()
        runner = InternPrefill.__new__(InternPrefill)
        runner.model = model
        runner.layout = {'input_length':5}
        runner.inputs = dict(inputs_embeds=torch.ones(1,5,3),attention_mask=torch.ones(1,5))
        runner.fingerprint = tensor_hash(runner.inputs['inputs_embeds'])
        runner.layers = []
        runner.hooks = 0
        actual, audit, trace = runner.forward()
        self.assertTrue(torch.equal(actual,torch.arange(7.)))
        self.assertEqual(audit,{})
        self.assertIsNone(trace)
        runner.verify()


if __name__ == '__main__':
    unittest.main()

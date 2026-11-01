from types import SimpleNamespace
import unittest

import torch

from .internvl_recovery_rescue import IController


class RescueContractTest(unittest.TestCase):
    def test_only_nonvisual_pre_p1_queries_are_relay_and_skip_is_disabled(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn=torch.nn.Identity()
        model=torch.nn.Module()
        model.language_model=torch.nn.Module()
        runner=SimpleNamespace(model=model,inputs={},layers=[Layer() for _ in range(28)],
            layout={'input_length':9,'positions':{'image':[2,3]}})
        control=IController(runner,[0])
        self.assertIs(control.model,model.language_model)
        self.assertEqual(control.config['relay_off'],True)
        self.assertEqual(control.config['skip'],False)
        mask=torch.zeros(1,1,9,9).masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),-1e30)
        # Exercise the actual controller's intervention definition without a model/GPU.
        from .engine import masked
        relay=[i for i in range(control.n-1) if i not in control.image]
        changed,count=masked(mask,relay,control.image)
        self.assertTrue(torch.equal(changed[:,:,8],mask[:,:,8]))
        self.assertTrue(torch.equal(changed[:,:,2:4],mask[:,:,2:4]))
        self.assertEqual(count,8)


if __name__=='__main__':
    unittest.main()

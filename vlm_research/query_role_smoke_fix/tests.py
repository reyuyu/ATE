import unittest
from unittest.mock import patch
import torch
from query_role_attention.engine import generate
from .run import summary, Recorder


class Tests(unittest.TestCase):
    def test_difference_summary(self):
        a=torch.tensor([1.,2.,3.],dtype=torch.float16)
        b=torch.tensor([1.,2.,4.],dtype=torch.float16)
        d=summary(a,b)
        self.assertFalse(d['exact'])
        self.assertEqual(d['unequal_elements'],1)
        self.assertEqual(d['max_absolute_error'],1.)
        self.assertFalse(d['top1_flip'])

    def test_same_shape_strict_not_tolerance(self):
        a=torch.tensor([1.,2.])
        b=a.clone();b[0]=torch.nextafter(b[0],torch.tensor(2.))
        self.assertFalse(summary(a,b)['exact'])
        self.assertTrue(summary(a,a.clone())['exact'])

    def test_shape_and_nonfinite_diagnostics(self):
        self.assertFalse(summary(torch.ones(2,3),torch.ones(1,3))['exact'])
        self.assertIsNone(summary(torch.ones(2,3),torch.ones(1,3))['unequal_elements'])
        self.assertFalse(summary(torch.tensor([float('nan')]),torch.ones(1))['finite'])

    def test_lm_head_argument_is_explicit(self):
        # Stop on the first model call, before needing an actual cache/model.
        inputs=dict(input_ids=torch.tensor([[1]+[32000]*576+[2,3,4]]))
        inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
        model=type('Model',(),{'config':type('Config',(),{'image_token_index':32000})()})()
        class Seen(Exception):pass
        class Fake:
            handles=[]
            def __init__(self,*a,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
        def call(self,**kwargs):
            self.seen=kwargs
            raise Seen()
        type(model).__call__=call
        with patch('query_role_attention.engine.RoleBypass',Fake):
            for keep in (0,1):
                with self.assertRaises(Seen):generate(model,None,inputs,'Full',logits_to_keep=keep)
                self.assertEqual(model.seen['logits_to_keep'],keep)
                self.assertFalse(model.seen['output_attentions'])
        with self.assertRaises(ValueError):generate(model,None,inputs,'Full',logits_to_keep=2)

    def test_recorder_cleans_exception(self):
        model=torch.nn.Identity()
        model.language_model=type('LM',(),{'lm_head':torch.nn.Identity()})()
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with Recorder(model):raise RuntimeError('intentional')
        self.assertFalse(model._forward_hooks)
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model.language_model.lm_head._forward_pre_hooks)

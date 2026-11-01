import unittest
from unittest.mock import Mock
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM

from .common import CONDITIONS,TASKS,select,prediction
from .engine import WindowHook,first_metrics
from vlm_research.internvl3_smoke.engine import MassObserver, reduce_mass, hooks_count


class Tests(unittest.TestCase):
    def test_selection(self):
        rows=[dict(task=t,question_id=f'{t}{i}',image_sha256=f'{t}_image_{i}') for t in TASKS for i in range(30)]
        a=select(rows)
        self.assertEqual(a,select(list(reversed(rows))))
        self.assertEqual(len(a),120)
        self.assertEqual(len({r['image_sha256'] for r in a}),120)
        for t in TASKS:
            self.assertEqual(sum(r['task']==t for r in a),20)

    def test_fixed_conditions(self):
        self.assertEqual(list(CONDITIONS),['Full','W0','W1','W2','W3','WM','WL'])
        self.assertEqual(CONDITIONS['W2'],[2,3,4,5])
        self.assertEqual(CONDITIONS['WL'],[24,25,26,27])

    def test_locality_decode_cleanup(self):
        layers=[Mock(self_attn=Mock(o_proj=torch.nn.Identity())) for _ in range(3)]
        x=torch.randn(1,5,7)
        with WindowHook(layers,[1],5) as hook:
            y=layers[1].self_attn.o_proj(x)
            self.assertTrue(torch.equal(y[:,:-1],x[:,:-1]))
            self.assertEqual(torch.count_nonzero(y[:,-1]),0)
            self.assertTrue(torch.equal(layers[0].self_attn.o_proj(x),x))
            self.assertEqual(torch.count_nonzero(layers[1].self_attn.o_proj(torch.ones(1,1,7))),0)
        self.assertEqual(hook.calls[1],[5,1])
        self.assertTrue(torch.equal(layers[1].self_attn.o_proj(x),x))
        with self.assertRaises(RuntimeError):
            with WindowHook(layers,[1],5):
                raise RuntimeError('test')
        self.assertFalse(layers[1].self_attn.o_proj._forward_hooks)

    def test_real_cpu_eager_noop_mass(self):
        torch.manual_seed(1)
        cfg=Qwen2Config(vocab_size=37,hidden_size=32,intermediate_size=48,num_hidden_layers=3,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,
                       use_sliding_window=False,sliding_window=None,attention_dropout=0.)
        cfg._attn_implementation='eager'
        model=Qwen2ForCausalLM(cfg).eval()
        x=torch.tensor([[1,2,3,4,5,6]])
        with torch.no_grad():
            full=model(x).logits
            with WindowHook(model.model.layers,[0,1],6,noop=True):
                self.assertTrue(torch.equal(full,model(x).logits))
            layout=dict(input_length=6,p_minus_1=5,positions=dict(anchor=[0],image=[1,2],question=[3],template=[4],self=[5]))
            with MassObserver({i:l.self_attn for i,l in enumerate(model.model.layers)},layout) as observer:
                self.assertTrue(torch.equal(full,model(x).logits))
            self.assertEqual(len(observer.rows),3)
            with WindowHook(model.model.layers,[1],6):
                off=model(x).logits
            self.assertTrue(torch.equal(full[:,:-1],off[:,:-1]))
            self.assertFalse(torch.equal(full[:,-1],off[:,-1]))
        self.assertEqual(hooks_count(model),0)

    def test_signed_not_clipped(self):
        value=first_metrics(torch.tensor([2.,1.]),torch.tensor([3.,1.]))
        self.assertLess(value['signed_logp_loss'],0)
        self.assertGreater(value['absolute_logp_change'],0)
        self.assertAlmostEqual(first_metrics(torch.tensor([2.,1.]),torch.tensor([2.,1.]))['js_divergence'],0.)

    def test_mass_partition_failure(self):
        with self.assertRaises(AssertionError):
            reduce_mass(torch.ones(2,5)/5,dict(anchor=[0],image=[1,2],question=[2],template=[3],self=[4]))

    def test_prior_scoring(self):
        self.assertEqual(prediction('Blue.'),'blue')
        self.assertEqual(prediction('red!'),'red!')
        self.assertEqual(prediction('A red bus'),'a red bus')


if __name__=='__main__':
    unittest.main()

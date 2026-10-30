import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from transformers import LlamaConfig,LlamaForCausalLM
from paper_rq1_early_joint_attention.engine import JointAttention,generate
from paper_rq1_early_joint_attention.protocol import CONDITIONS
from paper_rq1_decode_layerwise_quality.engine import is_decode
from paper_rq1_decode_layerwise_quality.statistics import bootstrap,cluster_deltas


class Attention(nn.Module):
    def __init__(self):super().__init__();self.o_proj=nn.Identity()
    def forward(self,hidden_states,cache_position=None,past_key_value=None):return self.o_proj(hidden_states*2)


class Tests(unittest.TestCase):
    def model(self):return SimpleNamespace(layers=[SimpleNamespace(self_attn=Attention()) for _ in range(32)])
    def call(self,m,k,decode=False,n=4):
        q=1 if decode else n
        cache=SimpleNamespace(get_seq_length=lambda layer:n) if decode else None
        return m.layers[k].self_attn(torch.ones(1,q,3),cache_position=torch.tensor([n]) if decode else torch.arange(n),past_key_value=cache)
    def scoped(self,mode,layers=(0,1,2,3,4),noop=False):
        m=self.model();outputs=[]
        with patch('paper_rq1_early_joint_attention.engine.model_layers',lambda m:m.layers):
            with JointAttention(m,layers,mode,4,noop,True) as h:
                for decode in (False,True):outputs.append([self.call(m,k,decode) for k in range(32)])
        return outputs,h,m
    def test_frozen_matrix(self):
        self.assertEqual(len(CONDITIONS),11)
        self.assertEqual(CONDITIONS['S8']['layers'],[2,3,4,5])
    def test_indices_32(self):
        for k in range(32):
            o,h,_=self.scoped('Combined',[k]);self.assertEqual(h.edits,2)
            self.assertEqual(o[0][k].sum(),0);self.assertEqual(o[1][k].sum(),0)
            self.assertGreater(o[0][(k+1)%32].sum(),0)
    def test_invalid(self):
        for layers in [[-1],[32],[1,1],[1.5]]:
            with self.assertRaises(ValueError):JointAttention(None,layers,'Combined',4)
    def test_simultaneous_p1(self):
        o,h,_=self.scoped('P-1-only');self.assertEqual(h.edits,5)
        for k in range(5):
            self.assertEqual(o[0][k][:,-1].sum(),0)
            self.assertTrue(torch.equal(o[0][k][:,:-1],torch.full((1,3,3),2.)))
            self.assertEqual(o[1][k].sum(),6)
    def test_prefill_all_only(self):
        o,h,_=self.scoped('Prefill-all-only');self.assertEqual(h.edits,5)
        for k in range(5):self.assertEqual(o[0][k].sum(),0);self.assertEqual(o[1][k].sum(),6)
    def test_decode_only(self):
        o,h,_=self.scoped('Decode-only');self.assertEqual(h.edits,5)
        for k in range(5):self.assertEqual(o[0][k].sum(),24);self.assertEqual(o[1][k].sum(),0)
    def test_combined(self):
        o,h,_=self.scoped('Combined');self.assertEqual(h.edits,10)
        for k in range(5):self.assertEqual(o[0][k].sum(),0);self.assertEqual(o[1][k].sum(),0)
    def test_noop(self):
        o,h,_=self.scoped('Combined',noop=True)
        self.assertTrue(all(v.sum()==24 for v in o[0]));self.assertTrue(all(v.sum()==6 for v in o[1]))
    def test_exception_cleanup(self):
        m=self.model()
        with patch('paper_rq1_early_joint_attention.engine.model_layers',lambda m:m.layers):
            with self.assertRaises(RuntimeError):
                with JointAttention(m,[0,1],'Combined',4):raise RuntimeError('test')
        for block in m.layers:
            self.assertFalse(block.self_attn._forward_pre_hooks);self.assertFalse(block.self_attn.o_proj._forward_hooks)
    def test_single_token_prefill(self):self.assertFalse(is_decode(torch.tensor([0]),1,0,1))
    def test_contaminated_cache_rejected(self):
        with self.assertRaises(AssertionError):is_decode(torch.tensor([4]),4,0,1)
    def test_cluster_ratio(self):
        sums,counts=cluster_deltas([1,1,-1],['a','a','b'])
        self.assertEqual(float(sums.sum()/counts.sum()),1/3)
    def test_bootstrap_paired(self):
        b=bootstrap(np.zeros((4,2)),list('abcd'),1000,family=8)
        np.testing.assert_array_equal(b['corrected'],np.zeros((2,2)))
    def test_bootstrap_family(self):
        b=bootstrap([1,0,-1,0],list('abcd'),2000,family=8)
        self.assertLessEqual(b['corrected'][0,0],b['ordinary'][0,0])
    def test_scoring(self):
        from paper_rq1.scorers import gqa_prediction
        self.assertEqual(gqa_prediction('Dog.'.strip()),'dog');self.assertNotEqual(gqa_prediction('a dog'),'dog')
    def test_selection(self):
        from paper_rq1_early_joint_attention.data import eligible
        r=dict(id='q',imageId='i',isBalanced=True,answer='yes',types=dict(detailed='chooseAttr'))
        self.assertEqual(len(eligible([r],set(),set())['ChooseAttr']),1)
        self.assertFalse(eligible([r],{'i'},set())['ChooseAttr']);self.assertFalse(eligible([r],set(),{'q'})['ChooseAttr'])
    def test_checkpoint_hash_resume(self):
        from paper_rq1_early_joint_attention.run import checkpoint_valid
        from benchmarks.dataset_registry import digest
        from paper_rq1_single_layer_gqa6.run import atomic_json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';sample={'question_id':'a'};g={'token_ids':[2]}
            row=dict(binding='b',sample_sha256=digest(sample),condition='S0',generation=g,result_sha256=digest(g))
            atomic_json(p,row);self.assertTrue(checkpoint_valid(p,'b',sample,'S0'))
            with self.assertRaises(ValueError):checkpoint_valid(p,'bad',sample,'S0')
            with self.assertRaises(ValueError):atomic_json(p,dict(row,binding='bad'))
    def test_real_cpu_cache_generation_mlp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        config=LlamaConfig(vocab_size=40,hidden_size=16,intermediate_size=24,num_hidden_layers=32,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,attn_implementation='eager',eos_token_id=2)
        model=LlamaForCausalLM(config).eval();prompt=dict(input_ids=torch.tensor([[1,5,6,7]]),attention_mask=torch.ones(1,4,dtype=torch.long))
        counts=[0]*32;handles=[]
        for k,b in enumerate(model.model.layers):
            def f(m,a,o,k=k):counts[k]+=1
            handles.append(b.mlp.register_forward_hook(f))
        with patch('paper_rq1_early_joint_attention.engine.model_layers',lambda m:m.model.layers):
            full,logits=generate(model,prompt,CONDITIONS['S0'],4,collect=True)
            self.assertTrue(all(c==len(full['token_ids']) for c in counts))
            noop,other=generate(model,prompt,CONDITIONS['S4'],4,noop=True,collect=True)
            self.assertEqual(full['token_ids'],noop['token_ids']);self.assertTrue(all(torch.equal(a,b) for a,b in zip(logits,other)))
            changed,_=generate(model,prompt,CONDITIONS['S8'],4,audit=True)
            repeat,_=generate(model,prompt,CONDITIONS['S8'],4,audit=True)
            self.assertEqual(changed,repeat)
            again,_=generate(model,prompt,CONDITIONS['S0'],4)
            self.assertEqual(full['token_ids'],again['token_ids'])
            eos,_=generate(model,prompt,CONDITIONS['S0'],1)
            self.assertEqual(eos['length'],1);self.assertEqual(eos['eos'],eos['token_ids'][-1]==2)
        for h in handles:h.remove()


if __name__=='__main__':unittest.main()

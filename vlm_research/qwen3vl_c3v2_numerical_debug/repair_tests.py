import unittest,torch
from types import SimpleNamespace
from .repair import pad_dead_key
from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward
class RepairTests(unittest.TestCase):
 def test_padding_cannot_change_legal_distribution(self):
  q=torch.arange(24,dtype=torch.float32).reshape(1,2,3,4)/16;k=q.clone();v=q.clone();mask=torch.full((1,1,3,3),torch.finfo(torch.float32).min).triu(1)
  kp,vp,mp=pad_dead_key(k,v,mask);a=SimpleNamespace(num_key_value_groups=1,training=False)
  reference,w=eager_attention_forward(a,q,k,v,mask,scaling=.5);candidate,p=eager_attention_forward(a,q,kp,vp,mp,scaling=.5)
  self.assertTrue(torch.equal(reference,candidate));self.assertTrue(torch.equal(w,p[...,:-1]));self.assertEqual(float(p[...,-1].abs().max()),0.)
 def test_no_input_mutation(self):
  k=torch.randn(1,2,3,4);v=k+1;m=torch.zeros(1,1,3,3);old=[x.clone() for x in [k,v,m]];pad_dead_key(k,v,m)
  for x,y in zip([k,v,m],old):self.assertTrue(torch.equal(x,y))
 def test_dummy_is_zero_and_not_original_p1(self):
  k=torch.ones(1,2,3,4);v=k*3;m=torch.zeros(1,1,3,3);kp,vp,mp=pad_dead_key(k,v,m)
  self.assertEqual(float(kp[...,-1,:].abs().sum()),0.);self.assertEqual(float(vp[...,-1,:].abs().sum()),0.);self.assertTrue((mp[...,-1]==torch.finfo(m.dtype).min).all())
 def test_physical_query_count_unchanged(self):
  k=torch.ones(1,2,121,4);m=torch.zeros(1,1,121,121);kp,vp,mp=pad_dead_key(k,k,m);self.assertEqual(mp.shape[-2],121);self.assertEqual(kp.shape[-2],122)
if __name__=='__main__':unittest.main(verbosity=2)

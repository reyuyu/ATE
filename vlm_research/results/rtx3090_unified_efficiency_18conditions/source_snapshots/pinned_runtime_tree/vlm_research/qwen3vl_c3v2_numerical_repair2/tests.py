import unittest,torch
from .probe import Probe,align,metrics
class PositionTests(unittest.TestCase):
 def probe(self):
  p=Probe.__new__(Probe);p.n=122;p.text=list(range(4))+list(range(100,122));p.images=set(range(4,100));p.step=1;p.layer=0;p.c='B';p.records={};return p
 def test_decode_absolute_position(self):self.assertEqual(self.probe().positions(1),[122])
 def test_compact_decode_keys(self):
  p=self.probe();self.assertEqual(p.positions(27),p.text+[122])
 def test_dense_decode_keys(self):self.assertEqual(self.probe().positions(123),list(range(123)))
 def test_decode_record_is_captured(self):
  p=self.probe();p.put('layer_input',torch.zeros(1,1,8));self.assertIn('S1/L00/layer_input',p.records)
 def test_late_prefill_capture(self):
  p=self.probe();p.step=0;p.layer=22;p.put('input_norm',torch.zeros(1,26,8));self.assertEqual(p.records['S0/L22/input_norm']['positions'],p.text)
 def test_early_prefill_not_recollected(self):
  p=self.probe();p.step=0;p.put('layer_input',torch.zeros(1,122,8));self.assertEqual(p.records,{})
 def test_mask_alignment_excludes_dummy(self):
  a=dict(positions=[122],axis=2,key_positions=[0,1,2],value=torch.tensor([[[[1.,2.,3.]]]]));b=dict(positions=[122],axis=2,key_positions=[0,-1,2],value=torch.tensor([[[[1.,0.,3.]]]]));x,y,_=align(a,b);self.assertTrue(torch.equal(x,y))
 def test_metric_preserves_failure(self):self.assertGreater(metrics(torch.zeros(10),torch.ones(10)*.03)['mean_abs'],.02)
class DecodePaddingTests(unittest.TestCase):
 def test_restore_exact_text_at_original_positions(self):
  from .repair import restore_dead_image_columns
  k=torch.arange(12.).reshape(1,1,3,4);v=k+100
  kp,vp,m,ix=restore_dead_image_columns(k,v,[0,3],4,1)
  self.assertEqual(list(kp.shape),[1,1,5,4]);self.assertTrue(torch.equal(kp.index_select(-2,ix),k));self.assertTrue(torch.equal(vp.index_select(-2,ix),v));self.assertTrue(torch.isneginf(m[...,1:3]).all());self.assertTrue((kp[...,1:3,:]==0).all())
 def test_dummy_probability_strict_zero(self):
  from .repair import restore_dead_image_columns
  k=torch.ones(1,1,3,4);kp,vp,m,ix=restore_dead_image_columns(k,k,[0,3],4,1);w=torch.softmax(torch.ones(1,1,1,5)+m,dim=-1);self.assertEqual(float(w[...,1:3].max()),0.)
 def test_original_key_is_not_modified(self):
  from .repair import restore_dead_image_columns
  k=torch.randn(1,2,3,4);copy=k.clone();restore_dead_image_columns(k,k,[0,3],4,1);self.assertTrue(torch.equal(k,copy))
if __name__=='__main__':unittest.main(verbosity=2)

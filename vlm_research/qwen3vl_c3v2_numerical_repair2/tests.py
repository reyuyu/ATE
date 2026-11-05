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
if __name__=='__main__':unittest.main(verbosity=2)

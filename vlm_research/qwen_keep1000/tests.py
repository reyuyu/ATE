import unittest,torch
from .common import *
from ..qwen3vl_keep_l0_21.engine import controller
from ..qwen3vl_semantic_visual_writers.engine import roles,blocked_edges,apply_mask
from ..qwen_recovery.engine import layer_mask
class Tests(unittest.TestCase):
 def setUp(self):
  self.l=dict(input_ids=[1,9,9,10,11,12,13],image_positions=[1,2],input_length=7,p_minus_1=6,question_positions=[4,5]);self.r=roles(self.l,10)
  self.base=torch.zeros(1,1,7,7,dtype=torch.bfloat16).masked_fill(torch.ones(7,7,dtype=torch.bool).triu(1),torch.finfo(torch.bfloat16).min)
 def test_historical_keep_equals_D_all36_masks(self):
  for i in range(36):
   b=blocked_edges(self.r,self.r['R'] if i<22 else [],7,0,'cpu');a=apply_mask(self.base,b,0);d=layer_mask(self.base,torch.tensor([1,2]),7,True,i,'D');self.assertTrue(torch.equal(a,d),i)
 def test_historical_decode_equals_D_all36(self):
  base=torch.zeros(1,1,1,9,dtype=torch.bfloat16)
  for i in range(36):
   b=blocked_edges(self.r,self.r['R'] if i<22 else [],7,8,'cpu');self.assertTrue(torch.equal(apply_mask(base,b,8),layer_mask(base,torch.tensor([1,2]),7,False,i,'D')))
 def test_all_writer_changes_only_late_R(self):
  allmask=apply_mask(self.base,blocked_edges(self.r,self.r['R'],7,0,'cpu'),0)
  for i in range(36):
   keep=apply_mask(self.base,blocked_edges(self.r,self.r['R'] if i<22 else [],7,0,'cpu'),0);expected=torch.zeros_like(keep,dtype=torch.bool)
   if i>=22:expected[0,0,3:6,1:3]=True
   self.assertTrue(torch.equal(allmask!=keep,expected))
 def test_vision_end_in_R_and_noop_native(self):
  self.assertEqual(self.r['R'],[3,4,5]);b=blocked_edges(self.r,self.r['R'],7,0,'cpu');self.assertIs(apply_mask(self.base,b,0,noop=True),self.base)
 def test_controller_exact_layers(self):self.assertEqual(controller(None,{}).open_layers,set(range(22)))
 def test_exact_current_manifest(self):self.assertEqual(sha(OUT/'sample_manifest.json'),sha(RECOVERY/'sample_manifest.json'))
 def test_no_timing_in_new_generation(self):
  text=(ROOT/'vlm_research/qwen_keep1000/engine.py').read_text()
  for forbidden in ['torch.cuda.Event','perf_counter','elapsed_time','reset_peak_memory','flops=']:self.assertNotIn(forbidden,text)
if __name__=='__main__':unittest.main()

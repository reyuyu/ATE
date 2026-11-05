import unittest,torch
from .engine import controller
from .common import *
from ..qwen3vl_semantic_visual_writers.engine import roles,blocked_edges,apply_mask

class Tests(unittest.TestCase):
 def setUp(self):
  self.r={'I':[1,2],'R':[3,4,5],'A':[6]};self.base=torch.zeros(1,1,7,7,dtype=torch.bfloat16).masked_fill(torch.ones(7,7,dtype=torch.bool).triu(1),torch.finfo(torch.bfloat16).min)
 def test_inclusive_endpoints(self):
  for end in [21,24,26]:self.assertEqual(controller(None,{},end).open_layers,set(range(end+1)))
 def test_all_layers_precise_prefill_masks(self):
  for end in [24,26]:
   for layer in range(36):
    block=blocked_edges(self.r,self.r['R'] if layer<=end else [],7,0,'cpu');mask=apply_mask(self.base,block,0);expected=torch.zeros_like(self.base,dtype=torch.bool);expected[0,0,6,1:3]=True
    if layer>end:expected[0,0,3:6,1:3]=True
    self.assertTrue(torch.equal(torch.isneginf(mask),expected));self.assertTrue(torch.equal(mask[~expected],self.base[~expected]))
 def test_decode_all36_both_conditions(self):
  for end in [24,26]:
   for layer in range(36):
    b=blocked_edges(self.r,self.r['R'] if layer<=end else [],7,8,'cpu');self.assertEqual(b.sum().item(),2);self.assertTrue(b[0,1:3].all())
 def test_manifest_exact_and_image_shards_disjoint(self):
  self.assertEqual(sha(OUT/'sample_manifest.json'),sha(PREVIOUS/'sample_manifest.json'));m=read(OUT/'sample_manifest.json');im=[{s['image_sha256'] for s in m['samples'] if s['rank']==r} for r in range(2)];self.assertFalse(im[0]&im[1])
 def test_nested_sets_change_only_added_R_layers(self):
  for layer in range(36):
   masks=[blocked_edges(self.r,self.r['R'] if layer<=end else [],7,0,'cpu') for end in [21,24,26]]
   for a,b,lo,hi in [(masks[0],masks[1],22,24),(masks[1],masks[2],25,26)]:self.assertEqual((a!=b).sum().item(),6 if lo<=layer<=hi else 0)
if __name__=='__main__':unittest.main()

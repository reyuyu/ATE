"""CPU mask-policy tests. These cannot replace real 7B GPU smoke tests."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest,torch,random
from .policy import LAYER_SETS,ALL,NEW,SEED
from .engine import bind
E=bind()
class PolicyTests(unittest.TestCase):
 def setUp(self):
  self.layout=dict(prompt_length=14,I=list(range(2,8)))
 def test_exact_transferred_layer_sets(self):
  self.assertEqual(LAYER_SETS['C']['Decode'],[0,11,12,13,14,15,16,17,18,19,20])
  self.assertEqual(LAYER_SETS['D']['Decode'],list(range(11)))
  self.assertEqual(LAYER_SETS['E']['Decode'],[0]+list(range(21,32)))
 def test_seed_reproduces_random_budget(self):
  self.assertEqual(LAYER_SETS['R']['Decode'],sorted([0]+random.Random(SEED).sample(range(1,32),10)))
  self.assertEqual(len(set(LAYER_SETS['R']['Decode'])),11)
 def test_all_new_prefill_entire_mask_native(self):
  native=torch.zeros((1,1,14,14),dtype=torch.float16)
  native.masked_fill_(torch.ones_like(native,dtype=torch.bool).triu(1),float('-inf'))
  for c in NEW:
   self.assertEqual(LAYER_SETS[c]['Pminus1'],ALL)
   for k in range(32):
    changed,m=E.apply_mask(native,self.layout,c,k)
    self.assertTrue(torch.equal(changed,native));self.assertEqual(m.sum().item(),0)
 def test_decode_changes_exact_image_edges_only(self):
  native=torch.zeros((1,1,1,18),dtype=torch.float16)
  for c in NEW:
   for k in range(32):
    changed,m=E.apply_mask(native,self.layout,c,k,past=17)
    self.assertTrue(torch.equal(changed[0,0][~m],native[0,0][~m]))
    self.assertEqual(m.sum().item(),0 if k in LAYER_SETS[c]['Decode'] else 6)
    if m.any():self.assertTrue(torch.isneginf(changed[0,0][m]).all().item())
 def test_forbidden_softmax_probabilities_exact_zero(self):
  native=torch.zeros((1,1,1,15),dtype=torch.float16)
  for c in NEW:
   for k in range(32):
    changed,m=E.apply_mask(native,self.layout,c,k,past=14)
    p=changed.softmax(-1)
    self.assertTrue((p[0,0][m]==0).all().item())
    self.assertTrue((p[0,0][~m]>0).all().item())
 def test_retained_layers_native_exact(self):
  native=torch.arange(15,dtype=torch.float16).reshape(1,1,1,15)/20
  for c in NEW:
   for k in LAYER_SETS[c]['Decode']:
    v,m=E.apply_mask(native,self.layout,c,k,past=14);self.assertIs(v,native);self.assertFalse(m.any())
 def test_noop_never_changes_mask(self):
  for c in LAYER_SETS:
   for past in [0,14,19]:
    q,n=(1,past+1) if past else (14,14);native=torch.zeros((1,1,q,n),dtype=torch.float16)
    for k in range(32):
     out,m=E.apply_mask(native,self.layout,c,k,past= past,noop=True);self.assertIs(out,native);self.assertFalse(m.any())
 def test_historical_baseline_phase_semantics(self):
  for k in range(32):
   self.assertFalse(E.blocked_layer('A',k,False));self.assertFalse(E.blocked_layer('A',k,True))
   self.assertFalse(E.blocked_layer('F',k,False));self.assertTrue(E.blocked_layer('F',k,True))
   self.assertTrue(E.blocked_layer('B',k,False));self.assertTrue(E.blocked_layer('B',k,True))
 def test_prompt_boundary_and_future_text_protection(self):
  for c in NEW:
   out=E.blocked_edges(self.layout,c,31,past=22)
   self.assertFalse(out[:,8:].any());self.assertFalse(out[:,:2].any());self.assertEqual(out.shape,(1,23))
  with self.assertRaises(AssertionError):E.blocked_edges(self.layout,'C',31,past=3)
 def test_cpu_has_not_initialized_cuda(self):
  self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'');self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()

import unittest
import torch
from .policy import CONDITIONS,query_rows
from .engine import blocked_edges
from textvqa_single_layer_visual_ablation.policy import query_rows as historical
class PolicyTests(unittest.TestCase):
 def setUp(self):self.layout=dict(prompt_length=610,I=list(range(5,581)),R=list(range(581,609)))
 def test_exact_six_layer_sets(self):
  self.assertEqual([len(CONDITIONS[c]) for c in ['A','B','C','D']],[10,11,11,12]);self.assertEqual(set(CONDITIONS['D'])-set(CONDITIONS['A']),{0,1})
 def test_all_prefill_native(self):
  for c in CONDITIONS:
   for k in range(32):self.assertEqual(query_rows(self.layout,c,k,0),[])
 def test_real_decode_all_layers(self):
  for c,v in CONDITIONS.items():
   for k in range(32):
    for past in [610,611,640]:self.assertEqual(query_rows(self.layout,c,k,past),[] if k in v else [0])
 def test_noop_all_layers(self):
  for c in CONDITIONS:
   for k in range(32):self.assertFalse(query_rows(self.layout,c,k,610,True))
 def test_original_576_image_keys_only(self):
  for c in CONDITIONS:
   for k in range(32):
    b=blocked_edges(self.layout,c,k,610);self.assertEqual(b.shape,(1,611));self.assertEqual(b.sum().item(),0 if k in CONDITIONS[c] else 576);self.assertFalse(b[0,:5].any());self.assertFalse(b[0,581:].any())
 def test_non_target_mask_identical_and_probability_zero(self):
  old=torch.zeros(1,611,dtype=torch.float16);old[0,600]=-65504
  b=blocked_edges(self.layout,'A',0,610);new=old.clone();new[b]=-torch.inf
  self.assertTrue(torch.equal(new[~b],old[~b]));p=torch.softmax(new.float(),-1);self.assertEqual(p[b].max().item(),0);self.assertAlmostEqual(p.sum().item(),1,places=6)
 def test_historical_A_B_equivalence(self):
  for c,legacy in [('A','Middle-minus-L0'),('B','Middle')]:
   for k in range(32):
    for past in [0,610,620]:self.assertEqual(query_rows(self.layout,c,k,past),historical(self.layout,legacy,k,past))
 def test_first_predicting_query_never_masked(self):
  for c in CONDITIONS:
   for k in range(32):self.assertFalse(blocked_edges(self.layout,c,k)[-1].any())
if __name__=='__main__':unittest.main()

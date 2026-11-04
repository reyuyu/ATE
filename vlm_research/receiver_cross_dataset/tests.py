import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import unittest,random,collections
import torch
torch.set_num_threads(1)
from .selection import *
from .scoring import score,normalize
from gqa3000_token_visual_writing.engine import cpu_mask
class ProtocolTests(unittest.TestCase):
 def test_firstword_is_not_anywhere_keyword(self):
  self.assertFalse(trigger('Is what he says true?',8,15));self.assertTrue(trigger('Which animal?',4,15))
 def test_exact_threshold_boundary(self):
  self.assertFalse(trigger('Is it blue?',14,15));self.assertTrue(trigger('Is it blue?',15,15));self.assertFalse(trigger('Is it blue?',17,18));self.assertTrue(trigger('Is it blue?',18,18))
 def test_punctuation_prefix_uses_frozen_regex(self):self.assertEqual(firstword('"WHAT is it?'),'what')
 def test_consensus_is_leave_one_out(self):
  for n,expected in [(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)]:
   for dataset in ['vqa_v2','textvqa']:self.assertAlmostEqual(score(dataset,'White',['white']*n+['black']*(10-n)),expected)
 def test_official_normalization(self):self.assertEqual(normalize('The two cats.'),'2 cats')
 def test_ten_answers_required(self):
  with self.assertRaises(AssertionError):score('vqa_v2','yes',['yes'])
 def test_random_strata_budget_and_determinism(self):
  specs=[]
  for n in [7,8,14,15,18]:
   for j in range(30):specs.append(dict(question_id=f'{n}_{j}',question='What is it?' if j<15 else 'Is it blue?',Q=list(range(n)),R=list(range(n+17))))
  a,audit=random_labels(specs);b,_=random_labels(list(reversed(specs)));self.assertEqual(a,b);self.assertGreater(audit['changed_labels'],0)
  for n in [7,8,14,15,18]:
   ss=[s for s in specs if len(s['Q'])==n];self.assertEqual(sum(a[s['question_id']] for s in ss),sum(trigger(s['question'],n,15) for s in ss))
 def test_pminus1_always_masked_even_all_writer(self):
  layout=dict(prompt_length=590,I=list(range(5,581)),R=list(range(581,589)))
  native,changed,blocked=cpu_mask(layout,layout['R']);self.assertEqual(int(blocked.sum()),576);self.assertTrue(blocked[589,5:581].all());self.assertTrue(torch.equal(changed[0,0][~blocked],native[0,0][~blocked]))
 def test_decode_always_blocked(self):
  layout=dict(prompt_length=590,I=list(range(5,581)),R=list(range(581,589)))
  for w in [[],[582],layout['R']]:
   native,changed,blocked=cpu_mask(layout,w,593);self.assertEqual(int(blocked.sum()),576);self.assertTrue(torch.isneginf(changed[0,0][blocked]).all());self.assertEqual(float(changed[0,0,0,593]),0)
 def test_no_image_queries_modified(self):
  layout=dict(prompt_length=590,I=list(range(5,581)),R=list(range(581,589)))
  native,changed,blocked=cpu_mask(layout,[]);self.assertTrue(torch.equal(native[:,:,:581],changed[:,:,:581]))
if __name__=='__main__':unittest.main(verbosity=2)

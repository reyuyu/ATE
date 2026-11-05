import unittest,torch
from .policy import CONDITIONS,MIDDLE,query_rows
from .engine import blocked_edges
class Tests(unittest.TestCase):
 def setUp(self):self.l=dict(prompt_length=12,I=[2,3,4,5],R=[6,7,8,9,10])
 def test_all_conditions(self):
  for c,d in CONDITIONS.items():
   for k in range(32):
    for past in [0,12,15]:
     b=blocked_edges(self.l,c,k,past);rows=query_rows(self.l,c,k,past)
     self.assertEqual(int(b.sum()),len(rows)*4)
     self.assertFalse(b[:,:2].any());self.assertFalse(b[:,6:].any())
     if not past:self.assertFalse(b[:6].any())
 def test_single_target(self):
  for f in ['All-Text','Ponly','Donly','Both']:
   for target in range(32):
    for k in range(32):
     if k!=target:self.assertFalse(blocked_edges(self.l,f'{f}-L{target}',k).any());self.assertFalse(blocked_edges(self.l,f'{f}-L{target}',k,12).any())
 def test_first_position_equivalence(self):
  for t in range(32):
   for k in range(32):
    self.assertTrue(torch.equal(blocked_edges(self.l,f'Ponly-L{t}',k),blocked_edges(self.l,f'Both-L{t}',k)))
    self.assertFalse(blocked_edges(self.l,f'Donly-L{t}',k).any())
 def test_middle(self):
  for target in MIDDLE:
   for k in range(32):
    self.assertFalse(blocked_edges(self.l,f'Middle-minus-L{target}',k).any())
    self.assertEqual(bool(blocked_edges(self.l,f'Middle-minus-L{target}',k,12).any()),k not in set(MIDDLE)-{target})
 def test_noop(self):
  for c in CONDITIONS:
   for past in [0,12]:self.assertFalse(blocked_edges(self.l,c,0,past,noop=True).any())
 def test_mask_unchanged(self):
  n=12;old=torch.zeros(n,n,dtype=torch.float16).masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)
  for c in CONDITIONS:
   for k in range(32):
    b=blocked_edges(self.l,c,k);new=old.clone().masked_fill(b,-torch.inf)
    self.assertTrue(torch.equal(old[~b],new[~b]));self.assertTrue((old[b]==0).all())
 def test_cpu_only(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main()

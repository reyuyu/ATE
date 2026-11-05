import unittest,torch
from .engine import source_mask,zero_answer
from .common import *

class Tests(unittest.TestCase):
 def setUp(self):
  self.base=torch.zeros(1,1,8,8,dtype=torch.float16).masked_fill(torch.ones(8,8,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min);self.image=[2,3]
 def test_native_mask_identity(self):self.assertIs(source_mask(self.base,self.image,8,0,'N'),self.base)
 def test_noop_mask_identity(self):
  for c in ['A','B']:self.assertIs(source_mask(self.base,self.image,8,0,c,noop=True),self.base)
 def test_prefill_only_original_P1_image_edges(self):
  expected=torch.zeros_like(self.base,dtype=torch.bool);expected[0,0,7,2:4]=True
  for c in ['A','B']:
   m=source_mask(self.base,self.image,8,0,c);self.assertTrue(torch.equal(torch.isneginf(m),expected));self.assertTrue(torch.equal(m[~expected],self.base[~expected]));self.assertTrue(torch.equal(m[:,:,:7],self.base[:,:,:7]))
 def test_decode_only_image_columns(self):
  base=torch.zeros(1,1,1,11,dtype=torch.float16)
  for c in ['A','B']:
   m=source_mask(base,self.image,8,10,c);self.assertTrue(torch.isneginf(m[0,0,0,2:4]).all());self.assertTrue(torch.equal(m[0,0,0,[0,1,4,5,6,7,8,9,10]],base[0,0,0,[0,1,4,5,6,7,8,9,10]]))
 def test_A_B_masks_identical(self):
  for past,base in [(0,self.base),(10,torch.zeros(1,1,1,11,dtype=torch.float16))]:self.assertTrue(torch.equal(source_mask(base,self.image,8,past,'A'),source_mask(base,self.image,8,past,'B')))
 def test_B_prefill_zero_only_last_row(self):
  x=torch.arange(32,dtype=torch.float16).reshape(1,8,4);saved=x.clone();y=zero_answer(x,8,True);self.assertTrue(torch.equal(x,saved));self.assertTrue(torch.equal(y[:,:7],x[:,:7]));self.assertEqual(y[:,7].count_nonzero().item(),0)
 def test_B_decode_zero_entire_real_query(self):
  x=torch.ones(1,1,4,dtype=torch.float16);self.assertEqual(zero_answer(x,8,False).count_nonzero().item(),0);self.assertEqual(x.count_nonzero().item(),4)
 def test_frozen_500_309_no_sample_filter(self):
  s=read(OUT/'sample_manifest.json');self.assertEqual(len(s),500);self.assertEqual(len({x['image_sha256'] for x in s.values()}),309);self.assertTrue(all(len(x['answer'])==10 for x in s.values()));self.assertEqual(sha(OUT/'sample_manifest.json'),sha(HISTORY/'sample_manifest.json'))
 def test_R_and_P1_disjoint_all_samples(self):
  for s in read(OUT/'sample_manifest.json').values():
   m=s['input_mapping'];self.assertEqual(m['R'],list(range(m['image'][-1]+1,m['Pminus1'])));self.assertNotIn(m['Pminus1'],m['R']);self.assertFalse(set(m['R'])&set(m['image']))
 def test_image_shards_disjoint_and_complete(self):
  s=read(OUT/'sample_manifest.json');p=read(OUT/'partition.json');self.assertEqual(len(p['image_shard']),309);self.assertEqual(sum(p['counts']),500)
  im=[{x['image_sha256'] for x in s.values() if p['image_shard'][x['image_sha256']]==r} for r in range(2)];self.assertFalse(im[0]&im[1])
if __name__=='__main__':unittest.main()

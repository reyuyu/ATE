import unittest,torch
from .engine import restore_prefill_columns
class PrefillPaddingTests(unittest.TestCase):
 def data(self):
  k=torch.arange(24.).reshape(1,2,3,4);v=k+20;m=torch.full((1,1,3,3),-torch.inf).triu(1);return k,v,m
 def test_text_keys_values_exact(self):
  k,v,m=self.data();a,b,c,ix=restore_prefill_columns(k,v,m,[0,3,4],5);self.assertTrue(torch.equal(a.index_select(-2,ix),k));self.assertTrue(torch.equal(b.index_select(-2,ix),v))
 def test_image_is_zero(self):
  k,v,m=self.data();a,b,c,ix=restore_prefill_columns(k,v,m,[0,3,4],5);self.assertEqual(int(a[...,1:3,:].count_nonzero()),0);self.assertEqual(int(b[...,1:3,:].count_nonzero()),0)
 def test_image_mask_is_negative_infinity(self):
  k,v,m=self.data();a,b,c,ix=restore_prefill_columns(k,v,m,[0,3,4],5);self.assertTrue(torch.isneginf(c[...,1:3]).all())
 def test_non_target_mask_exact(self):
  k,v,m=self.data();a,b,c,ix=restore_prefill_columns(k,v,m,[0,3,4],5);self.assertTrue(torch.equal(c.index_select(-1,ix),m))
 def test_query_rows_not_padded(self):
  k,v,m=self.data();a,b,c,ix=restore_prefill_columns(k,v,m,[0,3,4],5);self.assertEqual(c.shape[-2],3);self.assertEqual(c.shape[-1],5)
 def test_no_input_mutation(self):
  k,v,m=self.data();copies=[x.clone() for x in [k,v,m]];restore_prefill_columns(k,v,m,[0,3,4],5);self.assertTrue(all(torch.equal(x,y) for x,y in zip(copies,[k,v,m])))
 def test_strict_zero_probability(self):
  k,v,m=self.data();a,b,c,ix=restore_prefill_columns(k,v,m,[0,3,4],5);w=torch.softmax(torch.zeros(1,2,3,5)+c,-1);self.assertEqual(float(w[...,1:3].max()),0.)
 def test_reject_bad_mask_shape(self):
  k,v,m=self.data()
  with self.assertRaises(AssertionError):restore_prefill_columns(k,v,m[...,:2,:],[0,3,4],5)
 def test_reject_duplicate_position(self):
  k,v,m=self.data()
  with self.assertRaises(AssertionError):restore_prefill_columns(k,v,m,[0,0,4],5)
if __name__=='__main__':unittest.main()

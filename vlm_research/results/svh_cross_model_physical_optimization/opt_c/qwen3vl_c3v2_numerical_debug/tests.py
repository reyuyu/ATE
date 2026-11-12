import unittest,torch
from .probe import metrics,align
from .common import read,OUT,ORIGINAL
from ..qwen3vl_c3v2_physical.engine import CompactKV
class AuditTests(unittest.TestCase):
 def setUp(self):
  s=read(OUT/'sample_manifest.json');self.l=s['input_bindings'][ORIGINAL];self.n=self.l['input_length'];self.I=self.l['image_positions'];self.T=[i for i in range(self.n) if i not in self.I]
 def masks(self):
  base=torch.full((self.n,self.n),float('-inf')).triu(1);early=base.clone();early[-1,self.I]=-torch.inf;late=early.clone();late[self.I[-1]+1:,self.I[0]:self.I[-1]+1]=-torch.inf
  return early,late
 def test_old_case_exact_shape(self):self.assertEqual((self.n,len(self.I),len(self.T)),(122,96,26))
 def test_skip_keeps_prefix_causal_permissions(self):
  early,_=self.masks();cut=early[:-1,:-1];self.assertTrue(torch.equal(cut,torch.full((121,121),float('-inf')).triu(1)))
 def test_late_text_mask_matches_original_causal_submatrix(self):
  _,late=self.masks();cut=late[self.T][:,self.T];expected=torch.full((26,26),float('-inf')).triu(1);self.assertTrue(torch.equal(cut,expected))
 def test_deleted_image_keys_already_invisible_to_all_retained_queries(self):
  _,late=self.masks();self.assertTrue(torch.isneginf(late[self.T][:,self.I]).all())
 def test_physical_cache_returns_original_for_current_prefill(self):
  c=CompactKV(self.n,torch.tensor(self.T),60);k=torch.randn(1,8,self.n,128);a,b=c.update(k,k,0);self.assertEqual(a.data_ptr(),k.data_ptr());self.assertEqual(b.data_ptr(),k.data_ptr())
 def test_append_keeps_original_logical_length_separate(self):
  c=CompactKV(self.n,torch.tensor(self.T),60);k=torch.randn(1,8,self.n,128);c.update(k,k,0);c.logical_length=self.n;c.prefill=False;c.update(k[...,:1,:],k[...,:1,:],0);self.assertEqual(c.used[0],27);self.assertEqual(c.get_seq_length(),122)
 def test_aligned_text_rows_use_original_positions(self):
  x=torch.arange(self.n).reshape(1,self.n,1);y=x[:,self.T];a=dict(value=x,positions=list(range(self.n)),axis=1,retained_positions=self.T);b=dict(value=y,positions=self.T,axis=1);u,v,common=align(a,b);self.assertTrue(torch.equal(u,v));self.assertEqual(common,self.T)
 def test_align_p_minus_1_not_previous_r(self):
  a=dict(value=torch.zeros(1,122,1),positions=list(range(122)),axis=1);b=dict(value=torch.zeros(1,121,1),positions=list(range(121)),axis=1);u,v,p=align(a,b,'p_minus_1');self.assertIsNone(u);self.assertEqual(p,[])
 def test_mask_infinity_equality(self):
  x=torch.tensor([-torch.inf,0.,-3.]);m=metrics(x,x);self.assertTrue(m['equal']);self.assertEqual(m['mean_abs'],0.)
 def test_mask_infinity_mismatch_is_error(self):
  m=metrics(torch.tensor([-torch.inf]),torch.tensor([0.]));self.assertFalse(m['equal']);self.assertEqual(m['nonfinite_mismatch'],1)
 def test_equal_elements_different_memory_layout(self):
  x=torch.arange(12.).reshape(3,4);self.assertTrue(metrics(x,x.T.contiguous().T)['equal'])
 def test_rmse_relative_definition(self):
  m=metrics(torch.ones(4),torch.zeros(4));self.assertEqual(m['rmse'],1.);self.assertEqual(m['relative_l2'],1.)
if __name__=='__main__':unittest.main(verbosity=2)

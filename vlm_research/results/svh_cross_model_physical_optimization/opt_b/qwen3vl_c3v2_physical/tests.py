import unittest,torch,json,pathlib
from .engine import CompactKV,SKIP,Execution
from .worker import score,sha,O
class Checks(unittest.TestCase):
 def test_compact_full_prefill_returns_original(self):
  text=torch.tensor([0,3,4]);c=CompactKV(5,text,7);k=torch.arange(10.).reshape(1,1,5,2);v=k+1
  a,b=c.update(k,v,0);self.assertTrue(torch.equal(a,k));self.assertTrue(torch.equal(c.entries[0][0][...,:3,:],k.index_select(-2,text)))
 def test_inplace_append(self):
  c=CompactKV(5,torch.tensor([0,3,4]),7);k=torch.zeros(1,2,5,4);c.update(k,k,0);ptr=c.entries[0][0].data_ptr();c.prefill=False
  k=torch.ones(1,2,1,4);c.update(k,k,0);self.assertEqual(ptr,c.entries[0][0].data_ptr());self.assertEqual(c.used[0],4)
 def test_absent_has_no_dummy(self):
  c=CompactKV(5,torch.tensor([0,3,4]),7,absent=SKIP)
  with self.assertRaises(AssertionError):c.update(torch.zeros(1,1,1,2),torch.zeros(1,1,1,2),1)
 def test_native_causal_deleted_future_row(self):
  m=torch.ones(8,8).tril();self.assertTrue(torch.equal(m[:-1,:-1],torch.ones(7,7).tril()));self.assertEqual(float(m[:-1,-1].sum()),0)
 def test_original_position_gather(self):
  original=torch.arange(100);text=torch.tensor([0,1,2,88,89,90]);self.assertEqual(original[text].tolist(),[0,1,2,88,89,90])
 def test_mrope_gather_preserves_axes(self):
  ids=torch.arange(36).reshape(3,1,12);keep=torch.tensor([0,1,10,11]);r=ids.index_select(-1,keep);self.assertEqual(r[2,0].tolist(),[24,25,34,35])
 def test_storage_live_capacity(self):
  c=CompactKV(5,torch.tensor([0,3,4]),7,absent=tuple(range(1,36)));k=torch.zeros(1,2,5,4,dtype=torch.bfloat16);c.update(k,k,0);a=c.audit();self.assertEqual(a['live_bytes'],2*1*2*3*4*2);self.assertEqual(a['capacity_bytes'],2*1*2*7*4*2)
 def test_score_gqa_original(self):
  self.assertEqual(score(dict(dataset='GQA',answer='white'),'White.'),1.);self.assertEqual(score(dict(dataset='GQA',answer='white'),' white'),0.)
 def test_vqa_ten_consensus(self):
  s=dict(dataset='VQAv2',answers=['yes']*3+['no']*7);self.assertAlmostEqual(score(s,'yes'),.9);self.assertEqual(score(s,'no'),1.)
 def test_vqa_identical_answers_pinned_normalization(self):
  s=dict(dataset='VQAv2',answers=['cat']*10);self.assertEqual(score(s,'cat'),1.);self.assertEqual(score(s,'Cat'),1.)
 def test_frozen_data_sharding(self):
  m=json.loads((O/'sample_manifest.json').read_text());ss=m['samples'];h={}
  self.assertEqual(len({s['id'] for s in ss}),len(ss))
  for s in ss:
   self.assertFalse(s['historical_overlap']);self.assertEqual(h.setdefault(s['image_sha256'],s['shard']),s['shard'])
 def test_input_dynamic_image_indices(self):
  b=json.loads((O/'input_bindings.json').read_text());counts=set()
  for s in b.values():
   I=s['image_positions'];counts.add(len(I));self.assertEqual(I,list(range(I[0],I[-1]+1)));self.assertLess(I[-1],s['p_minus_1']);self.assertEqual(s['p_minus_1'],len(s['input_ids'])-1)
  self.assertGreater(len(counts),1)
if __name__=='__main__':unittest.main(verbosity=2)

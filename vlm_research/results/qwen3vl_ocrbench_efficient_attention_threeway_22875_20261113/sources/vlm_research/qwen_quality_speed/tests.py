import unittest,torch
from .common import *
from ..qwen3vl_c3v2_numerical_debug.repair import pad_dead_key
from ..qwen3vl_c3v2_numerical_repair2.repair import restore_dead_image_columns
from ..qwen3vl_c3v2_prefill_av_final.engine import restore_prefill_columns
from ..qwen3vl_c3v2_physical.engine import CompactKV
from ..qwen3vl_early_windows.experiment import gqa_prediction

class Tests(unittest.TestCase):
 def test_prefill_zero_mask_preserves_text(self):
  torch.manual_seed(7);k=torch.randn(1,2,3,4);v=torch.randn_like(k);mask=torch.zeros(1,1,3,3).masked_fill(torch.ones(3,3,dtype=torch.bool).triu(1),-torch.inf)
  kp,vp,mp,ix=restore_prefill_columns(k,v,mask,[0,4,5],6)
  self.assertTrue(torch.equal(kp.index_select(-2,ix),k));self.assertTrue(torch.equal(vp.index_select(-2,ix),v));self.assertTrue(torch.equal(mp.index_select(-1,ix),mask))
  self.assertTrue(torch.isneginf(mp[...,[1,2,3]]).all());self.assertEqual(int(kp[...,[1,2,3],:].count_nonzero()),0)
  q=torch.randn(1,2,3,4);p=(q@kp.transpose(-2,-1)+mp).softmax(-1)
  self.assertEqual(int(p[...,[1,2,3]].count_nonzero()),0)
 def test_decode_restore_uses_absolute_generated_positions(self):
  k=torch.arange(20.).reshape(1,1,5,4);v=k+1
  kp,vp,mp,ix=restore_dead_image_columns(k,v,[0,4,5],6,2)
  self.assertEqual(ix.tolist(),[0,4,5,6,7]);self.assertEqual(kp.shape[-2],8)
  self.assertTrue(torch.equal(kp.index_select(-2,ix),k));self.assertTrue(torch.isneginf(mp[...,[1,2,3]]).all())
 def test_early_dead_key_no_hidden_recovery(self):
  k=torch.randn(1,2,4,4);mask=torch.zeros(1,1,4,4)
  kp,vp,mp=pad_dead_key(k,k+1,mask)
  self.assertTrue(torch.equal(kp[...,:-1,:],k));self.assertEqual(kp.shape[-2],5);self.assertEqual(int(kp[...,-1,:].count_nonzero()),0)
  self.assertEqual(float((torch.randn(1,2,4,4)@kp.transpose(-2,-1)+mp).softmax(-1)[...,-1].max()),0.)
 def test_compact_cache_request_isolation_and_no_image_storage(self):
  text=torch.tensor([0,4,5]);a=CompactKV(6,text,5,tuple(range(1,7)));b=CompactKV(6,text,5,tuple(range(1,7)))
  k=torch.randn(1,2,6,4);a.update(k,k,0)
  self.assertIsNone(b.entries[0]);self.assertEqual(a.used[0],3)
  a.prefill=False;a.update(k[...,:1,:],k[...,:1,:],0);self.assertEqual(a.used[0],4)
  self.assertEqual(a.entries[0][0].shape[-2],5)
 def test_manifest_shards_and_benchmark(self):
  m=read(OUT/'sample_manifest.json');rows=m['samples'];self.assertEqual(len(rows),1000)
  groups={}
  for r in rows:groups.setdefault(r['image_sha256'],set()).add(r['rank'])
  self.assertTrue(all(len(v)==1 for v in groups.values()));self.assertEqual(len(set(m['smoke_question_ids'])),20);self.assertEqual(len(set(m['benchmark_question_ids'])),48)
  self.assertEqual(len({r['image_sha256'] for r in rows if r['question_id'] in m['benchmark_question_ids']}),48)
 def test_pinned_scoring(self):
  self.assertEqual(gqa_prediction('White.'),'white');self.assertEqual(gqa_prediction('a black cat'),'a black cat')
  self.assertNotEqual(gqa_prediction('The answer is white'),'white')
if __name__=='__main__':unittest.main()

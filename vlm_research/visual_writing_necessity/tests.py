import unittest
import tempfile
from pathlib import Path
import torch
from .common import CONDITIONS,writer_layers,shards,progress_write,read,write
from .engine import apply_mask,gold_logp,clone_prefix
from transformers.cache_utils import DynamicCache
from visual_write_window.engine import apply_mask as previous_mask


class NecessityTests(unittest.TestCase):
    def setUp(self):
        self.layout=dict(prompt_length=9,I=[2,3,4],R=[5,6,7],A=[8],other=[0,1])
        self.mask=torch.zeros(1,1,9,9,dtype=torch.float16).masked_fill(
            torch.ones(9,9,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)

    def test_exact_34_conditions(self):
        self.assertEqual(len(set(CONDITIONS)),34)
        for k in range(32):self.assertEqual(writer_layers(f'Drop-L{k}'),[j for j in range(32) if j!=k])
        with self.assertRaises(ValueError):writer_layers('Only-L1')

    def test_progress_replace_but_checkpoint_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'progress.json'
            progress_write(path,dict(questions=1));progress_write(path,dict(questions=2))
            self.assertEqual(read(path),dict(questions=2))
            with self.assertRaises(ValueError):write(path,dict(questions=3))

    def test_every_layer_and_query(self):
        for c in CONDITIONS:
            for layer in range(32):
                changed,b=apply_mask(self.mask,self.layout,layer,c,0)
                expected=torch.zeros(9,9,dtype=torch.bool);expected[8,[2,3,4]]=True
                if layer not in writer_layers(c):expected[5:8,2:5]=True
                self.assertTrue(torch.equal(b,expected))
                self.assertTrue(torch.equal(changed[0,0][~b],self.mask[0,0][~b]))
                self.assertTrue(torch.isneginf(changed[0,0][b]).all())

    def test_original_baselines_exact(self):
        for c in ['All-Writer','No-Writer']:
            for k in range(32):
                a,b=apply_mask(self.mask,self.layout,k,c,0);x,y=previous_mask(self.mask,self.layout,k,c,0)
                self.assertTrue(torch.equal(a,x) and torch.equal(b,y))

    def test_decode_only_image_and_self_preserved(self):
        for c in CONDITIONS:
            for past in [9,10,20]:
                old=torch.zeros(1,1,1,past+1,dtype=torch.float16)
                a,b=apply_mask(old,self.layout,0,c,past)
                self.assertEqual(torch.where(b[0])[0].tolist(),[2,3,4])
                self.assertEqual(float(a[0,0,0,past]),0.)
                self.assertEqual(float(a[0,0,0,0]),0.)

    def test_probabilities_zero_and_rows_valid(self):
        a,b=apply_mask(self.mask,self.layout,10,'Drop-L10',0)
        w=torch.softmax(torch.randn(1,4,9,9)+a.float(),-1)
        self.assertEqual(float(w[0,:,b].max()),0.)
        self.assertTrue(torch.allclose(w.sum(-1),torch.ones(1,4,9)))
        self.assertFalse(b.triu().any());self.assertFalse(b[:5].any())

    def test_noop_native_exact_and_input_untouched(self):
        original=self.mask.clone()
        for c in CONDITIONS+['Native']:
            a,_=apply_mask(self.mask,self.layout,0,c,0,noop=True)
            self.assertTrue(torch.equal(a,original))
        self.assertTrue(torch.equal(self.mask,original))

    def test_gold_full_vocabulary_fp32(self):
        vector=torch.tensor([2.,-1.,.5],dtype=torch.float16)
        self.assertEqual(gold_logp(vector,2),float(vector.float().log_softmax(-1)[2]))

    def test_cache_clone_isolation(self):
        cache=DynamicCache()
        cache.update(torch.randn(1,2,4,3),torch.randn(1,2,4,3),0)
        copied=clone_prefix(cache)
        copied.key_cache[0].add_(1)
        self.assertFalse(torch.equal(cache.key_cache[0],copied.key_cache[0]))
        self.assertEqual(cache.get_seq_length(),copied.get_seq_length())

    def test_shards_do_not_split_image_clusters(self):
        ss=[dict(question_id=str(i),image_sha256=str(i//3)) for i in range(20)]
        split,owners=shards(ss)
        self.assertEqual(sum(map(len,split)),20)
        self.assertFalse(set(split[0])&set(split[1]))
        for s in ss:self.assertIn(s['question_id'],split[owners[s['image_sha256']]])


if __name__=='__main__':unittest.main()

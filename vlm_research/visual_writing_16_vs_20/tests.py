import tempfile
import unittest
from pathlib import Path
import torch
from .common import CONDITIONS,writers,noninferiority,shards,progress_write,read,write
from .engine import apply_mask
from visual_writing_necessity.engine import apply_mask as previous


class KeepTests(unittest.TestCase):
    def setUp(self):
        self.layout=dict(prompt_length=9,I=[2,3,4],R=[5,6,7],A=[8],other=[0,1])
        self.mask=torch.zeros(1,1,9,9,dtype=torch.float16).masked_fill(
            torch.ones(9,9,dtype=torch.bool).triu(1),torch.finfo(torch.float16).min)

    def test_only_two_frozen_profiles(self):
        self.assertEqual(CONDITIONS,['Keep-0-15','Keep-0-19'])
        self.assertEqual(writers(CONDITIONS[0]),list(range(16)))
        self.assertEqual(writers(CONDITIONS[1]),list(range(20)))
        with self.assertRaises(ValueError):writers('Keep-0-23')

    def test_all_layers_and_exact_historical_composition(self):
        for c in CONDITIONS:
            for layer in range(32):
                a,b=apply_mask(self.mask,self.layout,layer,c,0)
                alias='All-Writer' if layer in writers(c) else 'No-Writer'
                x,y=previous(self.mask,self.layout,layer,alias,0)
                self.assertTrue(torch.equal(a,x) and torch.equal(b,y))
                expected=torch.zeros(9,9,dtype=torch.bool);expected[8,[2,3,4]]=True
                if layer not in writers(c):expected[5:8,2:5]=True
                self.assertTrue(torch.equal(b,expected));self.assertFalse(b[:5].any())
                self.assertTrue(torch.equal(a[0,0][~b],self.mask[0,0][~b]))

    def test_decode_always_blocks_original_images_only(self):
        for c in CONDITIONS+['All-Writer']:
            for layer in range(32):
                for past in [9,10,40]:
                    old=torch.zeros(1,1,1,past+1,dtype=torch.float16)
                    a,b=apply_mask(old,self.layout,layer,c,past)
                    self.assertEqual(torch.where(b[0])[0].tolist(),[2,3,4])
                    self.assertEqual(float(a[0,0,0,past]),0.)
                    self.assertEqual(float(a[0,0,0,0]),0.)

    def test_noop_exact_and_no_input_mutation(self):
        original=self.mask.clone()
        for c in CONDITIONS+['All-Writer','Native']:
            a,_=apply_mask(self.mask,self.layout,31,c,0,True)
            self.assertTrue(torch.equal(a,original))
        self.assertTrue(torch.equal(original,self.mask))

    def test_probability_zero_and_causal_preserved(self):
        a,b=apply_mask(self.mask,self.layout,31,CONDITIONS[0],0)
        p=torch.softmax(torch.randn(1,4,9,9)+a.float(),-1)
        self.assertEqual(float(p[0,:,b].max()),0.)
        self.assertTrue(torch.allclose(p.sum(-1),torch.ones(1,4,9)))
        self.assertFalse(b.triu().any());self.assertTrue((a[0,0].diagonal()==0).all())

    def test_ni_requires_strict_lower_bound(self):
        self.assertFalse(noninferiority(-1,1));self.assertTrue(noninferiority(-.999,1))
        self.assertFalse(noninferiority(-1.5,1));self.assertTrue(noninferiority(-1.5,2))

    def test_cluster_shards_disjoint_and_complete(self):
        ss=[dict(question_id=str(i),image_sha256=str(i//3)) for i in range(31)]
        split,owners=shards(ss)
        self.assertEqual(sum(map(len,split)),31);self.assertFalse(set(split[0])&set(split[1]))
        for s in ss:self.assertIn(s['question_id'],split[owners[s['image_sha256']]])

    def test_atomic_progress_not_checkpoint_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'progress.json';progress_write(p,{'q':1});progress_write(p,{'q':2})
            self.assertEqual(read(p),{'q':2})
            with self.assertRaises(ValueError):write(p,{'q':3})


if __name__=='__main__':unittest.main()

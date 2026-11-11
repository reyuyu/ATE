"""CPU regression checks; synthetic tensors are not research measurements."""
import unittest
from common import *
import torch
from engine import PrefixKV, DecodeKV, permissions, SKIP, ACTIVE
from freeze import grid


class Tests(unittest.TestCase):
    def test_mask_only_final_image_row(self):
        m = torch.zeros(1,1,8,8)
        m[..., -1,[1,2,3]] = -torch.inf
        copy = m.clone()
        out = permissions(m,[1,2,3],True)
        self.assertTrue(torch.equal(m,copy))
        self.assertTrue(torch.equal(out[...,:-1,:],m[...,:-1,:]))
        self.assertTrue(torch.equal(out[..., -1,[0,4,5,6,7]],m[..., -1,[0,4,5,6,7]]))
        self.assertEqual(out[...,-1,[1,2,3]].abs().max(),0)

    def test_unchanged_forbidden_mode(self):
        m = torch.zeros(1,1,1,8)
        self.assertIs(permissions(m,[1,2],False),m)

    def source(self):
        text = torch.tensor([0,4,5])
        source = PrefixKV(6,text,8,SKIP,selected=ACTIVE)
        for i in ACTIVE:
            k = torch.arange(48,dtype=torch.float32).reshape(1,1,6,8)+i
            v = k+100
            if i<22:
                source.update(k,v,i)
            else:
                source.visual[i] = tuple(t.index_select(-2,source.images).clone() for t in (k,v))
                source.update(k.index_select(-2,text),v.index_select(-2,text),i)
        source.logical_length=6
        return source

    def test_selected_cache_real_original_order(self):
        source=self.source()
        cache=DecodeKV(source,{11,24},3)
        for i in ACTIVE:
            expected=6 if i in {11,24} else 3
            self.assertEqual(cache.used[i],expected)
            if i in {11,24}:
                torch.testing.assert_close(cache.entries[i][0][...,:6,:],torch.arange(48,dtype=torch.float32).reshape(1,1,6,8)+i)

    def test_independent_cache_mutation(self):
        source=self.source()
        a,b=DecodeKV(source,{24},3),DecodeKV(source,{24},3)
        before=b.entries[24][0].clone()
        a.entries[24][0].fill_(999)
        self.assertTrue(torch.equal(before,b.entries[24][0]))
        self.assertNotEqual(a.entries[24][0].data_ptr(),b.entries[24][0].data_ptr())
        self.assertFalse(bool((source.entries[24][0][...,:3,:]==999).all()))

    def test_decode_append_and_capacity(self):
        cache=DecodeKV(self.source(),{24},2)
        x=torch.ones(1,1,1,8)
        a,b=cache.update(x,x,24)
        self.assertEqual(a.shape[-2],7)
        cache.update(x,x,24)
        with self.assertRaises(AssertionError):
            cache.update(x,x,24)

    def test_skip_kv_absent(self):
        cache=DecodeKV(self.source(),{24},2)
        self.assertTrue(all(cache.entries[i] is None for i in SKIP))
        self.assertTrue(all(cache.audit()['layers'][i]['capacity_bytes']==0 for i in SKIP))

    def test_fixed_grid_and_local_scope(self):
        g=grid()
        self.assertEqual(g['W11_24'],list(range(11,25)))
        self.assertEqual(g['B11_21'],list(range(11,22)))
        self.assertLessEqual(len(g),40)
        self.assertTrue(all(not set(x)&SKIP for x in g.values()))
        self.assertEqual(len(g),len({tuple(x) for x in g.values()}))

    def test_splits(self):
        path=OUT/'sample_manifest.json'
        if not path.exists():
            self.skipTest('Manifest not frozen yet')
        rows=json.loads(path.read_text())
        images={s:{r['image_sha256'] for r in rows if r['split']==s} for s in ['Discovery','Validation','Held-out']}
        self.assertFalse(images['Discovery']&images['Validation'] or images['Discovery']&images['Held-out'] or images['Validation']&images['Held-out'])
        placement={}
        for r in rows:
            self.assertEqual(placement.setdefault(r['cluster'],r['shard']),r['shard'])


if __name__=='__main__':
    unittest.main()

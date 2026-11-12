import unittest
import torch
from .run import GATE, compare_records, samples
from ..qwen3vl_svh_numerical_repair.engine import restore_columns
from .engine import restore_query_rows


class Tests(unittest.TestCase):
    def test_exact_frozen_scope(self):
        rows=samples()
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[-1]['question_id'],'20705853')
        self.assertEqual(rows[-1]['input_length'],122)
        self.assertEqual(rows[-1]['image_tokens'],96)

    def test_gate(self):
        self.assertEqual(GATE['mean_abs'],.02)
        self.assertEqual(GATE['max_abs'],.5)

    def test_restored_positions_and_forbidden_probability(self):
        k=torch.randn(1,2,3,4)
        v=torch.randn_like(k)
        mask=torch.zeros(1,1,3,3)
        kp,vp,mp,ix=restore_columns(k,v,mask,[0,3,5],6)
        self.assertTrue(torch.equal(kp[:,:,ix],k))
        self.assertTrue(torch.equal(vp[:,:,ix],v))
        prob=torch.softmax(mp,dim=-1)
        self.assertEqual(float(prob[:,:,:,[1,2,4]].abs().max()),0.)

    def test_first_difference_is_exact_not_tolerance(self):
        class T:pass
        a,b=T(),T()
        key=(0,4,'input_norm')
        a.records={key:dict(value=torch.zeros(1,1,2),axis=1,positions=[0],keys=None)}
        b.records={key:dict(value=torch.full((1,1,2),1e-10),axis=1,positions=[0],keys=None)}
        row=compare_records(a,b)[0]
        self.assertFalse(row['equal'])
        self.assertLess(row['max_abs'],GATE['mean_abs'])

    def test_no_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_AV_zero_rows_do_not_change_real_query_math(self):
        p=torch.randn(1,2,3,6,dtype=torch.float64)
        v=torch.randn(1,2,6,4,dtype=torch.float64)
        before=p.clone()
        expanded,ix=restore_query_rows(p,[0,3,5],6)
        self.assertTrue(torch.equal(p,before))
        self.assertTrue(torch.equal(expanded[:,:,ix],p))
        self.assertEqual(float(expanded[:,:,[1,2,4]].abs().max()),0.)
        self.assertTrue(torch.allclose(torch.matmul(p,v),torch.matmul(expanded,v)[:,:,ix],atol=1e-12,rtol=1e-12))

    def test_AV_positions_reject_duplicates(self):
        with self.assertRaises(ValueError):
            restore_query_rows(torch.ones(1,2,3,6),[0,0,5],6)


if __name__=='__main__':unittest.main()

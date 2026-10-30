import unittest
from unittest.mock import patch

import numpy as np
import torch

from gqa6_update_decomposition.data import select
from gqa6_update_decomposition.engine import NativeUpdates,measurements
from gqa6_update_decomposition.protocol import TASKS
from gqa6_update_decomposition.analyze import summarize
from paper_rq1_single_layer_gqa6.analyze import cluster_bootstrap,descriptive_correlations


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=torch.nn.Module();self.self_attn.o_proj=torch.nn.Identity()
        self.post_attention_layernorm=torch.nn.Identity();self.mlp=torch.nn.Identity()
    def forward(self,hidden_states):
        mid=hidden_states+self.self_attn.o_proj(hidden_states*.02)
        return (mid+self.mlp(self.post_attention_layernorm(mid)*-.01),)


def fixture():
    model=torch.nn.Module();model.language_model=torch.nn.Module();model.language_model.model=torch.nn.Module()
    model.language_model.model.layers=torch.nn.ModuleList([Layer() for _ in range(32)])
    return model


def native(model,x):
    for layer in model.language_model.model.layers:x=layer(x)[0]
    return x


class Tests(unittest.TestCase):
    def test_all_five_nodes_actual_query_and_native_unchanged(self):
        model=fixture();x=torch.arange(1,25).reshape(1,6,4).half()/100
        expected=native(model,x)
        with NativeUpdates(model,4) as trace:actual=native(model,x)
        self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.equal(x,torch.arange(1,25).reshape(1,6,4).half()/100))
        captured=trace.vectors();self.assertEqual(captured.shape,(32,5,4))
        self.assertTrue(torch.equal(captured[0,0],x[0,4]));self.assertFalse(torch.equal(captured[0,0],x[0,5]))
        rows=measurements(captured);self.assertEqual(len(rows),32)
        self.assertTrue(all(r['rounded_sums_exact'] for r in rows))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_cleanup_normal_and_exception(self):
        model=fixture()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with NativeUpdates(model,1):raise RuntimeError('fixture')
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        del model.language_model.model.layers[-1].post_attention_layernorm
        with self.assertRaises(AttributeError):
            with NativeUpdates(model,1):pass
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_incomplete_repeated_and_invalid_query_fail(self):
        trace=NativeUpdates(fixture(),2)
        with self.assertRaises(ValueError):trace.vectors()
        with self.assertRaises(ValueError):trace.capture(0,'h_in',torch.zeros(1,2,4))
        trace.capture(0,'h_in',torch.zeros(1,3,4))
        with self.assertRaises(ValueError):trace.capture(0,'h_in',torch.zeros(1,3,4))

    def test_fp32_metrics_and_cancellation(self):
        hin=torch.tensor([1.,2.,3.,4.]).half();a=torch.tensor([.1,.2,.3,.4]).half();m=-a
        mid=(hin+a).half();out=(mid+m).half();values=torch.stack([hin,a,mid,m,out]).repeat(32,1,1)
        r=measurements(values)[0]
        expected=float(a.float().norm()/hin.float().norm())
        self.assertEqual(r['attention_ratio'],expected);self.assertEqual(r['mlp_ratio'],expected)
        self.assertEqual(r['cancellation_ratio'],0);self.assertEqual(r['net_update_ratio'],0)
        self.assertAlmostEqual(r['cos_attention_mlp'],-1,places=6)

    def test_zero_vector_policy_and_nearzero_residual(self):
        hin=torch.ones(4).half();z=torch.zeros(4).half()
        values=torch.stack([hin,z,hin,z,hin]).repeat(32,1,1)
        row=measurements(values)[0]
        self.assertIsNone(row['cos_attention_residual']);self.assertIsNone(row['cancellation_ratio'])
        self.assertEqual(row['cos_input_output'],1);self.assertEqual(row['attention_ratio'],0)
        with self.assertRaises(ValueError):measurements(torch.zeros_like(values))

    def test_bad_sum_nonfinite_and_dtype_fail(self):
        v=torch.ones(32,5,4).half();v[:,2]=2;v[:,4]=3
        measurements(v)
        v[7,2,0]+=1
        with self.assertRaises(ValueError):measurements(v)
        v[7,2,0]=float('nan')
        with self.assertRaises(ValueError):measurements(v)
        with self.assertRaises(ValueError):measurements(torch.ones(32,5,4).float())

    def test_selection_is_input_only_and_order_independent(self):
        samples=[dict(task=t,question_id=f'{t}{i}',image_sha256=str(i),input_mapping=dict(prompt_ids=[i]),score=i)
                 for t in TASKS for i in range(500)]
        picked=select(samples);self.assertEqual(picked,select(samples[::-1]))
        ids=[s['question_id'] for s in picked]
        for s in samples:s['score']=-s['score']
        self.assertEqual(ids,[s['question_id'] for s in select(samples)])
        self.assertEqual(len(set(ids)),600)
        for t in TASKS:self.assertEqual(sum(s['task']==t for s in picked),100)

    def test_cluster_bootstrap_and_tie_ranks(self):
        ci=cluster_bootstrap([[0.],[1.],[1.]],['a','b','b'],100)
        self.assertEqual(ci.shape,(2,1));self.assertEqual(descriptive_correlations([1,2,2,3],[3,2,2,1])[1],-1)
        np.testing.assert_array_equal(cluster_bootstrap(np.zeros((4,32)),list('aabc'),100),0)

    def test_undefined_cosines_are_not_zero_filled_by_summary(self):
        hin=torch.ones(4).half();z=torch.zeros(4).half()
        rows=measurements(torch.stack([hin,z,hin,z,hin]).repeat(32,1,1))
        rows=[dict(r,bos_mass=.5,delta_logp=0.) for r in rows]
        items=[(dict(image_id='a'),rows),(dict(image_id='b'),rows)]
        with patch('gqa6_update_decomposition.analyze.bounds',side_effect=lambda v,ids:np.stack([np.asarray(v).reshape(len(v),-1).mean(0)]*2)):
            result=summarize(items,'fixture')
        self.assertIsNone(result[0]['mean_cos_attention_residual'])
        self.assertEqual(result[0]['n_valid_cos_attention_residual'],0)
        self.assertEqual(result[0]['mean_attention_ratio'],0)


if __name__=='__main__':unittest.main()

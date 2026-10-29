from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from paper_rq1_single_layer_gqa6.analyze import cluster_bootstrap, descriptive_correlations
from paper_rq1_single_layer_gqa6.data import select
from paper_rq1_single_layer_gqa6.engine import SingleLayerStep, metric_rows, positions
from paper_rq1_single_layer_gqa6.protocol import SEED, TASKS
from paper_rq1_single_layer_gqa6.run import atomic_json, save_logits


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=torch.nn.Module();self.self_attn.o_proj=torch.nn.Identity()
        self.mlp=torch.nn.Identity()
    def forward(self,hidden):
        updated=hidden+self.self_attn.o_proj(hidden*.1)
        return updated+self.mlp(updated*.2)


def model():
    m=torch.nn.Module();m.language_model=torch.nn.Module();m.language_model.model=torch.nn.Module()
    m.language_model.model.layers=torch.nn.ModuleList([Layer() for _ in range(32)])
    return m


class SingleLayerTests(unittest.TestCase):
    def test_correlations_ties_and_constant_scope(self):
        pearson,spearman=descriptive_correlations([1,2,2,4],[4,2,2,1])
        self.assertAlmostEqual(spearman,-1)
        self.assertLess(pearson,0)
        self.assertEqual(descriptive_correlations([1,1,1],[1,2,3]),(None,None))

    def test_only_one_layer_query_and_native_mlp(self):
        m=model();x=torch.randn(1,5,4);native=x.clone()
        for layer in m.language_model.model.layers:native=layer(native)
        for target in (0,15,31):
            calls=[]
            hooks=[layer.mlp.register_forward_hook(lambda m,a,o,i=i:calls.append(i)) for i,layer in enumerate(m.language_model.model.layers)]
            with SingleLayerStep(m,4,target,audit=True) as trace:
                changed=x.clone()
                for layer in m.language_model.model.layers:changed=layer(changed)
            self.assertTrue(torch.equal(native[:,:4],changed[:,:4]))
            self.assertFalse(torch.equal(native[:,4],changed[:,4]))
            self.assertEqual([r['layer'] for r in trace.edits],[target])
            self.assertTrue(trace.edits[0]['target_zero']);self.assertTrue(trace.edits[0]['other_queries_exact'])
            self.assertEqual(calls,list(range(32)))
            for h in hooks:h.remove()
            self.assertTrue(all(not module._forward_hooks for module in m.modules()))

    def test_noop_exact_and_cleanup_on_exception(self):
        m=model();x=torch.randn(1,3,4)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with SingleLayerStep(m,2,31,noop=True):
                self.assertTrue(torch.equal(m.language_model.model.layers[31].self_attn.o_proj(x),x))
                raise RuntimeError('fixture')
        self.assertTrue(all(not module._forward_hooks for module in m.modules()))

    def test_invalid_layer_and_query_rejected(self):
        for layer in (-1,32,3.5):
            with self.assertRaises(ValueError):SingleLayerStep(model(),0,layer)
        with self.assertRaises(ValueError):SingleLayerStep(model(),-1,0)

    def test_bos_discovered_not_assumed_index_zero(self):
        inputs=dict(input_ids=torch.tensor([[7,5,7,9,10]]),attention_mask=torch.tensor([[0,1,1,1,0]]))
        found=positions(inputs,7)
        self.assertEqual(found['bos'],[2]);self.assertEqual(found['query'],3)
        inputs['attention_mask'][0,0]=1
        with self.assertRaises(ValueError):positions(inputs,7)

    def test_post_softmax_bos_heads_and_row_audit(self):
        trace=SingleLayerStep(model(),3,observe=dict(bos=[2],valid_keys=[0,1,2,3]))
        weights=torch.ones(1,2,4,4)/4
        out=trace.bos_hook(12)(None,(),(torch.zeros(1,4,4),weights,None))
        self.assertIsNone(out[1]);self.assertEqual(trace.stats[0]['bos_mass'],.25)
        self.assertEqual(trace.stats[0]['bos_mass_per_head'],[.25,.25])
        weights[0,0,3,0]=.5
        with self.assertRaises(AssertionError):trace.bos_hook(12)(None,(),(torch.zeros(1,4,4),weights,None))

    def test_metrics_track_same_native_full_token_fp32(self):
        full=torch.tensor([4.,1.,-2.],dtype=torch.float16)
        changed=torch.stack([full,full.flip(0)])
        rows=metric_rows(full,changed)
        self.assertEqual(rows[0]['relative_delta_p'],0);self.assertEqual(rows[0]['delta_logp'],0)
        self.assertAlmostEqual(rows[0]['js'],0);self.assertFalse(rows[0]['top1_flip'])
        self.assertEqual(rows[1]['full_token_id'],0);self.assertEqual(rows[1]['ablated_token_id'],2)
        expected=changed[1].float().log_softmax(0)[0]-full.float().log_softmax(0)[0]
        self.assertEqual(rows[1]['delta_logp'],float(expected));self.assertTrue(rows[1]['top1_flip'])
        self.assertGreaterEqual(rows[1]['js'],0);self.assertLessEqual(rows[1]['js'],np.log(2)+1e-6)
        with self.assertRaises(ValueError):metric_rows(full,torch.tensor([[float('nan')]*3]))

    def test_unequal_image_cluster_ratio_not_average_of_image_means(self):
        values=np.array([[0.],[1.],[1.]])
        result=cluster_bootstrap(values,['a','b','b'],repetitions=1000)
        draws=np.random.default_rng(SEED).integers(0,2,(1000,2))
        expected=np.array([0.,2.])[draws].sum(1)/np.array([1,2])[draws].sum(1)
        np.testing.assert_array_equal(result,np.quantile(expected,[.025,.975])[:,None])
        zero=cluster_bootstrap(np.zeros((3,32)),['a','b','b'],repetitions=100)
        np.testing.assert_array_equal(zero,np.zeros((2,32)))

    def test_frozen_selection_counts_and_no_result_filter(self):
        rows=[]
        for task,types in TASKS.items():
            rows.extend(dict(id=f'{task}{i}',imageId=str(i),question='Real fixture?',answer='yes',
                types=dict(detailed=types[0]),isBalanced=True) for i in range(501))
        selected,_=select(rows);reversed_selection,_=select(rows[::-1])
        self.assertEqual(selected,reversed_selection)
        for task in TASKS:
            current=[s for s in selected if s['task']==task]
            self.assertEqual(len(current),500);self.assertEqual(sum(s['accuracy_subset'] for s in current),80)
        self.assertEqual(len({s['question_id'] for s in selected}),3000)
        first=next(iter(TASKS));missing=[r for r in rows if r['types']['detailed']!=TASKS[first][0]][:]
        missing.extend([r for r in rows if r['types']['detailed']==TASKS[first][0]][:499])
        with self.assertRaises(ValueError):select(missing)

    def test_atomic_checkpoint_and_all_logits_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.json';atomic_json(path,dict(conditions=list(range(32))))
            atomic_json(path,dict(conditions=list(range(32))))
            with self.assertRaises(ValueError):atomic_json(path,dict(conditions=[]))
            full=np.arange(9,dtype=np.float32);changed=np.tile(full,(32,1))
            raw=Path(directory)/'raw.npz';save_logits(raw,full,changed)
            with np.load(raw) as values:
                np.testing.assert_array_equal(values['full_logits'],full)
                np.testing.assert_array_equal(values['ablated_logits'],changed)
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])


if __name__=='__main__':unittest.main()

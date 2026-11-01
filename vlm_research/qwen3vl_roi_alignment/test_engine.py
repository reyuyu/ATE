from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .annotations import target_ids,clip_box
from .engine import AttentionRows,summarize
from .processor_check import roi_cells,reconstruct


class ObservationTests(unittest.TestCase):
    def test_semantic_targets_exclude_earlier_relation_anchor(self):
        sem=[dict(operation='select',argument='fence (1)',dependencies=[]),
             dict(operation='relate',argument='person,behind,o (2)',dependencies=[0]),
             dict(operation='choose rel',argument='racket,left|right,s (3)',dependencies=[1])]
        self.assertEqual(target_ids(sem),{'2','3'})
        sem=[dict(operation='select',argument='cat (-)',dependencies=[]),
             dict(operation='select',argument='dog (5)',dependencies=[]),
             dict(operation='or',argument='',dependencies=[0,1])]
        self.assertEqual(target_ids(sem),{'5'})

    def test_clip_records_change_and_rejects_empty(self):
        self.assertEqual(clip_box([0,17,501,358],[500,375]),([0.,17.,500.,358.],True))
        with self.assertRaises(AssertionError):clip_box([501,0,4,4],[500,375])

    def test_roi_positive_overlap_and_union(self):
        self.assertEqual(roi_cells([{'xywh':[0,0,32,32]}],64,64,2,2),[0])
        self.assertEqual(roi_cells([{'xywh':[31,31,2,2]}],64,64,2,2),[0,1,2,3])

    def test_uniform_roi_random_and_zero_mass(self):
        v=np.full((36,2,8),.05,dtype=np.float32)
        m=dict(roi_token_indices=[1,3],random_seed=42)
        metrics=summarize(v,m)
        np.testing.assert_allclose(metrics['roi_enrichment'],1)
        np.testing.assert_allclose(metrics['random_roi_enrichment'],1)
        np.testing.assert_allclose(metrics['image_mass'],.4)
        with self.assertRaises(AssertionError):summarize(v*0,m)

    def test_native_rows_read_only_and_cleanup(self):
        class Attn(torch.nn.Module):
            def forward(self,x):return x,torch.full((1,2,6,6),1/6)
        modules=[Attn() for _ in range(36)]
        model=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=m) for m in modules])))
        layout=dict(input_length=6,p_minus_1=5,image_positions=[1,2],question_positions=[3,4])
        x=torch.ones(1,6,3)
        with AttentionRows(model,layout) as observer:
            for m in modules:self.assertIs(m(x)[0],x)
        self.assertEqual(observer.arrays().shape,(36,2,2))
        np.testing.assert_allclose(observer.arrays(),1/6)
        with self.assertRaises(RuntimeError):
            with AttentionRows(model,layout):raise RuntimeError('fixture')
        self.assertTrue(all(not m._forward_hooks for m in modules))


if __name__=='__main__':unittest.main()

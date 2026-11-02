import math
import unittest

import torch

from .common import MODELS, SEEDS, conditions
from .engine import BudgetMask
from .history import token_metrics
from ..visual_pathways.test_sparse_accuracy import MaskTests


class BudgetTests(unittest.TestCase):
    def test_exact_nested_sets(self):
        for model,info in MODELS.items():
            c,perms=conditions(model);self.assertEqual(len(c),14)
            self.assertEqual(conditions(model),(c,perms))
            for seed in SEEDS:
                previous=set()
                self.assertEqual(sorted(perms[str(seed)]),list(range(info['layers'])))
                for p in [.2,.4,.6,.8]:
                    row=c[f'p{p:.1f}_s{seed}'];keep=set(row['keep'])
                    self.assertEqual(len(keep),math.floor(p*info['layers']+.5))
                    self.assertTrue(previous<keep);previous=keep
                    self.assertEqual(keep|set(row['off']),set(range(info['layers'])))
                    self.assertFalse(keep&set(row['off']))

    def test_l0_not_forced(self):
        rows=[r for m in MODELS for r in conditions(m)[0].values() if r['seed'] is not None]
        self.assertTrue(any(0 in r['keep'] for r in rows))
        self.assertTrue(any(0 not in r['keep'] for r in rows))

    def test_metrics_fp32_and_negative_loss(self):
        x=torch.tensor([2.,1.,0.]);full=dict(full_first_token=0,full_top1_logp=float(x.log_softmax(-1)[0]))
        self.assertEqual(token_metrics(x,full)['signed_logp_loss'],0)
        self.assertLess(token_metrics(torch.tensor([3.,1.,0.]),full)['signed_logp_loss'],0)

    def test_scope_reused_not_output_bypass(self):
        from ..visual_pathways.sparse_accuracy import GenerationMask
        self.assertTrue(issubclass(BudgetMask,GenerationMask))


if __name__=='__main__':unittest.main()

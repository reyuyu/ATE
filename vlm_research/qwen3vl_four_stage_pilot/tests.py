"""Small CPU-only fixtures for partition, native mass and stable signed metrics."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from .engine import MASS_TOLERANCE,SOURCES,partition,reduce_row,sensitivity
from .analyze import bootstrap,correlation,runs
from .run import OUT,choose,read,write,OLD
from ..qwen3vl_early_windows.tests import Checks


class PilotChecks(unittest.TestCase):
    def test_partition(self):
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _:5,all_special_ids=[5,9],convert_ids_to_tokens=str)
        layout = dict(input_ids=[5,9,8,8,7,6],image_positions=[2,3],question_positions=[4],p_minus_1=5)
        p = partition(tokenizer,layout,SimpleNamespace(bos_token_id=99))
        self.assertEqual(p["positions"],dict(anchor=[0],image=[2,3],question=[4],template=[1],self=[5]))
        self.assertEqual(p["actual_bos_positions"],[])

    def test_missing_anchor_rejected(self):
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _:5)
        with self.assertRaises(AssertionError):
            partition(tokenizer,dict(input_ids=[3,4]),None)

    def test_probability_reduction(self):
        row = torch.tensor([[.1,.2,.3,.15,.25]]*32)
        masses,error = reduce_row(row,{name:[i] for i,name in enumerate(SOURCES)})
        self.assertTrue(torch.equal(masses,row))
        self.assertLess(error,MASS_TOLERANCE)
        with self.assertRaises(AssertionError):
            reduce_row(row*2,{name:[i] for i,name in enumerate(SOURCES)})

    def test_signed_loss_not_clipped(self):
        m = sensitivity(torch.tensor([2.,1.,0.]),torch.tensor([3.,1.,0.]))
        self.assertLess(m["signed_logp_loss"],0)
        self.assertGreater(m["absolute_logp_change"],0)
        self.assertEqual(m["top1_flip"],0)
        zero = sensitivity(torch.tensor([1.,2.]),torch.tensor([1.,2.]))
        self.assertEqual(zero["signed_logp_loss"],0)

    def test_cluster_bootstrap(self):
        values = np.array([[1.,2.],[-1.,-2.],[0.,0.]])
        self.assertTrue(np.array_equal(bootstrap(values,["a","a","b"],100),np.zeros((2,2))))

    def test_runs_not_forced_segments(self):
        self.assertEqual(runs([False,True,True,True,False,True]),[[1,3]])
        self.assertEqual(runs([False]*36),[])

    def test_sampling_only_inputs(self):
        rows = read(OLD / "sample_manifest.json")["samples"]
        a = choose(rows)
        b = choose(list(reversed(rows)))
        self.assertEqual([r["input_binding"] for r in a],[r["input_binding"] for r in b])
        self.assertEqual(len(a),120)
        self.assertEqual(len({r["image_sha256"] for r in a}),120)

    def test_tied_rank_correlation(self):
        self.assertAlmostEqual(correlation([1,2,2,3],[3,2,2,1],True),-1)
        self.assertIsNone(correlation([1,1],[2,3]))


if __name__ == "__main__":
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (Checks,PilotChecks)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(OUT / "cpu_tests.json",dict(passed=result.wasSuccessful(),tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),scope="CPU only; includes reused query-locality/cleanup tests"))
    raise SystemExit(not result.wasSuccessful())

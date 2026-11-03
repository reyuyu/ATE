import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from .common import conditions,layer_of,seal,read,MODELS
from .engine import PrefillP1SingleLayerBypass,footprint
from .prepare import balanced


class Toy(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(n)])
        for layer in self.layers:
            layer.self_attn=torch.nn.Module()
            layer.self_attn.o_proj=torch.nn.Identity()
    def forward(self,input_ids,cache_position=None,past_key_values=None):
        out=input_ids.float().unsqueeze(-1).repeat(1,1,3)
        for layer in self.layers:
            out=out+layer.self_attn.o_proj(out)
        return out


class Checks(unittest.TestCase):
    def test_historical_tie_aware_correlations(self):
        from .analyze import historical_correlations
        pearson,spearman=historical_correlations([1,1,3,4],[4,4,2,1])
        self.assertAlmostEqual(pearson,-1.)
        self.assertAlmostEqual(spearman,-1.)
    def test_cluster_bootstrap_exact_null(self):
        from .analyze import bootstrap
        data=np.array([[0.,0.],[100.,100.],[100.,100.]])
        draws=bootstrap(data,['a','a','b'])
        self.assertTrue(np.array_equal(draws[:,0],draws[:,1]))
        self.assertEqual(draws.shape,(20000,2))
        self.assertTrue(set(np.unique(draws[:,0])).issubset({50.,100.,200/3}))
    def test_cluster_bootstrap_reproducible(self):
        from .analyze import bootstrap
        data=np.array([[0.,100.],[100.,0.]])
        self.assertTrue(np.array_equal(bootstrap(data,['a','b']),bootstrap(data,['a','b'])))
    def test_exact_layer_names(self):
        for kind,m in MODELS.items():
            self.assertEqual(len(conditions(kind)),m['layers']+1)
            self.assertEqual([layer_of(kind,c) for c in conditions(kind)],[None]+list(range(m['layers'])))
            with self.assertRaises(AssertionError):layer_of(kind,'Drop-P1-L99')
    def test_prefill_only(self):
        for n in (32,36):
            model=Toy(n); before=footprint(model)
            with PrefillP1SingleLayerBypass(model,model.layers,7,4) as ctl:
                x=torch.arange(1,8)[None]
                model(input_ids=x,cache_position=torch.arange(7))
                self.assertEqual(sum(e['active'] for e in ctl.events),1)
            self.assertEqual(footprint(model),before)
    def test_decode_identity(self):
        model=Toy(32)
        class Cache:
            def get_seq_length(self):return 7
        with PrefillP1SingleLayerBypass(model,model.layers,7,4) as ctl:
            model(input_ids=torch.ones(1,7,dtype=torch.long),cache_position=torch.arange(7))
            observed=model(input_ids=torch.ones(1,1,dtype=torch.long),cache_position=torch.tensor([7]),past_key_values=Cache())
            self.assertTrue(ctl.events[0]['active']);self.assertFalse(ctl.events[1]['active'])
            self.assertTrue(torch.equal(observed,model.layers[0].self_attn.o_proj(torch.tensor([[[2.**32]*3]]))))
    def test_noop_exact(self):
        m=Toy(32); x=torch.ones(1,9,dtype=torch.long);full=m(x)
        with PrefillP1SingleLayerBypass(m,m.layers,9,31,True):
            self.assertTrue(torch.equal(full,m(input_ids=x)))
    def test_exception_cleanup(self):
        m=Toy(32);before=footprint(m)
        with self.assertRaises(RuntimeError):
            with PrefillP1SingleLayerBypass(m,m.layers,9,1):raise RuntimeError('expected')
        self.assertEqual(before,footprint(m))
    def test_invalid_layers(self):
        m=Toy(32)
        for i in (-1,32,1.2,True):
            with self.assertRaises(ValueError):PrefillP1SingleLayerBypass(m,m.layers,7,i)
    def test_immutable_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'saved.json';seal(p,{'tokens':[1,2]});seal(p,{'tokens':[1,2]})
            with self.assertRaises(AssertionError):seal(p,{'tokens':[3]})
            self.assertEqual(read(p),{'tokens':[1,2]})
    def test_image_shards(self):
        samples=[dict(task='t',question_id=str(i),image_sha256=str(i//3)) for i in range(60)]
        parts=balanced(samples);self.assertEqual(parts,balanced(list(reversed(samples))))
        lookup={r['task']+'__'+r['question_id']:r['image_sha256'] for r in samples}
        self.assertTrue({lookup[k] for k in parts[0]}.isdisjoint({lookup[k] for k in parts[1]}))
        self.assertEqual(len(parts[0])+len(parts[1]),60)


if __name__=='__main__':unittest.main()

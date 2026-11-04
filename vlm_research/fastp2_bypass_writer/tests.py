import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaModel

from ..physical_handoff_fastpath.engine import Layout
from .engine import BypassWriter, LAYERS, CONDITIONS


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=4, vocab_size=64)
        config._attn_implementation = 'eager'
        self.decoder = LlamaModel(config).eval()
        self.a = SimpleNamespace(kind='llava', model=SimpleNamespace(language_model=SimpleNamespace(model=self.decoder)),
            layers=self.decoder.layers, inputs={'input_ids': torch.arange(14).unsqueeze(0)})
        self.layout = Layout(14, tuple(range(2, 6)), tuple([0,1]+list(range(6,14))), tuple(range(6,13)), (6,7,8), 6)
        self.selection = {'writers': {'Semantic-k': [6,7], **{'Random-k-'+str(s): [8,9] for s in (20261103,20261107,20261111)}}}
        self.h = torch.randn(1,14,32)

    def execute(self, name, decode=True, audit=True):
        ctl = BypassWriter(self.a, self.layout, name, self.selection, 4, audit)
        calls = [[0,0,0,0] for _ in self.a.layers]
        mlp = [0]*len(self.a.layers)
        handles = []
        for i, layer in enumerate(self.a.layers):
            for j, p in enumerate(('q_proj','k_proj','v_proj','o_proj')):
                def count(m, a, out, i=i, j=j): calls[i][j] += 1
                handles.append(getattr(layer.self_attn,p).register_forward_hook(count))
            def count_mlp(m, a, out, i=i): mlp[i] += 1
            handles.append(layer.mlp.register_forward_hook(count_mlp))
        try:
            with torch.inference_mode(), ctl.context():
                first = ctl.forward(inputs_embeds=self.h).last_hidden_state
                second = ctl.forward(inputs_embeds=torch.ones(1,1,32)).last_hidden_state if decode else None
        finally:
            for h in handles: h.remove()
        return first, second, ctl, calls, mlp

    def test_all_branch_exact_reference_and_no_qkvo(self):
        a = self.execute('A1')
        b = self.execute('A3')
        self.assertTrue(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]))
        for i in LAYERS: self.assertEqual(a[3][i],[0,0,0,0])
        self.assertEqual(a[4],[2]*8)
        self.assertEqual(a[2].cache.audit()['allocations'],8)

    def test_ia_partial_reference(self):
        a, b = self.execute('A2'), self.execute('A2-Reference')
        torch.testing.assert_close(a[0],b[0],atol=3e-7,rtol=3e-6)
        torch.testing.assert_close(a[1],b[1],atol=3e-7,rtol=3e-6)
        for i in LAYERS: self.assertEqual(a[3][i],[1,1,1,1])
        self.assertEqual(a[4],[2]*8)

    def test_split_writer_reference(self):
        a,b = self.execute('B1'),self.execute('B3')
        torch.testing.assert_close(a[0],b[0],atol=3e-7,rtol=3e-6)
        torch.testing.assert_close(a[1],b[1],atol=3e-7,rtol=3e-6)
        self.assertEqual(a[3],b[3])

    def test_mask_exact_edges(self):
        a = BypassWriter(self.a,self.layout,'A0',self.selection,4)
        b = BypassWriter(self.a,self.layout,'B1',self.selection,4)
        changed = a.prefill_early != b.prefill_early
        expected = torch.zeros_like(changed)
        for q in (8,9,10,11,12): expected[0,0,q,list(self.layout.image)] = True
        self.assertTrue(torch.equal(changed,expected))
        self.assertTrue(torch.isneginf(b.prefill_early[changed]).all())
        self.assertTrue((b.prefill_early == 0).any(-1).all())

    def test_ia_query_exact_scope(self):
        _,_,ctl,_,_ = self.execute('A2')
        for e in ctl.events:
            if e['layer'] in LAYERS and e['prefill']:
                self.assertEqual(e['query_positions'],[0,1,6,7,8,9,10,11,12])
            elif e['layer'] in LAYERS:
                self.assertEqual(e['computed_queries'],0)

    def test_exception_cleanup(self):
        old = [l.self_attn.forward for l in self.a.layers]
        ctl = BypassWriter(self.a,self.layout,'B3',self.selection,4)
        with self.assertRaises(ValueError):
            with ctl.context(): raise ValueError('intentional cleanup test')
        self.assertEqual(old,[l.self_attn.forward for l in self.a.layers])

    def test_all_conditions_finite_independent_cache(self):
        for name in CONDITIONS:
            a,b,ctl,_,mlp=self.execute(name)
            self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
            self.assertEqual(ctl.cache.logical_length,15)
            self.assertEqual(mlp,[2]*8)
            ctl.cache.audit()

    def test_official_textvqa_fixtures(self):
        from .common import score
        for n,v in ((0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)):
            self.assertAlmostEqual(score({'dataset':'textvqa300','answers':['white']*n+['black']*(10-n)},'White'),v)


if __name__ == '__main__':
    unittest.main(verbosity=2)

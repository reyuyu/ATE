import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaModel

from .common import CFG, KEEP, score
from .engine import MiddleWriter
from ..accuracy_autoresearch.engine import ResearchWriter
from ..ate_fulljoint_v2.engine import LAYERS
from ..physical_handoff_fastpath.engine import Layout


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(21)
        cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=32,
            num_attention_heads=4, num_key_value_heads=4, vocab_size=64)
        cfg._attn_implementation = 'eager'
        self.decoder = LlamaModel(cfg).eval()
        self.a = SimpleNamespace(kind='llava',
            model=SimpleNamespace(language_model=SimpleNamespace(model=self.decoder)),
            layers=self.decoder.layers, inputs={'input_ids': torch.arange(14).unsqueeze(0)})
        self.layout = Layout(14, tuple(range(2,6)), tuple([0,1]+list(range(6,14))),
            tuple(range(6,13)), (6,7,8), 20)
        self.h = torch.randn(1,14,32)

    def execute(self, keep=(), cutoff=20, dense=False, inherited=False):
        layout = replace(self.layout, cutoff=cutoff)
        ctl = (ResearchWriter(self.a, layout, CFG, list(layout.relay), 4, True) if inherited else
               MiddleWriter(self.a, layout, keep, 4, True, dense))
        calls, handles = [], []
        for i, layer in enumerate(self.a.layers):
            for op in ('q_proj','k_proj','v_proj','o_proj','mlp'):
                target = layer.mlp if op=='mlp' else getattr(layer.self_attn,op)
                def capture(m,a,o,i=i,op=op): calls.append((i,op,a[0].shape[1]))
                handles.append(target.register_forward_hook(capture))
        try:
            with torch.inference_mode(), ctl.context():
                first = ctl.forward(inputs_embeds=self.h).last_hidden_state.clone()
                pointers = [None if e is None else tuple(v.data_ptr() for v in e) for e in ctl.cache.entries]
                second = ctl.forward(inputs_embeds=torch.ones(1,1,32)).last_hidden_state.clone()
                self.assertEqual(pointers,[None if e is None else tuple(v.data_ptr() for v in e) for e in ctl.cache.entries])
        finally:
            for h in handles: h.remove()
        return first, second, ctl, calls

    def test_c3_regression_exact(self):
        a,b=self.execute(),self.execute(inherited=True)
        self.assertTrue(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]))

    def test_middle9_dense_reference(self):
        a,b=self.execute(KEEP['C3+Middle9']),self.execute(KEEP['C3+Middle9'],dense=True)
        torch.testing.assert_close(a[0],b[0].index_select(1,a[2].text_index),atol=3e-7,rtol=3e-6)
        torch.testing.assert_close(a[1],b[1],atol=3e-7,rtol=3e-6)

    def test_middle10_dense_reference(self):
        a,b=self.execute(KEEP['C3+Middle10'],21),self.execute(KEEP['C3+Middle10'],21,dense=True)
        torch.testing.assert_close(a[0],b[0].index_select(1,a[2].text_index),atol=3e-7,rtol=3e-6)
        torch.testing.assert_close(a[1],b[1],atol=3e-7,rtol=3e-6)

    def test_exact_mask_scope_and_l20_receiver_block(self):
        c=MiddleWriter(self.a,replace(self.layout,cutoff=21),KEEP['C3+Middle10'],4)
        for layer in range(21):
            mask=c.prefill_mask(layer)
            self.assertTrue((mask==0).any(-1).all())
            for q in self.layout.relay:
                self.assertEqual(bool(torch.isneginf(mask[0,0,q,list(self.layout.image)]).all()),layer>=20)
            self.assertEqual(bool((mask[0,0,-1,list(self.layout.image)]==0).all()),11<=layer<=20)
        self.assertEqual(c.prefill_mask(21).shape[-1],len(self.layout.text))

    def test_selective_persistence_and_no_copies(self):
        for label in ('C3+Middle9','C3+Middle10'):
            _,_,c,_=self.execute(KEEP[label],20 if label.endswith('9') else 21)
            audit=c.cache.audit()
            self.assertEqual(audit['retained_image_layers'],list(KEEP[label]))
            self.assertEqual(audit['historical_cache_copies'],0)
            self.assertEqual(audit['append_concatenations'],0)
            for r in audit['layers']:
                self.assertEqual(r['used_tokens'],0 if r['layer'] in LAYERS else
                    (self.layout.length if r['layer'] in KEEP[label] else len(self.layout.text))+1)
                if r['used_tokens']: self.assertEqual(r['original_positions'][-1],14)

    def test_query_shapes_and_complete_mlp(self):
        _,_,c,calls=self.execute(KEEP['C3+Middle10'],21)
        for i in range(32):
            self.assertEqual([n for l,op,n in calls if l==i and op=='mlp'],[14 if i<21 else 10,1])
            for op in ('q_proj','o_proj'):
                self.assertEqual([n for l,o,n in calls if l==i and o==op],
                    [13] if i in LAYERS else [14 if i<21 else 10,1])
            for op in ('k_proj','v_proj'):
                self.assertEqual([n for l,o,n in calls if l==i and o==op],
                    [14] if i in LAYERS else [14 if i<21 else 10,1])
        self.assertTrue(all(e['blocked_probability_max']==0 for e in c.events))

    def test_target_visual_probabilities(self):
        _,_,c,_=self.execute(KEEP['C3+Middle10'],21)
        for e in c.visual_events:
            if e['layer'] in KEEP['C3+Middle10']: self.assertGreater(e['answer_image_mass'],0)
            else: self.assertEqual(e['answer_image_mass'],0)
            if e['prefill'] and e['layer']>=20: self.assertEqual(e['relay_image_mass'],0)

    def test_other_prefill_queries_invariant(self):
        a,b=self.execute(),self.execute(KEEP['C3+Middle9'])
        for x,y in zip(a[2].hidden,b[2].hidden):
            self.assertTrue(torch.equal(x[:,:-1],y[:,:-1]))

    def test_cleanup_normal_and_exception(self):
        original=[l.self_attn.forward for l in self.a.layers]
        decoder=self.decoder.forward
        self.execute(KEEP['C3+Middle9'])
        self.assertEqual(original,[l.self_attn.forward for l in self.a.layers])
        self.assertEqual(decoder,self.decoder.forward)
        c=MiddleWriter(self.a,self.layout,KEEP['C3+Middle9'],4)
        with self.assertRaises(ValueError):
            with c.context(): raise ValueError('intentional exception')
        self.assertEqual(original,[l.self_attn.forward for l in self.a.layers])
        self.assertEqual(decoder,self.decoder.forward)

    def test_noop_and_causal_mask(self):
        c=MiddleWriter(self.a,self.layout,KEEP['C3+Middle9'],4)
        for i in range(20):
            mask=c.prefill_mask(i)[0,0]
            upper=torch.triu(torch.ones_like(mask,dtype=torch.bool),diagonal=1)
            self.assertTrue((mask[upper]<-1e20).all())

    def test_input_weights_and_finite(self):
        versions=[(p._version,p.data_ptr()) for p in self.decoder.parameters()]
        original=self.h.clone()
        a,b,_,_=self.execute(KEEP['C3+Middle10'],21)
        self.assertTrue(torch.isfinite(a).all() and torch.isfinite(b).all())
        self.assertTrue(torch.equal(original,self.h))
        self.assertEqual(versions,[(p._version,p.data_ptr()) for p in self.decoder.parameters()])

    def test_official_soft_score(self):
        for n,v in ((0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)):
            self.assertAlmostEqual(score({'answer':['white']*n+['black']*(10-n)},'White'),v)


if __name__=='__main__': unittest.main(verbosity=2)

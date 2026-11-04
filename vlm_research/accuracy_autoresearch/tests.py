from contextlib import ExitStack
import unittest
import torch

from ..ate_fulljoint_v2 import tests as v2_tests
from .engine import ResearchWriter, LAYERS
from .common import config


class Tests(v2_tests.Tests):
    def run_research(self,mode,reference=False):
        ctl=ResearchWriter(self.a,self.layout,config('qsa',mode),[6,7,8,9],4,True,reference)
        calls={}
        with ExitStack() as stack:
            for i,layer in enumerate(self.a.layers):
                def observe(m,a,out,i=i):
                    calls.setdefault(i,[]).append(a[0].shape[1])
                stack.callback(layer.self_attn.q_proj.register_forward_hook(observe).remove)
            with torch.inference_mode(),ctl.context():
                first=ctl.forward(inputs_embeds=self.h).last_hidden_state
                second=ctl.forward(inputs_embeds=torch.ones(1,1,32)).last_hidden_state
        return first,second,ctl,calls

    def test_decode_only_keeps_prefill(self):
        a,b,c=self.run_research('decode'),self.run_research('none'),self.run_research('decode',True)
        self.assertTrue(torch.equal(a[0],b[0]))
        torch.testing.assert_close(a[1],c[1],atol=3e-7,rtol=3e-6)
        for i in LAYERS:
            self.assertEqual(a[3][i],[14])
        self.assertTrue(all(r['used_tokens']==0 for r in a[2].cache.audit()['layers'] if r['layer'] in LAYERS))

    def test_both_matches_reference(self):
        a,b=self.run_research('both'),self.run_research('both',True)
        for x,y in zip(a[:2],b[:2]):
            torch.testing.assert_close(x,y,atol=3e-7,rtol=3e-6)
        for i in LAYERS:
            self.assertEqual(a[3][i],[13])

    def test_answer_reading_has_no_image_keys(self):
        ctl=ResearchWriter(self.a,self.layout,config('qsa','decode'),[6,7,8,9],4)
        self.assertTrue(torch.isneginf(ctl.prefill_early[..., -1,list(self.layout.image)]).all())
        for q in (10,11,12):
            self.assertTrue(torch.isneginf(ctl.prefill_early[...,q,list(self.layout.image)]).all())
        self.assertTrue((ctl.prefill_early==0).any(-1).all())


if __name__=='__main__':
    unittest.main(verbosity=2)

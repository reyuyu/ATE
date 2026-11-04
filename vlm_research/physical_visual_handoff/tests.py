import unittest
import torch

from .engine import CompactKV, Plan, plan, source_mask


class Tests(unittest.TestCase):
    def test_roles_and_causality(self):
        pos=torch.arange(9); image=torch.tensor([2,3,4]); relay=[5,6,7]; question=[6]
        for c in ('N0','D0','D1','D2','D3','P1','P2','P3','P4'):
            p=plan('llava',c); p.length=9
            for i in (0,19,20,31):
                p.layer=i
                m,b=source_mask(pos,pos,image,relay,question,p,torch.float32)
                self.assertTrue((m[0,0].triu(1)[torch.ones(9,9,dtype=torch.bool).triu(1)]<0).all())
                self.assertTrue((m[0,0]==0).any(-1).all())
                self.assertTrue((m[0,0].diagonal()==0).all())
                self.assertEqual(bool(b[8,2]),c!='N0')
                self.assertFalse(bool(b[:5].any()))
                if c=='D2' and i<20:
                    self.assertFalse(bool(b[6,2])); self.assertTrue(bool(b[5,2]))

    def test_cache_positions_and_storage(self):
        c=CompactKV(3); c.current_positions=torch.arange(9)
        k=torch.randn(1,2,9,4); v=k.clone()
        c.update(k,v,0); del k,v
        c.remove_visual(0,torch.tensor([2,3,4]))
        self.assertEqual(c.entries[0][2].tolist(),[0,1,5,6,7,8])
        c.logical_length=9; c.current_positions=torch.tensor([9])
        c.update(torch.ones(1,2,1,4),torch.ones(1,2,1,4),0)
        self.assertEqual(c.entries[0][0].shape[-2],7)
        self.assertEqual(c.get_seq_length(),9)
        self.assertEqual(c.audit()['old_tensor_references_alive'],0)
        c.drop_layer(0)
        self.assertEqual(c.audit()['live_kv_bytes'],0)

    def test_physical_and_functional_plans(self):
        self.assertEqual(plan('llava','P4').skip,(2,3,4,5))
        self.assertEqual(plan('qwen','P4-six').skip,(1,2,3,4,5,6))
        self.assertEqual(plan('qwen','D3-six').skip,(1,2,3,4,5,6))
        self.assertFalse(plan('llava','D3').physical_skip)
        self.assertTrue(plan('llava','P4',True).immediate)

    def test_decode_original_logical_position(self):
        p=plan('qwen','P2');p.length=9;p.layer=1
        m,b=source_mask(torch.tensor([10]),torch.tensor([0,1,5,6,7,8,9,10]),
                        torch.tensor([2,3,4]),[5,6,7],[6],p,torch.float32)
        self.assertEqual(m.shape,(1,1,1,8)); self.assertFalse(bool(b.any()))

    def test_native_llama_decoder_noop_and_compaction(self):
        import types
        from transformers import LlamaConfig, LlamaModel
        from .engine import Handoff
        torch.manual_seed(4)
        cfg=LlamaConfig(hidden_size=32,intermediate_size=48,num_hidden_layers=3,num_attention_heads=4,
                        num_key_value_heads=2,vocab_size=100,attn_implementation='eager')
        decoder=LlamaModel(cfg).eval()
        ctl=Handoff.__new__(Handoff)
        ctl.a=types.SimpleNamespace(kind='llava',layers=decoder.layers)
        ctl.decoder=decoder;ctl.n=9;ctl.p=plan('llava','N0');ctl.p.length=9
        ctl.image=torch.tensor([2,3,4]);ctl.relay=[5,6,7];ctl.question=[6]
        ctl.cache=CompactKV(3);ctl.events=[];ctl.hidden=[];ctl.mask_events=[];ctl.step=0
        ctl.audit_enabled=True;ctl.rotary_hashes=[]
        embed=torch.randn(1,9,32);pos=torch.arange(9)
        with torch.inference_mode():
            expected=decoder(inputs_embeds=embed,attention_mask=torch.ones(1,9),use_cache=True)
            actual=ctl.forward(inputs_embeds=embed,position_ids=pos[None],cache_position=pos)
        self.assertTrue(torch.equal(expected.last_hidden_state,actual.last_hidden_state))
        self.assertEqual(actual.past_key_values.get_seq_length(),9)


if __name__=='__main__': unittest.main()

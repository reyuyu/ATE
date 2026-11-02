import unittest
import torch
from .common import conditions, divergent
from .engine import DecodeMask
from ..visual_pathways.engine import masked


class Tests(unittest.TestCase):
    def test_sets(self):
        for kind,n,counts in [('qwen',36,[36,0,12,13,13,12,13,13]),
                              ('internvl',28,[28,0,10,10,10,10,10,10])]:
            c = conditions(kind)
            self.assertEqual(len(c),8)
            self.assertEqual(list(map(len,c.values())),counts)
            self.assertEqual(c,conditions(kind))
            for k,v in c.items():
                self.assertEqual(len(v),len(set(v)))
                self.assertTrue(set(v) <= set(range(n)))
                if k not in ('Full','Decode None'): self.assertEqual(v[0],0)

    def test_divergence(self):
        self.assertEqual(divergent([1,2],[1,2]),-1)
        self.assertEqual(divergent([1,2],[1,3]),1)
        self.assertEqual(divergent([1],[1,2]),1)

    def test_legal_only(self):
        m = torch.zeros(1,1,3,3).masked_fill(torch.ones(3,3,dtype=torch.bool).triu(1),-torch.inf)
        v,count = masked(m,[1],[0,2])
        self.assertEqual(count,1)
        self.assertTrue(torch.equal(v[:,:,0],m[:,:,0]))
        self.assertEqual(float(v[0,0,1,2]),-torch.inf)

    def exercise(self, noop=False, fail=False):
        class Attention(torch.nn.Module):
            def forward(self,hidden_states,attention_mask):
                return hidden_states,attention_mask.softmax(-1).expand(1,2,-1,-1)
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.self_attn=Attention()
            def forward(self,x,m): return self.self_attn(hidden_states=x,attention_mask=m)[0]
        class LM(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.layers=torch.nn.ModuleList([Layer(),Layer()])
            def forward(self,inputs_embeds,attention_mask,cache_position,past_key_values):
                for layer in self.layers: layer(inputs_embeds,attention_mask)
        class Cache:
            length=0
            def get_seq_length(self,i): return self.length
        lm,cache=LM(),Cache()
        try:
            with DecodeMask(lm,lm.layers,5,[1,2],[0],cache,noop) as control:
                lm(inputs_embeds=torch.ones(1,5,4),attention_mask=torch.zeros(1,1,5,5),
                   cache_position=torch.arange(5),past_key_values=cache)
                self.assertTrue(all(r['image_max'] > 0 and not r['masked'] for r in control.rows))
                cache.length=5
                lm(inputs_embeds=torch.ones(1,1,4),attention_mask=torch.zeros(1,1,1,6),
                   cache_position=torch.tensor([5]),past_key_values=cache)
                self.assertEqual(control.rows[2]['masked'],not noop)
                self.assertGreater(control.rows[3]['image_max'],0)
                if not noop: self.assertEqual(control.rows[2]['image_max'],0)
                if fail: raise RuntimeError('forced')
        except RuntimeError as exc:
            self.assertTrue(fail and str(exc)=='forced')
        self.assertFalse(lm._forward_pre_hooks)
        for layer in lm.layers:
            self.assertFalse(layer.self_attn._forward_hooks)
            self.assertFalse(layer.self_attn._forward_pre_hooks)

    def test_prefill_native_decode_off(self): self.exercise()
    def test_noop(self): self.exercise(noop=True)
    def test_exception_cleanup(self): self.exercise(fail=True)


if __name__ == '__main__': unittest.main()

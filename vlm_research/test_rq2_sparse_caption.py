import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from benchmarks.smoke import hook_footprint
from rq2_sparse_caption import generate, layers
from rq2_visual_sufficiency.engine import MultiImageKnockout
from test_rq2_source_knockout import Tiny, inputs


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(27)
        self.model=Tiny()
        config=LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=24,
            num_hidden_layers=32,num_attention_heads=2,num_key_value_heads=2,
            max_position_embeddings=64,attention_dropout=0.)
        config._attn_implementation='eager'; config.image_token_index=60
        self.model.config=config;self.model.language_model=LlamaForCausalLM(config).eval()
        self.data=inputs()

    def call(self,key,**kwargs):
        return generate(self.model,self.data,1,key,expected_visual=4,limit=3,**kwargs)

    def test_fixed_conditions(self):
        self.assertEqual(layers('DS',32),([], [k for k in range(32) if k not in [14,19]]))
        self.assertEqual(layers('DN',32),([],list(range(32))))
        self.assertEqual(layers('S',32),(layers('DS',32)[1],)*2)

    def test_noop_full_logits_and_independent_cache(self):
        a,af,_,at,ad=self.call('A',collect=True,audit=True)
        for key in ['S','DS','DN']:
            g,f,_,trace,d=self.call(key,noop=True,collect=True,audit=True)
            self.assertEqual(g['token_ids'],a['token_ids'])
            self.assertTrue(torch.equal(ad['prefill_logits'],d['prefill_logits']))
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(at,trace)))

    def test_scope_actual_zero_and_unaffected_queries(self):
        footprint=hook_footprint(self.model)
        _,af,_,_,ad=self.call('A',audit=True)
        for key in ['S','DS','DN']:
            g,f,rows,_,d=self.call(key,audit=True)
            self.assertTrue(torch.equal(ad['prefill_logits'][:,:-1],d['prefill_logits'][:,:-1]))
            self.assertEqual(ad['states'],d['states'])
            if key in ['DS','DN']: self.assertTrue(torch.equal(af,f))
            for r in rows:
                expected=r['layer'] in layers(key,32)[int(r['step']>0)]
                self.assertEqual(r['blocked'],expected)
                self.assertEqual(r['image_probability_max']==0.,expected)
            self.assertEqual(hook_footprint(self.model),footprint)

    def test_exception_cleanup_and_inputs(self):
        before=hook_footprint(self.model);ids=self.data['input_ids'].clone()
        with self.assertRaises(RuntimeError):
            with MultiImageKnockout(self.model,self.data,1,[0],expected_visual=4):
                raise RuntimeError('intentional')
        self.assertEqual(before,hook_footprint(self.model))
        self.call('DN')
        self.assertTrue(torch.equal(ids,self.data['input_ids']))


if __name__=='__main__': unittest.main()

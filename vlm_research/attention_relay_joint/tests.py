import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from benchmarks.smoke import hook_footprint
from paper_rq1_early_joint_attention.statistics import bootstrap
from test_rq2_source_knockout import Tiny, inputs
from .common import CONDITIONS, LAYERS
from .engine import generate
from .analyze import divergence, pair_flags


def tiny():
    model=Tiny()
    config=LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=24,
        num_hidden_layers=32,num_attention_heads=2,num_key_value_heads=2,
        max_position_embeddings=128,attention_dropout=0.,eos_token_id=2)
    config._attn_implementation='eager';config.image_token_index=60
    model.config=config;model.language_model=LlamaForCausalLM(config).eval()
    return model


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(19)

    def test_four_frozen_conditions_and_layers(self):
        self.assertEqual(list(CONDITIONS),['Full','A-Skip','Text-Only','Joint'])
        self.assertEqual(CONDITIONS['Joint'],(True,True))
        self.assertEqual(LAYERS,[2,3,4,5])

    def test_real_tiny_noop_and_native_hf(self):
        m=tiny();p=inputs()
        f,ff,ft,fd=generate(m,p,'Full',limit=4,audit=True,collect=True,expected_visual=4)
        n,nf,nt,nd=generate(m,p,'Joint',noop=True,limit=4,audit=True,collect=True,expected_visual=4)
        self.assertEqual(f['token_ids'],n['token_ids'])
        self.assertTrue(torch.equal(ff,nf));self.assertTrue(torch.equal(fd['prefill_logits'],nd['prefill_logits']))
        self.assertEqual(len(ft),len(nt));self.assertTrue(all(torch.equal(a,b) for a,b in zip(ft,nt)))
        self.assertEqual(fd['initial_cache_sha256'],nd['initial_cache_sha256'])
        with torch.inference_mode():
            ids=m.language_model.generate(**p,max_new_tokens=4,do_sample=False,use_cache=True,eos_token_id=2,pad_token_id=0)
        self.assertEqual(ids[0,9:].tolist(),f['token_ids'])
        with torch.inference_mode():
            aligned=m.language_model.generate(**p,max_new_tokens=4,do_sample=False,use_cache=True,
                eos_token_id=2,pad_token_id=0,logits_to_keep=0,return_dict_in_generate=True,output_scores=True)
        self.assertTrue(all(torch.equal(a[0].float(),b) for a,b in zip(aligned.scores,ft)))

    def test_joint_both_hooks_scope_cache_and_non_target(self):
        m=tiny();p=inputs();before=hook_footprint(m)
        full,_,_,fd=generate(m,p,'Full',limit=4,audit=True,expected_visual=4)
        for c,(skip,off) in CONDITIONS.items():
            g,_,_,d=generate(m,p,c,limit=4,audit=True,expected_visual=4)
            self.assertTrue(torch.equal(fd['prefill_logits'][:,:8],d['prefill_logits'][:,:8]))
            self.assertEqual(fd['other_query_states'],d['other_query_states'])
            self.assertEqual(fd['other_query_attention'],d['other_query_attention'])
            self.assertEqual(g['attention_output_edits'],4*g['length'] if skip else 0)
            for t,step in enumerate(g['steps']):
                self.assertEqual(step['absolute_query'],8+t)
                self.assertEqual(step['cache_after'],9+t)
                self.assertEqual(step['cache_before'],8+t if t else 0)
                self.assertEqual([r['applied'] for r in step['masks']],[off]*32)
                if off:self.assertTrue(all(r['image_probability_max']==0 for r in step['probabilities']))
            self.assertEqual(hook_footprint(m),before)
        after,_,_,_=generate(m,p,'Full',limit=4,expected_visual=4)
        self.assertEqual(full['token_ids'],after['token_ids'])

    def test_exception_both_hooks_removed(self):
        m=tiny();p=inputs();before=hook_footprint(m)
        with patch.object(m.language_model.model.layers[3].self_attn.q_proj,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                generate(m,p,'Joint',expected_visual=4)
        self.assertEqual(hook_footprint(m),before)
        generate(m,p,'Full',limit=2,expected_visual=4)

    def test_dynamic_mapping_does_not_assume_frozen_token_index(self):
        m=tiny();p=inputs()
        p['input_ids']=torch.cat([p['input_ids'],torch.tensor([[20,21,22]])],dim=-1)
        p['attention_mask']=torch.ones_like(p['input_ids'])
        g,_,_,_=generate(m,p,'Joint',limit=3,expected_visual=4)
        self.assertEqual(g['original_p_minus_one'],11)
        self.assertEqual(g['image_keys'],[2,3,4,5])
        if len(g['steps'])>1:self.assertEqual(g['steps'][1]['absolute_query'],12)

    def test_reject_unapproved_conditions_padding_and_budget(self):
        m=tiny();p=inputs()
        with self.assertRaises(ValueError):generate(m,p,'Other',expected_visual=4)
        with self.assertRaises(ValueError):generate(m,p,'Full',limit=33,expected_visual=4)
        p['attention_mask'][0,0]=0
        with self.assertRaises(ValueError):generate(m,p,'Full',expected_visual=4)

    def test_divergence_eos_and_joint_only_harm(self):
        self.assertEqual(divergence([10,2],[10,2]),('',False))
        self.assertEqual(divergence([10,2],[10,11,2]),(2,True))
        self.assertEqual(pair_flags(1,1,1,0)['joint_unique_harm'],1)
        self.assertEqual(pair_flags(1,0,1,0)['joint_unique_harm'],0)
        self.assertEqual(pair_flags(0,0,0,1)['joint_unique_rescue'],1)

    def test_signed_interaction_and_cluster_bootstrap(self):
        scores=np.array([[1,1,1,0],[1,0,1,0],[0,0,0,1]],dtype=float)
        f,a,t,j=scores.T
        values=np.column_stack([j-f,j-t-a+f])
        r=bootstrap(values,['a','a','b'],[1,1],repetitions=1000,seed=20261102)
        self.assertAlmostEqual(r[0]['mean'],-1/3)
        self.assertAlmostEqual(r[1]['mean'],0.)
        self.assertEqual(r[0]['clusters'],2)
        self.assertEqual(r,bootstrap(values,['a','a','b'],[1,1],repetitions=1000,seed=20261102))


if __name__=='__main__':unittest.main()

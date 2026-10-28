"""CPU numerical/scoping/cleanup tests for positive visual steering."""
import unittest
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import repeat_kv
from attention_sink_steering import VisualAttentionReweight, conditional, spatial_similarity, factors_for
from attention_sink_steering_case import make_conditions


class SteeringTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(724)
        config=LlamaConfig(vocab_size=45,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                           num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.)
        config._attn_implementation='eager'
        self.model=LlamaForCausalLM(config).eval()
        self.modules=[l.self_attn for l in self.model.model.layers]
        self.ids=torch.tensor([[1,3,4,5,6,7,8,9]])
        with torch.inference_mode():self.full=self.run_model()

    def run_model(self):
        return self.model(self.ids,use_cache=False,output_attentions=True,output_hidden_states=True)

    def context(self,factors,active=(1,2)):
        return VisualAttentionReweight(self.modules,[1,2,3],5,factors,active)

    def test_noop_exact_all_scopes_and_cleanup(self):
        for layers in [(0,),(1,2),(3,)]:
            with self.subTest(layers=layers),torch.inference_mode(),self.context(np.ones((4,3)),layers) as ctx:
                out=self.run_model();ctx.assert_complete()
                self.assertTrue(torch.equal(out.logits,self.full.logits))
                for a,b in zip(out.attentions,self.full.attentions):self.assertTrue(torch.equal(a,b))
            self.assertTrue(all('forward' not in m.__dict__ for m in self.modules))
        with torch.inference_mode():self.assertTrue(torch.equal(self.run_model().logits,self.full.logits))

    def test_positive_reweight_formula_mass_and_causal_scoping(self):
        factors=np.array([[.5,1,2],[2,1,.1],[1,4,.5],[.1,.5,1]])
        with torch.inference_mode(),self.context(factors) as ctx:
            out=self.run_model();ctx.assert_complete()
        self.assertEqual([r['layer'] for r in ctx.records],[1,2])
        self.assertTrue(torch.equal(out.attentions[0],self.full.attentions[0]))
        self.assertTrue(torch.equal(out.hidden_states[-1][:,:5],self.full.hidden_states[-1][:,:5]))
        self.assertFalse(torch.equal(out.logits[:,5],self.full.logits[:,5]))
        for pair,audit in zip(ctx.rows,ctx.records):
            old,new=pair
            expected=old*factors;expected=expected/expected.sum(-1,keepdims=True)*old.sum(-1,keepdims=True)
            np.testing.assert_allclose(new,expected,atol=1e-7,rtol=1e-5)
            self.assertTrue(audit['passed'] and audit['other_queries_bitwise_exact'] and audit['nonvisual_bitwise_exact'])
        # Literal reweighting occurs before P@V, using the returned modified weights.
        q=torch.randn(1,4,8,8);k=torch.randn(1,2,8,8);v=torch.randn(1,2,8,8)
        ctx=self.context(factors)
        with torch.inference_mode():
            output,weights=ctx.eager(1)(self.modules[1],q,k,v,None,scaling=.35)
        expected=torch.matmul(weights,repeat_kv(v,2)).transpose(1,2).contiguous()
        self.assertTrue(torch.equal(output,expected))

    def test_fp16_subnormal_mass_and_zero_support(self):
        ctx=VisualAttentionReweight(self.modules,range(576),599,np.tile(np.linspace(.1,1,576),(4,1)),[1])
        ctx.query=0
        weights=torch.zeros(1,4,1,600,dtype=torch.float16)
        weights[:,:,:,:576]=.0021/576
        weights[:,:,:,576]=1-weights[:,:,:,:576].sum(-1)
        out=ctx.replace(weights.clone(),1)
        self.assertTrue(ctx.records[0]['passed'])
        self.assertTrue(ctx.records[0]['formula_rounded_target_bitwise_exact'])
        self.assertTrue(ctx.records[0]['conditional_rounding_bound_passed'])
        self.assertTrue(torch.equal(out[:,:,:,576:],weights[:,:,:,576:]))
        weights[:,:,:,:576]=0;weights[:,:,:,576]=1
        ctx.replace(weights,1)
        self.assertTrue(ctx.records[-1]['passed'])
        self.assertEqual(float(weights[:,:,:,:576].sum()),0.)

    def test_failure_cleanup_and_isolation(self):
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with self.context(np.ones((4,3))):raise RuntimeError('deliberate')
        self.assertTrue(all('forward' not in m.__dict__ for m in self.modules))
        other=LlamaForCausalLM(self.model.config).eval();other.load_state_dict(self.model.state_dict())
        with torch.inference_mode(),self.context(np.tile([.1,2,1],(4,1))):
            result=other(self.ids,use_cache=False)
        self.assertTrue(torch.equal(result.logits,self.full.logits))

    def test_invalid_inputs(self):
        for x in [np.zeros((4,3)),np.full((4,3),np.nan),-np.ones((4,3))]:
            with self.assertRaises(ValueError):self.context(x)
        with self.assertRaises(ValueError):conditional([0,0])
        with self.assertRaises(ValueError):factors_for(dict(family='B',patch_indices=[0,0],strength=2),np.ones((4,3)))

    def test_similarity_and_factors_independent_numeric(self):
        b=np.array([[.2,.3,.5],[.1,.7,.2]])
        cosine,js=spatial_similarity(b,b)
        np.testing.assert_allclose(cosine,1);np.testing.assert_allclose(js,0)
        c,j=spatial_similarity([1,0],[0,1]);self.assertEqual(float(c),0);self.assertEqual(float(j),1)
        f=factors_for(dict(family='A',strength=0,epsilon=1e-6),b)
        np.testing.assert_array_equal(f,np.ones_like(b))
        f=factors_for(dict(family='A',strength=1,epsilon=1e-6),b)
        np.testing.assert_allclose(f,1/(b+1e-6))
        f=factors_for(dict(family='B',strength=2,patch_indices=[1]),b)
        np.testing.assert_array_equal(f,[[1,2,1],[1,2,1]])

    def test_frozen_grid_count_and_controls(self):
        from pathlib import Path
        import json
        root=Path(__file__).resolve().parent
        entry=next(s for s in json.loads((root/'results/phase3_4_roi_sink_diagnostic/sink_manifest.json').read_text())['samples'] if s['question_id']=='15133706')
        conditions=make_conditions(entry)
        self.assertEqual(len(conditions),31)
        for r in conditions:
            if r['family']=='A':continue
            expected=len(entry['roi_indices']) if r['family']=='B' else len(entry['sink_nonroi_indices'])
            self.assertEqual(len(r['patch_indices']),expected)
            if r['control_seed'] is not None:
                self.assertFalse(set(r['patch_indices'])&(set(entry['roi_indices'])|set(entry['sink_indices'])))


if __name__=='__main__':unittest.main(verbosity=2)

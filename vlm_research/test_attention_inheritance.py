"""CPU regression for post-softmax single-query visual inheritance."""
import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import eager_attention_forward
from attention_inheritance import VisualAttentionInheritance


class InheritanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        cfg=LlamaConfig(vocab_size=40,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
                        num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.)
        cfg._attn_implementation='eager'
        self.model=LlamaForCausalLM(cfg).eval()
        self.modules=[l.self_attn for l in self.model.model.layers]
        self.ids=torch.tensor([[1,3,4,5,6,7,8,9]])
        with torch.inference_mode():self.full=self.model(self.ids,use_cache=False,output_attentions=True,output_hidden_states=True)
        self.reference=self.full.attentions[0][0,:,5,1:4]

    def ctx(self, strength):
        return VisualAttentionInheritance(self.modules,[1,2,3],5,self.reference,strength,active_layers=[1,2])

    def run_model(self):
        return self.model(self.ids,use_cache=False,output_attentions=True,output_hidden_states=True)

    def test_zero_exact_and_no_leak(self):
        global_eager=eager_attention_forward
        with torch.inference_mode(),self.ctx(0) as ctx:
            out=self.run_model();ctx.assert_complete()
        self.assertTrue(torch.equal(out.logits,self.full.logits))
        for a,b in zip(out.attentions,self.full.attentions):self.assertTrue(torch.equal(a,b))
        self.assertTrue(all(r['lambda_zero_tensor_exact'] for r in ctx.records))
        self.assertTrue(all('forward' not in m.__dict__ for m in self.modules))
        self.assertIs(eager_attention_forward,global_eager)
        with torch.inference_mode():self.assertTrue(torch.equal(self.run_model().logits,self.full.logits))

    def test_blends_support_causality(self):
        for strength in [.5,1.]:
            with self.subTest(strength=strength),torch.inference_mode(),self.ctx(strength) as ctx:
                out=self.run_model();ctx.assert_complete()
                self.assertTrue(all(r['passed'] for r in ctx.records))
                self.assertTrue(torch.equal(out.attentions[0],self.full.attentions[0]))
                self.assertTrue(torch.equal(out.hidden_states[-1][:,:5],self.full.hidden_states[-1][:,:5]))
                for old_new in ctx.rows:
                    old,new=torch.from_numpy(old_new)
                    m=old.sum(-1,keepdim=True)
                    expected=m*((1-strength)*old/m+strength*self.reference/self.reference.sum(-1,keepdim=True))
                    torch.testing.assert_close(new,expected,atol=1e-7,rtol=1e-5)
                    torch.testing.assert_close(new.sum(-1),old.sum(-1))

    def test_exception_cleanup(self):
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with self.ctx(1):raise RuntimeError('deliberate')
        self.assertTrue(all('forward' not in m.__dict__ for m in self.modules))

    def test_instance_isolation(self):
        other=LlamaForCausalLM(self.model.config).eval();other.load_state_dict(self.model.state_dict())
        with torch.inference_mode(),self.ctx(1):
            out=other(self.ids,use_cache=False)
        self.assertTrue(torch.equal(out.logits,self.full.logits))

    def test_subnormal_conditioning_bound_and_actual_matmul(self):
        # Tiny visual mass makes a fixed conditional L1 tolerance inappropriate.
        ctx=VisualAttentionInheritance(self.modules,range(576),599,torch.softmax(torch.randn(4,576),-1),1,[1])
        ctx.query=0
        row=torch.zeros(1,4,1,600,dtype=torch.float16)
        row[:,:,:,:576]=.0021/576
        row[:,:,:,576]=1-row[:,:,:,:576].sum(-1)
        ctx.replace(row,1)
        self.assertTrue(ctx.records[0]['formula_rounded_target_bitwise_exact'])
        self.assertTrue(ctx.records[0]['conditional_rounding_bound_passed'])
        self.assertTrue(ctx.records[0]['conditional_nominal_exceeded'])
        # Independently reconstruct P@V to verify the edit acts before V.
        ctx=self.ctx(1)
        mod=self.modules[1]
        q=torch.randn(1,4,8,8);k=torch.randn(1,2,8,8);v=torch.randn(1,2,8,8)
        from transformers.models.llama.modeling_llama import repeat_kv
        with torch.inference_mode():
            out,weights=ctx.eager(1)(mod,q,k,v,None,scaling=.35)
            expected=torch.matmul(weights,repeat_kv(v,2)).transpose(1,2).contiguous()
        self.assertTrue(torch.equal(out,expected))

    def test_invalid_reference_and_indices(self):
        with self.assertRaises(ValueError):VisualAttentionInheritance(self.modules,[6],5,torch.ones(4,1),1,[1])
        with self.assertRaises(ValueError):VisualAttentionInheritance(self.modules,[1,2,3],5,torch.zeros(4,3),1,[1])

    def test_fp16_rounding_mass(self):
        torch.manual_seed(10)
        row=torch.softmax(torch.randn(1,32,4,600),-1).half()
        ref=torch.softmax(torch.randn(32,576),-1)
        ctx=VisualAttentionInheritance(self.modules,range(576),599,ref,1,[1])
        # Standalone probabilities: query index 599 needs 600 rows; avoid a large CPU matrix.
        ctx.query=3
        out=ctx.replace(row.clone(),1)
        self.assertTrue(ctx.records[0]['passed'])
        self.assertTrue(torch.equal(out[:,:,:3],row[:,:,:3]))

if __name__=='__main__':unittest.main(verbosity=2)

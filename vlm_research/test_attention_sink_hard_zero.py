"""CPU tests of hard-zero semantics, actual P@V, masks and cleanup."""
import json
from pathlib import Path
import unittest
from unittest import mock
import tempfile
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import repeat_kv, eager_attention_forward
from attention_sink_hard_zero import VisualSinkHardZero


class HardZeroTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(921)
        config = LlamaConfig(vocab_size=45, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=4, num_attention_heads=4,
                             num_key_value_heads=2, attention_dropout=0.)
        config._attn_implementation = "eager"
        self.model = LlamaForCausalLM(config).eval()
        self.modules = [layer.self_attn for layer in self.model.model.layers]
        self.ids = torch.tensor([[1,3,4,5,6,7,8,9]])
        with torch.inference_mode():
            self.full = self.forward()

    def forward(self):
        return self.model(self.ids, use_cache=False, output_attentions=True, output_hidden_states=True)

    def context(self, patches=(1,), active=(1,2)):
        return VisualSinkHardZero(self.modules, [1,2,3], 5, patches, active)

    def test_noop_all_logits_and_attentions_exact(self):
        native = eager_attention_forward
        with torch.inference_mode(), self.context((), range(4)) as edit:
            value = self.forward(); edit.assert_complete()
        self.assertTrue(torch.equal(value.logits, self.full.logits))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(value.attentions,self.full.attentions)))
        self.assertTrue(all(a["lambda_zero_tensor_exact"] for a in edit.records))
        self.assertTrue(all("forward" not in m.__dict__ for m in self.modules))
        self.assertIs(native, eager_attention_forward)

    def test_exact_zero_formula_local_mass_and_question_causality(self):
        with torch.inference_mode(), self.context() as edit:
            value = self.forward(); edit.assert_complete()
        self.assertEqual([r["layer"] for r in edit.records],[1,2])
        self.assertTrue(torch.equal(value.attentions[0],self.full.attentions[0]))
        self.assertTrue(torch.equal(value.hidden_states[-1][:,:5],self.full.hidden_states[-1][:,:5]))
        self.assertFalse(torch.equal(value.logits[:,5],self.full.logits[:,5]))
        for layer, pair, audit in zip([1,2], edit.rows, edit.records):
            old,new = pair
            remaining = old.copy(); remaining[:,1] = 0
            expected = remaining / remaining.sum(-1,keepdims=True) * old.sum(-1,keepdims=True)
            np.testing.assert_allclose(new,expected,atol=1e-7,rtol=1e-5)
            np.testing.assert_allclose(new.sum(-1),old.sum(-1),atol=1e-7)
            self.assertTrue(np.all(new[:,1]==0))
            self.assertTrue(audit["passed"] and audit["blocked_weights_exact_zero"])
            self.assertTrue(audit["other_queries_bitwise_exact"] and audit["nonvisual_bitwise_exact"])
            self.assertTrue(torch.all(value.attentions[layer][0,:,5,6:]==0))
        with torch.inference_mode():
            self.assertTrue(torch.equal(self.forward().logits,self.full.logits))

    def test_actual_edit_is_before_value_matmul(self):
        q=torch.randn(1,4,8,8);k=torch.randn(1,2,8,8);v=torch.randn(1,2,8,8)
        edit=self.context()
        with torch.inference_mode():
            out,weights=edit.eager(1)(self.modules[1],q,k,v,None,scaling=.35)
            expected=torch.matmul(weights,repeat_kv(v,2)).transpose(1,2).contiguous()
        self.assertTrue(torch.equal(out,expected))
        self.assertTrue(torch.all(weights[0,:,5,2]==0))

    def test_normal_exit_failure_cleanup_and_instance_isolation(self):
        other=LlamaForCausalLM(self.model.config).eval();other.load_state_dict(self.model.state_dict())
        with torch.inference_mode(), self.context():
            self.assertTrue(torch.equal(other(self.ids,use_cache=False).logits,self.full.logits))
        with self.assertRaisesRegex(RuntimeError,"deliberate"):
            with self.context():raise RuntimeError("deliberate")
        self.assertTrue(all("forward" not in m.__dict__ for m in self.modules))
        # Trigger an exception from inside the replacement, not just the context body.
        edit=self.context()
        edit.replace=lambda weights,layer: (_ for _ in ()).throw(RuntimeError("during forward"))
        with self.assertRaisesRegex(RuntimeError,"during forward"):
            with torch.inference_mode(),edit:self.forward()
        self.assertTrue(all("forward" not in m.__dict__ for m in self.modules))

    def test_fp16_subnormal_rounding_and_zero_mass(self):
        edit=VisualSinkHardZero(self.modules,range(576),599,list(range(13)),[1]);edit.query=0
        row=torch.zeros(1,4,1,600,dtype=torch.float16)
        row[:,:,:,:576]=.0021/576;row[:,:,:,576]=1-row[:,:,:,:576].sum(-1)
        out=edit.replace(row.clone(),1)
        self.assertTrue(edit.records[0]["passed"])
        self.assertTrue(edit.records[0]["formula_rounded_target_bitwise_exact"])
        self.assertTrue(edit.records[0]["conditional_rounding_bound_passed"])
        self.assertTrue(torch.all(out[:,:,:,:13]==0))
        self.assertTrue(torch.equal(out[:,:,:,576:],row[:,:,:,576:]))
        row[:,:,:,:576]=0;row[:,:,:,576]=1
        edit.replace(row,1);self.assertTrue(edit.records[-1]["passed"])

    def test_undefined_remaining_support_fails_before_edit(self):
        edit=self.context();edit.query=0
        row=torch.zeros(1,4,1,8);row[:,:,:,2]=.25;row[:,:,:,0]=.75
        old=row.clone()
        with self.assertRaisesRegex(ValueError,"remaining support"):edit.replace(row,1)
        self.assertTrue(torch.equal(row,old))

    def test_invalid_patches(self):
        for patches in [[0,0],[3],[-1],[True],[0,1,2]]:
            with self.subTest(patches=patches),self.assertRaises(ValueError):self.context(patches)

    def test_frozen_conditions_and_controls(self):
        from sink_hard_zero_case import make_conditions
        root=Path(__file__).resolve().parent
        protocol=json.loads((root/"results/attention_sink_steering_case_15133706/protocol.json").read_text())
        entry=protocol["sink_entry"];rows=make_conditions(entry)
        self.assertEqual(len(rows),144)
        single=[r for r in rows if r["target"]=="sink" and r["kind"]=="single"]
        self.assertEqual([r["active_layers"] for r in single],[[l] for l in range(32)])
        self.assertEqual(len({r["id"] for r in rows}),144)
        for row in rows:
            self.assertEqual(len(row["patch_indices"]),13)
            self.assertEqual(len(set(row["patch_indices"])),13)
            if row["target"]!="sink":
                self.assertFalse(set(row["patch_indices"]) & (set(entry["roi_indices"])|set(entry["sink_indices"])))


    def test_resource_gate_never_loads_seven_b_model(self):
        import sink_hard_zero_case as runner
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            protocol={"conditions":[{"id":"test_only"}],"condition_count":1,"minimum_free_mib":16000}
            runner.write(output/"protocol.json",protocol)
            runner.write(output/"status.json",dict(gpu_completed=False))
            with mock.patch.object(runner,"validate",return_value=protocol), \
                 mock.patch.object(runner,"committed_code",return_value="cpu-test-only"), \
                 mock.patch.object(runner,"cpu_checks"), \
                 mock.patch.object(runner,"resource_preflight",return_value=dict(ready=False,reason="CPU-test mocked busy GPU")), \
                 mock.patch.object(runner,"load_model") as load:
                runner.gpu(output)
                load.assert_not_called()
            state=runner.read(output/"status.json")
            self.assertEqual(state["status"],"gpu_resource_blocked")
            self.assertEqual(state["completed_conditions"],0)

    def test_checkpoint_binding_tamper_and_partial_rejected(self):
        import sink_hard_zero_case as runner
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);directory=run/"conditions"/"x";directory.mkdir(parents=True)
            condition={"id":"x"};data=directory/"data.txt";data.write_text("original")
            body=dict(binding="frozen",condition=condition,row={"condition":"x"},files_sha256={"data.txt":runner.sha(data)})
            runner.write(directory/"complete.json",dict(body=body,sha256=runner.digest(body)))
            protocol=dict(conditions=[condition])
            self.assertEqual(runner.read_checkpoints(run,protocol,"frozen")["x"]["condition"],"x")
            with self.assertRaisesRegex(ValueError,"binding"):
                runner.read_checkpoints(run,protocol,"different")
            data.write_text("modified")
            with self.assertRaisesRegex(ValueError,"artifact"):
                runner.read_checkpoints(run,protocol,"frozen")
            (directory/"complete.json").unlink()
            with self.assertRaisesRegex(ValueError,"Partial"):
                runner.read_checkpoints(run,protocol,"frozen")


if __name__ == "__main__":
    unittest.main(verbosity=2)

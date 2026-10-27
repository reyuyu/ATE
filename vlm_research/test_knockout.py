"""Phase 3 correctness tests on the installed real LLaVA/LLaMA code, CPU only.

Random tiny model results are implementation tests, NEVER LLaVA-7B findings.
"""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from vlm_common import ROOT, default_model
from knockout_probe import (VisualAttentionKnockout, answer_log_distribution,
    audit_attention_output, candidate_inputs, candidate_plans, knockout_mask,
    layout_from_inputs, model_layers, plot_sample, safe_output_dir,
    score_candidates, synchronized_forward, token_metrics, validate_intervention, evaluate_sample,
    CUTOFFS, replot_existing, write_completed_report)
from attention_position_probe import append_answer_inputs, encode_answer, render_prompt
from transformers import AutoProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
import torch
import numpy as np


def tiny_model(layers=32, dtype=torch.float32):
    torch.manual_seed(13)
    text = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=48,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0)
    vision = CLIPVisionConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                             num_attention_heads=4, image_size=8, patch_size=4)
    config = LlavaConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
        image_token_index=99, image_seq_length=4, vision_feature_layer=-2,
        vision_feature_select_strategy="default")
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    return LlavaForConditionalGeneration(config).to(dtype=dtype).eval()


def tiny_inputs():
    return {"input_ids": torch.tensor([[1, 11, 99, 99, 99, 99, 12, 13, 14, 15, 16, 17]]),
            "attention_mask": torch.ones((1, 12), dtype=torch.long),
            "pixel_values": torch.linspace(-1, 1, 3 * 8 * 8).reshape(1, 3, 8, 8)}


class MaskUnitTest(unittest.TestCase):
    def setUp(self):
        self.inputs = tiny_inputs()
        self.layout = layout_from_inputs(self.inputs, 99, 4)
        self.mask = torch.zeros(1, 1, 12, 12).masked_fill(
            torch.ones(12, 12, dtype=torch.bool).triu(1), torch.finfo(torch.float32).min)

    def test_exact_edge_edit_and_shared_mask_unchanged(self):
        before = self.mask.clone()
        changed, count = knockout_mask(self.mask, self.layout)
        self.assertTrue(torch.equal(self.mask, before))
        self.assertNotEqual(changed.data_ptr(), self.mask.data_ptr())
        self.assertEqual(count, 6 * 4)
        expected = torch.zeros((12, 12), dtype=torch.bool)
        expected[6:, 2:6] = True
        self.assertTrue(torch.equal(changed[0, 0][~expected], before[0, 0][~expected]))
        self.assertTrue(torch.isneginf(changed[0, 0][expected]).all())

    def test_locations_are_input_derived(self):
        self.inputs["input_ids"] = torch.tensor([[1, 99, 99, 99, 99, 20, 21, 22, 23, 24, 25, 26]])
        layout = layout_from_inputs(self.inputs, 99, 4)
        self.assertEqual(layout["visual_indices"].tolist(), [1, 2, 3, 4])
        self.assertEqual(torch.where(layout["text_queries"])[0].tolist(), list(range(5, 12)))

    def test_padding_and_all_masked_rows(self):
        self.inputs["attention_mask"][0, -1] = 0
        layout = layout_from_inputs(self.inputs, 99, 4)
        self.mask[:, :, :, -1] = torch.finfo(torch.float32).min
        changed, _ = knockout_mask(self.mask, layout)
        self.assertEqual(changed[0, 0, -2, -1], torch.finfo(torch.float32).min)
        self.mask[0, 0, 7, :] = torch.finfo(torch.float32).min
        self.mask[0, 0, 7, 2:6] = 0
        with self.assertRaisesRegex(ValueError, "no accessible key"):
            knockout_mask(self.mask, layout)

    def test_bad_masks_and_counts_fail_closed(self):
        for mask in (None, torch.ones(1, 12), self.mask.bool()):
            with self.assertRaises(ValueError):
                knockout_mask(mask, self.layout)
        with self.assertRaisesRegex(ValueError, "future"):
            knockout_mask(torch.zeros_like(self.mask), self.layout)
        with self.assertRaises(ValueError):
            layout_from_inputs(self.inputs, 99, 576)

    def test_old_results_protected(self):
        for path in (ROOT, ROOT / "results", ROOT / "results/phase2_attention/new",
                     ROOT / "results/phase2_1_attention", ROOT / "results/phase2_attention"):
            with self.assertRaises(ValueError):
                safe_output_dir(path)


class InstalledLlavaIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = tiny_model()
        cls.inputs = tiny_inputs()

    def test_full_32_layer_validation_on_real_transformers(self):
        parameter_copies = [p.detach().clone() for p in self.model.parameters()]
        input_copies = {k: v.clone() for k, v in self.inputs.items()}
        report = validate_intervention(self.model, self.inputs, expected_visual=4)
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(parameter_copies, self.model.parameters())))
        self.assertTrue(all(torch.equal(input_copies[k], v) for k, v in self.inputs.items()))
        self.assertTrue(report["passed"])
        self.assertEqual(report["layers"], 32)
        noop = next(r for r in report["controls"] if r["name"] == "K32_noop")
        self.assertEqual(noop["max_abs_logit_difference_from_full"], 0)
        for name, start in (("K0", 0), ("K16", 16)):
            result = next(r for r in report["controls"] if r["name"] == name)
            for layer, record in enumerate(result["attention"]):
                self.assertEqual(record["shape"], [1, 4, 12, 12])
                if layer >= start:
                    self.assertEqual(record["blocked_region_max"], 0)
            self.assertEqual([r["applied"] for r in result["hook_audit"]], [i >= start for i in range(32)])

    def test_hooks_removed_when_forward_raises(self):
        before = [len(x.self_attn._forward_pre_hooks) for x in model_layers(self.model)]
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with VisualAttentionKnockout(model_layers(self.model), self.inputs, 99, 0, 4):
                raise RuntimeError("deliberate")
        self.assertEqual(before, [len(x.self_attn._forward_pre_hooks) for x in model_layers(self.model)])

    def test_cache_and_bad_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "cache"):
            with VisualAttentionKnockout(model_layers(self.model), self.inputs, 99, 0, 4):
                with torch.inference_mode():
                    self.model(**self.inputs, use_cache=True)
        self.assertTrue(all(not x.self_attn._forward_pre_hooks for x in model_layers(self.model)))
        with self.assertRaises(ValueError):
            VisualAttentionKnockout(model_layers(self.model), self.inputs, 99, 33, 4)
        config = model_layers(self.model)[0].self_attn.config
        original = config._attn_implementation
        try:
            config._attn_implementation = "sdpa"
            with self.assertRaisesRegex(ValueError, "eager"):
                with VisualAttentionKnockout(model_layers(self.model), self.inputs, 99, 0, 4):
                    pass
        finally:
            config._attn_implementation = original

    def test_half_precision_eager_softmax_zero_and_normalization(self):
        model = tiny_model(layers=2, dtype=torch.float16)
        inputs = tiny_inputs()
        inputs["pixel_values"] = inputs["pixel_values"].half()
        out, _ = synchronized_forward(model, inputs, True, 0, 4)
        records = audit_attention_output(out, layout_from_inputs(inputs, 99, 4), 4, {0, 1})
        self.assertEqual(len(records), 2)
        self.assertIsNone(out.attentions)
        self.assertIsNone(out["attentions"])
        self.assertTrue(all(r["blocked_region_max"] == 0 for r in records))

    def test_inactive_layers_keep_same_shared_mask(self):
        seen = []
        def observe(module, args, kwargs):
            seen.append(kwargs["attention_mask"])
        handles = []
        try:
            with VisualAttentionKnockout(model_layers(self.model), self.inputs, 99, 16, 4) as ko:
                handles = [layer.self_attn.register_forward_pre_hook(observe, with_kwargs=True)
                           for layer in model_layers(self.model)]
                with torch.inference_mode():
                    self.model(**self.inputs, use_cache=False)
            ko.assert_one_forward()
            self.assertTrue(all(seen[i] is seen[0] for i in range(16)))
            self.assertTrue(all(seen[i].data_ptr() != seen[0].data_ptr() for i in range(16, 32)))
            self.assertTrue(torch.isfinite(seen[0]).all())
            self.assertTrue(all(torch.isneginf(seen[i][0, 0, 6:, 2:6]).all() for i in range(16, 32)))
        finally:
            for handle in handles:
                handle.remove()


class MetricsTest(unittest.TestCase):
    def test_full_artifact_pipeline_on_tiny_model(self):
        model = tiny_model()
        prompt = tiny_inputs()
        records = [{"answer_token_ordinal": i, "token_id": tid, "query_position": 11 + i,
                    "target_token_position": 12 + i, "token_text": str(tid),
                    "is_key_token": i == 1} for i, tid in enumerate([18, 19])]
        teacher = append_answer_inputs(prompt, [18, 19])
        metadata = {"question_mapping": [{"token_id": 12}], "prompt_ids": prompt["input_ids"][0].tolist()}
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "samples/tiny_fixture"
            with patch("knockout_probe.prepare_sample", return_value=(prompt, teacher, records, metadata)):
                result = evaluate_sample(model, None, {"id": "tiny_fixture"}, output_dir,
                                         cutoffs=CUTOFFS, validate=True, expected_visual=4)
            self.assertTrue(result["completed"])
            self.assertEqual(result["summaries"][-1]["answer_nll_increase"], 0)
            self.assertEqual(len(result["tokens"]), 16)
            self.assertTrue((output_dir / "correctness_validation.json").is_file())
            self.assertEqual(len(list(output_dir.glob("*.png"))), 3)  # No candidate fixture in this sample.
            write_completed_report(Path(temp), {"samples_completed": ["tiny_fixture"],
                                               "total_wall_seconds_including_load": 0.0})
            self.assertTrue((Path(temp) / "phase3_report.md").is_file())
            replot_existing(Path(temp))

    def test_next_token_alignment_and_kl_direction(self):
        logits = torch.zeros((1, 6, 3))
        logits[0, 2] = torch.tensor([1., 2., 4.])
        logits[0, 3] = torch.tensor([4., 2., 1.])
        records = [{"answer_token_ordinal": 0, "query_position": 2, "token_id": 2},
                   {"answer_token_ordinal": 1, "query_position": 3, "token_id": 0}]
        full = answer_log_distribution(logits, records)
        ko = torch.log(torch.tensor([[.2, .3, .5], [.6, .3, .1]], dtype=torch.float64))
        rows = token_metrics(ko, full, records, 0)
        self.assertAlmostEqual(rows[0]["log_probability"], np.log(.5))
        expected = (full[0].exp() * (full[0] - ko[0])).sum().item()
        reverse = (ko[0].exp() * (ko[0] - full[0])).sum().item()
        self.assertAlmostEqual(rows[0]["kl_full_to_knockout_nats"], expected)
        self.assertNotAlmostEqual(expected, reverse)
        self.assertAlmostEqual(rows[0]["token_nll_increase"], full[0, 2].item() - np.log(.5))

    def test_identical_distributions_zero_kl_and_delta(self):
        full = torch.log_softmax(torch.tensor([[1000., -1000., 0.]], dtype=torch.float64), -1)
        record = {"answer_token_ordinal": 0, "query_position": 2, "token_id": 1}
        row = token_metrics(full, full, [record], 32)[0]
        self.assertEqual(row["kl_full_to_knockout_nats"], 0)
        self.assertEqual(row["token_nll_increase"], 0)

    def test_multitoken_candidate_is_not_first_token_only(self):
        model = tiny_model(layers=2)
        prompt = tiny_inputs()
        plans = []
        for name, ids in [("correct", [18, 19]), ("other", [18, 20, 21])]:
            records = [{"token_id": tid, "query_position": 11 + i} for i, tid in enumerate(ids)]
            plans.append({"candidate": name, "is_target": name == "correct", "records": records,
                          "target_records": records, "candidate_token_ids": ids})
        padded = candidate_inputs(prompt, plans, 0)
        self.assertEqual([p["input_ids"].shape[-1] for p in padded], [15, 15])
        self.assertEqual(padded[0]["attention_mask"][0, -1], 0)
        results, margin = score_candidates(model, prompt, plans, 2, 0, 4)
        with torch.inference_mode():
            out = model(**padded[0], use_cache=False)
        lp = answer_log_distribution(out.logits, plans[0]["target_records"])
        expected = float(lp[0, 18] + lp[1, 19])
        self.assertAlmostEqual(results[0]["sequence_log_probability"], expected)
        self.assertNotAlmostEqual(expected, float(lp[0, 18]))
        self.assertAlmostEqual(margin["log_probability_margin"], expected - results[1]["sequence_log_probability"])

    def test_plot_files_from_explicit_synthetic_fixture(self):
        # Temporary files only. These charts must never enter experimental results.
        with tempfile.TemporaryDirectory() as temp:
            summaries = [{"cutoff": k, "answer_nll_increase": 0.2 if k == 0 else 0} for k in (0, 32)]
            tokens = [{"cutoff": k, "answer_token_ordinal": 0, "token_text": "fixture",
                       "log_probability": -1., "token_nll_increase": .2 if k == 0 else 0.,
                       "is_key_token": True} for k in (0, 32)]
            margins = [{"cutoff": k, "log_probability_margin": .5} for k in (0, 32)]
            plot_sample(Path(temp), summaries, tokens, margins)
            self.assertEqual(len(list(Path(temp).glob("*.png"))), 4)


class RealTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(default_model(), local_files_only=True)

    def test_red_mapping_and_contextual_candidates(self):
        plans = candidate_plans(self.processor.tokenizer, "The stop sign is red.", "red",
                                ["blue", "green", "light blue"], 595)
        target = plans[0]["target_records"]
        self.assertEqual([(r["query_position"], r["target_token_position"]) for r in target], [(598, 599)])
        self.assertGreater(len(plans[-1]["candidate_token_ids"]), 1)
        self.assertTrue(all(p["conditioning_prefix_ids"] == plans[0]["conditioning_prefix_ids"] for p in plans))

    def test_manifest_annotations_and_phase2_1_prefix_exact(self):
        manifest = json.loads((ROOT / "configs/phase3_samples.json").read_text())
        self.assertEqual(len(manifest["samples"]), 9)
        self.assertEqual(len({s["id"] for s in manifest["samples"]}), 9)
        for sample in manifest["samples"]:
            self.assertTrue((ROOT / sample["image"]).is_file())
            prompt = self.processor.tokenizer(render_prompt(self.processor, sample["question"]))["input_ids"]
            image_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
            ids = [v for tid in prompt for v in ([tid] * 576 if tid == image_id else [tid])]
            records = encode_answer(self.processor.tokenizer, sample["answer"], len(ids))
            if sample.get("candidates"):
                candidate_plans(self.processor.tokenizer, sample["answer"], sample["candidate_phrase"], sample["candidates"], len(ids))
            if sample.get("phase2_1_reference"):
                with np.load(ROOT / sample["phase2_1_reference"] / "selected_query_attention.npz") as saved:
                    actual = ids + [r["token_id"] for r in records]
                    self.assertEqual(actual, saved["input_ids"].tolist())
            if sample["id"] not in ("street_color", "street_description"):
                self.assertEqual(sample["annotation_source"], "assistant_visual_review_user_delegated")
                self.assertFalse(sample["human_verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

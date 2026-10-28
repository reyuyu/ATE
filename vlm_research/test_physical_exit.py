"""CPU implementation tests, not 7B research observations."""
import unittest

from test_knockout import tiny_model, tiny_inputs
from physical_exit import (physical_forward, reference_forward, measured_forward,
                           aligned, equivalence, cosine_distances)
from prepare_physical_exit_data import select_rows
from physical_exit_pilot import sequence_score
import torch


class PhysicalExitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = tiny_model()

    def test_local_full_matches_native_and_noop_mask(self):
        inputs = tiny_inputs()
        full = reference_forward(self.model, inputs, expected_visual=4)
        physical = physical_forward(self.model, inputs, 32, 4)
        mask = reference_forward(self.model, inputs, 32, 4)
        for other in [physical, mask]:
            self.assertTrue(torch.equal(full["hidden"], other["hidden"]))
            self.assertTrue(torch.equal(full["logits"], other["logits"]))

    def test_all_boundaries_mask_physical_and_real_shapes(self):
        inputs = tiny_inputs()
        keep = torch.where(inputs["input_ids"][0] != 99)[0]
        for cutoff in [0, 8, 16, 24, 32]:
            with self.subTest(cutoff=cutoff):
                mask = reference_forward(self.model, inputs, cutoff, 4)
                seen, handles = [], []
                for layer in self.model.language_model.model.layers:
                    handles.append(layer.register_forward_pre_hook(
                        lambda module, args, kwargs: seen.append(args[0].shape[1]), with_kwargs=True))
                try:
                    physical = measured_forward(self.model, inputs, "physical", cutoff, 4)
                finally:
                    for h in handles:
                        h.remove()
                self.assertEqual(seen, [12 if layer < cutoff else 8 for layer in range(32)])
                report = equivalence(aligned(mask, keep, "hidden"), aligned(physical, keep, "hidden"),
                                     aligned(mask, keep, "logits"), aligned(physical, keep, "logits"), strict=True)
                self.assertTrue(report["passed"], report)

    def test_original_positions_cache_positions_and_rotary_are_preserved(self):
        inputs = tiny_inputs()
        inputs["position_ids"] = torch.arange(12).unsqueeze(0) + 7
        keep = torch.where(inputs["input_ids"][0] != 99)[0]
        seen = []
        def observe(module, args, kwargs):
            seen.append((kwargs["position_ids"].clone(), kwargs["cache_position"].clone(),
                         tuple(x.clone() for x in kwargs["position_embeddings"])))
        layer = self.model.language_model.model.layers[16]
        handle = layer.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            physical_forward(self.model, inputs, 16, 4)
        finally:
            handle.remove()
        self.assertTrue(torch.equal(seen[0][0], inputs["position_ids"][:, keep]))
        self.assertTrue(torch.equal(seen[0][1], keep))
        decoder = self.model.language_model.model
        expected = decoder.rotary_emb(torch.zeros(1, 12, 32), inputs["position_ids"])
        for actual, full in zip(seen[0][2], expected):
            self.assertTrue(torch.equal(actual, full[:, keep]))

    def test_k0_text_independent_of_image_pixels(self):
        inputs = tiny_inputs()
        changed = dict(inputs, pixel_values=torch.randn_like(inputs["pixel_values"]))
        a = physical_forward(self.model, inputs, 0, 4)
        b = physical_forward(self.model, changed, 0, 4)
        self.assertTrue(torch.equal(a["hidden"], b["hidden"]))
        self.assertTrue(torch.equal(a["logits"], b["logits"]))

    def test_padding_retained_but_compare_only_valid_text(self):
        inputs = tiny_inputs()
        inputs["attention_mask"][0, -2:] = 0
        mask = reference_forward(self.model, inputs, 8, 4)
        physical = physical_forward(self.model, inputs, 8, 4)
        valid_text = torch.where((inputs["input_ids"][0] != 99) & inputs["attention_mask"][0].bool())[0]
        self.assertEqual(physical["original_positions"].tolist(), [0, 1, 6, 7, 8, 9, 10, 11])
        self.assertTrue(equivalence(aligned(mask, valid_text, "hidden"), aligned(physical, valid_text, "hidden"),
                                   aligned(mask, valid_text, "logits"), aligned(physical, valid_text, "logits"), True)["passed"])

    def test_mapping_rejects_deleted_queries_and_preserves_next_token_offset(self):
        physical = physical_forward(self.model, tiny_inputs(), 0, 4)
        with self.assertRaises(ValueError):
            aligned(physical, [2], "hidden")
        self.assertTrue(torch.equal(aligned(physical, [8, 9], "logits"), physical["logits"][:, [4, 5]]))
        self.assertTrue(torch.isfinite(cosine_distances(physical["hidden"], physical["hidden"])).all())

    def test_no_parameter_input_method_or_hook_mutation_and_error_cleanup(self):
        inputs = tiny_inputs()
        saved = {k: v.clone() for k, v in inputs.items()}
        params = [p.clone() for p in self.model.parameters()]
        method = self.model.language_model.model.forward.__func__
        physical_forward(self.model, inputs, 8, 4)
        reference_forward(self.model, inputs, 8, 4)
        self.assertTrue(all(torch.equal(v, inputs[k]) for k, v in saved.items()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(params, self.model.parameters())))
        self.assertIs(method, self.model.language_model.model.forward.__func__)
        def fail(module, args, kwargs):
            raise RuntimeError("injected")
        h = self.model.language_model.model.layers[2].register_forward_pre_hook(fail, with_kwargs=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                reference_forward(self.model, inputs, 8, 4)
        finally:
            h.remove()
        for layer in self.model.language_model.model.layers:
            self.assertEqual(len(layer._forward_pre_hooks), 0)
            self.assertEqual(len(layer.self_attn._forward_pre_hooks), 0)

    def test_reject_cache_non_eager_and_bad_boundary(self):
        inputs = tiny_inputs()
        for bad in [-1, 33, 1.5]:
            with self.assertRaises(ValueError):
                physical_forward(self.model, inputs, bad, 4)
        with self.assertRaises(ValueError):
            physical_forward(self.model, dict(inputs, past_key_values=None), 0, 4)
        config = self.model.language_model.model.layers[0].self_attn.config
        old = config._attn_implementation
        try:
            config._attn_implementation = "sdpa"
            with self.assertRaises(ValueError):
                physical_forward(self.model, inputs, 0, 4)
        finally:
            config._attn_implementation = old

    def test_fp16_cpu_shape_changing_control(self):
        model = tiny_model(dtype=torch.float16)
        inputs = tiny_inputs()
        inputs["pixel_values"] = inputs["pixel_values"].half()
        mask = reference_forward(model, inputs, 8, 4)
        physical = physical_forward(model, inputs, 8, 4)
        keep = torch.where(inputs["input_ids"][0] != 99)[0]
        self.assertTrue(equivalence(aligned(mask, keep, "hidden"), aligned(physical, keep, "hidden"),
                                   aligned(mask, keep, "logits"), aligned(physical, keep, "logits"))["passed"])


class FrozenSelectionTest(unittest.TestCase):
    def test_order_independent_seeded_balanced_unique_images(self):
        rows = [{"question_id": str(i), "img_id": str(i // 2), "answer": "left" if i % 2 else "right",
                 "true option": "left" if i % 2 else "right", "false option": "right" if i % 2 else "left"}
                for i in range(60)]
        selected = select_rows(rows)
        self.assertEqual(selected, select_rows(list(reversed(rows))))
        self.assertEqual(len({r["img_id"] for r in selected}), 12)
        self.assertEqual(sum(r["answer"] == "left" for r in selected), 6)

    def test_multitoken_probability_uses_whole_sequence(self):
        probabilities = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]], dtype=torch.float64)
        actual = sequence_score(probabilities.log(), [0, 2])
        self.assertAlmostEqual(actual, float(torch.tensor(0.56).double().log()), places=7)

    def test_equivalence_failure_cannot_be_reported_as_success(self):
        hidden = torch.tensor([[[1., 2., 3.]]])
        logits = torch.tensor([[[1., 2., 3., 4.]]])
        self.assertFalse(equivalence(hidden, -hidden, logits, logits + 1)["passed"])


if __name__ == "__main__":
    unittest.main()

"""Alignment, next-token shift and causal key partition tests; offline, no GPU."""
import unittest
import csv
import json
from pathlib import Path

from vlm_common import ROOT, default_model
from attention_position_probe import (append_answer_inputs, compare_arrays, encode_answer,
    locate_question_tokens, partition_keys, phrase_token_indices, render_prompt, select_queries)
from attention_probe import summarize_rows
from transformers import AutoProcessor
import numpy as np
import torch


class RealTokenizerAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(default_model(), local_files_only=True)
        cls.image_id = cls.processor.tokenizer.convert_tokens_to_ids("<image>")

    def ids(self, question):
        raw = self.processor.tokenizer(render_prompt(self.processor, question))["input_ids"]
        return [x for tid in raw for x in ([tid] * 576 if tid == self.image_id else [tid])]

    def test_real_question_offsets_and_exact_reconstruction(self):
        for question in ["What color is the stop sign?", "What is in this image?",
                         "Is ASSISTANT: a word on the hyperspectral stoplight?"]:
            ids = self.ids(question)
            records, alignment = locate_question_tokens(self.processor, question, ids, self.image_id)
            self.assertEqual(alignment["decoded_question_tokens"], question)
            self.assertTrue(all(r["token_id"] == ids[r["sequence_index"]] for r in records))
            self.assertTrue(all(r["sequence_index"] > np.flatnonzero(np.array(ids) == self.image_id)[-1] for r in records))
            # Template suffix must not accidentally be included, even when question mentions ASSISTANT.
            self.assertLess(records[-1]["sequence_index"], len(ids) - 1)

    def test_subwords_and_content_span(self):
        question = "What color is the hyperspectral stoplight?"
        records, _ = locate_question_tokens(self.processor, question, self.ids(question), self.image_id)
        positions, matches = phrase_token_indices(question, records, ["hyperspectral"], "question_char_span", "sequence_index")
        self.assertGreater(len(positions), 1)  # A word is not assumed to be one token.
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["char_span"], [question.index("hyperspectral"), question.index("hyperspectral") + len("hyperspectral")])

    def test_wrong_processor_ids_rejected(self):
        question = "What color is the stop sign?"
        ids = self.ids(question)
        ids[-1] += 1
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            locate_question_tokens(self.processor, question, ids, self.image_id)

    def test_red_prediction_shift_and_no_extra_bos(self):
        records = encode_answer(self.processor.tokenizer, "The stop sign is red.", 595)
        red = next(r for r in records if r["decoded_token"] == "red")
        self.assertEqual(red["answer_token_ordinal"], 4)
        self.assertEqual(red["target_token_position"], 599)
        self.assertEqual(red["query_position"], 598)
        self.assertEqual(records[3]["decoded_token"], "is")
        self.assertNotIn(self.processor.tokenizer.bos_token_id, [r["token_id"] for r in records])
        self.assertEqual(records[0]["query_position"], 594)

    def test_modes_and_answer_start_deduplication(self):
        question = "What color is the stop sign?"
        ids = self.ids(question)
        q, _ = locate_question_tokens(self.processor, question, ids, self.image_id)
        a = encode_answer(self.processor.tokenizer, "The stop sign is red.", len(ids))
        content, _ = phrase_token_indices(question, q, ["color", "stop sign"], "question_char_span", "sequence_index")
        for mode in ["all", "question_all", "question_last", "question_content", "answer_start", "answer_prediction"]:
            queries, groups = select_queries(mode, q, content, a, len(ids))
            positions = [r["query_position"] for r in queries]
            self.assertEqual(len(positions), len(set(positions)))
            self.assertEqual(positions.count(len(ids) - 1), 1)
            if mode == "all":
                self.assertEqual(len(positions), len(q) + len(a))
                self.assertEqual(groups["answer_start"][0], groups["answer_prediction"][0])
        with self.assertRaises(ValueError):
            select_queries("question_content", q, [], a, len(ids))


class CausalPartitionTest(unittest.TestCase):
    def test_prefix_and_pixels_preserved(self):
        prompt = {"input_ids": torch.tensor([[1, 99, 99, 4, 7]]),
                  "attention_mask": torch.ones((1, 5), dtype=torch.int64),
                  "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float16)}
        inputs = append_answer_inputs(prompt, [8, 9, 10])
        self.assertEqual(inputs["input_ids"].tolist(), [[1, 99, 99, 4, 7, 8, 9, 10]])
        self.assertEqual(inputs["attention_mask"].shape, (1, 8))
        self.assertIs(inputs["pixel_values"], prompt["pixel_values"])
        self.assertEqual(prompt["input_ids"].shape, (1, 5))

    def test_group_partition_all_query_types_and_future_answer_mask(self):
        ids = np.array([1, 11, 99, 99, 20, 21, 30, 40, 41, 42])
        # P=7; question indices 4,5; prediction a_i uses q=6+i.
        for query, expected_answer_count in [(4, 0), (5, 0), (6, 0), (7, 1), (8, 2)]:
            accessible, groups = partition_keys(ids, np.ones(10), query, 99, 1, 7, [4, 5])
            self.assertTrue(np.array_equal(sum(x.astype(int) for x in groups.values()), accessible.astype(int)))
            self.assertEqual(int(groups["previous_answer"].sum()), expected_answer_count)
            row = np.zeros((1, 10))
            row[:, accessible] = 1 / accessible.sum()
            stat = summarize_rows(row, accessible, groups["visual"], accessible & ~groups["visual"], 0)[0]
            self.assertEqual(stat["uniform_visual_baseline"], 2 / (query + 1))
            self.assertEqual(stat["visual_attention_density_vs_uniform"], 1.)
            # The target at query+1 is still a future key, even though included in teacher input.
            self.assertFalse(accessible[query + 1])
            row[0, query + 1] = .01
            with self.assertRaises(ValueError):
                summarize_rows(row, accessible, groups["visual"], accessible & ~groups["visual"], 0)

    def test_padding_not_counted(self):
        ids = np.array([1, 11, 99, 99, 20, 30, 40])
        a, groups = partition_keys(ids, [0, 1, 1, 1, 1, 1, 1], 4, 99, 1, 6, [4])
        self.assertEqual(int(a.sum()), 4)
        self.assertEqual(int(groups["bos"].sum()), 0)
        self.assertEqual(int(groups["question"].sum()), 1)

    def test_tensor_query_indexing_axis_order(self):
        matrix = torch.arange(1 * 3 * 9 * 9).reshape(1, 3, 9, 9)
        query = torch.tensor([4, 6, 7, 8])
        rows = matrix[0, :, query, :]
        self.assertEqual(rows.shape, (3, 4, 9))
        for i, q in enumerate(query):
            self.assertTrue(torch.equal(rows[:, i], matrix[0, :, q]))

    def test_comparison_rejects_real_change(self):
        result = compare_arrays([.5, .5], [.51, .49], 1e-4, 1e-4)
        self.assertFalse(result["allclose"])


class SavedExperimentTest(unittest.TestCase):
    """Independent checks of real saved rows and their serialized statistics."""
    @classmethod
    def setUpClass(cls):
        cls.directories = [ROOT / "results/phase2_1_attention", ROOT / "results/phase2_1_attention/description"]
        if not all((d / "position_metadata.json").exists() for d in cls.directories):
            raise unittest.SkipTest("Run both GPU experiments before auditing saved artifacts")

    def test_saved_mapping_prefix_and_next_token_targets(self):
        for directory in self.directories:
            m = json.loads((directory / "position_metadata.json").read_text())
            old_dir = Path(m["phase2_reference"])
            if not old_dir.exists():
                old_dir = ROOT / "results/phase2_attention"
                if directory.name == "description":
                    old_dir /= "description"
            old = json.loads((old_dir / "attention_statistics.json").read_text())
            ids = np.asarray(m["input_ids"])
            P = m["prompt_length"]
            self.assertEqual(ids[:P].tolist(), old["input_ids"])
            self.assertEqual(m["attention_mask"][:P], old["attention_mask"])
            self.assertEqual(int((ids == m["image_token_index"]).sum()), 576)
            self.assertEqual(int((ids == m["bos_token_id"]).sum()), 1)
            for i, target in enumerate(m["answer_tokens"]):
                self.assertEqual(target["target_token_position"], P + i)
                self.assertEqual(target["query_position"], P - 1 + i)
                self.assertEqual(ids[P + i], target["token_id"])
            checks = m["validation"]["forward_consistency"]
            self.assertTrue(checks["same_length_masked_prompt_vs_teacher_logits"]["exact_equal"])
            self.assertTrue(checks["same_length_masked_prompt_vs_teacher_prefix_attention"]["exact_equal"])
            self.assertTrue(checks["teacher_vs_phase2_visual"]["allclose"])
            self.assertTrue(checks["teacher_vs_phase2_nonvisual"]["allclose"])

    def test_all_saved_heads_causality_and_partition_masses_match_csv(self):
        for directory in self.directories:
            m = json.loads((directory / "position_metadata.json").read_text())
            bundle = np.load(directory / "selected_query_attention.npz")
            raw = bundle["attention"]
            Q = len(m["queries"])
            self.assertEqual(raw.shape, (32, 32, Q, m["teacher_forced_length"]))
            self.assertLess(Q, m["teacher_forced_length"])  # Selected rows, not full S*S matrices.
            self.assertTrue(np.isfinite(raw).all())
            self.assertTrue((raw >= 0).all())
            ids = np.asarray(m["input_ids"])
            indices = np.arange(len(ids))
            qpositions = [r["sequence_index"] for r in m["question_tokens"]]
            expected = {key: np.empty((32, 32, Q)) for key in ["visual", "bos", "question", "prompt_template", "previous_answer"]}
            accessible_counts = {}
            for query in m["queries"]:
                qi, pos = query["query_axis_index"], query["query_position"]
                r = np.take(raw, qi, axis=2).astype(np.float64)
                accessible = (indices <= pos) & (np.asarray(m["attention_mask"]) != 0)
                accessible_counts[qi] = int(accessible.sum())
                self.assertTrue(np.all(r[..., ~accessible] == 0))
                np.testing.assert_allclose(r[..., accessible].sum(-1), 1., atol=.002, rtol=0)
                masks = {"visual": accessible & (ids == m["image_token_index"]),
                         "bos": accessible & (ids == m["bos_token_id"]),
                         "question": accessible & np.isin(indices, qpositions),
                         "previous_answer": accessible & (indices >= m["prompt_length"])}
                masks["prompt_template"] = accessible & ~np.logical_or.reduce(list(masks.values()))
                self.assertTrue(np.array_equal(sum(v.astype(int) for v in masks.values()), accessible.astype(int)))
                for key, mask in masks.items():
                    expected[key][:, :, qi] = r[..., mask].sum(-1)
            count = 0
            with (directory / "head_statistics.csv").open() as handle:
                for record in csv.DictReader(handle):
                    index = (int(record["layer"]), int(record["head"]), int(record["query_axis_index"]))
                    for key, data in expected.items():
                        self.assertAlmostEqual(float(record[key + "_attention_mass"]), float(data[index]), places=12)
                    count_keys = accessible_counts[index[2]]
                    self.assertEqual(int(record["accessible_keys"]), count_keys)
                    self.assertEqual(float(record["uniform_visual_baseline"]), 576 / count_keys)
                    count += 1
            self.assertEqual(count, 32 * 32 * Q)


if __name__ == "__main__":
    unittest.main()

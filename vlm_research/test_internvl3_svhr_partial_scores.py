"""CPU regression tests for pair selection and partial-score units."""
import unittest

from vlm_research.internvl3_svhr_partial_scores import (
    OUT, TASKS, available_pairs, group_pairs, official_mme, pair_metrics, scoring,
)


class PartialScoreTests(unittest.TestCase):
    def pair(self, task, group):
        return [dict(category=task, group=group, question_id=task + "/" + group + ":" + str(i),
                     image_sha256="same-image") for i in range(2)]

    def test_acc_plus_is_per_image_not_global_accuracy_twice(self):
        pairs = [self.pair("a", "1"), self.pair("a", "2")]
        scores = dict(zip([r["question_id"] for rs in pairs for r in rs], [1, 1, 1, 0]))
        self.assertEqual(pair_metrics(pairs, scores),
                         dict(N=4, pairs=2, acc=75, acc_plus=50, score=125))

    def test_incomplete_common_pair_removed(self):
        rows = self.pair("a", "1") + self.pair("a", "2")
        groups = group_pairs(rows)
        valid = {r["question_id"] for r in rows[:-1]}
        self.assertEqual(list(available_pairs(groups, valid)), [("a", "1")])

    def test_category_part_of_group_identity(self):
        groups = group_pairs(self.pair("a", "1") + self.pair("b", "1"))
        self.assertEqual(len(groups), 2)

    def test_missing_task_is_missing_not_zero(self):
        self.assertIsNone(pair_metrics([], {})["score"])

    def test_duplicate_or_incomplete_group_rejected(self):
        pair = self.pair("a", "1")
        with self.assertRaises(AssertionError):
            group_pairs(pair + pair)
        with self.assertRaises(AssertionError):
            group_pairs(pair[:1])

    def test_wrong_answer_not_excluded(self):
        pair = self.pair("a", "1")
        values = pair_metrics([pair], {r["question_id"]: 0 for r in pair})
        self.assertEqual(values["N"], 2)
        self.assertEqual(values["score"], 0)

    def test_official_ocr_substring_and_case_rules(self):
        scoring.RUN = OUT
        rows = [dict(category="Regular Text Recognition", dataset_name="IIIT5K",
                     gold=["MiXeD"], text="prefix MIXED suffix")]
        self.assertEqual(scoring.official_ocr_batch(rows), [1])
        self.assertEqual(scoring.independent_score("OCRBench", rows[0]["text"], rows[0]), 1)

    def test_official_ocr_math_removes_spaces_but_keeps_case(self):
        scoring.RUN = OUT
        row = dict(category="Handwritten Mathematical Expression Recognition", dataset_name="HME100k",
                   gold=["x _ 2"], text="x_2")
        self.assertEqual(scoring.official_ocr_batch([row]), [1])
        row["text"] = "X_2"
        self.assertEqual(scoring.official_ocr_batch([row]), [0])

    def test_full_official_mme_reduction_keeps_14_tasks_not_component_totals(self):
        scoring.RUN = OUT
        rows, lookup = [], {}
        for task in TASKS:
            pair = self.pair(task, "fixture")
            for i, r in enumerate(pair):
                r.update(image_id="fixture.jpg", question="Question?", gold=("yes", "no")[i])
                lookup["MME", r["question_id"], "fixture"] = {"text": r["gold"]}
            rows.extend(pair)
        result = official_mme({"fixture": group_pairs(rows)}, lookup)
        self.assertEqual(len(result["methods"]["fixture"]["task_scores"]), 14)
        self.assertEqual(result["methods"]["fixture"]["total"], 2800)


if __name__ == "__main__":
    unittest.main()

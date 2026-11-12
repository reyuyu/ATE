"""Regression tests for fail-closed author protocol handling, without CUDA."""

import csv
import io
import json
import sys
import unittest
from .audit import REQUIRED, ROOT, OUT, command, eligibility, scienceqa_gold_letter


class ProtocolTests(unittest.TestCase):
    def test_output_is_under_vlm_research(self):
        self.assertEqual(ROOT.name, "vlm_research")
        self.assertEqual(OUT.parent.name, "results")
        self.assertTrue((ROOT / "qwen3vl_c3v2_physical/engine.py").is_file())

    def test_non_utf8_resource_output_is_preserved(self):
        result = command([sys.executable, "-c", "import os; os.write(1, bytes([160]))"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "\\xa0")
        self.assertEqual(len(result["stdout_raw_sha256"]), 64)

    def test_empty_recipe_is_not_official(self):
        self.assertEqual(eligibility({}, False)["status"], "BLOCKED_AUTHOR_PROTOCOL")

    def test_dataset_and_scorer_are_insufficient(self):
        self.assertEqual(eligibility({"scorer": "official", "dataset": "official"}, True)["status"], "BLOCKED_AUTHOR_PROTOCOL")

    def test_global_sampling_does_not_pin_task(self):
        self.assertEqual(eligibility({"sampling": {"seed": 3407}}, True)["status"], "BLOCKED_AUTHOR_PROTOCOL")

    def test_framework_candidate_is_not_author_attestation(self):
        recipe = dict.fromkeys(REQUIRED, "candidate")
        self.assertEqual(eligibility(recipe, True)["status"], "BLOCKED_AUTHOR_PROTOCOL")

    def test_every_required_field_is_mandatory(self):
        recipe = dict.fromkeys(REQUIRED, "verified")
        recipe["author_task_attestation"] = True
        for key in REQUIRED:
            with self.subTest(key=key):
                altered = {k: v for k, v in recipe.items() if k != key}
                self.assertEqual(eligibility(altered, True)["status"], "BLOCKED_AUTHOR_PROTOCOL")

    def test_hf_port_cannot_unlock_vllm(self):
        recipe = dict.fromkeys(REQUIRED, "verified")
        recipe["author_task_attestation"] = True
        self.assertEqual(eligibility(recipe, False)["status"], "BLOCKED_SVH_OFFICIAL_RUNTIME_PORT")

    def test_complete_recipe_only_unlocks_preflight_not_complete(self):
        recipe = dict.fromkeys(REQUIRED, "verified")
        recipe["author_task_attestation"] = True
        self.assertEqual(eligibility(recipe, True)["status"], "ELIGIBLE_FOR_GPU_PREFLIGHT")

    def test_scienceqa_zero_based_gold(self):
        self.assertEqual(scienceqa_gold_letter(0, ["x", "y"]), "A")
        self.assertEqual(scienceqa_gold_letter(1, ["x", "y"]), "B")

    def test_scienceqa_out_of_bounds_rejected(self):
        for value in [-1, 2, "A", True]:
            with self.assertRaises(ValueError):
                scienceqa_gold_letter(value, ["x", "y"])

    def test_missing_scores_are_empty_not_zero(self):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, ["score"])
        writer.writeheader()
        writer.writerow({"score": None})
        self.assertEqual(list(csv.DictReader(io.StringIO(stream.getvalue())))[0]["score"], "")
        self.assertIsNone(json.loads('{"score": null}')["score"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

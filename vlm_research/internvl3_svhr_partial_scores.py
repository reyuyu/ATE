"""Read-only CPU rescore of the frozen seven-benchmark predictions."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from importlib.metadata import version
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from vlm_research.internvl3_svhr_late9_sevenbench.common import (
    BENCHES, HISTORY, METHODS, OUT, ROOT, bind, csvout, manifest, read, sha,
    source_hashes, write,
)
from vlm_research.internvl3_svh_seven_benchmarks import scoring

BASE = "65c4cac2f925ebefb073c8db03caab119fcc765a"
STATE = "PARTIAL_5_OF_7_FULL_BENCHMARKS"
TASKS = ["existence", "count", "position", "color", "posters", "celebrity",
         "scene", "landmark", "artwork", "OCR", "commonsense_reasoning",
         "numerical_calculation", "text_translation", "code_reasoning"]
PRODUCTS = ["mme_common_subset_14tasks.csv", "mme_common_subset_totals.json",
            "mme_available_per_method.csv", "mme_missing_pair_audit.csv",
            "ocrbench_partial_per_question.csv", "ocrbench_partial_scores.json",
            "mme_official_scorer_output.json", "seven_benchmark_scores.csv",
            "score_recalculation_discrepancies.json"]
SCRIPT = Path(__file__).resolve()


def original(path):
    return subprocess.check_output(["git", "show", BASE + ":" + str(path)], cwd=ROOT)


def base_ledger():
    return json.loads(original(OUT.relative_to(ROOT) / "artifact_hashes.json"))


def verify_old_artifacts(allow_report_update=False):
    ledger = base_ledger()
    changed = []
    for name, expected in ledger.items():
        if sha(OUT / name) != expected:
            changed.append(name)
    assert not (set(changed) - ({"report_zh.md"} if allow_report_update else set())), changed
    return {"base_commit": BASE, "checked": len(ledger), "changed": changed}


def group_pairs(rows):
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        assert r["question_id"] not in seen
        seen.add(r["question_id"])
        groups[r["category"], r["group"]].append(r)
    for pair in groups.values():
        assert len(pair) == 2
        assert len({r["image_sha256"] for r in pair}) == 1
    return groups


def available_pairs(groups, valid_ids):
    return {k: rs for k, rs in groups.items()
            if all(r["question_id"] in valid_ids for r in rs)}


def pair_metrics(pairs, scores):
    if not pairs:
        return {"N": 0, "pairs": 0, "acc": None, "acc_plus": None, "score": None}
    values = [[scores[r["question_id"]] for r in rs] for rs in pairs]
    assert all(v in (0, 1) for pair in values for v in pair)
    acc = Fraction(100 * sum(sum(v) for v in values), 2 * len(values))
    plus = Fraction(100 * sum(all(v) for v in values), len(values))
    return {"N": 2 * len(values), "pairs": len(values), "acc": float(acc),
            "acc_plus": float(plus), "score": float(acc + plus)}


def load_predictions():
    scoring.RUN = OUT
    frozen = {(b, r["question_id"]): r for b in BENCHES for r in manifest(b)}
    processor = read(OUT / "processor_bindings.json")
    lookup, all_scores, counts, checksums, mismatches = {}, {}, Counter(), {}, []
    paths = [OUT / f"server_only/worker{i}.sqlite" for i in range(2)]
    before = {str(p.relative_to(OUT)): sha(p) for p in paths}
    protocol_sha = sha(OUT / "protocol.json")
    source_sha = sha(OUT / "source_manifest.json")
    for path in paths:
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        db.execute("PRAGMA query_only=ON")
        for b, q, m, checksum, raw in db.execute(
                "SELECT benchmark,qid,method,checksum,record FROM predictions"):
            key = b, q, m
            assert key not in all_scores and m in METHODS
            assert hashlib.sha256(raw.encode()).hexdigest() == checksum
            r, f = json.loads(raw), frozen[b, q]
            assert (r["benchmark"], r["question_id"], r["method"]) == key
            for field in ("sample_sha256", "prompt_sha256", "gold", "image_sha256"):
                assert r[field] == f[field], (key, field)
            assert r["protocol_sha256"] == protocol_sha
            assert r["source_manifest_sha256"] == source_sha
            assert r["input_tokens"] == len(f["input_ids"])
            assert r["max_new_tokens"] == f["max_new_tokens"]
            details = r["input_binding_details"]
            assert bind(details) == r["input_binding"]
            assert details["image_sha256"] == f["image_sha256"]
            assert details["prompt_sha256"] == f["prompt_sha256"]
            assert details["pixels_bf16_sha256"] == processor[f["image_sha256"]]["pixels_bf16_sha256"]
            assert isinstance(r["text"], str) and r["token_ids"]
            assert r["generated_tokens"] == len(r["token_ids"]) <= r["max_new_tokens"]
            assert r["eos"] == (r["token_ids"][-1] == 151645)
            parsed, value = scoring.score(b, r["text"], f)
            independent = scoring.independent_score(b, r["text"], f)
            assert math.isfinite(value) and abs(value - independent) < 1e-12
            if value != r["score"] or parsed != r["parsed_answer"]:
                mismatches.append({"benchmark": b, "qid": q, "method": m,
                                   "old_score": r["score"], "new_score": value,
                                   "old_parsed": r["parsed_answer"], "new_parsed": parsed})
            all_scores[key] = value
            checksums["/".join(key)] = checksum
            counts[b, m] += 1
            if b in ("MME", "OCRBench"):
                r["rescored"] = value
                lookup[key] = r
        db.close()
    after = {str(p.relative_to(OUT)): sha(p) for p in paths}
    assert before == after, "Prediction databases changed during read-only rescore"
    assert sum(counts.values()) == 106041
    return frozen, lookup, all_scores, counts, mismatches, before, bind(checksums)


def official_mme(pairs_by_method, lookup):
    path = OUT / "references/InternVL/internvl_chat/eval/mme/calculation.py"
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
    scope = scoring.ast_scope(path, {"calculate_metrics", "eval_type_dict"}, {
        "accuracy_score": accuracy_score, "confusion_matrix": confusion_matrix,
        "precision_score": precision_score, "recall_score": recall_score,
    })
    cal = scope["calculate_metrics"]()
    results = {}
    for method, groups in pairs_by_method.items():
        if any(not any(k[0] == task for k in groups) for task in TASKS):
            results[method] = {"status": "NOT_COMPUTABLE_MISSING_TASK"}
            continue
        with tempfile.TemporaryDirectory(prefix="mme_cpu_", dir=OUT / "server_only") as tmp:
            for task in TASKS:
                lines = []
                for (category, _), pair in sorted(groups.items()):
                    if category != task:
                        continue
                    for r in sorted(pair, key=lambda x: x["question_id"]):
                        text = lookup["MME", r["question_id"], method]["text"]
                        assert not any(c in text for c in ("\n", "\t", "\r"))
                        lines.append("\t".join([r["image_id"], r["question"], r["gold"], text]) + "\n")
                Path(tmp, task + ".txt").write_text("".join(lines))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                cal.process_result(tmp)
            text = stream.getvalue()
            task_scores = {}
            for line in text.splitlines():
                if " score:" in line:
                    task, score = line.strip().split(" score:")
                    if task.strip() in TASKS:
                        task_scores[task.strip()] = float(score)
            assert set(task_scores) == set(TASKS)
            results[method] = {"stdout": text, "task_scores": task_scores,
                               "total": sum(task_scores.values()), "status": "PASS"}
    return {"official_file_sha256": sha(path), "methods": results}


def score_mme(frozen, lookup, counts):
    groups = group_pairs([r for (b, _), r in frozen.items() if b == "MME"])
    ids = {m: {q for b, q, mm in lookup if b == "MME" and mm == m} for m in METHODS}
    common_ids = set.intersection(*ids.values())
    common = available_pairs(groups, common_ids)
    by_method = {m: available_pairs(groups, ids[m]) for m in METHODS}
    scores = {m: {q: int(r["rescored"]) for (b, q, mm), r in lookup.items()
                  if b == "MME" and mm == m} for m in METHODS}
    official = official_mme({m: common for m in METHODS}, lookup)
    tasks, available = [], []
    for task in TASKS:
        pairs = [rs for (category, _), rs in common.items() if category == task]
        row = {"task": task, "planned_questions": sum(2 for c, _ in groups if c == task),
               "common_valid_questions": 2 * len(pairs), "common_valid_pairs": len(pairs)}
        for method in METHODS:
            mm = pair_metrics(pairs, scores[method])
            row.update({method + "_" + k: v for k, v in mm.items()})
            if mm["score"] is not None:
                assert abs(mm["score"] - official["methods"][method]["task_scores"][task]) < 1e-10
        tasks.append(row)
    missing = []
    for (task, group), pair in sorted(groups.items()):
        if (task, group) in common:
            continue
        for r in pair:
            row = {k: r[k] for k in ("question_id", "category", "group", "image_sha256", "image_bytes")}
            for method in METHODS:
                row[method + "_present"] = r["question_id"] in ids[method]
            blocked = OUT / "blocked" / ("MME__" + r["question_id"].replace("/", "_") + ".json")
            assert blocked.exists(), r["question_id"]
            row["blocked_file"] = str(blocked.relative_to(OUT))
            row["blocked_sha256"] = sha(blocked)
            missing.append(row)
    totals = {"status": "PARTIAL", "view": "MME_COMMON_VALID_PAIRS", "units": "MME_points",
              "planned_N": 2374, "common_N": 2 * len(common), "common_pairs": len(common),
              "common_question_ids_sha256": bind(sorted(q for rs in common.values() for q in [r["question_id"] for r in rs])),
              "methods": {}, "differences": {}, "missing_questions": missing,
              "raw_records_N": {m: counts["MME", m] for m in METHODS},
              "all_methods_identical_question_sets": all(ids[m] == common_ids for m in METHODS)}
    totals["record_status"] = {}
    for method in METHODS:
        rs = [r for (b, _, m), r in lookup.items() if b == "MME" and m == method]
        totals["record_status"][method] = {
            "scorable_records": len(rs), "missing_records": 2374-len(rs),
            "eos": sum(r["eos"] for r in rs), "truncated": sum(r["truncated"] for r in rs),
            "empty": sum(r["empty"] for r in rs),
            "flagged_records": sum(bool(r["flags"]["flags"]) for r in rs),
            "flags": dict(Counter(f for r in rs for f in r["flags"]["flags"])),
            "non_pair_records_excluded": len(rs)-2*len(by_method[method]),
        }
    for method in METHODS:
        vals = [r[method + "_score"] for r in tasks]
        total = sum(vals) if all(x is not None for x in vals) else None
        if total is not None:
            assert abs(total - official["methods"][method]["total"]) < 1e-10
        totals["methods"][method] = total
        own = []
        for task in TASKS:
            mm = pair_metrics([rs for (c, _), rs in by_method[method].items() if c == task], scores[method])
            own.append(mm["score"])
            available.append(dict(view="MME_AVAILABLE_PER_METHOD", method=method, task=task, **mm))
        available.append(dict(view="MME_AVAILABLE_PER_METHOD", method=method, task="TOTAL_14_TASKS",
                              N=2*len(by_method[method]), pairs=len(by_method[method]),
                              score=sum(own) if all(v is not None for v in own) else None,
                              question_ids_sha256=bind(sorted(r["question_id"] for rs in by_method[method].values() for r in rs))))
    for ref in ("native", "physical_d"):
        a, b = totals["methods"]["late9"], totals["methods"][ref]
        totals["differences"]["late9_minus_" + ref] = a - b if a is not None and b is not None else None
    csvout(OUT / "mme_common_subset_14tasks.csv", tasks)
    csvout(OUT / "mme_available_per_method.csv", available)
    csvout(OUT / "mme_missing_pair_audit.csv", missing)
    write(OUT / "mme_official_scorer_output.json", official)
    write(OUT / "mme_common_subset_totals.json", totals)
    return totals


def score_ocr(frozen, lookup):
    ids = {m: {q for b, q, mm in lookup if b == "OCRBench" and mm == m} for m in METHODS}
    common = sorted(set.intersection(*ids.values()))
    rows, totals = [], {}
    for method in METHODS:
        batch = [dict(frozen["OCRBench", q], text=lookup["OCRBench", q, method]["text"]) for q in common]
        official = scoring.official_ocr_batch(batch)
        assert official == [lookup["OCRBench", q, method]["rescored"] for q in common]
        totals[method] = sum(official)
    for q in common:
        f = frozen["OCRBench", q]
        row = {k: f[k] for k in ("question_id", "category", "dataset_name", "image_sha256", "gold")}
        for method in METHODS:
            r = lookup["OCRBench", q, method]
            for field in ("text", "token_ids", "score", "eos", "truncated", "empty", "flags",
                          "input_binding", "sample_sha256", "prompt_sha256", "generation_origin"):
                row[method + "_" + field] = r[field]
            row[method + "_official_score"] = int(r["rescored"])
            row[method + "_failure"] = False
        rows.append(row)
    result = {"status": "PARTIAL", "common_N": len(common), "planned_N": 1000,
              "units": "raw_correct_on_observed_questions_NOT_full_0_1000_score",
              "methods": totals, "per_method_N": {m: len(v) for m, v in ids.items()},
              "all_methods_identical_question_sets": all(v == set(common) for v in ids.values()),
              "question_ids": common, "question_ids_sha256": bind(common),
              "official_scorer_sha256": sha(OUT / "references/OCRBench_official_LLaVA1_5.py"),
              "extrapolated": False}
    csvout(OUT / "ocrbench_partial_per_question.csv", rows)
    write(OUT / "ocrbench_partial_scores.json", result)
    return result


def seven_table(frozen, scores, mme, ocr):
    with (OUT / "accuracy_summary.csv").open(newline="") as f:
        old = {r["benchmark"]: r for r in csv.DictReader(f)}
    rows = []
    for bench in BENCHES:
        row = {"benchmark": bench, "planned": int(old[bench]["planned"]),
               "common_N": int(old[bench]["common_N"]), "status": old[bench]["status"]}
        if bench == "MME":
            row.update(mme["methods"], units="MME_points_COMMON_VALID_PAIRS", complete_pairs=mme["common_pairs"])
        elif bench == "OCRBench":
            row.update(ocr["methods"], units="raw_correct_out_of_" + str(ocr["common_N"]))
        else:
            rs = [r for (b, _), r in frozen.items() if b == bench]
            for method in METHODS:
                if bench == "MMBench":
                    groups = defaultdict(list)
                    for r in rs:
                        groups[r["group"]].append(scores[bench, r["question_id"], method])
                    value = 100 * sum(all(v) for v in groups.values()) / len(groups)
                    row["circular_groups"] = len(groups)
                else:
                    value = 100 * sum(scores[bench, r["question_id"], method] for r in rs) / len(rs)
                assert abs(value - float(old[bench][method + "_full_score"])) < 1e-8
                row[method] = float(old[bench][method + "_full_score"])
            row["units"] = "percent"
        row["late9_minus_physical_d"] = row["late9"] - row["physical_d"]
        rows.append(row)
    csvout(OUT / "seven_benchmark_scores.csv", rows)


def rescore():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only execution required"
    assert source_hashes() == read(OUT / "source_manifest.json")["runtime_source_sha256"]
    old = verify_old_artifacts(allow_report_update=True)
    frozen, lookup, scores, counts, mismatches, db_hashes, prediction_hash = load_predictions()
    write(OUT / "score_recalculation_discrepancies.json", mismatches)
    # This is the existing reporter's score tolerance, not a new model-logits tolerance.
    substantive = [r for r in mismatches if abs(r["new_score"]-r["old_score"]) >= 1e-10
                   or r["new_parsed"] != r["old_parsed"]]
    mme, ocr = score_mme(frozen, lookup, counts), score_ocr(frozen, lookup)
    seven_table(frozen, scores, mme, ocr)
    validation = {"status": "PASS" if not substantive else "SCORE_DISCREPANCIES_RECORDED",
                  "experiment_status": STATE, "base_commit": BASE,
                  "CPU_only": True, "model_inference_runs": 0, "GPU_allocation": False,
                  "prediction_count": sum(counts.values()), "prediction_checksum_map_sha256": prediction_hash,
                  "sqlite_sha256_before_and_after": db_hashes, "old_artifacts": old,
                  "score_exact_float_difference_count": len(mismatches),
                  "score_exact_float_difference_max": max([abs(r["new_score"]-r["old_score"]) for r in mismatches], default=0),
                  "score_comparison_tolerance_inherited_from_frozen_reporter": 1e-10,
                  "substantive_score_discrepancies": substantive, "runtime_source_unchanged": True,
                  "numerical_status": "EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER",
                  "historical_NUMERICAL_EQUIVALENCE_FAILED_not_overridden": True,
                  "mme_common_N": mme["common_N"], "mme_common_pairs": mme["common_pairs"],
                  "mme_all_14_tasks_scored": all(x is not None for x in mme["methods"].values()),
                  "ocr_common_N": ocr["common_N"],
                  "empty_truncated_or_wrong_answers_not_silently_excluded": True,
                  "CPU_environment": {"python": sys.version, "executable": sys.executable,
                                      "numpy": version("numpy"), "scikit-learn": version("scikit-learn"),
                                      "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]},
                  "source_sha256": sha(SCRIPT),
                  "test_source_sha256": sha(ROOT / "vlm_research/test_internvl3_svhr_partial_scores.py"),
                  "product_sha256": {name: sha(OUT / name) for name in PRODUCTS}}
    write(OUT / "score_recalculation_validation.json", validation)
    print(json.dumps({"validation": validation["status"], "MME": mme["methods"],
                      "pairs": mme["common_pairs"], "OCR": ocr["methods"], "N": ocr["common_N"]}))


def seal():
    validation = read(OUT / "score_recalculation_validation.json")
    assert validation["status"] == "PASS" and validation["source_sha256"] == sha(SCRIPT)
    assert validation["test_source_sha256"] == sha(ROOT / "vlm_research/test_internvl3_svhr_partial_scores.py")
    verify_old_artifacts(allow_report_update=True)
    for name, h in validation["product_sha256"].items():
        assert sha(OUT / name) == h
    for name, h in validation["sqlite_sha256_before_and_after"].items():
        assert sha(OUT / name) == h
    old = base_ledger()
    write(OUT / "pre_recalculation_artifact_hashes.json", old)
    names = set(old) | set(PRODUCTS) | {"score_recalculation_validation.json", "pre_recalculation_artifact_hashes.json"}
    for name in ("score_recalculation_cpu_tests.json", "score_recalculation_execution.md"):
        if (OUT / name).exists():
            names.add(name)
    new = {name: sha(OUT / name) for name in sorted(names)}
    write(OUT / "artifact_hashes.json", new)
    assert all(sha(OUT / name) == h for name, h in read(OUT / "artifact_hashes.json").items())
    print(json.dumps({"sealed_artifacts": len(new), "sha256": sha(OUT / "artifact_hashes.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    seal() if args.seal else rescore()

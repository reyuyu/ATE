#!/usr/bin/env python3
"""CPU-only portable audit verification. No model, Torch import or generation."""
from pathlib import Path
import argparse, ast, collections, copy, csv, gzip, hashlib, json

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_csv(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output-dir", type=Path, required=True, help="Separate new directory; archived evidence is never overwritten")
    ap.add_argument("--image-root", type=Path, help="Original OCRBench_Images directory, optional fresh 930-image SHA pass")
    ap.add_argument("--annotation", type=Path, help="Optional original OCRBench.json")
    a = ap.parse_args()
    root = a.audit_dir.resolve()
    out = a.output_dir.resolve()
    if out == root or root in out.parents or out.exists():
        raise SystemExit("Use a new output directory outside the archived audit")
    # Verify the publication and the unchanged original audit before executing pinned scoring AST.
    checks = {}
    for listing in ["PUBLICATION_SHA256SUMS", "SHA256SUMS"]:
        errors = []
        lines = (root / listing).read_text(encoding="utf-8").splitlines()
        for line in lines:
            expected, rel = line.split("  ", 1)
            p = (root / rel).resolve()
            if root not in p.parents or not p.is_file() or sha(p) != expected:
                errors.append(rel)
        assert not errors, (listing, errors)
        checks[listing] = {"status": "PASS", "files": len(lines)}
    rawpath = root / "official/Yuliang-Liu__MultimodalOCR/OCRBench/OCRBench/OCRBench.json"
    raw = json.loads(rawpath.read_text(encoding="utf-8"))
    expected_raw_sha = "df75957e75ea052e9f38f2c67a4f6676103998517345cf49ef7d03b62c835864"
    assert len(raw) == 1000 and sha(rawpath) == expected_raw_sha
    with gzip.open(root / "historical/native_stopv2/manifests/OCRBench.jsonl.gz", "rt", encoding="utf-8") as f:
        frozen = [json.loads(line) for line in f]
    assert len(frozen) == 1000
    qrows = read_csv(root / "question_image_sha256.csv")
    urows = read_csv(root / "unique_images_sha256.csv")
    qmap = {r["question_id"]: r for r in qrows}
    umap = {r["image_id"]: r for r in urows}
    assert len(qmap) == 1000 and len(umap) == 930
    for i, (r, x) in enumerate(zip(frozen, raw)):
        assert r["question_id"] == f"{i:04d}"
        for k, value in {
            "official_id": x["id"], "image_id": x["image_path"], "question": x["question"],
            "gold": x["answers"], "category": x["type"], "dataset_name": x["dataset_name"]
        }.items():
            assert r[k] == value, (r["question_id"], k)
        assert qmap[r["question_id"]]["sha256"] == r["image_sha256"]
        assert qmap[r["question_id"]]["image_id"] == r["image_id"]
        assert umap[r["image_id"]]["sha256"] == r["image_sha256"]
    checks["annotation_and_sha_lists"] = {"status": "PASS", "questions": 1000, "unique_images": 930}
    # Execute the exact pinned original official scoring loop rather than reimplement normalization.
    tree = ast.parse((root / "official/Yuliang-Liu__MultimodalOCR/OCRBench/example.py").read_text(encoding="utf-8"))
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and "data_type" in ast.unparse(n) and "data[i]['result']" in ast.unparse(n)
                and "dataset_name" in ast.unparse(n))
    scoring_code = compile(ast.Module(body=[loop], type_ignores=[]), "<pinned official OCRBench loop>", "exec")
    late_wide = read_csv(root / "historical/late9/ocrbench_partial_per_question.csv")
    late_long = [dict(question_id=r["question_id"], method=m, text=r[m + "_text"], score=r[m + "_score"])
                 for r in late_wide for m in ["native", "physical_d", "late9"]]
    groups = [
        ("native_stop_v2", read_csv(root / "historical/native_stopv2/OCRBench/per_question.csv.gz")),
        ("late9_partial", late_long)
    ]
    summaries = []
    for name, records in groups:
        for method in sorted({r["method"] for r in records}):
            rows = [r for r in records if r["method"] == method]
            assert len({r["question_id"] for r in rows}) == len(rows)
            data = []
            for r in rows:
                x = copy.deepcopy(raw[int(r["question_id"])])
                x["predict"] = r["text"]
                data.append(x)
            exec(scoring_code, {"data": data})
            scores = [int(x.get("result", 0)) for x in data]
            mismatches = [r["question_id"] for r, sc in zip(rows, scores) if sc != float(r["score"])]
            assert not mismatches, (name, method, mismatches)
            categories = collections.defaultdict(lambda: [0, 0])
            for r, sc in zip(rows, scores):
                c = raw[int(r["question_id"])]["type"]
                categories[c][0] += sc
                categories[c][1] += 1
            summaries.append({"source": name, "method": method, "N": len(rows),
                              "official_score": sum(scores), "stored_score": sum(float(r["score"]) for r in rows),
                              "mismatch_ids": mismatches, "categories": dict(categories)})
    expected = {
        ("native_stop_v2", "native"): (1000, 751),
        ("native_stop_v2", "physical_d"): (1000, 32),
        ("late9_partial", "native"): (3, 2),
        ("late9_partial", "physical_d"): (3, 0),
        ("late9_partial", "late9"): (3, 2)
    }
    assert {(r["source"], r["method"]): (r["N"], r["official_score"]) for r in summaries} == expected
    blocked = [r for r in read_csv(root / "historical/late9/blocked_requests.csv") if r["benchmark"] == "OCRBench"]
    done = {r["question_id"] for r in groups[1][1]}
    loc = json.loads((root / "historical/late9/local_image_map.json").read_text(encoding="utf-8"))
    unavailable = {r["question_id"] for r in frozen if r["image_sha256"] not in loc}
    assert len(blocked) == 997 and done == {"0053", "0963", "0988"}
    assert unavailable == {r["question_id"] for r in blocked}
    assert not unavailable & done and unavailable | done == set(qmap)
    checks["late9_blocking_partition"] = {"status": "PASS", "completed": 3, "blocked": 997}
    if a.annotation:
        assert sha(a.annotation) == expected_raw_sha
        assert json.loads(a.annotation.read_text(encoding="utf-8")) == raw
        checks["fresh_annotation"] = {"status": "PASS"}
    else:
        checks["fresh_annotation"] = {"status": "NOT_RUN", "reason": "--annotation not supplied"}
    if a.image_root:
        image_root = a.image_root.resolve()
        for r in urows:
            p = (image_root / r["image_id"]).resolve()
            assert image_root in p.parents and p.is_file(), r["image_id"]
            assert p.stat().st_size == int(r["bytes"]) and sha(p) == r["sha256"], r["image_id"]
        checks["fresh_original_images"] = {"status": "PASS", "unique_images_hashed": 930, "questions_covered": 1000}
    else:
        checks["fresh_original_images"] = {"status": "NOT_RUN", "reason": "--image-root not supplied; archived SHA evidence only"}
    out.mkdir(parents=True)
    result = {"status": "PASS", "checks": checks, "scoring": summaries, "new_generations": 0,
              "model_loaded": False, "gpu_initialized": False,
              "scope": "Archive integrity, exact official CPU rescoring, annotation/SHA mappings; no fresh processor/tokenizer or inference smoke"}
    (out / "reproduction_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "scoring": summaries, "checks": checks}, ensure_ascii=False))

if __name__ == "__main__":
    main()


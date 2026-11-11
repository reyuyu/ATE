"""Read-only inventory against the exact historical input and weight manifests."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict

from .common import *


def model_integrity():
    expected = read(LATE / "model_manifest.json")
    results = []
    for name, rec in expected["files"].items():
        path = MODEL / name
        if not path.is_file():
            results.append(dict(file=name, status="MISSING", required=name != ".gitattributes"))
            continue
        h = hashlib.sha1() if rec["hash_algorithm"] == "git-blob-SHA1" else hashlib.sha256()
        size = path.stat().st_size
        if rec["hash_algorithm"] == "git-blob-SHA1":
            h.update(f"blob {size}\0".encode())
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b""):
                h.update(chunk)
        results.append(dict(file=name, status="PASS" if h.hexdigest() == rec["hash"] and size == rec["bytes"] else "MISMATCH",
                            bytes=size, sha256=sha(path), expected_hash=rec["hash"], actual_hash=h.hexdigest(), required=True))
    report = dict(status="PASS" if all(r["status"] == "PASS" or not r["required"] for r in results) else "BLOCKED",
                  model=str(MODEL), revision=expected["revision"], files=results)
    write(OUT / "model_integrity.json", report)
    return report


def image_inventory():
    roots = [Path("/data/lk/ATE-resources/data/gqa-testdev-balanced-official/images"),
             Path("/data/lk/ATE-resources/benchmarks/phase4_0/images"),
             Path("/data/lk/ATE-audits/svh-cross-model-readiness-20261110/data")]
    base = Path("/data/lk/ATE-benchmarks-20261105/datasets")
    roots += [base / n for n in ["TextVQA", "ScienceQA-IMG", "POPE", "COCO2014", "MME", "MMBench", "OCRBench-v1"]]
    rows = {b: manifest(b) for b in BENCHES}
    sizes = {r["image_bytes"] for rs in rows.values() for r in rs}
    paths = set()
    for root in roots:
        if root.exists():
            for p in root.rglob("*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} and p.is_file() and p.stat().st_size in sizes:
                    paths.add(p)
    found = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for p, h in pool.map(lambda p: (str(p), sha(p)), sorted(paths)):
            found.setdefault(h, p)
    report = []
    mapping = {}
    for bench, rs in rows.items():
        missing = []
        for r in rs:
            if r["image_sha256"] in found:
                mapping[r["image_sha256"]] = found[r["image_sha256"]]
            else:
                missing.append({k: r[k] for k in ["question_id", "image_sha256", "image_path", "image_bytes"]})
        report.append(dict(benchmark=bench, planned=len(rs), matched=len(rs)-len(missing),
                           unique_images=len({r["image_sha256"] for r in rs}),
                           missing_images=len({r["image_sha256"] for r in missing}), missing=missing))
        print("IMAGES", bench, len(rs)-len(missing), "/", len(rs), flush=True)
    write(OUT / "local_image_map.json", mapping)
    write(OUT / "dataset_manifest.json", dict(status="PASS" if all(not x["missing"] for x in report) else "PARTIAL",
                                              scanned_roots=list(map(str, roots)), candidate_files=len(paths), benchmarks=report))
    return report


def inputs_and_history():
    import numpy as np
    from transformers import AutoTokenizer
    from vlm_research.internvl3_svh_seven_benchmarks.common import request_text, bind as oldbind
    from vlm_research.internvl3_visual_writing_figure3_3000.engine import role_layout
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False, local_files_only=True)
    expected = read(HISTORY / "dataset_manifest.json")
    checks = []
    for b in BENCHES:
        rs = manifest(b)
        assert sha(HISTORY / "manifests" / (b + ".jsonl.gz")) == expected[b]["manifest_sha256"]
        old = {(x["question_id"], x["method"]): x for x in history_rows(b)}
        assert len(old) == 2*len(rs)
        for r in rs:
            assert request_text(b, r) == r["request"]
            assert hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"]
            ids = tok(r["prompt"], return_tensors="np")["input_ids"]
            assert ids[0].tolist() == r["input_ids"]
            assert role_layout(r["input_ids"], tok.convert_tokens_to_ids("<IMG_CONTEXT>")) == r["roles"]
            ids_sha = hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest()
            for m in ["native", "physical_d"]:
                x = old[r["question_id"], m]
                d = json.loads(x["input_binding_details"])
                assert oldbind(d) == x["input_binding"] and d["input_ids_sha256"] == ids_sha
                assert x["sample_sha256"] == r["sample_sha256"]
                assert x["prompt_sha256"] == r["prompt_sha256"] and x["image_sha256"] == r["image_sha256"]
                assert int(x["max_new_tokens"]) == r["max_new_tokens"]
            assert old[r["question_id"], "native"]["input_binding"] == old[r["question_id"], "physical_d"]["input_binding"]
            checks.append(dict(benchmark=b, question_id=r["question_id"], image_sha256=r["image_sha256"], prompt_sha256=r["prompt_sha256"],
                               input_ids_sha256=ids_sha, request_exact=True, tokens_exact=True, methods_share_input=True))
        print("INPUTS", b, len(rs), flush=True)
    csvout(OUT / "prompt_alignment.csv", checks)
    result = dict(status="PASS_CPU_TEXT_BINDING", questions=len(checks), historical_predictions=2*len(checks),
                  note="Image bytes and BF16 processor hashes must also match before prediction reuse; CSV is not original SQLite checksum evidence.",
                  history_commit="3297559f21d3de55a6630a5491c9ecf66301c4b7")
    write(OUT / "protocol_alignment_audit.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["model", "images", "inputs", "all"], default="all")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    if args.stage in ["model", "all"]:
        print("MODEL", model_integrity()["status"], flush=True)
    if args.stage in ["images", "all"]:
        image_inventory()
    if args.stage in ["inputs", "all"]:
        inputs_and_history()
    write(OUT / "resource_preflight.json", dict(gpus=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free", "--format=csv"], text=True),
          compute_processes=subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv"], text=True),
          disk_free_bytes=shutil.disk_usage(OUT).free, no_data1_usage=True))


if __name__ == "__main__":
    main()

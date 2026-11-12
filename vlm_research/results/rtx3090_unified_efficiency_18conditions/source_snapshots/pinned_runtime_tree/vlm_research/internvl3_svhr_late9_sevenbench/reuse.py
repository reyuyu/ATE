"""Reconstruct original records and require the published raw-record SHA256.

CSV checksums alone are deliberately insufficient. Only records whose complete
original JSON, including all 28 KV layouts, matches the historical provenance
checksum are eligible for reuse.
"""
import csv
import gzip
import hashlib
import sqlite3
from collections import defaultdict
from .common import *


def reconstructed(bench):
    layout = defaultdict(list)
    with gzip.open(HISTORY / bench / "kv_layout_per_layer.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            values = {}
            for k, v in row.items():
                if k in {"benchmark", "question_id"}:
                    continue
                if k in {"k_shape", "v_shape"}:
                    values[k] = json.loads(v)
                elif k == "skip_attention":
                    values[k] = v == "True"
                elif k == "dtype":
                    values[k] = v or None
                elif k == "method":
                    values[k] = v
                else:
                    values[k] = int(v)
            method = {"Native": "native", "Physical-D": "physical_d"}[row["method"]]
            layout[row["question_id"], method].append(values)
    rs = {r["question_id"]: r for r in manifest(bench)}
    boolean = {"empty", "eos", "explanation", "parse_failure", "repetition", "truncated"}
    integers = {"generated_tokens", "input_tokens", "max_new_tokens", "visual_tokens"}
    structured = {"input_binding_details", "flags", "token_ids"}
    for x in history_rows(bench):
        rec = {}
        for k, v in x.items():
            if k in {"score", "question", "choices"}:
                continue
            if k in boolean:
                rec[k] = v == "True"
            elif k in integers:
                rec[k] = int(v)
            elif k in structured:
                rec[k] = json.loads(v)
            elif k == "KV_live_MiB":
                rec[k] = float(v)
            elif k == "gold":
                rec[k] = rs[x["question_id"]]["gold"]
            else:
                rec[k] = v
        rec["kv_layout"] = sorted(layout[x["question_id"], x["method"]], key=lambda x: x["layer"])
        assert len(rec["kv_layout"]) == 28
        yield rec


def main():
    folder = OUT / "server_only"
    folder.mkdir(exist_ok=True)
    path = folder / "verified_history.sqlite"
    tmp = path.with_suffix(".tmp.sqlite")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.execute("create table predictions(benchmark text,qid text,method text,checksum text,record text,primary key(benchmark,qid,method))")
    checks = []
    artifacts = read(HISTORY / "artifact_manifest.json")["uploaded_files"]
    artifact_map = {x["path"]: x["sha256"] for x in artifacts}
    for bench in BENCHES:
        provenance = HISTORY / bench / "prediction_provenance_v2.csv.gz"
        if not provenance.exists():
            checks.append(dict(benchmark=bench, reuse=0, status="REGENERATE_ORIGINAL_RAW_CHECKSUM_UNAVAILABLE"))
            continue
        for name in ["per_question.csv.gz", "prediction_provenance_v2.csv.gz", "kv_layout_per_layer.csv.gz"]:
            p = HISTORY / bench / name
            assert sha(p) == artifact_map[str(p.relative_to(ROOT))]
        with gzip.open(provenance, "rt") as f:
            expected = {(r["question_id"], r["method"]): r["raw_prediction_sha256"] for r in csv.DictReader(f)}
        good = bad = 0
        first_bad = None
        for rec in reconstructed(bench):
            raw = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            actual = hashlib.sha256(raw.encode()).hexdigest()
            if actual == expected[rec["question_id"], rec["method"]]:
                db.execute("insert into predictions values(?,?,?,?,?)", (bench, rec["question_id"], rec["method"], actual, raw))
                good += 1
            else:
                bad += 1
                if first_bad is None:
                    first_bad = dict(question_id=rec["question_id"], method=rec["method"], actual=actual, expected=expected[rec["question_id"], rec["method"]])
        db.commit()
        checks.append(dict(benchmark=bench, reuse=good, regenerate=bad, first_bad=first_bad, status="RAW_SHA_VERIFIED" if bad == 0 else "PARTIAL_RAW_SHA_VERIFIED"))
        print("RAW_HISTORY", checks[-1], flush=True)
    db.close()
    os.replace(tmp, path)
    write(OUT / "historical_reuse_audit.json", dict(checks=checks, verified_database_sha256=sha(path),
          policy="Only complete reconstructed original JSON matching published raw_prediction_sha256 is reused; other records regenerated."))


if __name__ == "__main__":
    main()

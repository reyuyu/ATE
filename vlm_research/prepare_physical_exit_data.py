"""Freeze a 12-image, balanced ChooseRel pilot before any model evaluation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess

from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_rows(rows, seed=20261028):
    # Sorting makes this independent of original CSV order. Label order is
    # fixed, and a shared used-image set guarantees distinct images overall.
    rng, selected, used = random.Random(seed), [], set()
    for label in ("left", "right"):
        pool = sorted([r for r in rows if r["answer"] == label and
                       r["true option"] == label and
                       {r["true option"], r["false option"]} == {"left", "right"}],
                      key=lambda r: (r["question_id"], r["img_id"]))
        rng.shuffle(pool)
        count = 0
        for row in pool:
            if row["img_id"] in used:
                continue
            selected.append(row)
            used.add(row["img_id"])
            count += 1
            if count == 6:
                break
        if count != 6:
            raise ValueError("Not enough distinct images for six examples per label")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Refuse to replace a frozen data selection")
    source = args.source_repo / "datasets/GQA_val_correct_question_with_relChooser_ChooseRel.csv"
    rows = list(csv.DictReader(source.open()))
    selected = select_rows(rows)
    args.output_dir.mkdir(parents=True)
    shutil.copyfile(source, args.output_dir / source.name)
    samples = []
    for r in selected:
        path = args.image_root / f'image_{r["img_id"]}.jpg'
        record = {k: r[k] for k in ("question_id", "img_id", "question", "answer", "true option", "false option")}
        record.update(candidates=["left", "right"], image_path=str(path), image_available=path.is_file(),
                      source_row_sha256=hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        if path.is_file():
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                record["image_size"] = list(image.size)
            record["image_sha256"] = sha256(path)
            record["source_image_size"] = [int(r["image_w"]), int(r["image_h"])]
            if record["image_size"] != record["source_image_size"]:
                raise ValueError(f'Local image dimensions differ from source: {r["img_id"]}')
        samples.append(record)
    manifest = {
        "seed": 20261028,
        "selection_rule": "Sort by question_id,img_id, shuffle with Python Random(20261028); choose first six unique images for left then right, sharing used-image set; no model outcomes or availability used in selection; fixed candidate order left,right; strict score tie is incorrect.",
        "source_repository": "https://github.com/FightingFighting/cross-modal-information-flow-in-MLLM",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source_repo, text=True).strip(),
        "source_file": "datasets/" + source.name,
        "source_csv_sha256": sha256(source), "source_rows": len(rows),
        "local_image_root": str(args.image_root),
        "local_image_search": "Searched /home,/data,/datasets,/mnt,/media for GQA/Visual Genome resources; existing MMEB GQA directory has every CSV image ID. No images downloaded or modified.",
        "caveat": "Upstream filename says correct_question; this is a preselected upstream cohort, not unbiased GQA. Current Full-correct subset is reported separately without resampling. Local image hashes and dimensions identify exact files, not independent certification of original ZIP bytes.",
        "samples": samples,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"samples": len(samples), "images_available": sum(r["image_available"] for r in samples),
                      "labels": [r["answer"] for r in samples], "question_ids": [r["question_id"] for r in samples]}, indent=2))


if __name__ == "__main__":
    main()

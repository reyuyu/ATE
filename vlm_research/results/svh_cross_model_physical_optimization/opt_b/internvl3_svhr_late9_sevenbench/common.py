import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "vlm_research/results/internvl3_svhr_late9_sevenbench_2x3090"
HISTORY = ROOT / "vlm_research/results/internvl3_svh_physical_d_seven_benchmarks_3090"
LATE = ROOT / "vlm_research/results/internvl3_svh_r_terminal_windows_textvqa_3090"
MODEL = Path("/data/lk/ATE-resources/models/internvl3-8b/853e3a797a661694b1b8ece0cb72dc2b23e3dac9")
BENCHES = ["GQA", "ScienceQA-IMG", "TextVQA", "POPE", "MME", "MMBench", "OCRBench"]
KEEP = list(range(19, 28))
METHODS = ["native", "physical_d", "late9"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def bind(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def csvout(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def manifest(bench):
    with gzip.open(HISTORY / "manifests" / (bench + ".jsonl.gz"), "rt") as f:
        return [json.loads(line) for line in f]


def history_rows(bench):
    with gzip.open(HISTORY / bench / "per_question.csv.gz", "rt") as f:
        return list(csv.DictReader(f))


def source_hashes():
    packages = ["internvl3_svhr_late9_sevenbench", "internvl3_svh_physical_d_pilot",
                "internvl3_visual_writing_figure3_3000", "internvl3_svh_r_terminal_windows_textvqa",
                "internvl3_svh_seven_benchmarks"]
    return {str(p.relative_to(ROOT)): sha(p) for pkg in packages
            for p in sorted((ROOT / "vlm_research" / pkg).glob("*.py"))}

"""Small artifact and token-layout helpers, independent of the LLaVA environment."""
import hashlib
import json
import os
from pathlib import Path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def question_positions(offsets, expanded_text, question):
    start = expanded_text.index(question)
    assert expanded_text.count(question) == 1
    end = start + len(question)
    result = [i for i, (a, b) in enumerate(offsets) if b > start and a < end and a != b]
    assert result and result == list(range(result[0], result[-1] + 1))
    return result


def select_samples(source, count_per_task=2, seed=20261101):
    raw = json.loads(Path(source).read_text(encoding="utf-8"))
    selected = []
    for task in sorted({x["task"] for x in raw["samples"]}):
        candidates = [x for x in raw["samples"] if x["task"] == task]
        candidates.sort(key=lambda x: hashlib.sha256(f"{seed}|{task}|{x['question_id']}".encode()).hexdigest())
        for row in candidates[:count_per_task]:
            fields = ("task", "question_id", "image_id", "question", "answer", "image_path", "image_sha256", "source")
            item = {k: row[k] for k in fields}
            assert sha256(item["image_path"]) == item["image_sha256"]
            selected.append(item)
    assert len(selected) == 6 * count_per_task
    assert len({(x["task"], x["question_id"]) for x in selected}) == len(selected)
    return {"seed": seed, "selection": "Two lowest SHA256(seed|task|question_id) per six original GQA task pools; no model output consulted.",
            "source_manifest": str(Path(source).resolve()), "source_manifest_sha256": sha256(source),
            "source_commit": "221cdde56dde53908aaa953fe26672fd33334f84", "samples": selected}

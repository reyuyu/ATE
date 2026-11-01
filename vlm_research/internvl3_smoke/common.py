"""Local artifact helpers; model payloads remain outside the Git worktree."""
import hashlib
import json
import os
from pathlib import Path

REPO = "OpenGVLab/InternVL3-8B"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "vlm_research/results/internvl3_8b_single_gpu_smoke"
MODELS = Path("/data/lk/ATE-resources/models/internvl3-8b")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()

"""Download only small pinned public source files; never imports model runtimes."""
import base64
import concurrent.futures
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPECS = {
    "open-compass/VLMEvalKit": (
        "051aa811ec27c494919bda944654de884fd70c41",
        ["vlmeval/inference.py", "vlmeval/dataset/image_base.py",
         "vlmeval/dataset/image_vqa.py", "vlmeval/dataset/image_yorn.py",
         "vlmeval/dataset/image_mcq.py", "vlmeval/dataset/utils/multiple_choice.py",
         "vlmeval/dataset/utils/vqa_eval.py", "vlmeval/dataset/utils/yorn.py",
         "vlmeval/utils/matching_util.py", "run.py"]),
    "EvolvingLMMs-Lab/lmms-eval": (
        "f9b577e76bac2fb5a17f6e2d5f9ad9b69156c292",
        ["lmms_eval/tasks/gqa", "lmms_eval/tasks/scienceqa",
         "lmms_eval/tasks/textvqa", "lmms_eval/tasks/pope",
         "lmms_eval/tasks/mme", "lmms_eval/tasks/mmbench",
         "lmms_eval/tasks/textvqa/_default_template_textvqa_yaml",
         "lmms_eval/tasks/mmbench/_default_template_mmbench_en_yaml",
         "lmms_eval/api/task.py"]),
}


def fetch(repo, commit, path):
    target = ROOT / "sources" / repo.replace("/", "__") / path
    if target.is_file():
        content = target.read_bytes()
        return [{"repo": repo, "commit": commit, "path": path,
                 "local_path": str(target.relative_to(ROOT)), "bytes": len(content),
                 "sha256": hashlib.sha256(content).hexdigest(), "reused": True}]
    url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={commit}"
    request = urllib.request.Request(url, headers={"User-Agent": "ATE-P0-static-audit"})
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        try:
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}", timeout=20) as response:
                content = response.read(1000000)
            if len(content) >= 1000000:
                raise RuntimeError("Small source limit exceeded")
            payload = {"content": base64.b64encode(content)}
        except (urllib.error.URLError, TimeoutError) as fallback:
            return [{"repo": repo, "commit": commit, "path": path,
                     "error": str(exc), "fallback_error": str(fallback)}]
    if isinstance(payload, list):
        rows = []
        for item in payload:
            if item["type"] == "file" and (item["name"].endswith((".py", ".yaml", ".yml", ".md")) or item["name"].startswith("_default_template")):
                rows.extend(fetch(repo, commit, item["path"]))
        return rows
    content = base64.b64decode(payload["content"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != content:
        raise RuntimeError(f"Refusing to overwrite changed archive: {target}")
    target.write_bytes(content)
    return [{"repo": repo, "commit": commit, "path": path,
             "local_path": str(target.relative_to(ROOT)), "bytes": len(content),
             "sha256": hashlib.sha256(content).hexdigest(), "api_url": url}]


if __name__ == "__main__":
    jobs = [(repo, commit, path) for repo, (commit, paths) in SPECS.items() for path in paths]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        groups = list(pool.map(lambda job: fetch(*job), jobs))
    rows = sorted([row for group in groups for row in group], key=lambda row: (row["repo"], row["path"]))
    (ROOT / "additional_source_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"files": sum("sha256" in row for row in rows),
                      "errors": [row for row in rows if "error" in row]}, indent=2))

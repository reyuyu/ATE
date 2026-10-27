"""Download a pinned official HF checkpoint, with optional mirror transport.

Uses resumable HTTP transfers and checks each weight shard against its
LFS SHA256. Model files stay outside the source tree.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = "llava-hf/llava-1.5-7b-hf"
REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--connections", type=int, default=4, help="Parallel range transfers per weight shard; 1 uses curl resume")
    parser.add_argument("--output", type=Path, default=Path.home() / ".cache/huggingface/manual/llava-1.5-7b-hf" / REVISION)
    args = parser.parse_args()
    if not 1 <= args.connections <= 8:
        parser.error("--connections must be between 1 and 8")
    args.output.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener()
    request = urllib.request.Request(f"{args.endpoint}/api/models/{REPO}/revision/{REVISION}", headers={"User-Agent": "vlm-research-phase1"})
    with opener.open(request, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise RuntimeError("Repository revision mismatch")
    names = [f["rfilename"] for f in metadata["siblings"] if f["rfilename"].endswith((".json", ".jinja", ".model", ".safetensors"))]

    def download(name):
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{args.endpoint}/{REPO}/resolve/{REVISION}/{name}"
        checksum = None
        total_size = None
        if name.endswith(".safetensors"):
            head_opener = urllib.request.build_opener(NoRedirect())
            try:
                response = head_opener.open(urllib.request.Request(url, method="HEAD"), timeout=60)
                headers = response.headers
                response.close()
            except urllib.error.HTTPError as exc:
                if exc.code not in (301, 302, 303, 307, 308):
                    raise
                headers = exc.headers
            checksum = headers.get("X-Linked-ETag", "").strip('"')
            total_size = int(headers.get("X-Linked-Size", "0"))
            if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise RuntimeError(f"Missing LFS SHA256 for {name}")
        partial = target.with_name(target.name + ".partial")
        if not target.exists():
            print(f"Downloading {name}", flush=True)
            if checksum and total_size and args.connections > 1:
                prefix_size = partial.stat().st_size if partial.exists() else 0
                if prefix_size > total_size:
                    raise RuntimeError(f"Oversized partial file: {name}")
                block_size = 256 * 1024 * 1024
                ranges = [(start, min(start + block_size, total_size) - 1) for start in range(prefix_size, total_size, block_size)]

                def fetch_range(bounds):
                    start, end = bounds
                    chunk = partial.with_name(f"{partial.name}.range-{start}-{end}")
                    expected = end - start + 1
                    for attempt in range(5):
                        offset = chunk.stat().st_size if chunk.exists() else 0
                        if offset == expected:
                            return chunk
                        if offset > expected:
                            raise RuntimeError(f"Oversized range file: {chunk.name}")
                        try:
                            request = urllib.request.Request(url, headers={"Range": f"bytes={start + offset}-{end}"})
                            with urllib.request.urlopen(request, timeout=90) as response:
                                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start + offset}-{end}/"):
                                    raise RuntimeError(f"Server ignored requested range for {name}")
                                with chunk.open("ab") as handle:
                                    for data in iter(lambda: response.read(1024 * 1024), b""):
                                        handle.write(data)
                            if chunk.stat().st_size != expected:
                                raise OSError("Incomplete range response")
                            print(f"Range complete {name}: {start}-{end}", flush=True)
                            return chunk
                        except (OSError, urllib.error.URLError):
                            if attempt == 4:
                                raise
                            time.sleep(3)

                with ThreadPoolExecutor(max_workers=args.connections) as transfers:
                    chunks = list(transfers.map(fetch_range, ranges))
                # Merge to a separate file so interruption never invalidates the
                # original prefix or the already completed range files.
                merged = partial.with_name(partial.name + ".merged")
                with merged.open("wb") as destination:
                    for source in ([partial] if prefix_size else []) + chunks:
                        with source.open("rb") as handle:
                            for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                                destination.write(data)
                if merged.stat().st_size != total_size:
                    raise RuntimeError(f"Merged file size mismatch: {name}")
                merged.replace(partial)
                for chunk in chunks:
                    chunk.unlink()
            else:
                subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "4", "--retry-delay", "3", "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "120", "--continue-at", "-", "--output", str(partial), url], check=True)
            partial.replace(target)
        if checksum:
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != checksum:
                raise RuntimeError(f"SHA256 mismatch: {name}; remove this file and retry")
        print(f"Verified {name}: {target.stat().st_size} bytes", flush=True)
        return {"name": name, "bytes": target.stat().st_size, "lfs_sha256": checksum}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, names))
    config = {"model_id": REPO, "revision": REVISION, "local_model_path": str(args.output.resolve()), "download_endpoint": args.endpoint, "files": files}
    (ROOT / "configs/model.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Ready: {args.output}", flush=True)


if __name__ == "__main__":
    main()

"""Pinned official payloads, with a transparent HF metadata mirror fallback."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import requests

REPO = "Qwen/Qwen3-VL-8B-Instruct"
HF_SHA = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MS_SHA = "1610d9f98460d59272cd9963efdf92f0cd3c8dc9"
HF = "https://hf-mirror.com"
MS = "https://modelscope.cn"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def session():
    result = requests.Session()
    result.trust_env = False
    return result


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def get(s, url):
    r = s.get(url, timeout=(15, 90))
    r.raise_for_status()
    return r


def download_file(info, root):
    path = root / info["path"]
    if path.exists() and path.stat().st_size == info["size"] and sha256(path) == info["sha256"]:
        print("Already verified", path.name, flush=True)
        return
    partial = path.with_suffix(path.suffix + ".partial")
    s = session()
    for attempt in range(6):
        start = partial.stat().st_size if partial.exists() else 0
        try:
            with s.get(info["url"], stream=True, headers={"Range": f"bytes={start}-"} if start else {}, timeout=(20, 90)) as r:
                r.raise_for_status()
                if start and r.status_code != 206:
                    start = 0
                if start:
                    assert r.headers["Content-Range"].startswith(f"bytes {start}-")
                t0 = last = time.monotonic()
                received = start
                with partial.open("ab" if start else "wb") as f:
                    for block in r.iter_content(4 * 1024**2):
                        f.write(block)
                        received += len(block)
                        assert received <= info["size"], "Oversize download"
                        if time.monotonic() - last > 15:
                            print(f"{path.name}: {received/1e9:.2f}/{info['size']/1e9:.2f} GB; {(received-start)/1e6/(time.monotonic()-t0):.1f} MB/s", flush=True)
                            last = time.monotonic()
            assert partial.stat().st_size == info["size"]
            assert sha256(partial) == info["sha256"], "SHA256 mismatch"
            os.replace(partial, path)
            print("SHA256 verified", path.name, flush=True)
            return
        except Exception as exc:
            print(f"Retry {attempt+1}: {path.name}: {exc}", flush=True)
            if partial.exists() and partial.stat().st_size >= info["size"]:
                partial.unlink()
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"Download failed: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.model_dir
    root.mkdir(parents=True, exist_ok=True)
    s = session()
    hf = get(s, f"{HF}/api/models/{REPO}/revision/{HF_SHA}?blobs=true").json()
    assert hf["sha"] == HF_SHA
    ms = get(s, f"{MS}/api/v1/models/{REPO}/repo/files?Revision={MS_SHA}&Recursive=true").json()
    ms_files = {x["Path"]: x for x in ms["Data"]["Files"] if x["Type"] == "blob"}
    write(args.out / "hf_metadata.json", hf)
    write(args.out / "modelscope_metadata.json", ms)
    files = []
    weights = []
    for entry in hf["siblings"]:
        name = entry["rfilename"]
        assert Path(name).name == name
        lfs = entry.get("lfs")
        if name.endswith(".safetensors"):
            assert lfs and ms_files[name]["Sha256"] == lfs["sha256"]
            assert ms_files[name]["Size"] == lfs["size"]
            info = {"path": name, "sha256": lfs["sha256"], "size": lfs["size"],
                    "url": f"{MS}/models/{REPO}/resolve/{MS_SHA}/{name}",
                    "source": "official ModelScope; exact pinned HF LFS SHA256 and size match"}
            weights.append(info)
        else:
            data = get(s, f"{HF}/{REPO}/resolve/{HF_SHA}/{name}").content
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if not lfs:
                assert blob == entry["blobId"], (name, blob, entry["blobId"])
            else:
                assert hashlib.sha256(data).hexdigest() == lfs["sha256"]
                assert len(data) == lfs["size"]
            info = {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                    "source": "pinned official HF snapshot via hf-mirror.com transport", "git_blob_sha1": blob,
                    "modelscope_same_content": ms_files.get(name, {}).get("Sha256") == hashlib.sha256(data).hexdigest()}
            (root / name).write_bytes(data)
        files.append(info)
    manifest = {"repo_id": REPO, "revision": HF_SHA, "local_path": str(root.resolve()),
                "hf_direct_access": "ConnectionError: network unreachable; no direct HF confirmation available",
                "metadata_transport": HF, "modelscope_revision": MS_SHA,
                "verification": "HF mirror supplies pinned HF Git/LFS identities; official ModelScope weights must exactly match all four HF LFS SHA256/size identities; actual files independently hashed. No mixing of weight identities.",
                "files": files, "weight_bytes": sum(x["size"] for x in weights)}
    write(args.out / "model_manifest.json", manifest)
    print("Frozen matching snapshot; weight GB", manifest["weight_bytes"] / 1e9, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda x: download_file(x, root), weights))
    actual_keys = set()
    actual_map = {}
    dtype_counts = {}
    total_tensors = 0
    for item in files:
        p = root / item["path"]
        assert p.stat().st_size == item["size"] and sha256(p) == item["sha256"]
        if p.suffix == ".safetensors":
            with p.open("rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                assert header_size < 100 * 1024**2
                header = json.loads(f.read(header_size))
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                assert key not in actual_keys
                actual_keys.add(key)
                actual_map[key] = p.name
                total_tensors += 1
                dtype_counts[meta["dtype"]] = dtype_counts.get(meta["dtype"], 0) + 1
                assert 0 <= meta["data_offsets"][0] <= meta["data_offsets"][1] <= p.stat().st_size - 8 - header_size
    index = json.loads((root / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == actual_keys
    assert index["weight_map"] == actual_map
    write(args.out / "download_validation.json", {"passed": True, "files": len(files), "weight_shards": len(weights),
          "tensors": total_tensors, "dtype_counts": dtype_counts, "all_sha256_verified": True,
          "weight_index_keys_exact": True, "logical_disk_bytes": sum(x["size"] for x in files),
          "physical_disk_bytes": sum((root/x["path"]).stat().st_blocks*512 for x in files)})
    print("Download and completeness validation PASS", flush=True)


if __name__ == "__main__":
    main()

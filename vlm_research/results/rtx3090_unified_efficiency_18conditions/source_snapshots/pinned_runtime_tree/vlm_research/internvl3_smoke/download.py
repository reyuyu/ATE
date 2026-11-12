"""Pin HF Git/LFS identities; only accept byte-identical official fallback files."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import shutil
import time
import traceback

import requests
from huggingface_hub import configure_http_backend, snapshot_download
from safetensors import safe_open

from .common import MODELS, OUT, REPO, read, sha256, write

OFFICIAL = "https://huggingface.co"
MIRROR = "https://hf-mirror.com"
MS = "https://modelscope.cn"


class Session(requests.Session):
    def __init__(self):
        super().__init__()
        self.trust_env = False

    def request(self, *args, **kwargs):
        kwargs["timeout"] = (5, 30)
        return super().request(*args, **kwargs)


def get(url):
    response = Session().get(url)
    response.raise_for_status()
    return response


def fallback_download(item, directory):
    path = directory / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = item["sha256"]
    if path.exists() and path.stat().st_size == item["size"] and sha256(path) == expected:
        return
    partial = path.with_suffix(path.suffix + ".partial")
    for attempt in range(5):
        start = partial.stat().st_size if partial.exists() else 0
        try:
            with Session().get(item["url"], stream=True,
                               headers={"Range": f"bytes={start}-"} if start else {}) as response:
                response.raise_for_status()
                if start and response.status_code != 206:
                    start = 0
                if start:
                    assert response.headers["Content-Range"].startswith(f"bytes {start}-")
                began = last = time.monotonic()
                received = start
                with partial.open("ab" if start else "wb") as handle:
                    for block in response.iter_content(4 * 1024**2):
                        handle.write(block)
                        received += len(block)
                        assert received <= item["size"]
                        if time.monotonic() - last > 20:
                            print(f"{path.name}: {received/1e9:.2f}/{item['size']/1e9:.2f}GB "
                                  f"{(received-start)/1e6/(time.monotonic()-began):.1f}MB/s", flush=True)
                            last = time.monotonic()
            assert partial.stat().st_size == item["size"]
            assert sha256(partial) == expected, f"SHA256 mismatch: {path.name}"
            os.replace(partial, path)
            print("Verified", path.name, flush=True)
            return
        except Exception as error:
            print(f"Retry {attempt+1} {path.name}: {error}", flush=True)
            if partial.exists() and partial.stat().st_size >= item["size"]:
                partial.unlink()
            time.sleep(5 * (attempt+1))
    raise RuntimeError(f"Download failed: {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(MODELS.parent).free
    assert available >= 25 * 1024**3
    configure_http_backend(Session)
    attempts = []
    try:
        metadata = get(f"{OFFICIAL}/api/models/{REPO}?blobs=true").json()
        endpoint = OFFICIAL
        attempts.append(dict(stage="official_api", passed=True))
    except Exception:
        attempts.append(dict(stage="official_api", passed=False, error=traceback.format_exc()))
        metadata = get(f"{MIRROR}/api/models/{REPO}?blobs=true").json()
        endpoint = MIRROR
    revision = metadata["sha"]
    assert metadata["id"] == REPO and len(revision) == 40
    pinned = get(f"{endpoint}/api/models/{REPO}/revision/{revision}?blobs=true").json()
    assert pinned["sha"] == revision
    directory = MODELS / revision
    directory.mkdir(parents=True, exist_ok=True)
    weight_bytes = sum(x.get("lfs", {}).get("size", 0) for x in pinned["siblings"])
    # local_dir stores one payload copy, and MS resumes into one .partial/file.
    assert available >= weight_bytes + 25 * 1024**3
    write(OUT / "hf_metadata.json", pinned)
    write(OUT / "download_manifest.json", dict(repo_id=REPO, revision=revision,
        local_path=str(directory), metadata_endpoint=endpoint, direct_official_api_verified=endpoint == OFFICIAL,
        available_before_bytes=available, expected_lfs_bytes=weight_bytes,
        peak_payload_copies=1, free_space_safety_bytes=25*1024**3, attempts=attempts))
    print("Pinned", REPO, revision, "LFS GB", weight_bytes/1e9, flush=True)
    try:
        snapshot_download(repo_id=REPO, revision=revision, local_dir=directory,
                          endpoint=OFFICIAL, max_workers=2)
        attempts.append(dict(stage="official_snapshot_download", passed=True))
    except Exception:
        attempts.append(dict(stage="official_snapshot_download", passed=False, error=traceback.format_exc()))
        try:
            snapshot_download(repo_id=REPO, revision=revision, local_dir=directory,
                endpoint=MIRROR, max_workers=2, ignore_patterns=["*.safetensors"])
            attempts.append(dict(stage="mirror_snapshot_metadata", passed=True))
        except Exception:
            attempts.append(dict(stage="mirror_snapshot_metadata", passed=False, error=traceback.format_exc()))
    files = []
    weights = []
    ms_metadata = get(f"{MS}/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true").json()
    assert ms_metadata["Success"]
    ms_revision = ms_metadata["Data"]["LatestCommitter"]["Id"] or ms_metadata["Data"]["LatestCommitter"]["ShortId"]
    # Resolve the full per-file revision for the latest commit, then pin listing.
    if len(ms_revision) != 40:
        candidates = [r["Revision"] for r in ms_metadata["Data"]["Files"] if r["Revision"].startswith(ms_revision)]
        assert candidates and len(set(candidates)) == 1
        ms_revision = candidates[0]
    ms_pinned = get(f"{MS}/api/v1/models/{REPO}/repo/files?Revision={ms_revision}&Recursive=true").json()
    ms_files = {x["Path"]: x for x in ms_pinned["Data"]["Files"] if x["Type"] == "blob"}
    write(OUT / "modelscope_metadata.json", dict(revision=ms_revision, listing=ms_pinned))
    for entry in pinned["siblings"]:
        name = entry["rfilename"]
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        lfs = entry.get("lfs")
        if name.endswith(".safetensors"):
            assert lfs and ms_files[name]["Sha256"] == lfs["sha256"]
            assert ms_files[name]["Size"] == lfs["size"]
            item = dict(path=name, size=lfs["size"], sha256=lfs["sha256"],
                url=f"{MS}/models/{REPO}/resolve/{ms_revision}/{name}",
                source="Official ModelScope fixed commit; exact pinned HF LFS SHA256 and size match")
            weights.append(item)
        else:
            path = directory / name
            data = path.read_bytes() if path.exists() else get(f"{endpoint}/{REPO}/resolve/{revision}/{name}").content
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if lfs:
                assert len(data) == lfs["size"] and hashlib.sha256(data).hexdigest() == lfs["sha256"]
            else:
                assert blob == entry["blobId"], f"Git blob mismatch: {name}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            item = dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest(), git_blob_sha1=blob,
                        source="Fixed official HF revision via documented metadata transport")
        files.append(item)
    manifest = read(OUT / "download_manifest.json")
    manifest.update(attempts=attempts, files=files, modelscope_revision=ms_revision,
        fallback_identity="Only bytes matching pinned HF Git blobs/LFS hashes accepted; no foreign-revision custom code",
        runtime_code_review_required=True)
    write(OUT / "download_manifest.json", manifest)
    print("All non-weight files pinned; downloading four exact-match official shards", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: fallback_download(item, directory), weights))
    all_keys = {}
    dtype_counts = {}
    for item in files:
        path = directory / item["path"]
        assert path.stat().st_size == item["size"] and sha256(path) == item["sha256"]
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as tensors:
                assert tensors.metadata() is not None
                for key in tensors.keys():
                    assert key not in all_keys
                    all_keys[key] = path.name
                    dtype = tensors.get_slice(key).get_dtype()
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    index = read(directory / "model.safetensors.index.json")
    assert index["weight_map"] == all_keys
    write(OUT / "model_files_sha256.json", dict(revision=revision, files=files))
    write(OUT / "download_validation.json", dict(passed=True, revision=revision, shards=len(weights),
        tensors=len(all_keys), dtype_counts=dtype_counts, weight_index_exact=True, all_sha256_verified=True,
        logical_disk_bytes=sum(x["size"] for x in files), actual_disk_bytes=sum(p.stat().st_blocks*512 for p in directory.rglob('*') if p.is_file()),
        local_path=str(directory), remote_code_executed=False))
    print("DOWNLOAD COMPLETE: all fixed-revision files and tensor index verified", flush=True)


if __name__ == "__main__":
    main()

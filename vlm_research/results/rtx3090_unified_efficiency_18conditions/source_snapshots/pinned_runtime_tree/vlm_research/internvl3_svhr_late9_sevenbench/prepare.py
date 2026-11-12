"""Freeze local paths, processor bindings, two shards and the continuation policy."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from .common import *


def main():
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    from vlm_research.internvl3_visual_writing_figure3_3000.engine import official_preprocessor, tensor_hash
    assert read(OUT / "model_integrity.json")["status"] == "PASS"
    local = read(OUT / "local_image_map.json")
    allrows = [r for b in BENCHES for r in manifest(b)]
    pixels_expected = {}
    for b in BENCHES:
        for r in history_rows(b):
            h = json.loads(r["input_binding_details"])["pixels_bf16_sha256"]
            assert pixels_expected.setdefault(r["image_sha256"], h) == h
    loader = official_preprocessor(MODEL)

    def check(item):
        h, path = item
        pix = loader(path, max_num=1).to(dtype=torch.bfloat16)
        assert list(pix.shape) == [1, 3, 448, 448]
        actual = tensor_hash(pix)
        assert actual == pixels_expected[h], ("PROCESSOR_BYTE_MISMATCH", h, actual, pixels_expected[h])
        return h, dict(path=path, pixels_bf16_sha256=actual, shape=list(pix.shape))

    with ThreadPoolExecutor(max_workers=4) as pool:
        pixels = dict(pool.map(check, sorted(local.items())))
    write(OUT / "processor_bindings.json", pixels)
    # MMBench circular rotations must remain intact even if a group has variant images.
    parent = {r["image_sha256"]: r["image_sha256"] for r in allrows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    groups = {}
    for r in allrows:
        if r["benchmark"] in {"MMBench", "MME"}:
            key = (r["benchmark"], r["group"])
            if key in groups:
                a, b = find(groups[key]), find(r["image_sha256"])
                parent[max(a, b)] = min(a, b)
            groups[key] = r["image_sha256"]
    clusters = {}
    for r in allrows:
        clusters.setdefault(find(r["image_sha256"]), []).append([r["benchmark"], r["question_id"]])
    loads = [0, 0]
    assignments = {}
    for h, rows in sorted(clusters.items(), key=lambda p: (-len(p[1]), p[0])):
        worker = min(range(2), key=lambda i: (loads[i], i))
        loads[worker] += len(rows)
        for b, q in rows:
            assignments[b + "/" + q] = worker
    write(OUT / "shard_assignment.json", dict(loads=loads, assignments=assignments, cluster="image_sha256 with MME/MMBench group unions"))
    chosen = []
    for h in sorted(local, key=lambda h: bind([20261110, "efficiency48", h]))[:48]:
        r = next(r for r in allrows if r["image_sha256"] == h)
        chosen.append(dict(benchmark=r["benchmark"], question_id=r["question_id"], image_sha256=h))
    write(OUT / "efficiency_manifest.json", chosen)
    # Remote-code chat-template reconstruction is CPU-only and uses the pinned file.
    spec = importlib.util.spec_from_file_location("pinned_internvl_conversation", MODEL / "conversation.py")
    convmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convmod)
    template = read(MODEL / "config.json")["template"]
    for r in allrows:
        c = convmod.get_conv_template(template)
        c.append_message(c.roles[0], r["request"])
        c.append_message(c.roles[1], None)
        prompt = c.get_prompt().replace("<image>", "<img>" + "<IMG_CONTEXT>"*256 + "</img>", 1)
        assert prompt == r["prompt"], ("CHAT_TEMPLATE_MISMATCH", r["question_id"])
    cpu = read(OUT / "cpu_validation.json")
    assert cpu["status"] == "PASS"
    protocol = dict(
        model="OpenGVLab/InternVL3-8B", revision="853e3a797a661694b1b8ece0cb72dc2b23e3dac9",
        model_path=str(MODEL), torch="2.6.0+cu124", transformers="4.49.0", dtype="bfloat16", attention="eager",
        image_shape=[1, 3, 448, 448], visual_tokens=256, batch_size=1, do_sample=False, num_beams=1,
        use_cache=True, logits_to_keep=1, eos_id=151645, methods=METHODS,
        late9=dict(keep=KEEP, cutoff=28, R_image_boundary=20, answer_attention_skip=[2, 3, 4, 5]),
        input_authority="6c294aae67ddb2a5cb3706ded76dcf3edb26c859", inherited_completed_baselines="3297559f21d3de55a6630a5491c9ecf66301c4b7",
        physical_authority="1407275cbdffca5d4a23cbf8b8d233002e2840dd", original_protocol_sha256=sha(HISTORY / "protocol.json"),
        caps=read(HISTORY / "protocol.json")["caps"], bootstrap=dict(repetitions=20000, seed=20261110, cluster="image_sha256", confidence=.95),
        gpu_uuids=["GPU-a63c6362-dcfd-4a26-c44b-014b40127c89", "GPU-a50cb044-679e-9a05-557d-4ddbd19ac3c6"],
        numerical_status="EXPLORATORY_QUALITY_WITH_NUMERICAL_WAIVER", strict_native_noop=True,
        continuation="Preserve and score all nonstructural empty/verbose/repeated/truncated outputs. Stop worker on NaN/Inf/OOM, hash, KV, mapping or structural failure. No retuning.",
        historical_reuse="Reconstruct complete original JSON including all 28 KV layouts, and require exact published raw_prediction_sha256. Reuse only verified records after local input/processor checks. Regenerate any record without a verifiable original raw checksum; retain historical files read-only.",
        unavailable_images="Record every missing-image request as BLOCKED. Never substitute a different SHA. Partial subsets are not full benchmark scores.",
        bindings={name: sha(OUT / name) for name in ["processor_bindings.json", "local_image_map.json", "shard_assignment.json", "efficiency_manifest.json", "historical_reuse_audit.json"]})
    write(OUT / "protocol.json", protocol)
    write(OUT / "source_manifest.json", dict(runtime_source_sha256=source_hashes(), frozen_before_GPU=True))
    print("PREPARED", loads, "verified image processors", len(pixels), flush=True)


if __name__ == "__main__":
    main()

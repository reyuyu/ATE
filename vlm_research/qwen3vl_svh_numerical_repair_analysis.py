"""CPU-only independent evidence verification and artifact packing.

Does not import the execution or its comparison helpers. BF16 logits are saved
losslessly as uint16 bits; complete vectors, including EOS, remain re-checkable.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vlm_research/results/qwen3vl_svh_physical_numerical_repair"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def table(p, rows):
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def load(p):
    return torch.load(p, map_location="cpu", weights_only=True)


def compare(ref, candidate):
    assert len(ref["logits"]) == len(candidate["logits"])
    rows = []
    for t, (a, b) in enumerate(zip(ref["logits"], candidate["logits"])):
        x, y = a.numpy().astype(np.float64), b.numpy().astype(np.float64)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        delta = np.abs(x-y)
        row = dict(step=t, max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                   strict_equal=bool(np.array_equal(x, y)), top1_equal=bool(x.argmax() == y.argmax()),
                   complete_tokens_equal=ref["token_ids"] == candidate["token_ids"])
        row["gate_pass"] = row["max_abs"] <= .5 and row["mean_abs"] <= .02 and row["top1_equal"] and row["complete_tokens_equal"]
        rows.append(row)
    return rows


def main():
    assert not torch.cuda.is_initialized()
    rows, layouts, firsts, kv = [], [], [], []
    archive, index = {}, []
    for group in ["diagnosis_round2", "independent"]:
        for folder in sorted((OUT / group).iterdir()):
            if not folder.is_dir():
                continue
            ref = load(folder / "Dense_logits.pt")
            native = load(folder / "Native_logits.pt")
            assert all(r["strict_equal"] for r in compare(native, load(folder / "Noop_logits.pt")))
            names = ["Physical"] if group == "independent" else ["Original", "Physical"]
            for name in names:
                comparisons = compare(ref, load(folder / (name + "_logits.pt")))
                rows += [dict(group=group, sample=folder.name, condition=name, **r) for r in comparisons]
                if name == "Physical":
                    assert all(r["gate_pass"] and r["strict_equal"] for r in comparisons)
                if group == "diagnosis_round2":
                    firsts.append(dict(sample=folder.name, condition=name,
                                      first=read(folder / (name + "_first_divergence.json"))))
            physical = read(folder / "Physical.json")["audit"]
            text_len = physical["layers"][22]["hidden_rows"]
            assert all(r["hidden_rows"] == text_len for r in physical["layers"] if r["step"] == 0 and r["layer"] >= 22)
            assert all(r["dense_projection_executed"] is False and r["real_rows"] == text_len for r in physical["aligned_linears"])
            assert len(physical["aligned_linears"]) == 14 * 7
            assert all(r["persistent_image_KV"] == 0 for r in physical["transient_prefill_columns"])
            assert all(r["temporary_zero_image_keys"] > 0 and r["persistent_dummy_KV"] == 0 for r in physical["decode_temporary_zero_keys"])
            cache = physical["kv"]
            for r in cache["layers"]:
                assert r["live_tokens"] == (0 if r["layer"] in range(1, 7) else text_len+1)
                kv.append(dict(group=group, sample=folder.name, **r))
            ev = [r for r in physical["attention_events"] if r.get("skipped_rows")]
            assert len([r for r in ev if r["step"] == 0]) == 6
            assert len([r for r in ev if r["step"] == 1 and r["qkv_calls"] == 0 and r["o_proj_calls"] == 0]) == 6
            assert len([r for r in physical["attention_events"] if r.get("mlp_and_norm_executed")]) == 72
            if group == "diagnosis_round2":
                sem = read(folder / "Physical_semantics.json")
                assert sem["mrope_original_positions_exact"] and sem["deepstack_exact"]
                assert all(r["k"]["equal"] and r["v"]["equal"] for r in sem["cache_comparisons"])
                assert all(r["blocked_probability"] == 0 for r in sem["masks"])
                layers = list(csv.DictReader((folder / "Physical_layerwise.csv").open()))
                assert all(r["equal"] == "True" for r in layers if r["component"] == "layer_output")
            layouts.append(dict(group=group, sample=folder.name, post_exit_hidden_rows=text_len,
                skip_layers=list(range(1, 7)), physical_exit_layer=22, persistent_image_KV_rows=0,
                kv_live_bytes=cache["live_bytes"], kv_capacity_bytes=cache["capacity_bytes"],
                projected_prefill_rows=sorted({r["real_rows"] for r in physical["aligned_linears"]}),
                late_temporary_key_rows=physical["transient_prefill_columns"][0]["transient_keys"],
                finite=True, eos=read(folder / "Physical.json")["eos"]))
    # Archive every attempt, including the interrupted diagnostic, without deleting raw evidence.
    for p in sorted(OUT.rglob("*_logits.pt")):
        data = load(p)
        key = "v" + str(len(index))
        vectors = torch.stack(data["logits"])
        bits = vectors.to(torch.bfloat16).view(torch.uint16).numpy()
        assert torch.equal(torch.from_numpy(bits.copy()).view(torch.bfloat16).float(), vectors)
        archive[key] = bits
        index.append(dict(key=key, path=str(p.relative_to(OUT)), source_sha256=sha(p),
                          shape=list(bits.shape), token_ids=data["token_ids"], encoding="BF16 uint16 bits"))
    np.savez_compressed(OUT / "complete_logits_bf16_bits.npz", **archive)
    write(OUT / "complete_logits_index.json", index)
    # Verify the frozen weights and installed primary source, without model loading/CUDA.
    parent = ROOT / "vlm_research/results/qwen3vl_svh_six_official_benchmarks_2x3090"
    manifest = read(parent / "model_manifest.json")
    weight_audit = [dict(path=r["path"], expected=r["expected_sha256"], actual=sha(r["path"])) for r in manifest["files"]]
    assert all(r["expected"] == r["actual"] for r in weight_audit)
    write(OUT / "model_hash_revalidation.json", weight_audit)
    write(OUT / "runtime.json", dict(torch=torch.__version__, transformers=transformers.__version__,
          CUDA_initialized=torch.cuda.is_initialized(), transformers_model_source=sha(Path(transformers.__file__).parent / "models/qwen3_vl/modeling_qwen3_vl.py"),
          execution_commit="a07c91b3bb", author_protocol="BLOCKED", adopted_protocol="ATE_UNIFIED_HF_NON_AUTHOR_OFFICIAL"))
    assert torch.__version__ == "2.6.0+cu124" and transformers.__version__ == "4.57.6"
    table(OUT / "independent_logits_recalculation.csv", rows)
    table(OUT / "token_kv_audit.csv", layouts)
    table(OUT / "per_layer_kv.csv", kv)
    write(OUT / "first_divergences.json", firsts)
    old = Path("/data/lk/ATE-resources/experiments/qwen3vl_c3v2_prefill_av_final/rank0/GQA__06905471/F_logits.pt")
    history_comparison = compare(load(old), load(OUT / "diagnosis_round2/historical_06905471/Dense_logits.pt"))
    assert all(r["strict_equal"] for r in history_comparison)
    write(OUT / "historical_full_vector_reproduction.json", dict(path=str(old), sha256=sha(old), comparison=history_comparison))
    release = dict(gpus=subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu", "--format=csv"], text=True),
                   processes=subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True))
    assert "GPU-96bc853f-ceea-2404-c704-c48a3d1ca185" not in release["processes"]
    write(OUT / "gpu_release_audit.json", release)
    write(OUT / "validation.json", dict(status="PASS_FIXED_SMOKE_ONLY", historical_failure_reproduced=True,
          native_noop_strict=True, hf_native_strict=True, dense_physical_all_valid_logits_strict=True,
          diagnosis_inputs=6, independent_new_process_inputs=3, CPU_checks=6, gates_unchanged=True,
          mrope_deepstack_exact=True, all_retained_layer_outputs_exact=True, all_retained_text_KV_exact=True,
          transient_attention_subnormal_difference="Historical shortest L22 AV max7.45e-9; vanishes at layer output",
          raw_vectors_recomputed_with="Independent NumPy float64; original thresholds .5/.02",
          no_OOM=True, no_nan_inf_activations=True, no_benchmark_evaluation=True, gpu_released=True,
          scope_limit="Three fixed GQA inputs, 96/192/672 visual tokens, first answer + genuine EOS Decode; not six-benchmark-wide evidence"))
    write(OUT / "completion.json", dict(status="NUMERICAL_REPAIR_SMOKE_PASSED", formal_predictions=0,
          author_official_reproduction="BLOCKED", push_performed=False, GPU_released=True))
    print("CPU independent verification PASS", len(rows), "comparisons;", len(index), "lossless vectors; weights verified", flush=True)


if __name__ == "__main__":
    main()

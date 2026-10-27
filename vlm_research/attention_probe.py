"""Phase 2: last-prompt-query LLM attention, without hooks or decoding."""
import argparse
import csv
import gc
import hashlib
import math
from pathlib import Path
import time

# Keep the phase-1 CUDA/determinism setup before importing torch.
from vlm_common import ROOT, gpu_memory, load_model, prepare_inputs, write_json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

SPATIAL_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 31]
EPSILON = 1e-12
SUM_ATOL = 2e-3  # Returned probabilities are FP16; accumulate in FP64 on CPU.


def key_masks(input_ids, attention_mask, query_index, image_token_index):
    """Uncached causal prefill: key is visible iff unpadded and key <= query."""
    ids = np.asarray(input_ids)
    mask = np.asarray(attention_mask)
    if ids.ndim != 1 or ids.shape != mask.shape or not 0 <= query_index < len(ids):
        raise ValueError("Expected one sequence and a valid query index")
    if not mask[query_index]:
        raise ValueError("Query must not be padding")
    accessible = (mask != 0) & (np.arange(len(ids)) <= query_index)
    visual = accessible & (ids == image_token_index)
    nonvisual = accessible & ~visual
    if not visual.any() or not nonvisual.any():
        raise ValueError("Both accessible visual and nonvisual keys are required")
    return accessible, visual, nonvisual


def summarize_rows(rows, accessible, visual, nonvisual, layer):
    """Use original probabilities for mass; condition on vision only for entropy."""
    rows = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(rows).all() or (rows < 0).any():
        raise ValueError("Non-finite or negative query attention")
    sums = rows[:, accessible].sum(axis=-1)
    if not np.allclose(sums, 1, atol=SUM_ATOL, rtol=0):
        raise ValueError(f"Layer {layer}: accessible attention sums {sums}")
    if np.any(rows[:, ~accessible] != 0):
        raise ValueError("Attention to causally inaccessible/padded keys")
    nv, nn = int(visual.sum()), int(nonvisual.sum())
    baseline = nv / int(accessible.sum())
    v = rows[:, visual]
    vm, nm = v.sum(-1), rows[:, nonvisual].sum(-1)
    conditional = np.divide(v, vm[:, None], out=np.zeros_like(v), where=vm[:, None] > 0)
    logp = np.zeros_like(conditional)
    np.log(conditional, out=logp, where=conditional > 0)
    entropy = -(conditional * logp).sum(-1)
    topk = math.ceil(0.1 * nv)
    topmass = np.sort(v, axis=-1)[:, -topk:].sum(-1)
    records = []
    for head in range(len(rows)):
        records.append({
            "layer": layer, "head": head,
            "accessible_keys": nv + nn, "visual_keys": nv, "nonvisual_keys": nn,
            "accessible_attention_sum": float(sums[head]),
            "visual_attention_mass": float(vm[head]),
            "nonvisual_attention_mass": float(nm[head]),
            "uniform_visual_baseline": baseline,
            "visual_attention_density_vs_uniform": float(vm[head] / baseline),
            "visual_mean_per_key": float(vm[head] / nv),
            "nonvisual_mean_per_key": float(nm[head] / nn),
            "visual_nonvisual_density_ratio": float((vm[head] / nv) / (nm[head] / nn + EPSILON)),
            "visual_entropy_nats": float(entropy[head]),
            "visual_entropy_normalized": float(entropy[head] / math.log(nv)) if nv > 1 else 0.,
            "visual_entropy_defined": bool(vm[head] > 0),
            "top10_patch_count": topk,
            "top10_visual_attention_mass": float(topmass[head]),
            "top10_visual_conditional_share": float(topmass[head] / vm[head]) if vm[head] > 0 else 0.,
        })
    return records


def layer_summary(records):
    fields = ["visual_attention_mass", "nonvisual_attention_mass", "uniform_visual_baseline",
              "visual_attention_density_vs_uniform", "visual_nonvisual_density_ratio",
              "visual_entropy_nats", "visual_entropy_normalized", "top10_visual_attention_mass",
              "top10_visual_conditional_share"]
    result = {"layer": records[0]["layer"]}
    for field in fields:
        values = np.array([r[field] for r in records])
        result[field + "_mean"] = float(values.mean())
        result[field + "_std"] = float(values.std())
    return result


def save_figure(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_plots(output_dir, layer_records, visual_rows, processed_image, question, grid_shape, head_records):
    layers = np.arange(len(layer_records))
    mass = np.array([r["visual_attention_mass_mean"] for r in layer_records])
    mass_std = np.array([r["visual_attention_mass_std"] for r in layer_records])
    baseline = layer_records[0]["uniform_visual_baseline_mean"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, mass, marker=".", label="Visual mass: mean of heads")
    ax.fill_between(layers, np.maximum(0, mass - mass_std), np.minimum(1, mass + mass_std), alpha=.2,
                    label="Across-head +/-1 SD (not confidence interval)")
    ax.axhline(baseline, color="gray", linestyle="--", label=f"Uniform baseline={baseline:.4f}")
    ax.set(xlabel="LLM layer (zero-based)", ylabel="Original attention mass", ylim=(0, 1.02), title=question)
    ax.legend(fontsize=8)
    save_figure(fig, output_dir / "layer_visual_attention_mass.png")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, field, label in zip(axes,
            ["visual_attention_density_vs_uniform", "visual_nonvisual_density_ratio"],
            ["Visual mass / uniform baseline", "R: visual / nonvisual density"]):
        values = np.array([r[field + "_mean"] for r in layer_records])
        ax.plot(layers, values, marker=".", label="Mean of per-head values")
        ax.axhline(1, color="gray", linestyle="--", label="Equal density reference = 1")
        ax.set(ylabel=label)
        if field == "visual_nonvisual_density_ratio":
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    axes[0].set_title(question)
    axes[-1].set_xlabel("LLM layer (zero-based)")
    fig.tight_layout()
    save_figure(fig, output_dir / "layer_normalized_attention.png")

    vm = visual_rows.sum(-1)
    # Head map uses original visual mass and its token-count-normalized density.
    ratios = np.array([r["visual_nonvisual_density_ratio"] for r in head_records]).reshape(vm.shape)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    from matplotlib.colors import LogNorm
    for ax, data, label in zip(axes, [vm, vm / baseline, ratios],
            ["Original visual mass", "Visual mass / uniform baseline", "R: density ratio (log color scale)"]):
        options = {"norm": LogNorm(vmin=max(float(ratios[ratios > 0].min()), 1e-12),
                                   vmax=float(ratios.max()))} if data is ratios else {}
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="viridis", **options)
        fig.colorbar(im, ax=ax, label=label)
        ax.set(xlabel="LLM layer (zero-based)", ylabel="Head", title=label)
    fig.suptitle(question)
    fig.tight_layout()
    save_figure(fig, output_dir / "head_layer_attention_heatmap.png")

    for layer in SPATIAL_LAYERS:
        raw = visual_rows[layer].mean(0).reshape(grid_shape)
        np.save(output_dir / f"spatial_attention_layer_{layer:02d}.npy", raw)
        # Local max scaling is strictly for display. Preserve raw probabilities in NPY.
        display = raw / max(float(raw.max()), EPSILON)
        rgb = (plt.get_cmap("inferno")(display)[..., :3] * 255).round().astype(np.uint8)
        Image.fromarray(rgb).save(output_dir / f"spatial_attention_grid_layer_{layer:02d}.png")
        up = np.asarray(Image.fromarray(display.astype(np.float32)).resize(
            processed_image.size, Image.Resampling.BILINEAR))
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
        grid_im = axes[0].imshow(display, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        axes[0].set(title=f"{grid_shape[0]}x{grid_shape[1]} patch grid", xlabel="Patch column", ylabel="Patch row")
        axes[1].imshow(processed_image)
        axes[1].imshow(up, cmap="inferno", vmin=0, vmax=1, alpha=.5)
        axes[1].set_title("Overlay on actual CLIP input (336x336)")
        axes[1].axis("off")
        fig.colorbar(grid_im, ax=axes, fraction=.025, pad=.04, label="Patch weight / layer max (display only)")
        fig.suptitle(f"Layer {layer} | original visual mass={mass[layer]:.6f}\n{question}", fontsize=11)
        save_figure(fig, output_dir / f"spatial_attention_layer_{layer:02d}.png")


def run_probe(model, processor, loading, image, question, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = "prepare_inputs"
    started = time.perf_counter()
    try:
        inputs, counts = prepare_inputs(model, processor, image, question)
        torch.cuda.synchronize()
        prepared = time.perf_counter()
        ids = inputs["input_ids"][0].cpu().numpy()
        mask = inputs["attention_mask"][0].cpu().numpy()
        seq_len = len(ids)
        query = seq_len - 1
        accessible, visual, nonvisual = key_masks(ids, mask, query, model.config.image_token_index)
        all_visual_indices = np.flatnonzero(ids == model.config.image_token_index)
        visual_indices = np.flatnonzero(visual)
        height, width = inputs["pixel_values"].shape[-2:]
        grid = (height // processor.patch_size, width // processor.patch_size)
        text = model.config.text_config
        if inputs["input_ids"].shape[0] != 1 or text.num_hidden_layers != 32 or text.num_attention_heads != 32:
            raise ValueError("This phase-2 experiment requires batch=1, LLaVA 32 layers / 32 heads")
        if grid != (24, 24) or len(all_visual_indices) != 576 or not np.array_equal(all_visual_indices, visual_indices):
            raise ValueError("Expected all 576 image patches accessible to the last prompt token")
        if not np.array_equal(visual_indices, np.arange(visual_indices[0], visual_indices[0] + 576)):
            raise ValueError("Expected one contiguous, row-major CLIP patch block")

        pixels = inputs["pixel_values"][0].float().cpu().numpy().transpose(1, 2, 0)
        ip = processor.image_processor
        pixels = pixels * np.array(ip.image_std) + np.array(ip.image_mean)
        processed_image = Image.fromarray((np.clip(pixels, 0, 1) * 255).round().astype(np.uint8))
        processed_image.save(output_dir / "clip_processed_image.png")

        stage = "control_forward_output_attentions_false"
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            control = model(**inputs, output_attentions=False, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        control_seconds = time.perf_counter() - t0
        control_memory = gpu_memory()
        control_logits = control.logits.float().cpu().numpy()
        if not np.isfinite(control_logits).all() or control.past_key_values is not None:
            raise ValueError("Control logits non-finite or unexpected KV cache")
        del control
        gc.collect()

        stage = "attention_forward_output_attentions_true"
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
        torch.cuda.synchronize()
        attention_seconds = time.perf_counter() - t0
        forward_memory = gpu_memory()
        if outputs.attentions is None or len(outputs.attentions) != 32 or outputs.past_key_values is not None:
            raise ValueError("Expected 32 LLM attentions and no KV cache")
        if list(outputs.image_hidden_states.shape) != [1, 576, text.hidden_size]:
            raise ValueError("Actual projected image features do not match image tokens")
        attention_logits = outputs.logits.float().cpu().numpy()
        if not np.isfinite(attention_logits).all():
            raise ValueError("Non-finite attention-forward logits")
        difference = np.abs(attention_logits - control_logits)
        last_diff = difference[0, query]
        control_next = int(control_logits[0, query].argmax())
        attention_next = int(attention_logits[0, query].argmax())
        comparison = {
            "same_inputs": True, "use_cache_both": False,
            "full_logits_shape": list(attention_logits.shape),
            "full_logits_exact_equal": bool(np.array_equal(control_logits, attention_logits)),
            "full_logits_allclose_atol_1e-4_rtol_1e-4": bool(np.allclose(control_logits, attention_logits, atol=1e-4, rtol=1e-4)),
            "full_logits_max_abs_difference": float(difference.max()),
            "full_logits_mean_abs_difference": float(difference.mean(dtype=np.float64)),
            "last_query_logits_max_abs_difference": float(last_diff.max()),
            "last_query_logits_mean_abs_difference": float(last_diff.mean(dtype=np.float64)),
            "full_prompt_argmax_equal": bool(np.array_equal(control_logits.argmax(-1), attention_logits.argmax(-1))),
            "next_token_argmax_equal": control_next == attention_next,
            "control_next_token_id": control_next, "attention_next_token_id": attention_next,
            "next_token_text": processor.tokenizer.decode([attention_next]),
        }
        if not comparison["full_logits_allclose_atol_1e-4_rtol_1e-4"] or not comparison["next_token_argmax_equal"]:
            write_json(output_dir / "failed_logits_comparison.json", comparison)
            raise ValueError("Attention flag changed logits beyond tolerance or next-token prediction; investigate before interpreting")
        del attention_logits, control_logits, difference, last_diff
        attentions = list(outputs.attentions)
        del outputs  # Release ModelOutput's tuple, logits and image features.
        stage = "per_layer_validation_and_aggregation"
        t0 = time.perf_counter()
        rows, layer_records, validation = [], [], []
        visual_rows, nonvisual_rows = [], []
        for layer in range(len(attentions)):
            matrix = attentions[layer]
            attentions[layer] = None  # Each full layer can be freed after its row is reduced.
            shape = list(matrix.shape)
            if shape != [1, text.num_attention_heads, seq_len, seq_len]:
                raise ValueError(f"Layer {layer}: unexpected shape {shape}")
            finite = bool(torch.isfinite(matrix).all().item())
            minimum = float(matrix.min().item())
            if not finite or minimum < 0:
                raise ValueError(f"Layer {layer}: non-finite/negative full attention")
            query_rows = matrix[0, :, query, :].float().cpu().numpy()
            records = summarize_rows(query_rows, accessible, visual, nonvisual, layer)
            rows.extend(records)
            layer_records.append(layer_summary(records))
            visual_rows.append(query_rows[:, visual].copy())
            nonvisual_rows.append(query_rows[:, nonvisual].copy())
            sums = query_rows[:, accessible].astype(np.float64).sum(-1)
            validation.append({"layer": layer, "shape": shape, "dtype": str(matrix.dtype),
                               "full_matrix_all_finite": finite, "full_matrix_min": minimum,
                               "query_accessible_sum_min": float(sums.min()),
                               "query_accessible_sum_max": float(sums.max()),
                               "query_max_sum_abs_error": float(np.abs(sums - 1).max()),
                               "query_inaccessible_mass_max": float(query_rows[:, ~accessible].sum(-1).max()) if (~accessible).any() else 0.})
            del matrix, query_rows
        del attentions
        gc.collect()
        torch.cuda.synchronize()
        aggregate_seconds = time.perf_counter() - t0
        run_peak_memory = gpu_memory()
        visual_rows = np.stack(visual_rows)
        nonvisual_rows = np.stack(nonvisual_rows)
        np.save(output_dir / "query_visual_attention.npy", visual_rows)
        # Small rows only (32*32*576); never save full S*S matrices.
        np.save(output_dir / "query_nonvisual_attention.npy", nonvisual_rows)
        with (output_dir / "attention_statistics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with (output_dir / "layer_statistics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(layer_records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(layer_records)
        del inputs
        gc.collect()
        cleanup_memory = gpu_memory()
        stage = "plotting"
        t0 = time.perf_counter()
        make_plots(output_dir, layer_records, visual_rows, processed_image, question, grid, rows)
        plot_seconds = time.perf_counter() - t0
        report = {
            "schema_version": 1, "experiment": "phase2_prompt_prefill_last_query",
            "question": question, "image": str(image.resolve()),
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "loading": loading, "input_counts": counts,
            "query": {"index": query, "token_id": int(ids[query]),
                      "token_text": processor.tokenizer.convert_ids_to_tokens(int(ids[query])),
                      "decoded_token": processor.tokenizer.decode([int(ids[query])]),
                      "interpretation": "Final prompt/template token; not the question word or a decoded answer token"},
            "input_ids": ids.tolist(), "attention_mask": mask.tolist(),
            "image_token_index": model.config.image_token_index,
            "visual_key_indices": visual_indices.tolist(),
            "nonvisual_keys": [{"index": int(i), "token_id": int(ids[i]),
                                "token_text": processor.tokenizer.convert_ids_to_tokens(int(ids[i]))}
                               for i in np.flatnonzero(nonvisual)],
            "accessible_key_indices": np.flatnonzero(accessible).tolist(),
            "grid_shape": list(grid), "patch_order": "CLIP row-major, CLS removed, no token permutation",
            "preprocessing": {"original_image_size": counts["original_image_size"],
                "do_resize": ip.do_resize, "size": ip.size, "do_center_crop": ip.do_center_crop,
                "crop_size": ip.crop_size, "image_mean": ip.image_mean, "image_std": ip.image_std,
                "overlay_source": "Denormalized actual FP16 pixel_values; RGB rounded to uint8, not original image coordinates"},
            "definitions": {
                "query_accessible_keys": "attention_mask != 0 and key_index <= query_index; no past cache",
                "visual_attention_mass": "Sum of original probabilities over accessible image-token keys",
                "nonvisual_attention_mass": "Sum over all other accessible keys, including BOS, role/template and query itself",
                "uniform_visual_baseline": "N_visual / (N_visual + N_nonvisual)",
                "visual_attention_density_vs_uniform": "M_visual / uniform_visual_baseline; reference 1",
                "visual_nonvisual_density_ratio": "(M_visual/N_visual)/(M_nonvisual/N_nonvisual + epsilon); reference 1",
                "epsilon": EPSILON,
                "visual_entropy_nats": "-sum(p_visual*ln(p_visual)), p_visual=a_visual/M_visual; zero mass is flagged undefined",
                "visual_entropy_normalized": "Conditional entropy / ln(N_visual)",
                "top10_visual_attention_mass": "Original mass of largest ceil(0.1*N_visual)=58 patch weights, per head",
                "top10_visual_conditional_share": "Top-10% original mass / M_visual",
                "layer_summary": "Arithmetic mean and population SD over heads; mean(R_head) is not ratio(mean masses)",
                "spatial_display": "Head-mean raw patch probabilities / that layer's maximum; only display is scaled",
            },
            "validation": {"passed": True, "attention_layers": len(validation),
                "attention_heads": text.num_attention_heads, "sum_atol": SUM_ATOL,
                "all_visual_tokens_accessible": True, "projected_image_features_shape": [1, 576, text.hidden_size],
                "layers": validation, "logits_comparison": comparison},
            "timings_seconds": {"preprocess": prepared - started, "control_forward": control_seconds,
                "attention_forward": attention_seconds, "validation_and_layer_aggregation": aggregate_seconds,
                "plotting": plot_seconds, "total_excluding_model_load": time.perf_counter() - started,
                "note": "CUDA-synchronized forward only; CPU transfer/comparison/file IO are in total, not forward"},
            "memory": {"control_forward": control_memory, "attention_forward": forward_memory,
                       "attention_forward_and_validation_peak": run_peak_memory, "after_cleanup": cleanup_memory,
                       "note": "MiB; allocated/reserved peaks include resident model; driver process value is a sample, not a peak"},
            "layer_statistics": layer_records, "head_statistics": rows,
            "limitations": "Single image, two prompts, last template-token query, prefill only; descriptive attention, no causal information-flow claims",
        }
        write_json(output_dir / "attention_statistics.json", report)
        print(f"Saved {output_dir}: {len(validation)} layers, [1,32,{seq_len},{seq_len}], "
              f"forward={attention_seconds:.6f}s, peak={run_peak_memory['peak_allocated_mib']:.3f}MiB, "
              f"max logit diff={comparison['full_logits_max_abs_difference']}", flush=True)
        return report
    except torch.OutOfMemoryError:
        write_json(output_dir / "failure.json", {"stage": stage, "error": "CUDA OOM", "memory": gpu_memory()})
        raise RuntimeError(f"CUDA OOM at {stage}; inspect failure.json and GPU occupancy; no quantization/offloading attempted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "test_images/example.jpg")
    parser.add_argument("--question", default="What color is the stop sign?")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/phase2_attention")
    args = parser.parse_args()
    model, processor, loading = load_model()
    run_probe(model, processor, loading, args.image, args.question, args.output_dir)


if __name__ == "__main__":
    main()

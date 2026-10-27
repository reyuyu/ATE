"""Compare two saved phase-2 prompt-prefill probes on the same image; no model load."""
import argparse
import csv
import json
from pathlib import Path

from attention_probe import SPATIAL_LAYERS, EPSILON, save_figure
from vlm_common import write_json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def conditional_spatial_distances(a, b):
    """Head-mean maps conditioned on vision, not averages of per-head entropies."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("Spatial comparison requires positive visual mass")
    p, q = a.astype(np.float64) / a.sum(), b.astype(np.float64) / b.sum()
    m = (p + q) / 2
    def kl(x, y):
        keep = x > 0
        return float(np.sum(x[keep] * np.log(x[keep] / y[keep])))
    return p, q, {
        "conditional_spatial_total_variation": float(.5 * np.abs(p - q).sum()),
        "conditional_spatial_js_divergence_nats": .5 * (kl(p, m) + kl(q, m)),
        "conditional_spatial_cosine_similarity": float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))),
    }


def compare(first, second, output_dir):
    reports = [json.loads((d / "attention_statistics.json").read_text()) for d in [first, second]]
    if reports[0]["image_sha256"] != reports[1]["image_sha256"]:
        raise ValueError("Use the same image for both prompts")
    if reports[0]["grid_shape"] != reports[1]["grid_shape"]:
        raise ValueError("Grid mismatch")
    if reports[0]["loading"]["model_revision"] != reports[1]["loading"]["model_revision"]:
        raise ValueError("Model revision mismatch")
    maps = [np.load(d / "query_visual_attention.npy").astype(np.float64).mean(1) for d in [first, second]]
    if maps[0].shape != maps[1].shape:
        raise ValueError("Attention map shape mismatch")
    images = [np.asarray(Image.open(d / "clip_processed_image.png")) for d in [first, second]]
    if not np.array_equal(*images):
        raise ValueError("Actual CLIP input images differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison, conditional = [], []
    for layer, (a, b) in enumerate(zip(*maps)):
        p, q, distances = conditional_spatial_distances(a, b)
        record = {"layer": layer, "first_visual_mass": float(a.sum()), "second_visual_mass": float(b.sum()),
                  "visual_mass_difference_second_minus_first": float(b.sum() - a.sum()), **distances}
        for field in ["visual_attention_density_vs_uniform", "visual_nonvisual_density_ratio"]:
            record["first_" + field] = reports[0]["layer_statistics"][layer][field + "_mean"]
            record["second_" + field] = reports[1]["layer_statistics"][layer][field + "_mean"]
        comparison.append(record)
        conditional.append((p, q))
    with (output_dir / "question_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)
    write_json(output_dir / "question_comparison.json", {
        "first": {"directory": str(first.resolve()), "question": reports[0]["question"]},
        "second": {"directory": str(second.resolve()), "question": reports[1]["question"]},
        "difference_direction": "second minus first",
        "definitions": {"conditional_spatial": "Normalize each head-mean raw visual map by its own total visual mass",
                        "total_variation": "0.5*sum(abs(p-q)); range [0,1]",
                        "js_divergence_nats": "0.5*(KL(p||m)+KL(q||m)), m=(p+q)/2; range [0,ln(2)]",
                        "cosine_similarity": "dot(p,q)/(norm(p)*norm(q)); no spatial neighborhood weighting"},
        "layers": comparison,
        "limitations": "Descriptive comparison; questions also differ in token count/positions. No object ROI annotation or causal intervention."})

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for report, label in zip(reports, ["First: " + reports[0]["question"], "Second: " + reports[1]["question"]]):
        records = report["layer_statistics"]
        for ax, field in zip(axes, ["visual_attention_mass", "visual_attention_density_vs_uniform", "visual_nonvisual_density_ratio"]):
            ax.plot([x["layer"] for x in records], [x[field + "_mean"] for x in records], marker=".", label=label)
    for ax, ylabel in zip(axes, ["Original visual mass", "Visual mass / uniform baseline", "Mean per-head R (log scale)"]):
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    for report in reports:
        axes[0].axhline(report["layer_statistics"][0]["uniform_visual_baseline_mean"], linestyle="--", alpha=.4)
    axes[0].set_ylim(0, 1.02)
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[2].axhline(1, color="gray", linestyle="--")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("LLM layer (zero-based)")
    fig.tight_layout()
    save_figure(fig, output_dir / "layer_question_comparison.png")

    grid = reports[0]["grid_shape"]
    differences = np.stack([conditional[l][1] - conditional[l][0] for l in SPATIAL_LAYERS]).reshape(-1, *grid)
    vmax = max(float(np.abs(differences).max()), EPSILON)
    fig, axes = plt.subplots(3, 3, figsize=(11, 12))
    for ax, layer, diff in zip(axes.flat, SPATIAL_LAYERS, differences):
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        x = comparison[layer]
        ax.set_title(f"Layer {layer} | TV={x['conditional_spatial_total_variation']:.3f}\n"
                     f"raw mass: {x['first_visual_mass']:.4f} -> {x['second_visual_mass']:.4f}", fontsize=10)
        ax.set(xlabel="Patch column", ylabel="Patch row")
    fig.suptitle("Conditional spatial difference: second minus first\nShared scale; red=second higher, blue=first higher", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=.025, pad=.035, label="Difference in conditional patch probability")
    save_figure(fig, output_dir / "spatial_attention_question_difference.png")

    fig, axes = plt.subplots(3, 6, figsize=(16, 12))
    for i, layer in enumerate(SPATIAL_LAYERS):
        for j in range(2):
            ax = axes[i // 3, 2 * (i % 3) + j]
            raw = maps[j][layer].reshape(grid)
            display = raw / max(float(raw.max()), EPSILON)
            up = np.asarray(Image.fromarray(display.astype(np.float32)).resize((336, 336), Image.Resampling.BILINEAR))
            ax.imshow(images[j])
            ax.imshow(up, cmap="inferno", vmin=0, vmax=1, alpha=.5)
            ax.set_title(f"L{layer}, {'first' if j == 0 else 'second'}\noriginal mass={raw.sum():.4f}", fontsize=10)
            ax.axis("off")
    fig.suptitle("First: " + reports[0]["question"] + "\nSecond: " + reports[1]["question"] +
                 "\nEach map divided by its own maximum for display only", fontsize=12)
    fig.subplots_adjust(top=.88, bottom=.025, left=.01, right=.99, hspace=.30, wspace=.08)
    save_figure(fig, output_dir / "spatial_attention_question_comparison.png")
    print(f"Saved comparison: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.first, args.second, args.output_dir)


if __name__ == "__main__":
    main()

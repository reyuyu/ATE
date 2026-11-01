"""CPU-only plot finalization; does not change frozen inference sources."""
from pathlib import Path

import numpy as np

from .qwen3vl_four_stage_pilot.analyze import bootstrap, csv_read, csv_write
from .qwen3vl_four_stage_pilot.run import (
    LLAVA_MASS, LLAVA_SENS, OUT, read, sha256, source_hashes, write,
)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocol = read(OUT / "protocol.json")
    assert protocol["code_sha256"] == source_hashes()
    assert read(OUT / "execution.json")["status"] == "completed"
    assert all(sha256(path) == value for path, value in protocol["llava_bindings"].items())
    manifest = read(OUT / "sample_manifest.json")["samples"]
    per_question = csv_read(OUT / "attention_mass_per_question.csv")
    lookup = {(r["task"], r["question_id"], int(r["layer"])): r for r in per_question}
    values = np.array([
        [[float(lookup[(s["task"], str(s["question_id"]), layer)][key])
          for key in ("anchor", "image", "prompt", "self")]
         for layer in range(36)] for s in manifest
    ])
    # Historical LLaVA Template includes Self; compare equal text source unions.
    values = np.stack([values[:, :, 0], values[:, :, 1],
                       values[:, :, 2] + values[:, :, 3]], axis=-1)
    assert np.allclose(values.sum(-1), 1, atol=protocol["mass_tolerance"], rtol=0)
    means = values.mean(0)
    intervals = bootstrap(values, [s["image_sha256"] for s in manifest])
    llava_mass = [r for r in csv_read(LLAVA_MASS) if r["group"] == "Overall"]
    llava_sens = csv_read(LLAVA_SENS)
    qwen_sens = csv_read(OUT / "single_layer_sensitivity.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6), constrained_layout=True,
                             sharey="row")
    colors = ["#9e3e75", "#277f8e", "#be782c"]
    source_rows = []
    definitions = [
        ("bos_mass", "BOS", "Start Anchor"),
        ("image_mass", "Image", "Image"),
        ("prompt_mass", "Prompt + Self", "Prompt + Self"),
    ]
    for j, (source, old_label, new_label) in enumerate(definitions):
        rows = sorted([r for r in llava_mass if r["source"] == source],
                      key=lambda r: int(r["layer"]))
        assert len(rows) == 32
        layers = [int(r["layer"]) for r in rows]
        axes[0, 0].plot(layers, [float(r["mean"]) for r in rows],
                        label=old_label, color=colors[j], linewidth=1.4)
        axes[0, 0].fill_between(layers, [float(r["ci95_low"]) for r in rows],
                                [float(r["ci95_high"]) for r in rows],
                                color=colors[j], alpha=.12)
        axes[0, 1].plot(range(36), means[:, j], label=new_label,
                        color=colors[j], linewidth=1.4)
        axes[0, 1].fill_between(range(36), intervals[0, :, j], intervals[1, :, j],
                                color=colors[j], alpha=.12)
        for row in rows:
            source_rows.append(dict(model="LLaVA-1.5-7B", layer=int(row["layer"]),
                source=old_label, mean=float(row["mean"]),
                ci95_low=float(row["ci95_low"]), ci95_high=float(row["ci95_high"]),
                n_questions=int(row["n_questions"]), n_images=int(row["n_images"])))
        for layer in range(36):
            source_rows.append(dict(model="Qwen3-VL-8B", layer=layer, source=new_label,
                mean=float(means[layer, j]), ci95_low=float(intervals[0, layer, j]),
                ci95_high=float(intervals[1, layer, j]), n_questions=120, n_images=120))
    axes[0, 0].set_title("LLaVA-1.5-7B: historical n=3000")
    axes[0, 1].set_title("Qwen3-VL-8B: pilot n=120")
    for ax in axes[0]:
        ax.set_ylim(0, 1)
        ax.set_ylabel("P-1 attention mass")
        ax.legend(frameon=False, ncol=3, fontsize=8)
    old_signed = np.array([-np.mean([float(r["mean_delta_logp"]) for r in llava_sens
                                     if int(r["layer"]) == layer]) for layer in range(32)])
    axes[1, 0].plot(range(32), old_signed, color="#245b84")
    axes[1, 0].set_title("Historical mean; sign converted")
    assert [int(r["layer"]) for r in qwen_sens] == list(range(36))
    axes[1, 1].plot(range(36), [float(r["mean_signed_logp_loss"]) for r in qwen_sens],
                    color="#245b84")
    axes[1, 1].fill_between(range(36), [float(r["signed_ci95_low"]) for r in qwen_sens],
                            [float(r["signed_ci95_high"]) for r in qwen_sens],
                            color="#245b84", alpha=.18)
    axes[1, 1].set_title("Pilot mean and descriptive 95% CI")
    for ax in axes[1]:
        ax.set_ylabel("Signed Full-token LogP loss (nat)")
        ax.axhline(0, color="#666666", linewidth=.6)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
        ax.set_xlabel("Language decoder layer")
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"figure3_llava_vs_qwen.{extension}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    # Mark both observed flip maxima, rather than only the first argmax tie.
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), constrained_layout=True)
    panels = [
        ("mean_signed_logp_loss", "signed", "Signed LogP loss", "Loss (nat)", 1),
        ("mean_absolute_logp_change", "absolute", "Absolute LogP change", "Absolute change (nat)", 1),
        ("top1_flip_rate", "flip", "Top-1 flip rate", "Flip rate (%)", 100),
    ]
    for ax, (key, prefix, title, ylabel, factor) in zip(axes, panels):
        y = np.array([float(r[key]) for r in qwen_sens]) * factor
        ax.plot(range(36), y, color="#245b84", linewidth=1.5)
        ax.fill_between(range(36), [float(r[prefix+"_ci95_low"])*factor for r in qwen_sens],
                        [float(r[prefix+"_ci95_high"])*factor for r in qwen_sens],
                        color="#245b84", alpha=.18)
        for peak in np.flatnonzero(y == y.max()):
            ax.scatter([peak], [y[peak]], s=20, color="#ae3548", zorder=3)
            ax.annotate(f"L{peak}", (peak, y[peak]), xytext=(3, 7),
                        textcoords="offset points", fontsize=8)
        ax.axhline(0, color="#666666", linewidth=.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Language decoder layer")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    for extension in ("pdf", "png"):
        fig.savefig(OUT / f"figure2_attention_sensitivity.{extension}", dpi=350, bbox_inches="tight")
    plt.close(fig)
    csv_write("cross_model_harmonized_mass.csv", source_rows)
    write(OUT / "comparison_partition_compatibility.json", dict(
        passed=True, inference_changed=False, frozen_inference_sources_unchanged=True,
        figure="figure3_llava_vs_qwen", source_sha256=sha256(Path(__file__)),
        sensitivity_peak_labels="All exact maximum ties marked; L17 and L24 have identical Top-1 flip maxima",
        historical_partition_source="/data/lk/ATE-worktrees/first-answer-attention-mass/vlm_research/first_answer_mass/engine.py",
        harmonization="LLaVA historical Prompt includes P-1 Self. Figure3 uses Prompt + Self for both models; Qwen Figure1 and primary tables still separate Self.",
        qwen_combined_interval="10000 image-cluster bootstrap draws of per-question Prompt+Self, preserving covariance; not a sum of marginal intervals",
        anchor_caveat="LLaVA BOS and Qwen Start Anchor are different tokens, not equivalent across architectures",
        no_additional_forward=True))
    print("Figure3 source unions harmonized; frozen inference sources unchanged", flush=True)


if __name__ == "__main__":
    main()

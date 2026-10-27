"""Offline Phase 3 summaries; reads completed runs without loading a model."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    run = json.loads((root / "run.json").read_text())
    if not run["completed"]:
        raise ValueError("Only completed runs can be compared")
    cutoffs = run["cutoffs"]
    metrics = {}
    for name in run["samples_completed"]:
        data = json.loads((root / "samples" / name / "metrics.json").read_text())
        if not data["completed"] or [r["cutoff"] for r in data["summaries"]] != cutoffs:
            raise ValueError(f"Incomplete cutoff sweep: {name}")
        if data["summaries"][-1]["answer_nll_increase"] != 0:
            raise ValueError(f"K32 baseline mismatch: {name}")
        metrics[name] = data
    validation = json.loads((root / "samples" / run["samples_completed"][0] /
                             "correctness_validation.json").read_text())
    if not validation["passed"]:
        raise ValueError("GPU intervention validation failed")
    formal = [r for data in metrics.values()
              for r in data["summaries"] + data["candidate_scores"] + [data["baseline"]]]
    controls = [validation["baseline"]] + validation["controls"]
    checks = validation["baseline_attention"] + [r for c in validation["controls"] for r in c["attention"]]
    summary = {
        "source_run": root.name,
        "samples": len(metrics), "cutoffs": cutoffs,
        "wall_seconds_including_load_validation_plots": run["total_wall_seconds_including_load"],
        "formal_forward_count_including_baselines_and_candidates": len(formal),
        "formal_forward_seconds_sum": sum(r["forward_seconds"] for r in formal),
        "formal_forward_seconds_min": min(r["forward_seconds"] for r in formal),
        "formal_forward_seconds_max": max(r["forward_seconds"] for r in formal),
        "formal_peak_allocated_mib": max(r["memory"]["peak_allocated_mib"] for r in formal),
        "formal_peak_reserved_mib": max(r["memory"]["peak_reserved_mib"] for r in formal),
        "validation_peak_allocated_mib": max(r["memory_after_attention_validation"]["peak_allocated_mib"] for r in controls),
        "validation_peak_reserved_mib": max(r["memory_after_attention_validation"]["peak_reserved_mib"] for r in controls),
        "max_attention_row_sum_error": max(r["max_row_sum_error"] for r in checks),
        "K32_max_abs_logit_error": next(r["max_abs_logit_difference_from_full"]
                                        for r in validation["controls"] if r["name"] == "K32_noop"),
        "sample_nll_delta": {name: [r["answer_nll_increase"] for r in data["summaries"]]
                             for name, data in metrics.items()},
    }
    # Same prompt and answer prefix, different pixels: K0 must remove direct
    # image dependence at every post-image text position in this paired control.
    pair = ["street_spatial_left", "street_spatial_right_mirrored"]
    if all(name in metrics for name in pair):
        scores = [{r["candidate"]: r["sequence_log_probability"]
                   for r in metrics[name]["candidate_scores"] if r["cutoff"] == 0} for name in pair]
        error = max(abs(scores[0][word] - scores[1][word]) for word in scores[0])
        if error != 0:
            raise ValueError(f"Mirrored K0 candidate scores differ: {error}")
        summary["mirrored_K0_shared_candidate_logp_max_abs_error"] = error
    (root / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    values = np.array(list(summary["sample_nll_delta"].values()))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    vmax = max(float(np.abs(values).max()), 1e-8)
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set(xticks=range(len(cutoffs)), xticklabels=cutoffs, xlabel="Cutoff K",
           yticks=range(len(metrics)), yticklabels=list(metrics))
    for i in range(len(metrics)):
        for j in range(len(cutoffs)):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean answer NLL increase (nats/token)")
    fig.tight_layout()
    fig.savefig(root / "all_samples_nll_heatmap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ["street_color", "street_description"]:
        if name not in metrics:
            continue
        data = metrics[name]
        axes[0].plot(cutoffs, [r["answer_nll_increase"] for r in data["summaries"]], "o-", label=name)
        ordinals = sorted({r["answer_token_ordinal"] for r in data["tokens"] if r["is_key_token"]})
        for ordinal in [0] + ordinals:
            rows = [r for r in data["tokens"] if r["answer_token_ordinal"] == ordinal]
            axes[1].plot(cutoffs, [r["token_nll_increase"] for r in rows], "o-",
                         label=f"{name.removeprefix('street_')}:{ordinal}:{rows[0]['token_text']}")
    for ax in axes:
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_xlabel("Cutoff K")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("Mean answer NLL increase (nats/token)")
    axes[1].set_ylabel("Token NLL increase (nats)")
    fig.tight_layout()
    fig.savefig(root / "original_tasks_comparison.png", dpi=160)
    plt.close(fig)

    # Historical Phase 2.1 attention is descriptive and measured with its own
    # FP16 shape. Suffix KO perturbs multiple layers; it is not per-layer causality.
    project = Path(__file__).resolve().parent
    rows = []
    for name, phase2_dir in [("street_color", project / "results/phase2_1_attention"),
                             ("street_description", project / "results/phase2_1_attention/description")]:
        path = phase2_dir / "query_layer_statistics.csv"
        if name not in metrics or not path.exists():
            continue
        with path.open() as f:
            historical = list(csv.DictReader(f))
        for token in metrics[name]["tokens"]:
            if not token["is_key_token"]:
                continue
            selected = [r for r in historical if int(r["query_position"]) == token["query_position"]
                        and int(r["layer"]) >= token["cutoff"]]
            if len(selected) != 32 - token["cutoff"]:
                raise ValueError("Historical query attention mapping incomplete")
            rows.append({"sample": name, "cutoff": token["cutoff"],
                         "answer_token_ordinal": token["answer_token_ordinal"],
                         "token_text": token["token_text"], "query_position": token["query_position"],
                         "historical_suffix_visual_mass_sum": sum(float(r["visual_attention_mass_mean"]) for r in selected),
                         "token_nll_increase": token["token_nll_increase"],
                         "note": "Historical descriptive mass summed across layers; not a probability or isolated layer causal effect"})
    if rows:
        with (root / "historical_attention_vs_knockout.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

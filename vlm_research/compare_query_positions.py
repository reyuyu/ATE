"""Offline Phase 2.1 plots and comparisons from selected query rows; no model load."""
import argparse
import csv
import json
from pathlib import Path
import re

from vlm_common import write_json
from attention_probe import EPSILON, layer_summary, make_plots, save_figure
from compare_attention import conditional_spatial_distances
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_statistics(directory):
    metadata = json.loads((directory / "position_metadata.json").read_text())
    with (directory / "query_layer_statistics.csv").open() as handle:
        records = list(csv.DictReader(handle))
    nquery = len(metadata["queries"])
    fields = [key for key in records[0] if key.endswith("_mean")]
    cube = {key: np.full((32, nquery), np.nan) for key in fields}
    for row in records:
        for field in fields:
            cube[field][int(row["layer"]), int(row["query_axis_index"])] = float(row[field])
    if any(not np.isfinite(x).all() for x in cube.values()):
        raise ValueError("Missing/non-finite layer-query summaries")
    return metadata, cube


def analysis_views(metadata):
    """Analysis views can overlap; they are not the mutually exclusive KEY groups."""
    views = {k: list(v) for k, v in metadata["analysis_groups"].items() if v}
    available = {q["query_position"] for q in metadata["queries"]}
    for match in metadata["content_matches"]:
        name = "question_content:" + match["phrase"]
        views.setdefault(name, [])
        views[name] += [p for p in match["token_indices"] if p in available]
    for match in metadata["semantic_answer_matches"]:
        name = "answer_prediction:" + match["phrase"]
        views.setdefault(name, [])
        views[name] += [metadata["answer_tokens"][i]["query_position"] for i in match["token_indices"]
                        if metadata["answer_tokens"][i]["query_position"] in available]
    if metadata["answer_tokens"]:
        views["answer_prediction:first_target"] = [metadata["answer_tokens"][0]["query_position"]]
    return {key: sorted(set(values)) for key, values in views.items() if values}


def plot_matrix(values, labels, path, title, fields):
    fig, axes = plt.subplots(len(fields), 1, figsize=(max(9, len(labels) * .34), 3.8 * len(fields)), squeeze=False)
    for ax, (field, label) in zip(axes.flat, fields):
        data = values[field]
        vmax = max(1., float(data.max())) if "density" in field else 1.
        im = ax.imshow(data, origin="upper", cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=8)
        ax.set_ylabel("LLM layer (zero-based)")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_title(label)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_figure(fig, path)


def write_groups(directory, metadata, cube, views):
    axis = {q["query_position"]: q["query_axis_index"] for q in metadata["queries"]}
    records = []
    for name, positions in views.items():
        qi = [axis[p] for p in positions]
        for layer in range(32):
            records.append({"view": name, "layer": layer, "query_count": len(qi),
                            **{key: float(data[layer, qi].mean()) for key, data in cube.items()}})
    with (directory / "query_group_layer_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    write_json(directory / "query_group_layer_statistics.json", {
        "aggregation": "Unweighted average of token-specific, per-head-mean statistics; descriptive summary only",
        "views_are_overlapping": True, "query_positions": views, "layers": records})


def plot_positions(directory):
    metadata, cube = load_statistics(directory)
    axis = {q["query_position"]: q["query_axis_index"] for q in metadata["queries"]}
    question = [r for r in metadata["question_tokens"] if r["sequence_index"] in axis]
    if question:
        qi = [axis[r["sequence_index"]] for r in question]
        labels = [r["token_text"] + f"\nq={r['sequence_index']}" for r in question]
        plot_matrix({k: v[:, qi] for k, v in cube.items()}, labels, directory / "layer_question_token_heatmap.png",
            metadata["question"] + "\nEach column is a separate question input-token query; heads averaged",
            [("visual_attention_mass_mean", "Original visual attention mass"),
             ("visual_attention_density_vs_uniform_mean", "Visual mass / query-specific uniform baseline")])
    answer = [r for r in metadata["answer_tokens"] if r["captured"]]
    if answer:
        qi = [axis[r["query_position"]] for r in answer]
        labels = [r["token_text"] + f"\nq{r['query_position']} -> t{r['target_token_position']}" for r in answer]
        plot_matrix({k: v[:, qi] for k, v in cube.items()}, labels, directory / "layer_answer_prediction_heatmap.png",
            "Answer TARGET tokens (attention comes from the previous position)\n" + metadata["answer_source"],
            [("visual_attention_mass_mean", "Original visual attention mass"),
             ("visual_attention_density_vs_uniform_mean", "Visual mass / query-specific uniform baseline"),
             ("bos_attention_mass_mean", "BOS attention mass")])

    views = analysis_views(metadata)
    write_groups(directory, metadata, cube, views)
    fields = [("visual_attention_mass_mean", "Original visual mass"),
              ("visual_attention_density_vs_uniform_mean", "Visual mass / uniform baseline"),
              ("bos_attention_mass_mean", "BOS mass")]
    last = metadata["analysis_groups"]["question_last"]
    start = metadata["analysis_groups"]["answer_start"]
    if last and start:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for label, pos in [("Question last", last[0]), ("Answer-start (predict first target)", start[0])]:
            for ax, (field, ylabel) in zip(axes, fields):
                ax.plot(cube[field][:, axis[pos]], marker=".", label=f"{label}, q={pos}")
                ax.set_ylabel(ylabel)
        for ax in axes:
            ax.legend(fontsize=8)
        axes[1].axhline(1, color="gray", linestyle="--")
        axes[-1].set_xlabel("LLM layer (zero-based)")
        fig.suptitle(metadata["question"])
        fig.tight_layout()
        save_figure(fig, directory / "question_last_vs_answer_start.png")

    preferred = ["question_content", "question_last", "answer_start", "answer_prediction"]
    preferred += [name for name in views if name.startswith("question_content:") or name.startswith("answer_prediction:")]
    selected_views = [name for name in preferred if name in views]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    compare_fields = fields + [("previous_answer_attention_mass_mean", "Previous-answer mass")]
    for name in selected_views:
        qi = [axis[p] for p in views[name]]
        for ax, (field, ylabel) in zip(axes.flat, compare_fields):
            ax.plot(cube[field][:, qi].mean(1), label=name)
            ax.set(xlabel="LLM layer", ylabel=ylabel)
    axes[0, 1].axhline(1, color="gray", linestyle="--")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Query-specific comparisons (token averages are descriptive only)\n" + metadata["question"])
    fig.tight_layout(rect=(0, .15, 1, .93))
    save_figure(fig, directory / "query_type_comparison.png")

    bundle = np.load(directory / "selected_query_attention.npz")
    raw = bundle["attention"]
    vi = bundle["visual_key_indices"]
    # np.take keeps axis order [layer,head,query,patch]. Avoid mixed advanced indexing.
    visual = np.take(raw, vi, axis=-1)
    del raw
    # Spatial differences are conditional on vision and separate from visual MASS.
    spatial_records = []
    head_mean_visual = visual.astype(np.float64).mean(1)
    for family in ["question_all", "answer_prediction"]:
        if family not in views or len(views[family]) < 2:
            continue
        qi = [axis[p] for p in views[family]]
        for layer in range(32):
            distributions = head_mean_visual[layer, qi]
            distances = [conditional_spatial_distances(distributions[i], distributions[j])[2]
                         for i in range(len(qi)) for j in range(i + 1, len(qi))]
            masses = distributions.sum(-1)
            spatial_records.append({"family": family, "layer": layer, "query_count": len(qi),
                "pair_count": len(distances), "visual_mass_range": float(masses.max() - masses.min()),
                "mean_pairwise_conditional_js_nats": float(np.mean([d["conditional_spatial_js_divergence_nats"] for d in distances])),
                "mean_pairwise_conditional_tv": float(np.mean([d["conditional_spatial_total_variation"] for d in distances]))})
    if spatial_records:
        with (directory / "query_specificity_by_layer.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(spatial_records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(spatial_records)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for family in sorted({r["family"] for r in spatial_records}):
            subset = [r for r in spatial_records if r["family"] == family]
            axes[0].plot([r["layer"] for r in subset], [r["visual_mass_range"] for r in subset], label=family)
            axes[1].plot([r["layer"] for r in subset], [r["mean_pairwise_conditional_js_nats"] for r in subset], label=family)
        axes[0].set_ylabel("Across-query visual mass range")
        axes[1].set(ylabel="Mean pairwise conditional spatial JS (nats)", xlabel="LLM layer")
        for ax in axes:
            ax.legend()
        fig.suptitle("Descriptive query variation; different causal contexts, no causal attribution")
        fig.tight_layout()
        save_figure(fig, directory / "query_specificity_by_layer.png")
    image = Image.open(directory / "clip_processed_image.png").convert("RGB")
    semantic = [r for r in answer if r["answer_token_ordinal"] in metadata["semantic_answer_ordinals"]]
    semantic_positions = {r["query_position"] for r in semantic}
    head_records = {p: [] for p in semantic_positions}
    with (directory / "head_statistics.csv").open() as handle:
        for record in csv.DictReader(handle):
            pos = int(record["query_position"])
            if pos in head_records:
                numeric = {key: float(value) for key, value in record.items()
                           if key not in {"query_types", "visual_entropy_defined", "answer_target_token_text"}}
                numeric["layer"], numeric["head"] = int(numeric["layer"]), int(numeric["head"])
                head_records[pos].append(numeric)
    artifacts = []
    for target in semantic:
        pos, ordinal = target["query_position"], target["answer_token_ordinal"]
        token_slug = re.sub(r"[^a-zA-Z0-9]+", "_", target["decoded_token"]).strip("_") or "token"
        out = directory / f"semantic_answer_{ordinal:02d}_{token_slug}"
        out.mkdir(exist_ok=True)
        layer_records = [layer_summary([r for r in head_records[pos] if r["layer"] == l]) for l in range(32)]
        title = f"Predict {target['token_text']} | query={pos}, target={target['target_token_position']}"
        make_plots(out, layer_records, visual[:, :, axis[pos]].astype(np.float32), image, title, [24, 24], head_records[pos])
        artifacts.append({"answer_token_ordinal": ordinal, "token_id": target["token_id"],
                          "query_position": pos, "directory": out.name})
    montage_views = [k for k in ["question_content", "question_last", "answer_start"] if k in views]
    montage_views += [k for k in views if k.startswith("answer_prediction:") and not k.endswith("first_target")][:1]
    if montage_views:
        layers = [4, 12, 20, 31]
        fig, axes = plt.subplots(len(layers), len(montage_views), figsize=(3.3 * len(montage_views), 13), squeeze=False)
        for col, name in enumerate(montage_views):
            qi = [axis[p] for p in views[name]]
            for row, layer in enumerate(layers):
                patch = visual[layer].astype(np.float64)[:, qi, :].mean(axis=(0, 1)).reshape(24, 24)
                display = patch / max(float(patch.max()), EPSILON)
                up = np.asarray(Image.fromarray(display.astype(np.float32)).resize(image.size, Image.Resampling.BILINEAR))
                ax = axes[row, col]
                ax.imshow(image)
                ax.imshow(up, cmap="inferno", vmin=0, vmax=1, alpha=.5)
                ax.set_title(f"L{layer}: {name}\noriginal visual mass={patch.sum():.6f}", fontsize=9)
                ax.axis("off")
        fig.suptitle("Each display map divided by its own maximum; compare original mass separately")
        fig.subplots_adjust(top=.94, bottom=.01, left=.01, right=.99, wspace=.05, hspace=.30)
        save_figure(fig, directory / "query_spatial_comparison.png")
    write_json(directory / "plot_manifest.json", {"semantic_answer_artifacts": artifacts,
        "note": "All spatial panels use actual CLIP input, individual display scaling and original visual mass labels"})
    print(f"Saved query-position plots: {directory}")


def compare_experiments(first, second, output_dir):
    """Compare matching descriptive query views; prompt lengths differ and remain a confound."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [load_statistics(d) for d in [first, second]]
    if sources[0][0]["image_sha256"] != sources[1][0]["image_sha256"]:
        raise ValueError("Experiments must use the same image")
    fields = [("visual_attention_mass_mean", "Original visual mass"),
              ("visual_attention_density_vs_uniform_mean", "Visual mass / per-query baseline"),
              ("bos_attention_mass_mean", "BOS mass")]
    views = ["question_all", "question_last", "answer_start", "answer_prediction"]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), sharex=True)
    records = []
    for j, view in enumerate(views):
        for i, (meta, cube) in enumerate(sources):
            indices = {q["query_position"]: q["query_axis_index"] for q in meta["queries"]}
            qi = [indices[p] for p in meta["analysis_groups"][view]]
            if not qi:
                continue
            for row, (field, ylabel) in enumerate(fields):
                curve = cube[field][:, qi].mean(1)
                axes[row, j].plot(curve, label=f"{'A' if i == 0 else 'B'}: {meta['question']}")
                axes[row, j].set(ylabel=ylabel, xlabel="LLM layer")
                if row == 0:
                    axes[row, j].set_title(view)
                for layer, value in enumerate(curve):
                    records.append({"sample": "A" if i == 0 else "B", "view": view, "layer": layer,
                                    "metric": field, "value": float(value), "query_count": len(qi)})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Question / answer query views across prompts\nDescriptive means; token counts, positions and answer histories differ")
    fig.tight_layout(rect=(0, .06, 1, .94))
    save_figure(fig, output_dir / "cross_question_query_comparison.png")
    with (output_dir / "cross_question_query_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.input_dir:
        plot_positions(args.input_dir)
    elif args.first and args.second and args.output_dir:
        compare_experiments(args.first, args.second, args.output_dir)
    else:
        parser.error("Use --input-dir, or --first --second --output-dir")


if __name__ == "__main__":
    main()

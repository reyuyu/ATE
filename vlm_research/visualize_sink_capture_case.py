"""Offline image overlays for the frozen 15133706 Sink steering experiment.

Only numpy/Pillow/matplotlib and existing artifacts; never loads a model.
Writes an independent child directory and refuses to overwrite existing output.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
ROOT = BASE / "results/attention_sink_steering_case_15133706"
OUT = ROOT / "intuitive_visualization"
P34 = BASE / "results/phase3_4_roi_sink_diagnostic/attention_rows"
CONDITIONS = [
    ("B_roi_f2", "ROI enhancement x2", range(14, 24)),
    ("C24_sink_f0.1", "Deep non-ROI Sink x0.1", range(24, 32)),
    ("D_sink_f0.1", "Early non-ROI Sink x0.1", range(2, 5)),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditional(a):
    a = np.asarray(a, dtype=np.float64)
    assert np.isfinite(a).all() and (a >= 0).all()
    total = a.sum(-1, keepdims=True)
    assert (total > 0).all()
    return a / total


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve completed outputs: {OUT}")
    frozen = read(ROOT / "artifact_sha256.json")["files_sha256"]
    for name, digest in frozen.items():
        assert sha(ROOT / name) == digest, name
    protocol = read(ROOT / "protocol.json")
    positions = read(ROOT / "roi_sink_positions.json")
    status = read(ROOT / "status.json")
    assert status["gpu_completed"] and status["completed_conditions"] == 31
    run = ROOT / status["latest_attempt"]
    with (ROOT / "per_condition_results.csv").open() as f:
        results = {r["condition"]: r for r in csv.DictReader(f)}
    roi, sink, overlap = [positions[k] for k in
        ("roi_indices", "sink_nonroi_indices", "overlap_indices")]
    assert len(roi) == 133 and len(sink) == 13 and overlap == [364]
    assert not set(roi) & set(sink)
    image = np.asarray(Image.open(ROOT / "processor_crop.png").convert("RGB"))
    assert image.shape == (336, 336, 3)
    full_file = P34 / "15133706_task.npz"
    neutral_file = P34 / "15133706_neutral.npz"
    raw = np.load(full_file)["attention"].astype(np.float64)
    neutral = np.load(neutral_file)["attention"].astype(np.float64)
    assert raw.shape == (32, 32, 2, 576)
    assert neutral.shape == (3, 32, 2, 576)
    assert sha(neutral_file) == protocol["sink_entry"]["neutral_npz_sha256"]
    full = raw[:, :, 1]  # P-1, never an already-input answer token.
    inputs = [ROOT / "protocol.json", ROOT / "roi_sink_positions.json",
              ROOT / "processor_crop.png", ROOT / "per_condition_results.csv",
              ROOT / "status.json", ROOT / "artifact_sha256.json", full_file, neutral_file,
              Path(__file__).resolve()]
    records, maps, local = [], {}, {}

    def register(name, weights):
        p = conditional(weights).reshape(-1, 576).mean(0)
        assert np.isclose(p.sum(), 1., atol=1e-12)
        maps[name] = p
        records.append(dict(map=name, raw_visual_mass=float(weights.sum(-1).mean()),
                            roi_share=float(p[roi].sum()), sink_nonroi_share=float(p[sink].sum()),
                            overlap_share=float(p[overlap].sum()),
                            nonroi_sinks_in_top58=len(set(np.argsort(p)[-58:]) & set(sink))))
        return p

    for name, layers in [("Full L2-L4", range(2, 5)), ("Full L14-L23", range(14, 24)),
                         ("Full L27-L31", range(27, 32))]:
        register(name, full[list(layers)])
    # Neutral cache contains head means, not per-head weights. Do not invent heads.
    register("Neutral discovery", neutral[:2])
    register("Neutral held-out", neutral[2:])
    for name, label, layers in CONDITIONS:
        file = run / f"rows_{name}.npz"
        inputs.append(file)
        with np.load(file) as saved:
            assert saved["layers"].tolist() == list(layers)
            a = saved["attention_before_after"].astype(np.float64)
        assert a.shape == (len(layers), 2, 32, 576)
        assert np.allclose(a[:, 0].sum(-1), a[:, 1].sum(-1), rtol=.001, atol=2e-5)
        values = [full[list(layers)], a[:, 0], a[:, 1]]
        local[name] = [register(f"{name}/{stage}", w) for stage, w in
                       zip(["native_full", "local_before", "local_after"], values)]
        for stage in ["before", "after"]:
            r = records[-2 if stage == "before" else -1]
            for key in ["roi", "sink_nonroi", "overlap"]:
                assert abs(r[f"{key}_share"] - float(results[name][f"{key}_{stage}_share"])) < 1e-12
    OUT.mkdir()
    vmin, vmax = 1e-5, max(float(p.max()) for p in maps.values())
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("inferno")
    legend = [Patch(facecolor="none", edgecolor=c, label=t, linewidth=2) for c, t in
              [("cyan", "Boy ROI"), ("magenta", "Girl ROI"),
               ("lime", "13 non-ROI Sink candidates"), ("#ff9e00", "ROI / Sink overlap #364 (not suppressed)")]]

    def draw(ax, p=None, annotate=False, difference=False, diffnorm=None):
        ax.imshow(image, extent=(-.5, 23.5, 23.5, -.5))
        if p is not None:
            ax.imshow(p.reshape(24, 24), norm=diffnorm if difference else norm,
                      cmap="RdBu_r" if difference else cmap, alpha=.68,
                      interpolation="nearest")
        for obj, color in zip(protocol["sample"]["objects"], ["cyan", "magenta"]):
            x0, y0, x1, y1 = np.asarray(obj["crop_xyxy"]) / 14 - .5
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                   edgecolor=color, linewidth=2))
        for indices, color in [(sink, "lime"), (overlap, "#ff9e00")]:
            for i in indices:
                y, x = divmod(i, 24)
                ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, fill=False, edgecolor=color,
                                      linewidth=1.2 if color == "lime" else 2.2))
                if annotate or i == 364:
                    ax.text(x+.6, y+.1, str(i), color="white", fontsize=7,
                            bbox=dict(facecolor="black", alpha=.8, pad=.5, linewidth=0))
        ax.set(xlim=(-.5, 23.5), ylim=(23.5, -.5), xticks=[], yticks=[])

    def shares(p):
        return f"ROI {p[roi].sum():.1%} | non-ROI Sink {p[sink].sum():.1%} | overlap {p[overlap].sum():.1%}"

    # Figure 1: neutral-prompt discovery/held-out stability and real-task hotspots.
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 11.2))
    fig.subplots_adjust(left=.025, right=.92, top=.87, bottom=.12, wspace=.15, hspace=.29)
    draw(axes[0, 0], annotate=True)
    axes[0, 0].set_title("Exact Processor crop / frozen locations\nGold: right; image pixels unchanged", fontsize=11)
    for ax, name in zip([axes[0, 1], axes[0, 2], *axes[1]],
                        ["Neutral discovery", "Neutral held-out", "Full L2-L4", "Full L14-L23", "Full L27-L31"]):
        p = maps[name]
        draw(ax, p)
        count = len(set(np.argsort(p)[-58:]) & set(sink))
        ax.set_title(f"{name}\n{shares(p)}\n{count}/13 non-ROI candidates in top-58 patches", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.94,.21,.014,.54]),
                 label="Visual-conditional probability per patch (shared log scale)")
    fig.legend(handles=legend, loc="lower center", ncol=2, bbox_to_anchor=(.49,.035), fontsize=10)
    fig.suptitle("15133706: do the frozen Sink candidates match recurrent hotspots?\n"
                 "Neutral held-out retention 13/14; Jaccard 0.867. This is stability, not labeled Sink accuracy.", fontsize=14)
    fig.text(.03,.015,"Task: P-1 only; normalize each head, then average all heads/layers. Neutral: normalize saved head means, then average prompts/queries/layers.", fontsize=9)
    fig.savefig(OUT / "sink_capture.png", dpi=180)
    plt.close(fig)

    # Figure 2: true Full, current intervened-forward weights before edit, after edit.
    fig, axes = plt.subplots(3, 3, figsize=(15.8, 16.5))
    fig.subplots_adjust(left=.025, right=.92, top=.90, bottom=.10, wspace=.15, hspace=.32)
    for row, (name, label, layers) in enumerate(CONDITIONS):
        for col, (stage, p) in enumerate(zip(["Native Full", "Local BEFORE edit", "Local AFTER edit"], local[name])):
            draw(axes[row, col], p)
            mass = next(r["raw_visual_mass"] for r in records if r["map"] ==
                        f"{name}/{['native_full','local_before','local_after'][col]}")
            axes[row, col].set_title(f"{stage} | {label}, L{layers.start}-L{layers.stop-1}\n"
                                   f"{shares(p)}\nRaw visual mass = {mass:.4f}", fontsize=10)
        axes[row, 2].text(.5,-.07,f"Result Margin = {float(results[name]['margin']):+.7f}",
                         transform=axes[row,2].transAxes, ha="center", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.94,.22,.014,.53]),
                 label="Visual-conditional probability per patch (same scale as capture figure)")
    fig.legend(handles=legend, loc="lower center", ncol=2, bbox_to_anchor=(.49,.035), fontsize=10)
    fig.suptitle("15133706: original attention vs ROI enhancement / Sink suppression\n"
                 "Answer P-1 only, all 32 heads and every layer in the specified window; native Full Margin = -0.03125", fontsize=14)
    fig.text(.03,.015,"BEFORE is inside the intervened forward: earlier edited layers can change it. AFTER preserves local per-head visual mass; x0.1 is not deletion.\n"
             "Orange #364 belongs to both ROI and frozen Sink: it is never suppressed. Shared color/opacity; no panel-wise rescaling.", fontsize=9)
    fig.savefig(OUT / "intervention_comparison.png", dpi=180)
    plt.close(fig)

    # Signed changes use a separate, explicitly shared linear diverging color scale.
    changes = [local[name][2]-local[name][1] for name, _, _ in CONDITIONS]
    max_change = max(float(np.abs(p).max()) for p in changes)
    dn = TwoSlopeNorm(vmin=-max_change, vcenter=0, vmax=max_change)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.2))
    fig.subplots_adjust(left=.025,right=.92,top=.78,bottom=.22,wspace=.15)
    for ax, change, (name, label, layers) in zip(axes, changes, CONDITIONS):
        draw(ax, change, difference=True, diffnorm=dn)
        ax.set_title(f"{label}, L{layers.start}-L{layers.stop-1}\nAFTER minus local BEFORE", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(norm=dn,cmap="RdBu_r"), cax=fig.add_axes([.94,.25,.014,.47]),
                 label="Change in visual-conditional patch probability (linear)")
    fig.legend(handles=legend,loc="lower center",ncol=2,bbox_to_anchor=(.49,.065),fontsize=10)
    fig.suptitle("Where does the attention go?  Red = gain; blue = loss\n"
                 "These are changes of attention weights, not changes of image pixels or removed tokens.",fontsize=14)
    fig.savefig(OUT / "attention_change.png",dpi=180)
    plt.close(fig)

    with (OUT / "map_statistics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n");writer.writeheader();writer.writerows(records)
    np.savez_compressed(OUT / "plotted_maps.npz", **maps)
    metadata = dict(question_id="15133706",cpu_only=True,new_forward=False,
                    input_sha256={str(p.relative_to(BASE)):sha(p) for p in inputs},
                    old_artifacts_unchanged=len(frozen), normalization="per-head visual-conditional, then head/layer mean; neutral cache has head means",
                    shared_log_color_range=[vmin,vmax],signed_change_range=[-max_change,max_change],
                    visual_mass_preservation="current intervened-forward per head, not equality to native Full",
                    selected_existing_conditions=[c[0] for c in CONDITIONS],
                    selection="fixed representative ROI x2, deep L24-L31 Sink x0.1, early L2-L4 Sink x0.1; no new conditions")
    for name,digest in frozen.items():
        assert sha(ROOT/name)==digest,name
    (OUT / "validation.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(records,indent=2))
    print(f"Verified {len(frozen)} historical artifacts; wrote {OUT}")


if __name__ == "__main__":
    main()

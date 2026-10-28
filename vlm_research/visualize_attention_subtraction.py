"""Cached P-1 attention subtraction for case15133706; CPU plotting only.

Per head: normalize visual rows, subtract Full L2 from Full L22, clip negatives,
normalize the positive residual. This never runs or intervenes on a model.
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
SOURCE = BASE / "results/attention_sink_steering_case_15133706"
CACHE = BASE / "results/phase3_4_roi_sink_diagnostic/attention_rows/15133706_task.npz"
OUT = BASE / "results/attention_subtraction_case_15133706"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite previous outputs: {OUT}")
    historical = load(SOURCE / "artifact_sha256.json")["files_sha256"]
    assert all(sha(SOURCE / name) == digest for name, digest in historical.items())
    protocol = load(SOURCE / "protocol.json")
    positions = load(SOURCE / "roi_sink_positions.json")
    roi, sink, overlap = (positions[k] for k in
                          ["roi_indices", "sink_nonroi_indices", "overlap_indices"])
    assert len(roi) == 133 and len(sink) == 13 and overlap == [364]
    assert not set(roi) & set(sink)
    cache_fingerprints = load(CACHE.parent.parent / "artifact_sha256.json")["files"]
    assert sha(CACHE) == cache_fingerprints["attention_rows/15133706_task.npz"]["sha256"]
    raw = np.load(CACHE)["attention"].astype(np.float64)
    assert raw.shape == (32, 32, 2, 576)
    assert np.isfinite(raw).all() and (raw >= 0).all()
    image = np.asarray(Image.open(SOURCE / "processor_crop.png").convert("RGB"))
    assert image.shape == (336, 336, 3)
    original = raw[[2, 22], :, 1, :]
    mass = original.sum(-1, keepdims=True)
    assert (mass > 0).all()
    per_head = original / mass
    early, late = per_head
    difference = late - early  # beta=1, fixed; not selected after trying strengths.
    assert np.allclose(difference.sum(-1), 0., atol=1e-12)
    residual = np.maximum(difference, 0.)
    residual_sum = residual.sum(-1, keepdims=True)
    if (residual_sum <= 0).any():
        raise ValueError("At least one head has zero positive residual; no fallback invented")
    # Independent identity: positive residual mass = total variation distance.
    assert np.allclose(residual_sum[:, 0], np.abs(difference).sum(-1) / 2, atol=1e-12)
    result = residual / residual_sum
    assert np.isfinite(result).all() and (result >= 0).all()
    assert np.allclose(result.sum(-1), 1., atol=1e-12)
    # Offline, hypothetical restoration of current L22 per-head visual mass only.
    restored = result * mass[1]
    assert np.allclose(restored.sum(-1), mass[1, :, 0], atol=1e-12)
    maps = [early.mean(0), late.mean(0), difference.mean(0), result.mean(0)]
    names = ["Full L2", "Full L22", "L22 minus L2", "Clipped / normalized residual"]

    def summary(name, p):
        positive = p[p > 0]
        return dict(distribution=name, roi_share=float(p[roi].sum()),
                    sink_nonroi_share=float(p[sink].sum()), overlap_share=float(p[overlap].sum()),
                    normalized_entropy=float(-(positive * np.log(positive)).sum()/np.log(576)),
                    max_patch_share=float(p.max()), zero_mean_patches=int((p == 0).sum()),
                    nonroi_sinks_in_top58=len(set(np.argsort(p)[-58:]) & set(sink)))

    summaries = [summary(names[i], maps[i]) for i in [0, 1, 3]]
    head_rows = []
    for head in range(32):
        for name, p in [(names[0], early[head]), (names[1], late[head]), (names[3], result[head])]:
            head_rows.append(dict(head=head, **summary(name, p)))
    # Diagnostic sensitivity only: subtraction after averaging is a different map.
    alt = np.maximum(maps[1] - maps[0], 0.)
    assert alt.sum() > 0
    alt /= alt.sum()
    alternative = summary("Sensitivity: average heads before clipping", alt)
    patch_rows = [dict(patch=i, row=i//24, col=i%24, roi=i in roi, sink_nonroi=i in sink,
                       roi_sink_overlap=i in overlap, l2_share=float(maps[0][i]),
                       l22_share=float(maps[1][i]), signed_difference=float(maps[2][i]),
                       residual_share=float(maps[3][i]),
                       zeroed_head_count=int((result[:, i] == 0).sum())) for i in range(576)]
    OUT.mkdir()
    norm = LogNorm(vmin=1e-5, vmax=max(maps[i].max() for i in [0, 1, 3]))
    bound = float(np.abs(maps[2]).max())
    diffnorm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("black")  # Zero is explicitly black, not a positive log probability.

    def mark(ax):
        for obj, color in zip(protocol["sample"]["objects"], ["cyan", "magenta"]):
            x0, y0, x1, y1 = np.asarray(obj["crop_xyxy"]) / 14 - .5
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                   edgecolor=color, linewidth=1.8))
        for indices, color in [(sink, "lime"), (overlap, "#ff9e00")]:
            for i in indices:
                y, x = divmod(i, 24)
                ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, fill=False,
                                       edgecolor=color, linewidth=1.1 if i != 364 else 2.))
                if i == 364:
                    ax.text(x+.6, y+.1, "364", color="white", fontsize=7,
                            bbox=dict(facecolor="black", alpha=.8, pad=.5, linewidth=0))
        ax.set(xlim=(-.5,23.5), ylim=(23.5,-.5), xticks=[], yticks=[])

    fig, axes = plt.subplots(2, 4, figsize=(19, 10.4))
    fig.subplots_adjust(left=.02, right=.91, top=.83, bottom=.16, wspace=.16, hspace=.22)
    for col, (name, p) in enumerate(zip(names, maps)):
        for row in range(2):
            ax = axes[row, col]
            if row == 0:
                ax.imshow(image, extent=(-.5,23.5,23.5,-.5))
            values = p.reshape(24,24) if col == 2 else np.ma.masked_less_equal(p.reshape(24,24),0)
            ax.imshow(values, cmap="RdBu_r" if col == 2 else cmap,
                      norm=diffnorm if col == 2 else norm, interpolation="nearest",
                      alpha=.68 if row == 0 else 1.)
            mark(ax)
        if col == 2:
            label = "Signed difference; NOT a probability\nRed: increase; blue: decrease"
        else:
            s = summaries[[0,1,3].index(col)]
            label = f"ROI {s['roi_share']:.1%} | non-ROI Sink {s['sink_nonroi_share']:.1%}\nOverlap #364 {s['overlap_share']:.2%} | entropy {s['normalized_entropy']:.3f}"
        axes[0,col].set_title(f"{name}\n{label}", fontsize=10)
    axes[1,0].set_xlabel(f"Native visual mass = {mass[0].mean():.4f}")
    axes[1,1].set_xlabel(f"Native visual mass = {mass[1].mean():.4f}")
    axes[1,2].set_xlabel("Subtract per-head visual-conditional rows")
    axes[1,3].set_xlabel("Clip + renormalize EACH head, then average")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=fig.add_axes([.93,.47,.012,.32]),
                 label="Conditional patch probability: shared log scale (L2, L22, residual)")
    fig.colorbar(plt.cm.ScalarMappable(norm=diffnorm,cmap="RdBu_r"),cax=fig.add_axes([.93,.19,.012,.20]),
                 label="Signed difference: separate linear scale")
    legend = [Patch(facecolor="none",edgecolor=c,label=t,linewidth=2) for c,t in
              [("cyan","Boy ROI"),("magenta","Girl ROI"),("lime","Frozen non-ROI Sink candidates"),
               ("#ff9e00","ROI / Sink overlap #364")]]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(.47,.065), fontsize=10)
    fig.suptitle("15133706: does Full L22 minus Full L2 remove recurring hotspots?\n"
                 "Answer-prediction P-1 only; all 32 heads; beta = 1; upper: image overlay / lower: exact 24 x 24 maps", fontsize=14)
    fig.text(.02,.025,"Offline transformation of cached Full attention only: no model forward, no edited logits, no measured answer improvement.\n"
             "Visual-conditional distributions, not raw attention mass. Colors are shared across positive maps; black = zero / below display floor. ROI / Sink labels do not select subtracted patches.",fontsize=10)
    fig.savefig(OUT / "l22_minus_l2.png", dpi=180)
    plt.close(fig)

    # Direct patch shares make residual frozen hotspots visible without log-color ambiguity.
    selected = sorted(sink + overlap)
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(13,5.2))
    ax.bar(x-.18,100*maps[1][selected],.36,label="Native L22",color="#666666")
    ax.bar(x+.18,100*maps[3][selected],.36,label="Clipped / normalized residual",color="#2279aa")
    ax.set(xticks=x,xticklabels=[str(i) for i in selected],xlabel="Frozen candidate patch index (364 is inside ROI)",
           ylabel="Share of visual-conditional attention (%)",
           title="15133706: some frozen hotspots remain after subtraction\nAll 32 heads; compare normalized distributions, not raw layer masses")
    for tick, i in zip(ax.get_xticklabels(),selected):
        if i == 364:
            tick.set_color("#b36600")
            tick.set_fontweight("bold")
    ax.legend();ax.grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(OUT / "sink_patch_comparison.png",dpi=180);plt.close(fig)

    write_csv(OUT / "distribution_summary.csv",summaries)
    write_csv(OUT / "per_head_statistics.csv",head_rows)
    write_csv(OUT / "per_patch_statistics.csv",patch_rows)
    np.savez_compressed(OUT / "distributions.npz",early_raw=original[0],late_raw=original[1],
                        early_conditional=early,late_conditional=late,signed_difference=difference,
                        positive_residual=residual,residual_conditional=result,
                        hypothetical_l22_mass_restored=restored,alternative_clip_after_head_mean=alt)
    inputs = [CACHE,CACHE.parent.parent/"artifact_sha256.json",SOURCE/"protocol.json",SOURCE/"roi_sink_positions.json",SOURCE/"processor_crop.png",Path(__file__).resolve()]
    validation = dict(question_id="15133706",cpu_only=True,new_forward=False,answer_effect="not measured",
                      source_layer=2,target_layer=22,query="P-1",beta=1.,heads=32,patches=576,
                      layer_indexing="zero-based: L2 is third transformer layer; L22 is twenty-third",
                      formula="p_l=raw_l/sum(raw_l); r_h=max(p22_h-p2_h,0); q_h=r_h/sum(r_h)",
                      aggregation="normalize/subtract/clip/renormalize per head, then mean all 32 heads",
                      cross_layer_head_semantic_correspondence="assumption, not verified",
                      degenerate_heads=int((residual_sum[:,0]<=0).sum()),
                      positive_residual_mass_range=[float(residual_sum.min()),float(residual_sum.max())],
                      row_normalization_passed=True,total_variation_identity_passed=True,
                      hypothetical_visual_mass_restoration_passed=True,
                      alternative_aggregation_sensitivity=alternative,
                      shared_probability_color_range=[norm.vmin,float(norm.vmax)],
                      signed_color_range=[-bound,bound],
                      inputs_sha256={str(p.relative_to(BASE)):sha(p) for p in inputs},
                      historical_files_unchanged=len(historical))
    assert all(sha(SOURCE/name)==digest for name,digest in historical.items())
    (OUT / "validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    print(json.dumps(dict(summaries=summaries,alternative=alternative,
                         retained_nonroi_sink_patches=int(np.count_nonzero(maps[3][sink]>0)),
                         residual_mass_range=validation['positive_residual_mass_range']),indent=2))
    print("WROTE",OUT)


if __name__ == "__main__":
    main()

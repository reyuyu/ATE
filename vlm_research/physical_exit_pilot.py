"""Frozen 12-image Physical Exit pilot. No cache, decoding, training or tuning."""
import argparse
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from vlm_common import ROOT, load_model, prepare_inputs, write_json
from attention_position_probe import locate_question_tokens, encode_answer, append_answer_inputs
from knockout_probe import write_csv, resource_preflight
from physical_exit import (TOLERANCES, measured_forward, aligned, equivalence, cosine_distances)
from prepare_physical_exit_data import sha256
import numpy as np
import torch

CUTOFFS = [0, 8, 16, 24, 32]
OUTPUT = ROOT / "results/phase3_1_physical_exit_pilot"


def sequence_score(logits, token_ids):
    """All answer positions, full vocabulary, complete candidate sequence."""
    lp = logits.double().log_softmax(-1)
    return float(sum(lp[i, tid] for i, tid in enumerate(token_ids)))


def pack_forward(model, inputs, mode, cutoff, text_positions, question_positions, answer_records):
    out = measured_forward(model, inputs, mode, cutoff)
    logits = aligned(out, text_positions, "logits").detach().cpu()
    hidden = aligned(out, text_positions, "hidden").detach().cpu()
    qhidden = aligned(out, question_positions, "hidden").detach().cpu()
    score = sequence_score(aligned(out, [r["query_position"] for r in answer_records], "logits")[0].cpu(),
                           [r["token_id"] for r in answer_records])
    result = {"logits": logits, "hidden": hidden, "question_hidden": qhidden, "logp": score,
              "measurement": out["measurement"], "layers": out["layers"],
              "output_original_positions": out["original_positions"].cpu().tolist()}
    del out
    return result


def candidate_inputs(processor, prompt, labels):
    p = prompt["input_ids"].shape[1]
    records = [encode_answer(processor.tokenizer, label, p) for label in labels]
    size = max(len(r) for r in records)
    results = []
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id
    for rows in records:
        inputs = append_answer_inputs(prompt, [r["token_id"] for r in rows])
        pad = size - len(rows)
        if pad:
            ids = inputs["input_ids"]
            inputs["input_ids"] = torch.cat([ids, torch.full((1, pad), pad_id, device=ids.device, dtype=ids.dtype)], 1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros((1, pad), device=ids.device, dtype=inputs["attention_mask"].dtype)], 1)
        results.append((inputs, rows))
    return results


def evaluate_sample(model, processor, sample, validate_pixel_control=False):
    prompt, counts = prepare_inputs(model, processor, sample["image_path"], sample["question"])
    ids = prompt["input_ids"][0].cpu().numpy()
    qrows, qaudit = locate_question_tokens(processor, sample["question"], ids, model.config.image_token_index)
    qpositions = [r["sequence_index"] for r in qrows]
    labels = sample["candidates"]
    candidates = candidate_inputs(processor, prompt, labels)
    text_positions = [torch.where((x["input_ids"][0] != model.config.image_token_index) & x["attention_mask"][0].bool())[0]
                      for x, records in candidates]
    full = [pack_forward(model, x, "full", 32, positions, qpositions, records)
            for (x, records), positions in zip(candidates, text_positions)]
    target = labels.index(sample["answer"])
    wrong = 1 - target
    full_margin = full[target]["logp"] - full[wrong]["logp"]
    base_correct = full_margin > 0
    metadata = {"sample": sample, "counts": counts, "question_mapping": qrows, "question_audit": qaudit,
                "prompt_ids": ids.tolist(), "candidate_mapping": [
                    {"label": label, "answer_records": records, "input_ids": x["input_ids"][0].cpu().tolist(),
                     "attention_mask": x["attention_mask"][0].cpu().tolist(), "original_valid_text_positions": positions.cpu().tolist()}
                    for label, (x, records), positions in zip(labels, candidates, text_positions)],
                "question_source": "First candidate's common-length teacher-forced forward. Question queries cannot causally see future candidate tokens. Hidden states are after final LLaMA RMSNorm.",
                "full_candidate_correct": base_correct, "full_margin": full_margin,
                "full_candidates": [{"candidate": label, "log_probability": r["logp"], "measurement": r["measurement"], "layers": r["layers"]}
                                    for label, r in zip(labels, full)]}
    rows, details = [], []
    pixel_control = None
    for k in CUTOFFS:
        modes = {}
        for mode in ("mask", "physical"):
            values = [pack_forward(model, x, mode, k, positions, qpositions, records)
                      for (x, records), positions in zip(candidates, text_positions)]
            if k == 32:
                for a, b in zip(full, values):
                    if not torch.equal(a["logits"], b["logits"]) or not torch.equal(a["hidden"], b["hidden"]):
                        raise AssertionError(f"K32 {mode} differs from native Full")
            if validate_pixel_control and mode == "physical" and k == 0:
                x, records = candidates[0]
                altered = dict(x, pixel_values=x["pixel_values"].flip(-1))
                control = pack_forward(model, altered, mode, k, text_positions[0], qpositions, records)
                same = torch.equal(control["hidden"], values[0]["hidden"]) and torch.equal(control["logits"], values[0]["logits"])
                pixel_control = {"passed": same, "method": "Same frozen sample/prompt, horizontally flipped pixel_values tensor at K0; diagnostic only, not a new GQA image", "max_abs_text_logit_error": float((control["logits"].float() - values[0]["logits"].float()).abs().max()), "measurement": control["measurement"]}
                if not same:
                    raise AssertionError("K0 physical text output still depends on image pixels")
            modes[mode] = values
        paired = [equivalence(a["hidden"], b["hidden"], a["logits"], b["logits"]) for a, b in zip(modes["mask"], modes["physical"])]
        logp_error = max(abs(a["logp"] - b["logp"]) for a, b in zip(modes["mask"], modes["physical"]))
        margins = {mode: values[target]["logp"] - values[wrong]["logp"] for mode, values in modes.items()}
        margin_error = abs(margins["mask"] - margins["physical"])
        passed = all(r["passed"] for r in paired) and logp_error <= TOLERANCES["candidate_joint_logp_atol_nats"] and margin_error <= TOLERANCES["candidate_margin_atol_nats"]
        pair = {"question_id": sample["question_id"], "cutoff": k, "passed": passed,
                "candidate_checks": paired, "candidate_logp_max_abs_error": logp_error,
                "margin_abs_error": margin_error}
        if not passed:
            error = AssertionError("Mask/Physical numerical equivalence exceeds frozen tolerances; investigate implementation before interpretation")
            error.equivalence_diagnostic = pair
            raise error
        for mode, values in modes.items():
            drift = cosine_distances(full[0]["question_hidden"], values[0]["question_hidden"])[0]
            margin = margins[mode]
            row = {"question_id": sample["question_id"], "img_id": sample["img_id"], "answer": sample["answer"],
                   "mode": mode, "cutoff": k, "full_correct": base_correct, "candidate_correct": margin > 0,
                   "full_margin": full_margin, "answer_margin": margin, "answer_margin_drop": full_margin - margin,
                   "correct_answer_log_probability": values[target]["logp"],
                   "correct_answer_sequence_probability": float(np.exp(values[target]["logp"])),
                   "correct_probability_within_two_candidates": float(torch.sigmoid(torch.tensor(margin, dtype=torch.float64))),
                   "question_mean_cosine_distance": float(drift.mean()), "question_max_cosine_distance": float(drift.max()),
                   "equivalence_passed": passed, "mask_physical_max_abs_logit_error": max(r["max_abs_logit_error"] for r in paired),
                   "mask_physical_max_text_cosine_distance": max(r["max_text_cosine_distance"] for r in paired),
                   "mask_physical_margin_abs_error": margin_error,
                   "forward_seconds_sum_two_candidates": sum(v["measurement"]["forward_seconds"] for v in values),
                   "peak_allocated_mib": max(v["measurement"]["gpu_memory"]["peak_allocated_mib"] for v in values),
                   "peak_reserved_mib": max(v["measurement"]["gpu_memory"]["peak_reserved_mib"] for v in values)}
            rows.append(row)
            details.append({**row, "equivalence": pair, "question_per_token_cosine_distance": drift.tolist(),
                            "candidates": [{"candidate": label, "token_count": len(records), "log_probability": v["logp"],
                                            "measurement": v["measurement"], "layer_sequence_lengths": v["layers"],
                                            "output_original_positions": v["output_original_positions"]}
                                           for label, (_, records), v in zip(labels, candidates, values)]})
    return rows, {"metadata": metadata, "records": details, "K0_pixel_control": pixel_control}


def summarize(output, rows):
    summary = []
    for cohort in ("all_12", "full_correct"):
        for mode in ("mask", "physical"):
            for k in CUTOFFS:
                selected = [r for r in rows if r["mode"] == mode and r["cutoff"] == k and (cohort == "all_12" or r["full_correct"])]
                summary.append({"cohort": cohort, "mode": mode, "cutoff": k, "n": len(selected),
                    **{key: float(np.mean([r[key] for r in selected])) if selected else None for key in
                       ["candidate_correct", "question_mean_cosine_distance", "answer_margin_drop", "correct_answer_sequence_probability"]}})
    write_json(output / "summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, filename, ylabel in [
        ("question_mean_cosine_distance", "question_representation_drift_vs_K.png", "Mean Question-token cosine distance vs Full"),
        ("answer_margin_drop", "answer_margin_drop_vs_K.png", "Full margin - intervention margin (nats)")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for cohort in ("all_12", "full_correct"):
            for mode in ("mask", "physical"):
                selected = [r for r in summary if r["cohort"] == cohort and r["mode"] == mode]
                if not selected[0]["n"]:
                    continue
                ax.plot(CUTOFFS, [r[key] for r in selected], marker="o" if mode == "physical" else "x",
                        linestyle="-" if mode == "physical" else "--", label=f'{cohort} (n={selected[0]["n"]}), {mode}')
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set(xlabel="K: visual tokens retained through first K LLM layers", ylabel=ylabel, xticks=CUTOFFS)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/phase3_1_data/manifest.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.output_dir.resolve() != OUTPUT.resolve():
        raise ValueError("This fixed pilot writes only to its independent Phase 3.1 directory")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if len(manifest["samples"]) != 12 or len({s["img_id"] for s in manifest["samples"]}) != 12:
        raise ValueError("Pilot requires exactly 12 distinct fixed images")
    registration = {"frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    "manifest_sha256": sha256(args.manifest), "cutoffs": CUTOFFS, "tolerances": TOLERANCES,
                    "candidate_order": ["left", "right"], "question_readout": "Final RMSNorm Question hidden states from first candidate forward; full causal mask prevents future-answer access",
                    "no_timing_claim": "No warmup/repeated trials. Diagnostic wall time includes frontend, decoder, size observers, mask validation/cloning and position hashing; not a stable acceleration benchmark.",
                    "scope": "12 images; full uncached prefill/teacher forcing only. CLIP/projector still run even for K0. No Router, decode, ROI or directed path sweep.",
                    "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    prereg = output / "preregistration.json"
    if args.freeze_only:
        if prereg.exists():
            raise ValueError("Frozen configuration already exists")
        write_json(prereg, registration)
        print("Frozen selection, tolerances and protocol before 7B outputs.")
        return
    frozen = json.loads(prereg.read_text())
    if frozen["manifest_sha256"] != registration["manifest_sha256"] or frozen["tolerances"] != TOLERANCES or frozen["cutoffs"] != CUTOFFS:
        raise ValueError("Frozen selection/tolerances changed")
    if (output / "run.json").exists():
        raise ValueError("Never overwrite an existing GPU pilot")
    for sample in manifest["samples"]:
        if not sample["image_available"] or sha256(sample["image_path"]) != sample["image_sha256"]:
            raise ValueError("Frozen image unavailable or changed; do not resample")
    preflight = resource_preflight()
    write_json(output / "resource_preflight.json", preflight)
    if not preflight["ready"]:
        raise RuntimeError("GPU free memory below existing Phase 3 preflight threshold")
    torch.set_num_threads(4)
    started, stage = time.perf_counter(), "model_loading"
    run = {"completed": False, "samples_completed": [], "preregistration": frozen}
    rows, details = [], []
    try:
        model, processor, loading = load_model()
        if len(model.language_model.model.layers) != 32:
            raise ValueError("Expected the existing 32-layer model")
        run["loading"] = loading
        write_json(output / "run.json", run)
        for index, sample in enumerate(manifest["samples"]):
            stage = sample["question_id"]
            print(f'Running {index + 1}/12: {stage}', flush=True)
            sample_rows, detail = evaluate_sample(model, processor, sample, validate_pixel_control=index == 0)
            rows.extend(sample_rows)
            details.append(detail)
            run["samples_completed"].append(stage)
            write_json(output / "per_sample_results.json", {"completed": False, "samples": details})
            write_csv(output / "per_sample_results.csv", rows)
            write_json(output / "run.json", run)
        summary = summarize(output, rows)
        run.update(completed=True, total_wall_seconds_including_load_plots=time.perf_counter() - started,
                   full_correct_question_ids=[d["metadata"]["sample"]["question_id"] for d in details if d["metadata"]["full_candidate_correct"]])
        write_json(output / "per_sample_results.json", {"completed": True, "samples": details})
        write_json(output / "run.json", run)
        print(f'Completed 12 images; Full-correct={len(run["full_correct_question_ids"])}; {run["total_wall_seconds_including_load_plots"]:.3f}s', flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"stage": stage, "type": type(exc).__name__, "error": str(exc),
                   "equivalence_diagnostic": getattr(exc, "equivalence_diagnostic", None),
                   "completed_samples": run["samples_completed"], "physical_results_claimed_valid": False})
        raise


if __name__ == "__main__":
    main()

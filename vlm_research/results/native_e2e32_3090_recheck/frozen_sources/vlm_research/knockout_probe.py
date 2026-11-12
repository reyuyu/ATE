"""Phase 3: full-prefill, suffix-of-layers Text->Visual attention knockout.

Only masks are intervened on. No weights, token positions, cache, or installed
Transformers files are changed. Import vlm_common before torch.
"""
import argparse
from contextlib import AbstractContextManager
import csv
import hashlib
import inspect
import json
from pathlib import Path
import platform
import re
import subprocess
import time

from vlm_common import (ROOT, REPO, REVISION, default_model, gpu_memory,
                        load_model, prepare_inputs, write_json)
from attention_position_probe import (append_answer_inputs, encode_answer,
                                       locate_question_tokens)
import numpy as np
import torch
import transformers

CUTOFFS = [0, 8, 12, 16, 20, 24, 28, 32]
ROW_ATOL = 0.002


def write_csv(path, records):
    if records:
        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def layout_from_inputs(inputs, image_token_id, expected_visual=576):
    ids, valid = inputs["input_ids"], inputs["attention_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or valid.shape != ids.shape:
        raise ValueError("Only batch=1 full-prefill inputs with a 2D padding mask are supported")
    if not torch.all((valid == 0) | (valid == 1)):
        raise ValueError("Padding mask must contain only 0/1")
    visual = (ids[0] == image_token_id) & valid[0].bool()
    if ((ids[0] == image_token_id) & ~valid[0].bool()).any():
        raise ValueError("Visual image IDs must not be padding")
    indices = torch.where(visual)[0]
    if len(indices) != expected_visual or not len(indices):
        raise ValueError(f"Expected {expected_visual} visual tokens; found {len(indices)}")
    if not torch.equal(indices, torch.arange(indices[0], indices[-1] + 1, device=ids.device)):
        raise ValueError("This single-image experiment requires one contiguous image block")
    positions = torch.arange(ids.shape[1], device=ids.device)
    text_queries = (positions > indices[-1]) & ~visual & valid[0].bool()
    if not text_queries.any():
        raise ValueError("No post-image text queries")
    return {"visual": visual, "text_queries": text_queries, "valid": valid[0].bool(),
            "visual_indices": indices, "sequence_length": ids.shape[1]}


def knockout_mask(original, layout):
    """Clone a real additive causal mask, changing only legal Text->Visual edges."""
    n = layout["sequence_length"]
    if original is None or original.shape != (1, 1, n, n) or not original.is_floating_point():
        raise ValueError(f"Expected a floating additive [1,1,{n},{n}] causal mask")
    if torch.isnan(original).any() or torch.isposinf(original).any():
        raise ValueError("Invalid additive mask")
    # The installed eager implementation produces 0 / finfo.min (or -inf).
    legal = original[0, 0] == 0
    if not torch.all((original == 0) | (original == torch.finfo(original.dtype).min)
                     | torch.isneginf(original)):
        raise ValueError("Unexpected additive bias: inspect the installed mask implementation")
    future = torch.ones((n, n), dtype=torch.bool, device=original.device).triu(1)
    if legal[future].any() or legal[:, ~layout["valid"]].any():
        raise ValueError("Original mask permits future or padding keys")
    blocked = layout["text_queries"][:, None] & layout["visual"][None, :] & legal
    modified = original.clone()
    # -inf guarantees exact zero softmax probability even in FP32 softmax.
    modified[0, 0].masked_fill_(blocked, -torch.inf)
    if not (modified[0, 0] == 0).any(dim=-1).all():
        raise ValueError("Intervention would leave a query with no accessible key")
    if not torch.equal(modified[0, 0][~blocked], original[0, 0][~blocked]):
        raise AssertionError("A connection outside the intervention changed")
    return modified, int(blocked.sum())


class VisualAttentionKnockout(AbstractContextManager):
    """Install one audited pre-hook per LLM layer, including no-op K=L hooks."""
    def __init__(self, layers, inputs, image_token_id, cutoff, expected_visual=576,
                 only_layers=None):
        self.layers = list(layers)
        if not 0 <= cutoff <= len(self.layers):
            raise ValueError("Cutoff must be in [0, number of layers]")
        self.layout = layout_from_inputs(inputs, image_token_id, expected_visual)
        self.cutoff = cutoff
        self.active = set(range(cutoff, len(self.layers))) if only_layers is None else set(only_layers)
        if not self.active.issubset(range(len(self.layers))):
            raise ValueError("Invalid layer selection")
        self.handles, self.records = [], []

    def __enter__(self):
        try:
            for index, layer in enumerate(self.layers):
                module = layer.self_attn
                if module.config._attn_implementation != "eager":
                    raise ValueError("This intervention requires LLaMA eager attention")
                signature = inspect.signature(module.forward)
                if "attention_mask" not in signature.parameters:
                    raise ValueError("Unsupported LlamaAttention.forward signature")
                self.handles.append(module.register_forward_pre_hook(self._hook(index), with_kwargs=True))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _hook(self, index):
        def hook(module, args, kwargs):
            if "attention_mask" not in kwargs or "hidden_states" not in kwargs:
                raise ValueError("Expected installed LlamaDecoderLayer keyword invocation")
            if kwargs.get("past_key_value") is not None or kwargs.get("use_cache", False):
                raise ValueError("KV cache / decode is not supported")
            n = self.layout["sequence_length"]
            hidden = kwargs["hidden_states"]
            if hidden.shape[:2] != (1, n):
                raise ValueError("Query length differs from the complete input")
            positions = kwargs.get("cache_position")
            if positions is None or not torch.equal(positions, torch.arange(n, device=hidden.device)):
                raise ValueError("Only the original full-prefill positions are supported")
            old = kwargs["attention_mask"]
            if old is None or old.shape != (1, 1, n, n):
                raise ValueError("Missing full additive mask")
            touched = index in self.active
            new_kwargs = kwargs
            count = 0
            if touched:
                changed, count = knockout_mask(old, self.layout)
                if changed.data_ptr() == old.data_ptr():
                    raise AssertionError("Shared mask was not cloned")
                new_kwargs = dict(kwargs, attention_mask=changed)
            self.records.append({"layer": index, "applied": touched,
                                 "blocked_connections_per_head": count,
                                 "mask_shape": list(old.shape),
                                 "cloned_mask": touched, "positions_unchanged": True})
            return args, new_kwargs
        return hook

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def assert_one_forward(self):
        if [r["layer"] for r in self.records] != list(range(len(self.layers))):
            raise AssertionError("Expected exactly one invocation of each LLM attention layer")


def model_layers(model):
    return model.language_model.model.layers


def synchronized_forward(model, inputs, attentions=False, cutoff=None, expected_visual=576,
                         only_layers=None):
    cuda = inputs["input_ids"].is_cuda
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before_hooks = [len(x.self_attn._forward_pre_hooks) for x in model_layers(model)]
    started = time.perf_counter()
    intervention = None
    try:
        with torch.inference_mode():
            if cutoff is None:
                output = model(**inputs, output_attentions=attentions, use_cache=False, return_dict=True)
            else:
                with VisualAttentionKnockout(model_layers(model), inputs, model.config.image_token_index,
                                             cutoff, expected_visual, only_layers) as intervention:
                    output = model(**inputs, output_attentions=attentions, use_cache=False, return_dict=True)
                intervention.assert_one_forward()
    except Exception as exc:
        exc.phase3_forward_context = {"cutoff": cutoff, "output_attentions": attentions,
            "input_sequence_length": inputs["input_ids"].shape[1],
            "only_layers": sorted(only_layers) if only_layers is not None else None}
        raise
    if cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if before_hooks != [len(x.self_attn._forward_pre_hooks) for x in model_layers(model)]:
        raise AssertionError("Hook contamination detected")
    if output.past_key_values is not None or not torch.isfinite(output.logits).all():
        raise AssertionError("Unexpected cache or non-finite logits")
    return output, {"forward_seconds": seconds,
                    "memory": gpu_memory() if cuda else None,
                    "hook_audit": intervention.records if intervention else [],
                    "hooks_removed": True, "use_cache": False}


def audit_attention_output(output, layout, heads, active):
    """Validate all rows, then release each full attention tensor; never write matrices."""
    matrices = list(output.attentions or ())
    # ModelOutput also owns tensors in its dict. Attribute assignment of None
    # alone leaves that dict entry alive in Transformers 4.49.0.
    output["attentions"] = None
    n = layout["sequence_length"]
    future = torch.ones((n, n), device=layout["visual"].device, dtype=torch.bool).triu(1)
    blocked = layout["text_queries"][:, None] & layout["visual"][None, :]
    end = int(layout["visual_indices"][-1]) + 1
    records = []
    for layer in range(len(matrices)):
        attn = matrices[layer]
        matrices[layer] = None
        if tuple(attn.shape) != (1, heads, n, n):
            raise AssertionError(f"Unexpected attention shape in layer {layer}: {attn.shape}")
        if not torch.isfinite(attn).all() or (attn < 0).any():
            raise AssertionError("Non-finite or negative attention")
        row_error = float((attn.float().sum(-1) - 1).abs().max())
        if row_error > ROW_ATOL or torch.count_nonzero(attn[0, :, future]):
            raise AssertionError("Normalization or causal-mask violation")
        if torch.count_nonzero(attn[0, :, :, ~layout["valid"]]):
            raise AssertionError("Padding key has nonzero attention")
        blocked_max = float(attn[0, :, blocked].max())
        if layer in active and blocked_max != 0:
            raise AssertionError(f"Nonzero knocked-out attention at layer {layer}")
        records.append({"layer": layer, "shape": list(attn.shape),
                        "max_row_sum_error": row_error, "future_attention_max": 0.0,
                        "blocked_region_max": blocked_max,
                        "full_attention_sha256": tensor_hash(attn),
                        "preimage_and_visual_rows_sha256": tensor_hash(attn[:, :, :end])})
        del attn
    return records


def validate_intervention(model, inputs, expected_visual=576):
    """Single-layer smoke first, then K=L, K=0, K=16, and post-cleanup controls."""
    layers = model_layers(model)
    n_layers, heads = len(layers), model.config.text_config.num_attention_heads
    layout = layout_from_inputs(inputs, model.config.image_token_index, expected_visual)
    baseline, timing = synchronized_forward(model, inputs, True, expected_visual=expected_visual)
    baseline_attn = audit_attention_output(baseline, layout, heads, set())
    if inputs["input_ids"].is_cuda:
        timing["memory_after_attention_validation"] = gpu_memory()
    if len(baseline_attn) != n_layers:
        raise AssertionError("Wrong number of language attention layers")
    baseline_logits = baseline.logits.detach().cpu()
    del baseline
    report = {"scope": "loaded_model", "layers": n_layers, "baseline": timing,
              "baseline_attention": baseline_attn, "controls": []}
    cases = [("single_layer", n_layers - 1, {n_layers - 1}),
             ("K32_noop" if n_layers == 32 else "K_L_noop", n_layers, set()),
             ("K0", 0, set(range(n_layers))),
             ("K16" if n_layers == 32 else "K_mid", n_layers // 2, set(range(n_layers // 2, n_layers)))]
    for name, cutoff, active in cases:
        out, measurement = synchronized_forward(model, inputs, True, cutoff, expected_visual,
                                                 active if name == "single_layer" else None)
        stats = audit_attention_output(out, layout, heads, active)
        if inputs["input_ids"].is_cuda:
            measurement["memory_after_attention_validation"] = gpu_memory()
        if len(stats) != n_layers:
            raise AssertionError("Missing attention layers")
        for index, (a, b) in enumerate(zip(baseline_attn, stats)):
            if a["preimage_and_visual_rows_sha256"] != b["preimage_and_visual_rows_sha256"]:
                raise AssertionError("Attention before/inside the image changed")
            if index < cutoff and a["full_attention_sha256"] != b["full_attention_sha256"]:
                raise AssertionError("A layer before the cutoff changed")
        logits = out.logits.detach().cpu()
        delta = float((logits.float() - baseline_logits.float()).abs().max())
        if cutoff == n_layers and not torch.equal(logits, baseline_logits):
            raise AssertionError(f"No-op hook changed logits by {delta}")
        report["controls"].append({"name": name, "cutoff": cutoff,
            "max_abs_logit_difference_from_full": delta, "attention": stats, **measurement})
        del out, logits
    after, measurement = synchronized_forward(model, inputs, False, expected_visual=expected_visual)
    after_logits = after.logits.detach().cpu()
    if not torch.equal(after_logits, baseline_logits):
        raise AssertionError("Cleanup or output_attentions flag changed the baseline logits")
    report["cleanup_and_output_attentions_control"] = {"max_abs_logit_difference": 0.0, **measurement}
    report["passed"] = True
    del after, after_logits, baseline_logits
    return report


def answer_log_distribution(logits, records):
    queries = torch.tensor([r["query_position"] for r in records], device=logits.device)
    selected = logits[0].index_select(0, queries).detach().cpu().double()
    if not torch.isfinite(selected).all():
        raise ValueError("Non-finite answer logits")
    return torch.log_softmax(selected, dim=-1)


def token_metrics(log_distribution, full_distribution, records, cutoff):
    if log_distribution.shape != full_distribution.shape:
        raise ValueError("KL distributions must have the same shape")
    kl = (full_distribution.exp() * (full_distribution - log_distribution)).sum(-1)
    if not torch.isfinite(kl).all() or float(kl.min()) < -1e-10:
        raise AssertionError("Invalid KL(Full || Knockout)")
    rows = []
    for ordinal, record in enumerate(records):
        target = record["token_id"]
        lp = float(log_distribution[ordinal, target])
        full_lp = float(full_distribution[ordinal, target])
        rows.append({"cutoff": cutoff, **record, "log_probability": lp,
                     "probability": float(np.exp(lp)), "full_log_probability": full_lp,
                     "token_nll_increase": full_lp - lp,
                     "kl_full_to_knockout_nats": max(0.0, float(kl[ordinal]))})
    return rows


def candidate_plans(tokenizer, answer, phrase, alternatives, prompt_length):
    """Tokenize full contextual replacements; score EVERY token in a candidate phrase."""
    if answer.count(phrase) != 1:
        raise ValueError("Candidate phrase must occur exactly once in the reference answer")
    start = answer.index(phrase)
    ref = encode_answer(tokenizer, answer, prompt_length)
    reference_prefix = [r["token_id"] for r in ref if r["answer_char_span"][1] <= start]
    plans = []
    for candidate in dict.fromkeys([phrase, *alternatives]):
        if not candidate or candidate != candidate.strip():
            raise ValueError("Candidates must be nonempty without surrounding whitespace")
        replaced = answer[:start] + candidate + answer[start + len(phrase):]
        records = encode_answer(tokenizer, replaced, prompt_length)
        targets = [r for r in records if r["answer_char_span"][0] < start + len(candidate)
                   and r["answer_char_span"][1] > start]
        prefix = [r["token_id"] for r in records[:targets[0]["answer_token_ordinal"]]]
        if prefix != reference_prefix:
            raise ValueError("Candidate tokenization changed the conditioning prefix")
        # SentencePiece offsets include the leading space in e.g. `▁red`.
        # Accept that whitespace, while rejecting any crossed lexical boundary.
        token_start = targets[0]["answer_char_span"][0]
        token_end = targets[-1]["answer_char_span"][1]
        if replaced[token_start:start].strip() or replaced[start + len(candidate):token_end].strip():
            raise ValueError("Candidate token crosses a reference phrase boundary")
        plans.append({"candidate": candidate, "is_target": candidate == phrase,
                      "answer": replaced, "records": records, "target_records": targets,
                      "candidate_token_ids": [r["token_id"] for r in targets],
                      "conditioning_prefix_ids": prefix})
    if len(plans) < 2:
        raise ValueError("Provide at least one competing candidate")
    return plans


def candidate_inputs(prompt_inputs, plans, pad_token_id):
    max_tokens = max(len(p["records"]) for p in plans)
    prepared = []
    for plan in plans:
        ids = [r["token_id"] for r in plan["records"]]
        inputs = append_answer_inputs(prompt_inputs, ids)
        padding = max_tokens - len(ids)
        if padding:
            inputs["input_ids"] = torch.nn.functional.pad(inputs["input_ids"], (0, padding), value=pad_token_id)
            inputs["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, padding), value=0)
        prepared.append(inputs)
    return prepared


def score_candidates(model, prompt_inputs, plans, cutoff, pad_token_id, expected_visual=576):
    results = []
    for plan, inputs in zip(plans, candidate_inputs(prompt_inputs, plans, pad_token_id)):
        output, measurement = synchronized_forward(model, inputs, False, cutoff, expected_visual)
        distribution = answer_log_distribution(output.logits, plan["target_records"])
        if inputs["input_ids"].is_cuda:
            measurement["memory"] = gpu_memory()
        lp = sum(float(distribution[i, r["token_id"]]) for i, r in enumerate(plan["target_records"]))
        results.append({"cutoff": cutoff, "candidate": plan["candidate"], "is_target": plan["is_target"],
            "token_ids": plan["candidate_token_ids"], "token_count": len(plan["target_records"]),
            "sequence_log_probability": lp, "sequence_probability": float(np.exp(lp)),
            "mean_token_log_probability": lp / len(plan["target_records"]),
            "common_input_length": inputs["input_ids"].shape[1],
            "peak_allocated_mib": measurement["memory"]["peak_allocated_mib"] if measurement["memory"] else None,
            "peak_reserved_mib": measurement["memory"]["peak_reserved_mib"] if measurement["memory"] else None,
            **measurement})
        del output, distribution, inputs
    target = next(r for r in results if r["is_target"])
    competitor = max((r for r in results if not r["is_target"]), key=lambda r: r["sequence_log_probability"])
    return results, {"cutoff": cutoff, "target": target["candidate"],
        "strongest_competitor": competitor["candidate"],
        "target_log_probability": target["sequence_log_probability"],
        "competitor_log_probability": competitor["sequence_log_probability"],
        "log_probability_margin": target["sequence_log_probability"] - competitor["sequence_log_probability"]}


def plot_sample(output_dir, summaries, tokens, margins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cutoffs = [r["cutoff"] for r in summaries]
    for filename, values, ylabel in [
        ("cutoff_answer_nll_increase.png", [r["answer_nll_increase"] for r in summaries], "Mean answer NLL increase (nats/token)"),
        ("cutoff_key_token_log_probability.png", None, "Target token log probability (nats)")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        if values is not None:
            ax.plot(cutoffs, values, marker="o")
            ax.axhline(0, color="gray", linewidth=0.7)
        else:
            ordinals = sorted({r["answer_token_ordinal"] for r in tokens if r.get("is_key_token") or r["answer_token_ordinal"] == 0})
            for ordinal in ordinals:
                rows = [r for r in tokens if r["answer_token_ordinal"] == ordinal]
                ax.plot([r["cutoff"] for r in rows], [r["log_probability"] for r in rows], marker="o",
                        label=f'{ordinal}: {rows[0]["token_text"]}')
            ax.legend(fontsize=8)
        ax.set(xlabel="Cutoff K (layers >= K blocked; K=32 full)", ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
    if margins:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r["cutoff"] for r in margins], [r["log_probability_margin"] for r in margins], marker="o")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(xlabel="Cutoff K", ylabel="Target - strongest competitor log P (nats)")
        fig.tight_layout()
        fig.savefig(output_dir / "cutoff_candidate_margin.png", dpi=160)
        plt.close(fig)
    first = [r for r in tokens if r["cutoff"] == cutoffs[0]]
    values = [[r["token_nll_increase"] for r in tokens if r["cutoff"] == k] for k in cutoffs]
    values = np.asarray(values)
    vmax = max(float(np.abs(values).max()), 1e-8)
    fig, ax = plt.subplots(figsize=(max(8, len(first) * 0.3), 4.8))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set(yticks=range(len(cutoffs)), yticklabels=cutoffs, ylabel="Cutoff K",
           xticks=range(len(first)), xticklabels=[f'{r["answer_token_ordinal"]}:{r["token_text"]}' for r in first])
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.colorbar(im, ax=ax, label="Token NLL increase vs Full (nats)")
    fig.tight_layout()
    fig.savefig(output_dir / "token_knockout_sensitivity_heatmap.png", dpi=160)
    plt.close(fig)


def resource_preflight(required_free_mib=16000):
    report = {"required_free_mib": required_free_mib, "python": platform.python_version(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "cuda_runtime": torch.version.cuda, "model_id": REPO, "model_revision": REVISION,
              "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                                 "--format=csv"], capture_output=True, text=True, timeout=10, check=True)
        report["nvidia_smi"] = result.stdout
        report["processes"] = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one available CUDA GPU")
        free, total = torch.cuda.mem_get_info()
        report.update(logical_device=0, gpu=torch.cuda.get_device_name(0), free_mib=free / 1024**2,
                      total_mib=total / 1024**2)
        report["ready"] = free >= required_free_mib * 1024**2
        if not report["ready"]:
            report["reason"] = "Insufficient free GPU memory before loading; no model load was attempted"
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        report.update(ready=False, reason=str(exc))
    return report


def prepare_sample(model, processor, sample):
    image = ROOT / sample["image"]
    prompt, counts = prepare_inputs(model, processor, image, sample["question"])
    ids = prompt["input_ids"][0].cpu().tolist()
    question_records, alignment = locate_question_tokens(processor, sample["question"], ids, model.config.image_token_index)
    records = encode_answer(processor.tokenizer, sample["answer"], len(ids))
    inputs = append_answer_inputs(prompt, [r["token_id"] for r in records])
    # Compare exact IDs with the saved Phase 2.1 inputs, without modifying them.
    reference = sample.get("phase2_1_reference")
    verified = None
    if reference:
        with np.load(ROOT / reference / "selected_query_attention.npz") as saved:
            if not np.array_equal(saved["input_ids"], inputs["input_ids"][0].cpu().numpy()):
                raise AssertionError("Teacher-forced IDs differ from Phase 2.1")
        verified = True
    for record in records:
        left, right = record["answer_char_span"]
        spans = [match.span() for phrase in sample.get("key_phrases", [])
                 for match in re.finditer(re.escape(phrase), sample["answer"], re.IGNORECASE)]
        record["is_key_token"] = any(start < right and end > left for start, end in spans)
    metadata = {"sample": sample, "counts": counts, "prompt_ids": ids,
                "input_ids": inputs["input_ids"][0].cpu().tolist(),
                "question_mapping": question_records, "answer_mapping": records,
                "alignment": alignment, "phase2_1_exact_ids_match": verified,
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "pixel_values_sha256": tensor_hash(prompt["pixel_values"])}
    return prompt, inputs, records, metadata


def evaluate_sample(model, processor, sample, output_dir, cutoffs=CUTOFFS, validate=False, expected_visual=576):
    output_dir.mkdir(parents=True, exist_ok=False)
    prompt, inputs, records, metadata = prepare_sample(model, processor, sample)
    write_json(output_dir / "input_mapping.json", metadata)
    write_csv(output_dir / "question_mapping.csv", metadata["question_mapping"])
    write_csv(output_dir / "answer_prediction_mapping.csv", records)
    if validate:
        write_json(output_dir / "correctness_validation.json", validate_intervention(model, inputs, expected_visual))
    full, baseline_measurement = synchronized_forward(model, inputs, expected_visual=expected_visual)
    full_distribution = answer_log_distribution(full.logits, records)
    del full
    plans = candidate_plans(processor.tokenizer, sample["answer"], sample["candidate_phrase"],
        sample["candidates"], len(metadata["prompt_ids"])) if sample.get("candidates") else []
    if plans:
        write_json(output_dir / "candidate_mapping.json", plans)
    all_tokens, summaries, candidate_scores, margins = [], [], [], []
    full_mean_nll = -sum(float(full_distribution[i, r["token_id"]]) for i, r in enumerate(records)) / len(records)
    for cutoff in cutoffs:
        output, measurement = synchronized_forward(model, inputs, False, cutoff, expected_visual)
        distribution = answer_log_distribution(output.logits, records)
        if inputs["input_ids"].is_cuda:
            measurement["memory"] = gpu_memory()
        if cutoff == len(model_layers(model)) and not torch.equal(distribution, full_distribution):
            raise AssertionError("Formal K32 sweep differs from unhooked baseline")
        rows = token_metrics(distribution, full_distribution, records, cutoff)
        all_tokens.extend(rows)
        mean_nll = -sum(r["log_probability"] for r in rows) / len(rows)
        summaries.append({"sample_id": sample["id"], "cutoff": cutoff, "answer_tokens": len(rows),
            "answer_mean_nll": mean_nll, "answer_sum_nll": mean_nll * len(rows),
            "full_mean_nll": full_mean_nll, "answer_nll_increase": mean_nll - full_mean_nll,
            "mean_kl_full_to_knockout_nats": float(np.mean([r["kl_full_to_knockout_nats"] for r in rows])),
            "peak_allocated_mib": measurement["memory"]["peak_allocated_mib"] if measurement["memory"] else None,
            "peak_reserved_mib": measurement["memory"]["peak_reserved_mib"] if measurement["memory"] else None,
            **measurement})
        del output, distribution
        if plans:
            pad_id = processor.tokenizer.pad_token_id
            scores, margin = score_candidates(model, prompt, plans, cutoff,
                processor.tokenizer.eos_token_id if pad_id is None else pad_id, expected_visual)
            candidate_scores.extend(scores)
            margins.append(margin)
        # Checkpoint the sweep after EACH cutoff. Partial runs remain explicitly partial.
        write_json(output_dir / "metrics.json", {"completed": False, "cutoffs_done": [r["cutoff"] for r in summaries],
            "summaries": summaries, "tokens": all_tokens, "candidate_scores": candidate_scores, "margins": margins})
    write_csv(output_dir / "token_statistics.csv", all_tokens)
    write_csv(output_dir / "sample_statistics.csv", [{k: v for k, v in r.items() if k not in ("memory", "hook_audit")}
              for r in summaries])
    write_csv(output_dir / "candidate_scores.csv", [{k: v for k, v in r.items() if k not in ("memory", "hook_audit")}
              for r in candidate_scores])
    write_csv(output_dir / "candidate_margins.csv", margins)
    result = {"completed": True, "baseline": baseline_measurement, "summaries": summaries,
              "tokens": all_tokens, "candidate_scores": candidate_scores, "margins": margins,
              "kl_direction": "KL(Full || Knockout), full vocabulary at each reference answer prediction position",
              "nll_definition": "Mean and sum over reference answer tokens only; no prompt loss, no EOS",
              "candidate_definition": "Joint conditional probability of every candidate phrase token under its own prefix; common padded input length across candidates; suffix excluded from score"}
    write_json(output_dir / "metrics.json", result)
    plot_sample(output_dir, summaries, all_tokens, margins)
    return result


def safe_output_dir(path):
    path = path.resolve()
    results = (ROOT / "results").resolve()
    if path == results or path in results.parents:
        raise ValueError("Output directory must not be an ancestor of prior results")
    for name in ("phase2_attention", "phase2_1_attention"):
        protected = results / name
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError("Output must not overwrite prior experiment results")
    if path.exists() and any((path / name).exists() for name in ("run.json", "samples")):
        raise ValueError("Use a new output directory; existing experiments are never overwritten")
    path.mkdir(parents=True, exist_ok=True)
    return path


def audit_manifest_inputs(manifest, output_dir):
    """Real local processor/tokenizer on CPU; no model weights or predictions."""
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor
    from attention_position_probe import render_prompt
    from transformers.models.llama import modeling_llama
    config = AutoConfig.from_pretrained(default_model(), local_files_only=True)
    processor = AutoProcessor.from_pretrained(default_model(), local_files_only=True)
    processor.patch_size = config.vision_config.patch_size
    processor.num_additional_image_tokens = 1
    processor.vision_feature_select_strategy = config.vision_feature_select_strategy
    processor.tokenizer.padding_side = "left"
    report = {"scope": "CPU input audit only, no model loaded or run", "samples": [],
        "llama_attention_signature": str(inspect.signature(modeling_llama.LlamaAttention.forward)),
        "installed_llama_source_sha256": hashlib.sha256(Path(modeling_llama.__file__).read_bytes()).hexdigest(),
        "eager_attention_source": inspect.getsource(modeling_llama.eager_attention_forward)}
    review_dir = output_dir / "image_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for sample in manifest["samples"]:
        image_path = ROOT / sample["image"]
        with Image.open(image_path) as image:
            inputs = processor(images=image.convert("RGB"), text=render_prompt(processor, sample["question"]), return_tensors="pt")
        prompt_ids = inputs["input_ids"][0].tolist()
        layout = layout_from_inputs(inputs, config.image_token_index)
        question_records, alignment = locate_question_tokens(processor, sample["question"], prompt_ids, config.image_token_index)
        answer_records = encode_answer(processor.tokenizer, sample["answer"], len(prompt_ids))
        teacher_ids = prompt_ids + [r["token_id"] for r in answer_records]
        reference_match = None
        if sample.get("phase2_1_reference"):
            with np.load(ROOT / sample["phase2_1_reference"] / "selected_query_attention.npz") as saved:
                if not np.array_equal(saved["input_ids"], teacher_ids):
                    raise AssertionError("CPU audit differs from Phase 2.1 IDs")
            reference_match = True
        plans = candidate_plans(processor.tokenizer, sample["answer"], sample["candidate_phrase"],
                               sample["candidates"], len(prompt_ids)) if sample.get("candidates") else []
        if sample["image"] not in seen:
            # Invert the real CLIP tensor to review exactly the resize/crop used.
            pixels = inputs["pixel_values"][0].numpy().transpose(1, 2, 0)
            pixels = pixels * np.asarray(processor.image_processor.image_std) + np.asarray(processor.image_processor.image_mean)
            filename = f"{len(seen)}_clip_processed.png"
            Image.fromarray(np.uint8(np.round(np.clip(pixels, 0, 1) * 255))).save(review_dir / filename)
            seen[sample["image"]] = filename
        report["samples"].append({"id": sample["id"], "sample": sample,
            "prompt_tokens": len(prompt_ids), "teacher_tokens": len(teacher_ids),
            "visual_token_indices": layout["visual_indices"].tolist(),
            "question_mapping": question_records, "answer_prediction_mapping": answer_records,
            "input_ids": teacher_ids, "phase2_1_exact_ids_match": reference_match,
            "alignment": alignment, "candidate_plans": plans,
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "fp16_pixel_sha256": tensor_hash(inputs["pixel_values"].half()),
            "clip_image": "image_review/" + seen[sample["image"]]})
    report["passed"] = True
    write_json(output_dir / "input_audit.json", report)
    return report


def replot_existing(output_dir):
    for sample_dir in sorted((output_dir / "samples").glob("*")):
        metrics = json.loads((sample_dir / "metrics.json").read_text())
        if not metrics["completed"]:
            raise ValueError(f"Incomplete sample cannot be plotted as complete: {sample_dir}")
        plot_sample(sample_dir, metrics["summaries"], metrics["tokens"], metrics["margins"])


def write_completed_report(output_dir, run):
    lines = ["# Phase 3：Layer-wise Visual Attention Knockout 实测报告", "",
        "以下数值来自本目录已完成的 7B GPU 运行；随机小模型测试不参与结果统计。", "",
        "## 设置", "",
        f"模型 `{REPO}`，revision `{REVISION}`；FP16/eager、batch=1、eval/inference_mode、use_cache=False。",
        "层索引 >= K 的 LLM 层中，所有后图像文本 Query 到视觉 Key 的合法连接被置为 -inf；其他连接 Mask 不变。",
        "答案损失只包括固定参考答案 Token，不包括 Prompt 或 EOS；KL 方向为 KL(Full || Knockout)。",
        "候选评分累加完整候选词序列的条件 log P；候选间使用相同 padding 后输入长度，并共享此前答案 Token。", "",
        "## Cutoff 结果", "",
        "下表为每条样本的 mean answer NLL increase（nats/token）；不能将同图多问当作独立图片。", "",
        "| Sample | " + " | ".join(f"K={k}" for k in CUTOFFS) + " |",
        "| --- | " + " | ".join("---:" for _ in CUTOFFS) + " |"]
    all_measurements = []
    for sample_id in run["samples_completed"]:
        folder = output_dir / "samples" / sample_id
        metrics = json.loads((folder / "metrics.json").read_text())
        by_k = {r["cutoff"]: r for r in metrics["summaries"]}
        lines.append("| " + sample_id + " | " + " | ".join(f'{by_k[k]["answer_nll_increase"]:.6f}' for k in CUTOFFS) + " |")
        all_measurements.extend(metrics["summaries"] + metrics["candidate_scores"])
    lines.extend(["", "## 逐 Token、候选与数值校验", ""])
    for sample_id in run["samples_completed"]:
        metrics = json.loads((output_dir / "samples" / sample_id / "metrics.json").read_text())
        lines.extend([f"### {sample_id}", "",
            f"[逐 Token CSV](samples/{sample_id}/token_statistics.csv) · [完整指标 JSON](samples/{sample_id}/metrics.json)", "",
            f"![NLL](samples/{sample_id}/cutoff_answer_nll_increase.png)", "",
            f"![Key tokens](samples/{sample_id}/cutoff_key_token_log_probability.png)", "",
            f"![Token sensitivity](samples/{sample_id}/token_knockout_sensitivity_heatmap.png)", ""])
        if metrics["margins"]:
            lines.extend([f"![Candidate margin](samples/{sample_id}/cutoff_candidate_margin.png)", ""])
        improved = [r["cutoff"] for r in metrics["summaries"] if r["answer_nll_increase"] < -0.01]
        lines.append(f"相对 Full mean NLL 降低超过 0.01 nats/token 的 cutoff：{improved}。这是原始变化描述，不是显著性检验。")
        ordered = sorted(metrics["summaries"], key=lambda r: r["cutoff"])
        reversals = [(a["cutoff"], b["cutoff"]) for a, b in zip(ordered, ordered[1:])
                     if b["answer_mean_nll"] > a["answer_mean_nll"] + 0.01]
        lines.extend([f"K 增大时 mean NLL 反而升高超过 0.01 的相邻区间：{reversals}。", ""])
    peak = max((m["memory"]["peak_allocated_mib"] for m in all_measurements if m["memory"]), default=0)
    lines.extend(["## 资源与解释边界", "",
        f"正式 sweep/candidate forwards 最大 allocated：{peak:.3f} MiB；各次同步 Forward 时间与显存见 JSON。校验阶段峰值另见首样本 correctness_validation.json。",
        f"总墙钟时间（含加载、校验、绘图）：{run['total_wall_seconds_including_load']:.3f} 秒。",
        "新 QA 为用户委托 AI 目视复核，未声称人工核验；黑色汽车有蓝色反光，颜色标签存在歧义。",
        "空间配对图为水平镜像派生图，文字也反转；不是新增独立自然图。原描述轨迹包含不能确认的 fire hydrant，不能作为完整真值。",
        "当前干预测量固定答案历史下直接视觉访问被屏蔽后的预测变化，含 softmax 重分配和分布外效应。",
        "某个 K 下损失接近 Full 只能说明这些样本/轨迹的边际变化较小；无预注册稳定阈值，不自动判定 Visual Fusion Saturation Depth。",
        "不据此宣称文本吸收全部视觉信息、安全删除视觉 Token 或真实推理加速。未开展 Question-only/Answer-only、Decode、Pruning 或 Router。", ""])
    (output_dir / "phase3_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/phase3_samples.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/phase3_knockout")
    parser.add_argument("--sample-ids", nargs="+", help="Optional subset; color + description first by default")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-inputs-only", action="store_true", help="CPU processor and alignment audit, no model")
    parser.add_argument("--replot-only", action="store_true", help="Offline plots from completed Phase 3 sample JSON")
    args = parser.parse_args()
    if args.replot_only:
        if not (args.output_dir / "run.json").is_file():
            raise ValueError("Replot requires an existing Phase 3 run.json")
        replot_existing(args.output_dir)
        return
    output_dir = safe_output_dir(args.output_dir)
    if args.audit_inputs_only:
        audit_manifest_inputs(json.loads(args.manifest.read_text()), output_dir)
        print(f"CPU input audit saved to {output_dir / 'input_audit.json'}")
        return
    preflight = resource_preflight()
    write_json(output_dir / "resource_preflight.json", preflight)
    if not preflight["ready"]:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"]
    if args.sample_ids:
        if set(args.sample_ids) - {s["id"] for s in samples}:
            raise ValueError("Unknown sample IDs")
        samples = [s for s in samples if s["id"] in args.sample_ids]
    stage = "model_loading"
    started = time.perf_counter()
    try:
        model, processor, loading = load_model()
        if len(model_layers(model)) != 32:
            raise ValueError("This experiment requires the existing 32-layer LLaVA checkpoint")
        run = {"completed": False, "loading": loading, "manifest": manifest,
               "cutoffs": CUTOFFS, "samples_completed": [], "git_base": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        write_json(output_dir / "run.json", run)
        totals = []
        for index, sample in enumerate(samples):
            stage = f'validate_and_sweep:{sample["id"]}'
            result = evaluate_sample(model, processor, sample, output_dir / "samples" / sample["id"], validate=index == 0)
            run["samples_completed"].append(sample["id"])
            totals.extend(result["summaries"])
            write_json(output_dir / "run.json", run)
        run.update(completed=True, total_wall_seconds_including_load=time.perf_counter() - started)
        write_json(output_dir / "run.json", run)
        write_csv(output_dir / "all_samples.csv", [{k: v for k, v in r.items() if k not in ("memory", "hook_audit")} for r in totals])
        write_completed_report(output_dir, run)
        print(f"Completed {len(samples)} samples. Results: {output_dir}")
    except Exception as exc:
        try:
            memory = gpu_memory() if torch.cuda.is_available() else None
        except Exception as memory_error:
            memory = {"sampling_error": str(memory_error)}
        write_json(output_dir / "failure.json", {"stage": stage, "type": type(exc).__name__, "error": str(exc),
            "forward_context": getattr(exc, "phase3_forward_context", None), "memory": memory,
            "partial_results_preserved": True})
        raise


if __name__ == "__main__":
    main()

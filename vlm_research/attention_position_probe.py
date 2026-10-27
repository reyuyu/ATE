"""Phase 2.1: query-specific question and teacher-forced answer attention."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import re
import time

from vlm_common import ROOT, gpu_memory, load_model, prepare_inputs, write_json
from attention_probe import key_masks, summarize_rows, layer_summary, SUM_ATOL
import numpy as np
from PIL import Image
import torch

ATTN_ATOL, ATTN_RTOL = 5e-4, 5e-3
# Sequence-length-dependent FP16 error is validated with an equal-length masked
# control, an absolute probability tolerance AND a bound on each row's L1 error.
CROSS_LENGTH_ATTN_ATOL = 3e-3
MAX_CROSS_LENGTH_ROW_L1 = 2e-2
LOGIT_ATOL, LOGIT_RTOL = 2e-2, 2e-3
GROUPS = ("visual", "bos", "question", "prompt_template", "previous_answer")


def render_prompt(processor, question):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def locate_question_tokens(processor, question, actual_ids, image_token_id):
    """Offsets in the unexpanded template, then ID-verified image-block expansion."""
    tokenizer = processor.tokenizer
    if not tokenizer.is_fast or not question or question != question.strip():
        raise ValueError("Use a fast tokenizer and a nonempty question without leading/trailing whitespace")
    marker = "__ATE_QUESTION_BOUNDARY_71326__"
    if marker in question:
        raise ValueError("Reserved marker in question")
    marked = render_prompt(processor, marker)
    if marked.count(marker) != 1:
        raise ValueError("Cannot uniquely locate question slot in template")
    prefix, suffix = marked.split(marker)
    prompt = render_prompt(processor, question)
    if prompt != prefix + question + suffix:
        raise ValueError("Template transformed question; offsets require investigation")
    start, end = len(prefix), len(prefix) + len(question)
    raw = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    actual_ids = np.asarray(actual_ids)
    nvisual = int((actual_ids == image_token_id).sum())
    if raw["input_ids"].count(image_token_id) != 1 or nvisual != 576:
        raise ValueError("Expected one original image placeholder and 576 actual patches")
    expanded_ids, expanded_offsets = [], []
    for tid, offset in zip(raw["input_ids"], raw["offset_mapping"]):
        copies = nvisual if tid == image_token_id else 1
        expanded_ids.extend([tid] * copies)
        expanded_offsets.extend([offset] * copies)
    if not np.array_equal(expanded_ids, actual_ids):
        raise ValueError("Expanded tokenizer IDs do not exactly match actual Processor input")
    records = []
    for index, (tid, (left, right)) in enumerate(zip(expanded_ids, expanded_offsets)):
        if left < end and right > start:
            if tid in tokenizer.all_special_ids or left < start or right > end:
                raise ValueError("Question overlaps special token/template boundary")
            records.append({"question_token_ordinal": len(records), "sequence_index": index,
                "token_id": tid, "token_text": tokenizer.convert_ids_to_tokens(tid),
                "decoded_token": tokenizer.decode([tid], clean_up_tokenization_spaces=False),
                "prompt_char_span": [left, right], "question_char_span": [left - start, right - start],
                "source_text": prompt[left:right]})
    decoded = tokenizer.decode([r["token_id"] for r in records], clean_up_tokenization_spaces=False)
    covered = np.zeros(len(question), dtype=bool)
    for r in records:
        left, right = r["question_char_span"]
        covered[left:right] = True
    if decoded != question or not covered.all():
        raise ValueError(f"Question token reconstruction mismatch: {decoded!r} != {question!r}")
    return records, {"prompt": prompt, "question_char_span": [start, end],
        "decoded_question_tokens": decoded, "all_question_characters_covered": True,
        "expanded_ids_exactly_match_processor": True,
        "method": "Template marker -> original character offsets -> verified single-image ID expansion"}


def phrase_token_indices(text, token_records, phrases, span_key, index_key):
    selected = set()
    matches = []
    for phrase in phrases:
        if not phrase.strip():
            raise ValueError("Empty content phrase")
        for match in re.finditer(re.escape(phrase), text, re.IGNORECASE):
            indices = [r[index_key] for r in token_records
                       if r[span_key][0] < match.end() and r[span_key][1] > match.start()]
            if not indices:
                raise ValueError(f"Phrase {phrase!r} has no aligned tokens")
            selected.update(indices)
            matches.append({"phrase": phrase, "char_span": list(match.span()), "token_indices": indices})
    return sorted(selected), matches


def encode_answer(tokenizer, answer, prompt_length):
    if not answer or answer != answer.strip():
        raise ValueError("Answer must be nonempty without surrounding whitespace")
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if tokenizer.bos_token_id in ids or any(x in tokenizer.all_special_ids for x in ids):
        raise ValueError("Answer must not contain BOS or other special tokens")
    if tokenizer.decode(ids, clean_up_tokenization_spaces=False) != answer:
        raise ValueError("Answer reconstruction does not match the supplied text")
    return [{"answer_token_ordinal": i, "token_id": tid,
             "token_text": tokenizer.convert_ids_to_tokens(tid),
             "decoded_token": tokenizer.decode([tid], clean_up_tokenization_spaces=False),
             "answer_char_span": list(span), "source_text": answer[span[0]:span[1]],
             "target_token_position": prompt_length + i,
             "query_position": prompt_length - 1 + i}
            for i, (tid, span) in enumerate(zip(ids, encoded["offset_mapping"]))]


def append_answer_inputs(prompt_inputs, answer_ids):
    """Append IDs only, never re-run image processing or insert an extra BOS."""
    inputs = dict(prompt_inputs)
    ids = prompt_inputs["input_ids"]
    mask = prompt_inputs["attention_mask"]
    addition = torch.tensor([answer_ids], device=ids.device, dtype=ids.dtype)
    inputs["input_ids"] = torch.cat([ids, addition], dim=1)
    inputs["attention_mask"] = torch.cat([mask, torch.ones_like(addition, dtype=mask.dtype)], dim=1)
    if not torch.equal(inputs["input_ids"][:, :ids.shape[1]], ids):
        raise ValueError("Teacher-forced prompt prefix changed")
    if not torch.equal(inputs["attention_mask"][:, :mask.shape[1]], mask):
        raise ValueError("Teacher-forced mask prefix changed")
    if inputs["pixel_values"] is not prompt_inputs["pixel_values"]:
        raise ValueError("Teacher-forced pixels changed")
    return inputs


def partition_keys(ids, mask, query, image_token_id, bos_token_id, prompt_length, question_positions):
    accessible, visual, _ = key_masks(ids, mask, query, image_token_id)
    positions = np.arange(len(ids))
    bos = accessible & (np.asarray(ids) == bos_token_id)
    question = accessible & np.isin(positions, question_positions) & ~visual & ~bos
    template = accessible & (positions < prompt_length) & ~visual & ~bos & ~question
    previous_answer = accessible & (positions >= prompt_length) & ~visual & ~bos
    groups = dict(zip(GROUPS, [visual, bos, question, template, previous_answer]))
    coverage = np.stack(list(groups.values())).sum(0)
    if not np.array_equal(coverage, accessible.astype(int)):
        raise ValueError("Key groups must be disjoint and cover exactly the accessible keys")
    return accessible, groups


def select_queries(mode, question_records, content_positions, answer_records, prompt_length):
    qpos = [r["sequence_index"] for r in question_records]
    modes = {"question_all": qpos, "question_last": qpos[-1:], "question_content": content_positions,
             "answer_start": [prompt_length - 1],
             "answer_prediction": [r["query_position"] for r in answer_records]}
    if mode == "question_content" and not content_positions:
        raise ValueError("question_content requires matching --content-phrases")
    if mode in {"all", "answer_prediction"} and not answer_records:
        raise ValueError("Supply --answer for answer_prediction/all mode")
    selected = set(x for values in modes.values() for x in values) if mode == "all" else set(modes[mode])
    # Always capture answer-start for validation; flag it if outside the requested analysis mode.
    selected.add(prompt_length - 1)
    records = []
    for index, pos in enumerate(sorted(selected)):
        types = [key for key, values in modes.items() if pos in values]
        records.append({"query_axis_index": index, "query_position": pos, "query_types": types,
                        "validation_only": mode != "all" and pos not in modes[mode]})
    groups = {key: [p for p in values if p in selected] for key, values in modes.items()}
    return records, groups


def compare_arrays(a, b, atol, rtol):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Comparison requires equal shapes and finite values")
    diff = np.abs(a - b)
    return {"shape": list(a.shape), "exact_equal": bool(np.array_equal(a, b)),
            "max_abs_difference": float(diff.max()), "mean_abs_difference": float(diff.mean()),
            "max_row_l1_difference": float(diff.sum(axis=-1).max()),
            "atol": atol, "rtol": rtol, "allclose": bool(np.allclose(a, b, atol=atol, rtol=rtol))}


def extract_forward(model, inputs, query_positions):
    """One forward; validate full matrices, copy only selected rows, free each layer."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model(**inputs, output_attentions=True, use_cache=False, return_dict=True)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - t0
    forward_memory = gpu_memory()
    seq = inputs["input_ids"].shape[1]
    if output.attentions is None or len(output.attentions) != 32 or output.past_key_values is not None:
        raise ValueError("Expected 32 LLM attention layers without KV cache")
    if list(output.image_hidden_states.shape) != [1, 576, model.config.text_config.hidden_size]:
        raise ValueError("Projected image shape mismatch")
    if not bool(torch.isfinite(output.logits).all().item()):
        raise ValueError("Non-finite logits")
    qi = torch.tensor(query_positions, device=inputs["input_ids"].device)
    logits = output.logits[0, qi].float().cpu().numpy()
    attentions = list(output.attentions)
    del output
    rows = np.empty((32, 32, len(query_positions), seq), dtype=np.float16)
    checks = []
    t0 = time.perf_counter()
    for layer in range(32):
        matrix, attentions[layer] = attentions[layer], None
        if list(matrix.shape) != [1, 32, seq, seq]:
            raise ValueError(f"Layer {layer} unexpected shape: {list(matrix.shape)}")
        if not bool(torch.isfinite(matrix).all().item()) or float(matrix.min().item()) < 0:
            raise ValueError(f"Layer {layer} non-finite/negative full attention matrix")
        rows[layer] = matrix[0, :, qi, :].cpu().numpy()
        checks.append({"layer": layer, "shape": list(matrix.shape), "full_matrix_all_finite": True,
                       "full_matrix_nonnegative": True})
        del matrix
    del attentions, qi
    gc.collect()
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - t0
    memory = gpu_memory()
    return rows, logits, {"forward_seconds": forward_seconds, "matrix_validation_and_cpu_copy_seconds": validation_seconds,
                         "forward_memory": forward_memory, "forward_and_validation_memory": memory, "layers": checks}


def find_phase2_reference(question, image, explicit=None):
    image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    candidates = [explicit] if explicit else [ROOT / "results/phase2_attention", ROOT / "results/phase2_attention/description"]
    for directory in candidates:
        path = directory / "attention_statistics.json"
        if path.exists():
            report = json.loads(path.read_text())
            if report["question"] == question and report["image_sha256"] == image_hash:
                return directory, report
    if explicit:
        raise ValueError("Explicit Phase 2 reference has different question/image or is missing")
    return None, None


def write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def run(args):
    output_dir = args.output_dir.resolve()
    protected = (ROOT / "results/phase2_attention").resolve()
    if output_dir == protected or protected in output_dir.parents or output_dir == (ROOT / "results").resolve():
        raise ValueError("Output must not overwrite Phase 1/2 results")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = "model_loading"
    try:
        model, processor, loading = load_model()
        stage = "input_alignment"
        started = time.perf_counter()
        prompt_inputs, counts = prepare_inputs(model, processor, args.image, args.question)
        tokenizer = processor.tokenizer
        prompt_ids = prompt_inputs["input_ids"][0].cpu().numpy()
        prompt_length = len(prompt_ids)
        question_records, alignment = locate_question_tokens(processor, args.question, prompt_ids, model.config.image_token_index)
        qpositions = [r["sequence_index"] for r in question_records]
        content_positions, content_matches = phrase_token_indices(args.question, question_records, args.content_phrases,
                                                                 "question_char_span", "sequence_index")
        answer_records = encode_answer(tokenizer, args.answer, prompt_length) if args.answer else []
        answer_ids = [r["token_id"] for r in answer_records]
        answer_provenance = {"kind": "caller_provided_reference"}
        if args.answer and args.answer_source == "model_generated":
            archive = ROOT / "results/validation.json"
            archived = json.loads(archive.read_text())["tests"] if archive.exists() else {}
            matched = [(name, test) for name, test in archived.items()
                       if test.get("question") == args.question and test.get("answer") == args.answer]
            if matched:
                name, test = matched[0]
                original_ids = list(test["new_token_ids"])
                if original_ids and original_ids[-1] == tokenizer.eos_token_id:
                    original_ids.pop()
                if original_ids != answer_ids:
                    raise ValueError("Re-encoded answer does not match Phase 1 generated token IDs")
                answer_provenance = {"kind": "phase1_archived_model_output", "file": str(archive),
                                     "test": name, "ids_exactly_match_generated_excluding_eos": True}
            else:
                answer_provenance = {"kind": "caller_labeled_model_generated", "archive_match_verified": False}
        semantic_ordinals, semantic_matches = phrase_token_indices(args.answer or "", answer_records, args.semantic_answer_phrases,
                                                                  "answer_char_span", "answer_token_ordinal")
        queries, analysis_groups = select_queries(args.query_mode, question_records, content_positions, answer_records, prompt_length)
        teacher_inputs = append_answer_inputs(prompt_inputs, answer_ids)
        ids = teacher_inputs["input_ids"][0].cpu().numpy()
        mask = teacher_inputs["attention_mask"][0].cpu().numpy()
        if len(ids) > model.config.text_config.max_position_embeddings:
            raise ValueError("Teacher-forced input exceeds context limit")
        if (ids == tokenizer.bos_token_id).sum() != 1 or (ids == model.config.image_token_index).sum() != 576:
            raise ValueError("BOS or visual tokens duplicated during answer append")
        if list(teacher_inputs["pixel_values"].shape) != [1, 3, 336, 336]:
            raise ValueError("Expected one 336x336 image")
        visual_indices = np.flatnonzero(ids == model.config.image_token_index)
        if not np.array_equal(visual_indices, np.arange(visual_indices[0], visual_indices[0] + 576)):
            raise ValueError("Image block is not contiguous")
        for query in queries:
            pos = query["query_position"]
            query.update({"query_input_token_id": int(ids[pos]), "query_input_token_text": tokenizer.convert_ids_to_tokens(int(ids[pos]))})
        query_positions = [r["query_position"] for r in queries]
        start_axis = query_positions.index(prompt_length - 1)
        # Same inverse normalization as Phase 2; this is the actual model input, not original image coordinates.
        pixels = prompt_inputs["pixel_values"][0].float().cpu().numpy().transpose(1, 2, 0)
        ip = processor.image_processor
        pixels = pixels * np.asarray(ip.image_std) + np.asarray(ip.image_mean)
        Image.fromarray((np.clip(pixels, 0, 1) * 255).round().astype(np.uint8)).save(output_dir / "clip_processed_image.png")
        preprocessing_seconds = time.perf_counter() - started
        stage = "prompt_control_forward"
        control_rows, control_logits, control_run = extract_forward(model, prompt_inputs, [prompt_length - 1])
        stage = "teacher_forced_forward"
        rows, logits, teacher_run = extract_forward(model, teacher_inputs, query_positions)
        consistency = {"teacher_vs_prompt_last_logits": compare_arrays(logits[start_axis], control_logits[0], LOGIT_ATOL, LOGIT_RTOL),
                       "teacher_vs_prompt_last_attention": compare_arrays(rows[:, :, start_axis, :prompt_length], control_rows[:, :, 0], CROSS_LENGTH_ATTN_ATOL, ATTN_RTOL),
                       "strict_cross_length_diagnostic": {"comparison": compare_arrays(rows[:, :, start_axis, :prompt_length], control_rows[:, :, 0], ATTN_ATOL, ATTN_RTOL)}}
        consistency["first_target_argmax_equal"] = bool(logits[start_axis].argmax() == control_logits[0].argmax())
        padded_run = None
        if answer_ids:
            stage = "equal_length_masked_prompt_control"
            padded = dict(teacher_inputs)
            padded["input_ids"] = teacher_inputs["input_ids"].clone()
            padded["attention_mask"] = teacher_inputs["attention_mask"].clone()
            padded["input_ids"][:, prompt_length:] = tokenizer.pad_token_id
            padded["attention_mask"][:, prompt_length:] = 0
            padded_rows, padded_logits, padded_run = extract_forward(model, padded, [prompt_length - 1])
            consistency["same_length_masked_prompt_vs_teacher_logits"] = compare_arrays(padded_logits[0], logits[start_axis], 0, 0)
            consistency["same_length_masked_prompt_vs_teacher_prefix_attention"] = compare_arrays(
                padded_rows[:, :, 0, :prompt_length], rows[:, :, start_axis, :prompt_length], 0, 0)
            cache_positions = torch.arange(len(ids), device=teacher_inputs["input_ids"].device)
            mask_options = dict(sequence_length=len(ids), target_length=len(ids), dtype=torch.float16,
                device=cache_positions.device, cache_position=cache_positions, batch_size=1)
            build_mask = model.language_model.model._prepare_4d_causal_attention_mask_with_cache_position
            teacher_mask = build_mask(teacher_inputs["attention_mask"], **mask_options)
            padded_mask = build_mask(padded["attention_mask"], **mask_options)
            consistency["equal_length_actual_prefix_causal_masks_equal"] = bool(torch.equal(
                teacher_mask[:, :, :prompt_length], padded_mask[:, :, :prompt_length]))
            del padded, padded_rows, padded_logits, teacher_mask, padded_mask, cache_positions
        reference_dir, reference = find_phase2_reference(args.question, args.image, args.phase2_reference)
        if reference is not None:
            if reference["loading"]["model_revision"] != loading["model_revision"] or reference["input_ids"] != prompt_ids.tolist():
                raise ValueError("Phase 2 model/input IDs changed")
            ref_visual = np.load(reference_dir / "query_visual_attention.npy")
            ref_nonvisual = np.load(reference_dir / "query_nonvisual_attention.npy")
            nonvisual_indices = np.array([r["index"] for r in reference["nonvisual_keys"]])
            consistency["prompt_vs_phase2_visual"] = compare_arrays(control_rows[:, :, 0][:, :, visual_indices], ref_visual, ATTN_ATOL, ATTN_RTOL)
            consistency["teacher_vs_phase2_visual"] = compare_arrays(rows[:, :, start_axis][:, :, visual_indices], ref_visual, CROSS_LENGTH_ATTN_ATOL, ATTN_RTOL)
            consistency["teacher_vs_phase2_nonvisual"] = compare_arrays(rows[:, :, start_axis][:, :, nonvisual_indices], ref_nonvisual, CROSS_LENGTH_ATTN_ATOL, ATTN_RTOL)
            if not np.array_equal(np.asarray(Image.open(reference_dir / "clip_processed_image.png")),
                                  np.asarray(Image.open(output_dir / "clip_processed_image.png"))):
                raise ValueError("Actual CLIP image differs from Phase 2")
        consistency["cross_length_row_l1_limit"] = MAX_CROSS_LENGTH_ROW_L1
        consistency["cross_length_row_l1_passed"] = consistency["teacher_vs_prompt_last_attention"]["max_row_l1_difference"] <= MAX_CROSS_LENGTH_ROW_L1
        checks_passed = all(v["allclose"] for v in consistency.values() if isinstance(v, dict) and "allclose" in v)
        checks_passed = checks_passed and consistency["first_target_argmax_equal"] and consistency["cross_length_row_l1_passed"]
        checks_passed = checks_passed and consistency.get("equal_length_actual_prefix_causal_masks_equal", True)
        write_json(output_dir / "forward_consistency.json", consistency)
        if not checks_passed:
            raise ValueError("Prompt/teacher/Phase2 mismatch: investigate input IDs, masks, positions and numerical precision before interpreting")
        stage = "query_statistics"
        summaries, query_validation = [], []
        head_csv = (output_dir / "head_statistics.csv").open("w", newline="")
        writer = None
        targets_by_query = {t["query_position"]: t for t in answer_records}
        try:
            for layer in range(32):
                for query in queries:
                    qi, pos = query["query_axis_index"], query["query_position"]
                    accessible, groups = partition_keys(ids, mask, pos, model.config.image_token_index,
                        tokenizer.bos_token_id, prompt_length, qpositions)
                    if int(groups["visual"].sum()) != 576:
                        raise ValueError("Query cannot access all 576 visual tokens")
                    r = rows[layer, :, qi, :].astype(np.float64)
                    stats = summarize_rows(r, accessible, groups["visual"], accessible & ~groups["visual"], layer)
                    for name, group in groups.items():
                        mass = r[:, group].sum(-1)
                        for head, stat in enumerate(stats):
                            stat[name + "_attention_mass"] = float(mass[head])
                            stat[name + "_key_count"] = int(group.sum())
                    for stat in stats:
                        target = targets_by_query.get(pos)
                        stat.update({"query_position": pos, "query_axis_index": qi,
                            "query_types": "|".join(query["query_types"]),
                            "answer_target_token_id": target["token_id"] if target else "",
                            "answer_target_token_text": target["token_text"] if target else "",
                            "answer_target_position": target["target_token_position"] if target else "",
                            "prompt_text_attention_mass": stat["question_attention_mass"] + stat["prompt_template_attention_mass"],
                            "other_text_attention_mass": stat["question_attention_mass"] + stat["prompt_template_attention_mass"] + stat["previous_answer_attention_mass"]})
                        if not np.isclose(sum(stat[g + "_attention_mass"] for g in GROUPS), stat["accessible_attention_sum"], atol=1e-12, rtol=0):
                            raise ValueError("Grouped mass does not cover accessible attention")
                    if writer is None:
                        writer = csv.DictWriter(head_csv, fieldnames=list(stats[0]), lineterminator="\n")
                        writer.writeheader()
                    writer.writerows(stats)
                    summary = layer_summary(stats)
                    summary.update({"query_axis_index": qi, "query_position": pos})
                    for field in [g + "_attention_mass" for g in GROUPS if g != "visual"] + ["prompt_text_attention_mass", "other_text_attention_mass"]:
                        values = np.array([s[field] for s in stats])
                        summary[field + "_mean"], summary[field + "_std"] = float(values.mean()), float(values.std())
                    summaries.append(summary)
                    query_validation.append({"layer": layer, "query_position": pos,
                        "max_sum_abs_error": float(np.abs(r[:, accessible].sum(-1) - 1).max()),
                        "future_or_padding_mass_max": float(r[:, ~accessible].sum(-1).max()) if (~accessible).any() else 0.,
                        "accessible_keys": int(accessible.sum()), "group_counts": {k: int(v.sum()) for k, v in groups.items()},
                        "groups_disjoint_and_exhaustive": True})
        finally:
            head_csv.close()
        for target in answer_records:
            pos = target["query_position"]
            target["captured"] = pos in query_positions
            if target["captured"]:
                prediction = logits[query_positions.index(pos)]
                target["argmax_token_id"] = int(prediction.argmax())
                target["argmax_matches_target"] = bool(target["argmax_token_id"] == target["token_id"])
                z = prediction.astype(np.float64) - float(prediction.max())
                target["target_probability"] = float(np.exp(z[target["token_id"]]) / np.exp(z).sum())
        write_csv(output_dir / "question_token_mapping.csv", question_records)
        if answer_records:
            write_csv(output_dir / "answer_prediction_mapping.csv", answer_records)
        write_csv(output_dir / "query_layer_statistics.csv", summaries)
        np.savez_compressed(output_dir / "selected_query_attention.npz", attention=rows,
                            query_positions=query_positions, input_ids=ids, visual_key_indices=visual_indices)
        statistics_seconds = time.perf_counter() - started - preprocessing_seconds - sum(
            x["forward_seconds"] + x["matrix_validation_and_cpu_copy_seconds"] for x in [control_run, teacher_run])
        del prompt_inputs, teacher_inputs, rows, logits, control_rows, control_logits
        gc.collect()
        cleanup_memory = gpu_memory()
        metadata = {"schema_version": 1, "stage": "phase2_1", "question": args.question, "answer": args.answer,
            "answer_source": args.answer_source if args.answer else None,
            "answer_provenance": answer_provenance,
            "image": str(args.image.resolve()), "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
            "loading": loading, "input_counts": counts, "prompt_length": prompt_length, "teacher_forced_length": len(ids),
            "input_ids": ids.tolist(), "attention_mask": mask.tolist(), "image_token_index": model.config.image_token_index,
            "bos_token_id": tokenizer.bos_token_id, "visual_key_indices": visual_indices.tolist(), "grid_shape": [24, 24],
            "question_alignment": alignment, "question_tokens": question_records, "answer_tokens": answer_records,
            "query_mode": args.query_mode, "queries": queries, "analysis_groups": analysis_groups,
            "content_phrases": args.content_phrases, "content_matches": content_matches,
            "semantic_answer_matches": semantic_matches, "semantic_answer_ordinals": semantic_ordinals,
            "phase2_reference": str(reference_dir.resolve()) if reference_dir else None,
            "validation": {"passed": checks_passed, "prefix_ids_and_mask_exact": True, "same_pixel_values_object": True,
                "bos_count": 1, "visual_tokens": 576, "sum_atol": SUM_ATOL, "query_checks": query_validation,
                "forward_consistency": consistency},
            "runs": {"prompt_control": control_run, "teacher_forced": teacher_run, "equal_length_masked_control": padded_run},
            "timings_seconds": {"preprocessing_and_alignment": preprocessing_seconds,
                "statistics_comparison_serialization_and_overhead": statistics_seconds - (
                    padded_run["forward_seconds"] + padded_run["matrix_validation_and_cpu_copy_seconds"] if padded_run else 0),
                "total_before_plotting_excluding_load": time.perf_counter() - started},
            "memory_after_cleanup": cleanup_memory,
            "definitions": {"raw_attention_axes": "[layer, head, unique_query, all_sequence_keys]; FP16 selected rows only, not full S*S",
                "target_alignment": "target position=P+i; prediction query=P-1+i; answer includes no added BOS/EOS",
                "exclusive_key_groups": list(GROUPS),
                "question": "Accessible question tokens located by template character offsets and exact ID expansion",
                "prompt_template": "All other accessible prompt keys excluding image, BOS and question",
                "previous_answer": "Answer input keys <= query; strictly before current predicted target",
                "prompt_text_attention_mass": "Derived subtotal question + prompt_template; do not add again to the five exclusive groups",
                "other_text_attention_mass": "Derived subtotal prompt_text + previous_answer; excludes BOS and vision",
                "statistics": "Reuse Phase 2 summarize_rows/layer_summary, with per-query accessible counts and baseline",
                "query_groups": "Overlapping analysis views; answer_start and prediction of a0 are the same physical query, stored once",
                "question_aggregate": "Unweighted mean of token-specific statistics; not a complete-question causal attention measure"},
            "limitations": "Fixed teacher-forced answer history, not a measured free-generation decode trajectory; descriptive attention only"}
        write_json(output_dir / "position_metadata.json", metadata)
        prior_failure = output_dir / "failure.json"
        if prior_failure.exists():
            # Keep earlier failures as history; never leave a stale active failure after success.
            diagnostics = output_dir / "diagnostics"
            diagnostics.mkdir(exist_ok=True)
            prior_failure.rename(diagnostics / f"resolved_failure_{time.time_ns()}.json")
        stage = "plotting"
        from compare_query_positions import plot_positions
        t0 = time.perf_counter()
        plot_positions(output_dir)
        metadata["timings_seconds"]["plotting"] = time.perf_counter() - t0
        metadata["timings_seconds"]["total_excluding_model_load"] = time.perf_counter() - started
        write_json(output_dir / "position_metadata.json", metadata)
        print(f"Saved {output_dir}: {len(question_records)} question tokens, {len(answer_records)} answer targets, "
              f"{len(queries)} unique queries; teacher attention [1,32,{len(ids)},{len(ids)}]; "
              f"forward={teacher_run['forward_seconds']:.6f}s, "
              f"peak={teacher_run['forward_and_validation_memory']['peak_allocated_mib']:.3f}MiB", flush=True)
    except Exception as exc:
        info = {"stage": stage, "error_type": type(exc).__name__, "message": str(exc)}
        if torch.cuda.is_available():
            info["memory"] = gpu_memory()
        write_json(output_dir / "failure.json", info)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "test_images/example.jpg")
    parser.add_argument("--question", default="What color is the stop sign?")
    parser.add_argument("--answer")
    parser.add_argument("--answer-source", choices=["model_generated", "ground_truth_teacher_forced"], default="ground_truth_teacher_forced")
    parser.add_argument("--query-mode", choices=["all", "question_all", "question_last", "question_content", "answer_start", "answer_prediction"], default="all")
    parser.add_argument("--content-phrases", nargs="*", default=[])
    parser.add_argument("--semantic-answer-phrases", nargs="*", default=["red"])
    parser.add_argument("--phase2-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/phase2_1_attention")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

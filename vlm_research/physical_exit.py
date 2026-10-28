"""Physical visual-token exit for Transformers 4.49 LLaVA/LLaMA, prefill only.

No model methods or third-party files are replaced. The local decoder loop
preserves the installed implementation's multimodal embedding and final norm.
"""
from contextlib import nullcontext
import time

from vlm_common import gpu_memory
from knockout_probe import VisualAttentionKnockout, layout_from_inputs, model_layers, tensor_hash
import torch


# Frozen before any 7B pilot outputs. Shape-changing FP16 GEMMs need not be
# bitwise equal; the CPU float32 and same-shape K=L controls are stricter.
TOLERANCES = {
    "fp16_logit_atol": 0.10,
    "fp16_logit_rtol": 0.003,
    "max_text_cosine_distance": 1e-4,
    "max_text_relative_rmse": 0.01,
    "candidate_joint_logp_atol_nats": 0.10,
    "candidate_margin_atol_nats": 0.15,
    "cpu_fp32_atol": 2e-6,
    "cpu_fp32_rtol": 2e-5,
}


def multimodal_embeddings(model, inputs):
    """Same get_image_features/masked_scatter as installed LLaVA.forward."""
    ids = inputs["input_ids"]
    hidden = model.get_input_embeddings()(ids)
    features = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        image_sizes=inputs.get("image_sizes"),
    ).to(hidden.device, hidden.dtype)
    image_mask = (ids == model.config.image_token_index).unsqueeze(-1).expand_as(hidden)
    if int(image_mask.sum()) != features.numel():
        raise ValueError("Image features and expanded image token count differ")
    return hidden.masked_scatter(image_mask, features)


def physical_forward(model, inputs, cutoff, expected_visual=576):
    if model.training or any(layer.training for layer in model_layers(model)):
        raise ValueError("Exit requires eval mode")
    if any(k in inputs for k in ("past_key_values", "cache_position", "inputs_embeds", "labels", "use_cache")):
        raise ValueError("Only complete uncached input_ids/pixels prefill is supported")
    layers = model_layers(model)
    if not isinstance(cutoff, int) or not 0 <= cutoff <= len(layers):
        raise ValueError("Invalid exit boundary")
    if any(layer.self_attn.config._attn_implementation != "eager" for layer in layers):
        raise ValueError("Physical exit requires the existing eager backend")
    layout = layout_from_inputs(inputs, model.config.image_token_index, expected_visual)
    decoder = model.language_model.model
    with torch.inference_mode():
        hidden = multimodal_embeddings(model, inputs)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        cache_position = positions.clone()
        position_ids = inputs.get("position_ids", cache_position.unsqueeze(0))
        if position_ids.shape != (1, hidden.shape[1]):
            raise ValueError("Expected batch-one position_ids")
        causal = decoder._update_causal_mask(inputs["attention_mask"], hidden, cache_position, None, False)
        if causal is None or causal.shape != (1, 1, hidden.shape[1], hidden.shape[1]):
            raise ValueError("Unexpected installed causal mask; inspect source before proceeding")
        rotary = decoder.rotary_emb(hidden, position_ids)
        keep = torch.where(~layout["visual"])[0]  # Keep ALL text, including padding.
        original_text_positions = position_ids.index_select(1, keep).clone()
        trace = []
        for index, layer in enumerate(layers):
            if index == cutoff and cutoff < len(layers):
                hidden = hidden.index_select(1, keep)
                causal = causal.index_select(2, keep).index_select(3, keep)
                position_ids = position_ids.index_select(1, keep)
                cache_position = cache_position.index_select(0, keep)
                # Select original cos/sin; never renumber or recompute with a
                # shorter context, which also protects dynamic RoPE semantics.
                rotary = tuple(x.index_select(1, keep) for x in rotary)
                positions = positions.index_select(0, keep)
                if not torch.equal(position_ids, original_text_positions):
                    raise AssertionError("Original logical text positions changed")
            trace.append({"layer": index, "sequence_length": hidden.shape[1],
                          "mask_shape": list(causal.shape),
                          "position_ids_sha256": tensor_hash(position_ids)})
            hidden = layer(hidden, attention_mask=causal, position_ids=position_ids,
                           past_key_value=None, output_attentions=False, use_cache=False,
                           cache_position=cache_position, position_embeddings=rotary)[0]
        hidden = decoder.norm(hidden)
        logits = model.language_model.lm_head(hidden)
    return {"hidden": hidden, "logits": logits, "original_positions": positions,
            "layers": trace, "physical_deletion": cutoff < len(layers),
            "original_text_position_ids": original_text_positions[0].cpu().tolist()}


def reference_forward(model, inputs, cutoff=None, expected_visual=576):
    """Native Full or existing Phase 3 Mask Exit, with temporary size observers."""
    trace, handles = [], []
    layers = model_layers(model)
    before = [(len(x._forward_pre_hooks), len(x.self_attn._forward_pre_hooks)) for x in layers]
    def observer(index):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            trace.append({"layer": index, "sequence_length": hidden.shape[1],
                          "mask_shape": list(kwargs["attention_mask"].shape),
                          "position_ids_sha256": tensor_hash(kwargs["position_ids"])})
        return hook
    try:
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(observer(i), with_kwargs=True))
        context = (nullcontext() if cutoff is None else VisualAttentionKnockout(
            layers, inputs, model.config.image_token_index, cutoff, expected_visual))
        with torch.inference_mode(), context as intervention:
            out = model(**inputs, output_attentions=False, output_hidden_states=True,
                        use_cache=False, return_dict=True)
        if cutoff is not None:
            intervention.assert_one_forward()
        if out.past_key_values is not None:
            raise AssertionError("Unexpected KV cache")
        result = {"hidden": out.hidden_states[-1], "logits": out.logits,
                  "original_positions": torch.arange(inputs["input_ids"].shape[1], device=out.logits.device),
                  "layers": trace, "physical_deletion": False}
        del out
    finally:
        for handle in handles:
            handle.remove()
    if before != [(len(x._forward_pre_hooks), len(x.self_attn._forward_pre_hooks)) for x in layers]:
        raise AssertionError("Observer/Mask hooks leaked")
    return result


def measured_forward(model, inputs, mode, cutoff, expected_visual=576):
    cuda = inputs["input_ids"].is_cuda
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if mode == "physical":
        result = physical_forward(model, inputs, cutoff, expected_visual)
    elif mode in ("full", "mask"):
        result = reference_forward(model, inputs, None if mode == "full" else cutoff, expected_visual)
    else:
        raise ValueError("Unknown mode")
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not torch.isfinite(result["hidden"]).all() or not torch.isfinite(result["logits"]).all():
        raise AssertionError("Non-finite output")
    n = inputs["input_ids"].shape[1]
    for row in result["layers"]:
        expected = n - expected_visual if mode == "physical" and row["layer"] >= cutoff else n
        if row["sequence_length"] != expected:
            raise AssertionError("Observed decoder sequence length is wrong")
    result["measurement"] = {"forward_seconds": elapsed, "gpu_memory": gpu_memory() if cuda else None,
                              "use_cache": False, "mode": mode, "cutoff": cutoff}
    return result


def aligned(result, original_positions, field):
    available = result["original_positions"]
    requested = torch.as_tensor(original_positions, device=available.device, dtype=torch.long)
    indices = torch.searchsorted(available, requested)
    if (indices >= len(available)).any() or not torch.equal(available[indices], requested):
        raise ValueError("Requested original position was deleted or is invalid")
    return result[field].index_select(1, indices)


def cosine_distances(reference, changed):
    a, b = reference.double(), changed.double()
    return (1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).clamp(min=0)


def equivalence(mask_hidden, physical_hidden, mask_logits, physical_logits, strict=False):
    """Compare valid retained text only; padded query rows are not observations."""
    a, b = mask_hidden.double(), physical_hidden.double()
    relative = ((a - b).square().mean(-1).sqrt() / a.square().mean(-1).sqrt().clamp_min(1e-12))
    cosine = cosine_distances(a, b)
    delta = (mask_logits.double() - physical_logits.double()).abs()
    atol = TOLERANCES["cpu_fp32_atol"] if strict else TOLERANCES["fp16_logit_atol"]
    rtol = TOLERANCES["cpu_fp32_rtol"] if strict else TOLERANCES["fp16_logit_rtol"]
    close = bool(torch.allclose(mask_logits.double(), physical_logits.double(), atol=atol, rtol=rtol))
    passed = close and float(cosine.max()) <= TOLERANCES["max_text_cosine_distance"] and float(relative.max()) <= TOLERANCES["max_text_relative_rmse"]
    return {"passed": passed, "logits_allclose": close, "max_abs_logit_error": float(delta.max()),
            "max_text_cosine_distance": float(cosine.max()), "max_text_relative_rmse": float(relative.max()),
            "logit_atol": atol, "logit_rtol": rtol}

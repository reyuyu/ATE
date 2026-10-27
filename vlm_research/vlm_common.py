"""Shared phase-one loading, input preparation, and measurements. No hooks."""
import os

# Logical cuda:0 is physical GPU 0 by default. Set this BEFORE importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gc
import json
from pathlib import Path
import platform
import subprocess
import time

from PIL import Image
import torch
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration

ROOT = Path(__file__).resolve().parent
REPO = "llava-hf/llava-1.5-7b-hf"
REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
MIB = 1024**2


def default_model():
    path = ROOT / "configs/model.json"
    if path.exists():
        return json.loads(path.read_text())["local_model_path"]
    return REPO


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def gpu_memory():
    free, total = torch.cuda.mem_get_info()
    report = {
        "allocated_mib": torch.cuda.memory_allocated() / MIB,
        "reserved_mib": torch.cuda.memory_reserved() / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        "device_free_mib": free / MIB,
        "device_total_mib": total / MIB,
    }
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
        for line in result.stdout.splitlines():
            pid, used = line.split(",")
            if int(pid.strip()) == os.getpid():
                report["nvidia_smi_process_mib"] = int(used.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return report


def load_model(model_path=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Activate vlm_research and check the driver / CUDA_VISIBLE_DEVICES.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=0. CPU offloading is disabled.")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    path = model_path or default_model()
    local = Path(path).is_dir()
    options = {"local_files_only": True} if local else {"revision": REVISION}
    processor = AutoProcessor.from_pretrained(path, **options)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = LlavaForConditionalGeneration.from_pretrained(
        path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        use_safetensors=True,
        **options,
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    locations = {str(parameter.device) for parameter in model.parameters()}
    dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if locations != {"cuda:0"} or dtypes != {"torch.float16"}:
        raise RuntimeError(f"Expected all FP16 parameters on cuda:0; devices={locations}, dtypes={dtypes}")
    if model.config.text_config._attn_implementation != "eager" or model.config.vision_config._attn_implementation != "eager":
        raise RuntimeError("Both the vision encoder and LLM must use eager attention")
    # Required by Transformers >=4.46 to expand <image> before LLM input.
    processor.patch_size = model.config.vision_config.patch_size
    processor.num_additional_image_tokens = 1  # CLIP adds a CLS token.
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    processor.tokenizer.padding_side = "left"
    # This checkpoint has sampling defaults; greedy decoding overrides them.
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_id": REPO,
        "model_revision": REVISION,
        "model_path": str(path),
        "load_seconds": load_seconds,
        "parameter_devices": sorted(locations),
        "parameter_dtypes": sorted(dtypes),
        "hf_device_map": {key: str(value) for key, value in model.hf_device_map.items()},
        "memory_after_load": gpu_memory(),
    }
    return model, processor, report


def prepare_inputs(model, processor, image_path, question):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    visual_tokens = int((inputs["input_ids"] == model.config.image_token_index).sum().item())
    height, width = inputs["pixel_values"].shape[-2:]
    patches = (height // processor.patch_size) * (width // processor.patch_size)
    expected = patches if processor.vision_feature_select_strategy == "default" else patches + 1
    if visual_tokens != expected or visual_tokens == 0:
        raise RuntimeError(f"Image tokens were not expanded correctly: got {visual_tokens}, expected {expected}")
    counts = {
        "visual_tokens": visual_tokens,
        "llm_input_tokens": int(inputs["attention_mask"].sum().item()),
        "non_visual_tokens_including_special": int(inputs["attention_mask"].sum().item()) - visual_tokens,
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "original_image_size": list(image.size),
    }
    return inputs.to("cuda:0", dtype=torch.float16), counts


def generate_once(model, processor, image_path, question, max_new_tokens=64):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    inputs, counts = prepare_inputs(model, processor, image_path, question)
    if counts["llm_input_tokens"] + max_new_tokens > model.config.text_config.max_position_embeddings:
        raise ValueError("Input plus generation exceeds the LLM context length")
    torch.cuda.synchronize()
    prepared = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    torch.cuda.synchronize()
    finished = time.perf_counter()
    # Check all generated-token scores without altering logits or using hooks.
    scores_finite = bool(torch.stack([torch.isfinite(score).all() for score in generated.scores]).all().item()) if generated.scores else False
    if not scores_finite:
        raise RuntimeError("Non-finite (NaN/Inf) generation scores detected")
    new_ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
    token_ids = new_ids[0].cpu().tolist()
    answer = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    peak = gpu_memory()
    del generated, new_ids, inputs
    gc.collect()
    torch.cuda.synchronize()
    return {
        "image": str(Path(image_path).resolve()),
        "question": question,
        "answer": answer,
        "new_token_ids": token_ids,
        "new_tokens": len(token_ids),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "generation_seconds": finished - prepared,
        "preprocess_seconds": prepared - started,
        "preprocess_and_generation_seconds": finished - started,
        "scores_all_finite": scores_finite,
        **counts,
        "inference_memory": peak,
        "memory_after_cleanup": gpu_memory(),
    }


def inspect_structure(model, processor, image_path, question):
    inputs, counts = prepare_inputs(model, processor, image_path, question)
    text = model.config.text_config
    vision = model.config.vision_config
    report = {
        "model_id": REPO,
        "model_revision": REVISION,
        "model_class": type(model).__name__,
        "checkpoint_transformers_version": model.config.transformers_version,
        "runtime_transformers_version": transformers.__version__,
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_vision": sum(p.numel() for p in model.vision_tower.parameters()),
        "parameters_projector": sum(p.numel() for p in model.multi_modal_projector.parameters()),
        "parameters_language_model": sum(p.numel() for p in model.language_model.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "vision_encoder_class": type(model.vision_tower).__name__,
        "vision_model_type": vision.model_type,
        "vision_layers": vision.num_hidden_layers,
        "vision_hidden_size": vision.hidden_size,
        "vision_feature_dimension_before_projector": vision.hidden_size,
        "vision_feature_dimension_after_projector": text.hidden_size,
        "vision_feature_layer": model.config.vision_feature_layer,
        "vision_feature_select_strategy": model.config.vision_feature_select_strategy,
        "vision_patch_size": vision.patch_size,
        "vision_image_size": vision.image_size,
        "vision_cls_tokens_before_selection": 1,
        "language_model_class": type(model.language_model).__name__,
        "language_model_config_name": text._name_or_path,
        "language_model_type": text.model_type,
        "llm_transformer_layers": text.num_hidden_layers,
        "llm_hidden_size": text.hidden_size,
        "llm_attention_heads": text.num_attention_heads,
        "llm_kv_heads": text.num_key_value_heads,
        "llm_attention_implementation": text._attn_implementation,
        "vision_attention_implementation": vision._attn_implementation,
        "image_token_id": model.config.image_token_index,
        "max_position_embeddings": text.max_position_embeddings,
        "module_paths_for_future_reference": {
            "vision": "vision_tower.vision_model.encoder.layers",
            "projector": "multi_modal_projector",
            "llm": "language_model.model.layers",
        },
        "question_for_token_counts": question,
        **counts,
    }
    del inputs
    return report

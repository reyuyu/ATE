"""Official local Qwen3-VL generation only: no intervention hooks or ablations."""
import argparse
import csv
import gc
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

from common import question_positions, select_samples, sha256, write


def smi(*fields):
    return subprocess.check_output(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], text=True).strip()


class MemoryMonitor:
    def __init__(self, physical, csv_path, label):
        self.physical, self.csv_path, self.label = physical, csv_path, label
        self.stop = threading.Event()
        self.peak = 0
        self.errors = []

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def loop(self):
        exists = self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["unix_time", "mode", "physical_gpu", "pid", "process_mib", "device_used_mib", "device_util_percent"])
            while not self.stop.is_set():
                try:
                    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True, timeout=5)
                    own = [int(row.split(",")[1]) for row in rows.splitlines() if row.split(",")[0].strip() == str(os.getpid())]
                    used, util = subprocess.check_output(["nvidia-smi", "-i", str(self.physical), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5).strip().split(",")
                    process = sum(own)
                    self.peak = max(self.peak, process)
                    writer.writerow([time.time(), self.label, self.physical, os.getpid(), process, int(used), int(util)])
                    f.flush()
                except Exception as exc:
                    self.errors.append(str(exc))
                self.stop.wait(0.25)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=10)


def freeze(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert json.loads((out / "download_validation.json").read_text())["passed"], "Validate the complete download before final freeze"
    manifest = select_samples(args.source_manifest)
    write(out / "sample_manifest.json", manifest)
    write(out / "resource_preflight.json", {"gpus": smi("index", "uuid", "name", "memory.total", "memory.used", "memory.free", "utilization.gpu"),
          "disk": subprocess.check_output(["df", "-B1", "/data", "/data1", "/home/likai"], text=True),
          "gpu1_forbidden": True})
    sources = {p.name: sha256(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()}
    write(out / "protocol.json", {"scope": "Single-GPU technical feasibility, not performance evaluation or ablation", "source_sha256": sources,
          "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "model_revision": args.model_dir.name, "model_path": str(args.model_dir.resolve()),
          "model_manifest_sha256": sha256(out / "model_manifest.json"),
          "download_validation_sha256": sha256(out / "download_validation.json"),
          "sample_manifest_sha256": sha256(out / "sample_manifest.json"), "modes": ["sdpa", "eager"],
          "dtype": "bfloat16", "offload": False, "quantization": False, "visible_devices": 1,
          "batch_size": 1, "do_sample": False, "num_beams": 1, "max_new_tokens": 32, "use_cache": True,
          "image_budget": "Unmodified official processor defaults first; reduced resolution only after an actual documented Eager OOM",
          "attention_weights": "Never request full output_attentions; no hooks or source masks",
          "timing": "First generated token streamer callback after CUDA synchronization is Prefill wall time; later callbacks give Decode wall time including callback overhead, not kernel-only timing",
          "memory": "CUDA allocator peaks plus nvidia-smi process peak sampled every approximately250ms; sampled peak is not guaranteed to catch transient spikes"})


def architecture(model, processor, layouts):
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
    source = Path(inspect.getfile(modeling))
    lines = source.read_text().splitlines()
    def snippets(term, radius=3):
        return [{"line": i+1, "text": "\n".join(lines[max(0, i-radius):i+radius+1])} for i, line in enumerate(lines) if term in line]
    language = model.model.language_model
    text = model.config.text_config
    paths = [f"model.language_model.layers.{i}.self_attn" for i in range(len(language.layers))]
    modules = dict(model.named_modules())
    assert all(p in modules for p in paths)
    return {"language_decoder_layers": len(language.layers), "attention_paths": paths,
            "attention_class": type(language.layers[0].self_attn).__name__, "query_heads": text.num_attention_heads,
            "kv_heads": text.num_key_value_heads, "grouped_query_attention": text.num_attention_heads != text.num_key_value_heads,
            "head_dim": text.head_dim, "qk_norm": ["q_norm", "k_norm"],
            "vision_encoder_path": "model.visual", "vision_depth": model.config.vision_config.depth,
            "vision_patch_size": model.config.vision_config.patch_size,
            "temporal_patch_size": model.config.vision_config.temporal_patch_size,
            "spatial_merge_size": model.config.vision_config.spatial_merge_size,
            "vision_feature_connection": "Merged image features replace input embeddings at actual image-token positions; DeepStack additionally adds vision features at visual positions after early decoder layers.",
            "deepstack_vision_extraction_layers": list(model.config.vision_config.deepstack_visual_indexes),
            "deepstack_language_injection_after_layers": list(range(len(model.config.vision_config.deepstack_visual_indexes))),
            "deepstack_evidence": snippets("self._deepstack_process", 6) + snippets("hidden_states[visual_pos_masks", 2),
            "mrope": text.rope_scaling, "rope_theta": text.rope_theta, "rope_evidence": snippets("get_rope_index(", 3),
            "mask_evidence": snippets("create_causal_mask", 4),
            "special_token_ids": {k: getattr(model.config, k, None) for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")},
            "configured_bos_token_id": text.bos_token_id, "actual_bos_counts": [x["bos_count"] for x in layouts],
            "template": processor.chat_template, "input_layouts": layouts,
            "implementation_source": str(source), "implementation_sha256": sha256(source),
            "future_source_mask_caveats": ["Dynamic image-token counts, not576; use actual grid and token IDs.",
                "36 language layers, GQA32 Query/8 KV heads and Q/K RMSNorm; do not paste LLaVA hooks.",
                "M-RoPE has three position axes and cache-dependent deltas; distinguish packed token index and position coordinates.",
                "DeepStack vision features enter after language L0/L1/L2 at visual positions; removing a P-1 image-key path would not remove all visual processing.",
                "Actual chat begins with im_start, not necessarily a BOS token; do not equate index0 to BOS.",
                "Future hard mask must compose with native causal mask, account for true query/key lengths and preserve retained-source renormalization; not tested here.",
                "Passing small-image inference is not evidence that retaining all attention matrices or long-context interventions fits24GB."]}


def cpu_probe(args):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    rows = []
    for case in json.loads((args.out / "sample_manifest.json").read_text())["samples"]:
        with Image.open(case["image_path"]) as handle:
            image = handle.convert("RGB")
        question = case["question"]
        messages = [{"role": "user", "content": [{"type": "image", "image": case["image_path"]},
                    {"type": "text", "text": question + "\nAnswer using a single word or short phrase."}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = processor(text=[text], images=[image], return_tensors="pt", return_offsets_mapping=True)
        ids = batch["input_ids"][0].tolist()
        count = ids.count(config.image_token_id)
        grid = batch["image_grid_thw"][0].tolist()
        assert count == grid[0]*grid[1]*grid[2]//processor.image_processor.merge_size**2
        expanded = text.replace(processor.image_token, processor.image_token*count)
        assert processor.tokenizer(expanded)["input_ids"] == ids
        qpos = question_positions(batch["offset_mapping"][0].tolist(), expanded, question)
        assert processor.tokenizer.decode([ids[i] for i in qpos]).strip() == question.strip()
        assert all(value.device.type == "cpu" for value in batch.values())
        rows.append({"question_id": case["question_id"], "tokens": len(ids), "image_tokens": count,
                     "question_positions": qpos, "p_minus_1": len(ids)-1})
    write(args.out / "processor_cpu_validation.json", {"passed": True, "cases": rows,
          "config_class": type(config).__name__, "official_model_class_available": Qwen3VLForConditionalGeneration.__name__,
          "model_weights_loaded": False, "gpu_used": False})
    print("Offline CPU processor/model-class validation PASS", len(rows), flush=True)


def run(args):
    assert args.gpu in (0, 2, 3), "GPU1 forbidden"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.gpu)
    out = args.out
    protocol = json.loads((out / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        assert sha256(Path(__file__).parent / name) == expected, "Frozen source changed"
    assert sha256(out / "sample_manifest.json") == protocol["sample_manifest_sha256"]
    assert sha256(out / "model_manifest.json") == protocol["model_manifest_sha256"]
    assert sha256(out / "download_validation.json") == protocol["download_validation_sha256"]
    assert json.loads((out / "download_validation.json").read_text())["passed"]
    assert json.loads((out / "processor_cpu_validation.json").read_text())["passed"]
    pre = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
    uuid, used, free = [x.strip() for x in pre.split(",")]
    assert int(used) < 500 and int(free) > 23000, f"Selected GPU no longer idle: {pre}"
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.generation.streamers import BaseStreamer
    assert torch.cuda.device_count() == 1
    assert "3090" in torch.cuda.get_device_name(0)
    assert torch.cuda.is_bf16_supported()
    torch.cuda.set_device(0)
    torch.manual_seed(20261101)
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
          "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
          "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
          "physical_gpu": args.gpu, "gpu_uuid": uuid, "process_cuda": "cuda:0", "pid": os.getpid(),
          "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": torch.cuda.get_device_name(0),
          "bf16_supported": torch.cuda.is_bf16_supported(), "preload_gpu": pre}
    write(out / "environment.json", environment)
    write(out / f"environment_{args.mode}.json", environment)
    torch.cuda.reset_peak_memory_stats()
    report = {"mode": args.mode, "physical_gpu": args.gpu, "gpu_uuid": uuid, "passed": False, "cases": [],
              "image_budget": "official defaults" if args.max_pixels is None else {"max_pixels": args.max_pixels},
              "load_passed": False, "first_token_forward_passed": False, "generate_passed": False}
    name = "native_smoke.json" if args.mode == "sdpa" else "eager_smoke.json"
    with MemoryMonitor(args.gpu, out / "gpu_memory_report.csv", args.mode) as monitor:
        try:
            t0 = time.perf_counter()
            processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
                    dtype=torch.bfloat16, attn_implementation=args.mode, device_map={"": "cuda:0"},
                    use_safetensors=True, low_cpu_mem_usage=True).eval()
            torch.cuda.synchronize()
            report["load_seconds"] = time.perf_counter() - t0
            report["parameter_devices"] = sorted({str(p.device) for p in model.parameters()})
            report["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
            report["buffer_devices"] = sorted({str(b.device) for b in model.buffers()})
            report["device_map"] = model.hf_device_map
            report["parameter_count"] = model.num_parameters()
            assert report["parameter_devices"] == ["cuda:0"]
            assert report["parameter_dtypes"] == ["torch.bfloat16"]
            assert report["buffer_devices"] == ["cuda:0"]
            assert set(map(str, model.hf_device_map.values())) <= {"cuda:0", "0"}
            implementations = {"text": model.config.text_config._attn_implementation,
                               "vision": model.config.vision_config._attn_implementation}
            assert set(implementations.values()) == {args.mode}
            report["attention_implementations"] = implementations
            report["load_passed"] = True
            samples = json.loads((out / "sample_manifest.json").read_text())["samples"]
            cases = [{**samples[0], "question_id": "demo-local-image", "question": "What is visible in this picture?", "task": "local-demo"}] + samples
            layouts = []

            class Timer(BaseStreamer):
                def __init__(self):
                    self.times = []
                def put(self, value):
                    torch.cuda.synchronize()
                    self.times.append(time.perf_counter())
                    if len(self.times) == 2:
                        report["first_token_forward_passed"] = True
                def end(self):
                    pass

            for case in cases:
                assert sha256(case["image_path"]) == case["image_sha256"]
                with Image.open(case["image_path"]) as handle:
                    image = handle.convert("RGB")
                question = case["question"]
                messages = [{"role": "user", "content": [{"type": "image", "image": case["image_path"]},
                            {"type": "text", "text": question + "\nAnswer using a single word or short phrase."}]}]
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                kwargs = {} if args.max_pixels is None else {"images_kwargs": {"max_pixels": args.max_pixels}}
                inputs = processor(text=[rendered], images=[image], return_tensors="pt", return_offsets_mapping=True, **kwargs)
                offsets = inputs.pop("offset_mapping")[0].tolist()
                ids = inputs["input_ids"][0].tolist()
                visual = [i for i, token in enumerate(ids) if token == model.config.image_token_id]
                grid = inputs["image_grid_thw"][0].tolist()
                count = grid[0]*grid[1]*grid[2] // processor.image_processor.merge_size**2
                assert count == len(visual) and visual == list(range(visual[0], visual[-1]+1))
                expanded = rendered.replace(processor.image_token, processor.image_token * count)
                assert processor.tokenizer(expanded)["input_ids"] == ids
                qpos = question_positions(offsets, expanded, question)
                assert processor.tokenizer.decode([ids[i] for i in qpos]).strip() == question.strip()
                pminus1 = int(inputs["attention_mask"][0].nonzero()[-1])
                special = {str(token): [i for i, value in enumerate(ids) if value == token] for token in processor.tokenizer.all_special_ids if token in ids}
                layout = {"question_id": case["question_id"], "task": case["task"], "image_sha256": case["image_sha256"],
                          "question": question, "image_path": case["image_path"], "original_image_size": list(image.size),
                          "input_ids": ids, "input_length": len(ids), "image_grid_thw": grid,
                          "image_token_positions": visual, "image_tokens": count, "question_positions": qpos,
                          "p_minus_1": pminus1, "p_minus_1_token_id": ids[pminus1],
                          "p_minus_1_token": processor.tokenizer.convert_ids_to_tokens(ids[pminus1]),
                          "special_token_positions": special,
                          "bos_count": ids.count(model.config.text_config.bos_token_id),
                          "template_or_instruction_positions": [i for i in range(len(ids)) if i not in set(visual + qpos)],
                          "partition_note": "Question is raw question only; instruction and chat delimiters are template. P-1 self is an explicit label and also a template token, not an extra token.",
                          "rendered_chat_template": rendered}
                # M-RoPE is read directly through the official model's index utility, not guessed from sequence indices.
                positions, deltas = model.model.get_rope_index(inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"], attention_mask=inputs["attention_mask"])
                layout["mrope_position_shape"] = list(positions.shape)
                layout["p_minus_1_mrope_coordinates"] = positions[:, 0, pminus1].tolist()
                layout["rope_deltas"] = deltas.tolist()
                layouts.append(layout)
                write(out / f"input_layouts_{args.mode}.json", layouts)
                report["current_input"] = {"question_id": case["question_id"], "input_length": len(ids), "visual_tokens": count}
                write(out / name, report)
                inputs = inputs.to("cuda:0")
                timer = Timer()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    result = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=32, use_cache=True,
                            return_dict_in_generate=True, output_scores=True, output_attentions=False, output_hidden_states=False, streamer=timer)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                tokens = result.sequences[0, len(ids):].tolist()
                answer = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                assert tokens and answer, "Empty generation"
                assert all(torch.isfinite(score).all().item() for score in result.scores), "Nonfinite generation scores"
                assert len(result.scores) == len(tokens) and len(timer.times) == len(tokens) + 1
                eos_config = model.generation_config.eos_token_id
                eos = set(eos_config if isinstance(eos_config, list) else [eos_config])
                row = {"question_id": case["question_id"], "task": case["task"], "answer": answer,
                       "generated_token_ids": tokens, "eos_reached": tokens[-1] in eos,
                       "truncated": len(tokens) == 32 and tokens[-1] not in eos,
                       "all_scores_finite": True, "input_length": len(ids), "visual_tokens": count,
                       "wall_seconds": elapsed, "prefill_first_token_wall_seconds": timer.times[1] - t0,
                       "decode_wall_seconds": timer.times[-1] - timer.times[1],
                       "decode_steps": len(tokens)-1,
                       "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                       "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2,
                       "first_token_id": tokens[0], "first_token_score": float(result.scores[0][0, tokens[0]]),
                       "cache_class": type(result.past_key_values).__name__ if result.past_key_values is not None else None}
                assert row["cache_class"] is not None
                row["final_cache_seq_length"] = result.past_key_values.get_seq_length()
                assert row["final_cache_seq_length"] == len(ids) + len(tokens) - 1
                report["cases"].append(row)
                report["first_token_forward_passed"] = True
                write(out / name, report)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                del result, inputs
            report["generate_passed"] = True
            write(out / "architecture_audit.json", architecture(model, processor, layouts))
            # CPU-only processor audit: demonstrate dynamic token counts without an extra model forward.
            reduced = processor(text=[rendered], images=[image], return_tensors="pt", images_kwargs={"max_pixels": 65536})
            write(out / "visual_budget_layout_check.json", {"model_forward_run": False, "official_default_count": count,
                  "max_pixels_65536_count": int((reduced["input_ids"] == model.config.image_token_id).sum()),
                  "default_processor": processor.image_processor.to_dict()})
            report["passed"] = True
            del model, processor, reduced
        except Exception as exc:
            report["exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            print(traceback.format_exc(), flush=True)
        finally:
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated()/1024**2
            report["peak_reserved_mib"] = torch.cuda.max_memory_reserved()/1024**2
            gc.collect()
            torch.cuda.empty_cache()
    report["nvidia_smi_sampled_process_peak_mib"] = monitor.peak
    report["memory_monitor_errors"] = monitor.errors
    write(out / name, report)
    if not report["passed"]:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["freeze", "cpu", "run"])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-manifest", type=Path)
    ap.add_argument("--mode", choices=["sdpa", "eager"])
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--max-pixels", type=int)
    args = ap.parse_args()
    {"freeze": freeze, "cpu": cpu_probe, "run": run}[args.action](args)


if __name__ == "__main__":
    main()

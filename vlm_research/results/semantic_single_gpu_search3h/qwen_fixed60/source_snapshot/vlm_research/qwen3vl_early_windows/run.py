"""Freeze -> CPU tests -> GPU1 smoke -> immutable checkpointed generation."""
import argparse
import ast
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

from .experiment import (AttentionBypass, CONDITIONS, HISTORICAL, MODEL, OUT, REVISION, SEED, SOURCE,
                         digest, generate, gqa_prediction, hook_counts, prepare, select, sha256,
                         tensor_digest, write)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sources():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths.append(Path(__file__).parents[1] / "qwen3vl_smoke/common.py")
    return {str(p.relative_to(Path(__file__).parents[2])): sha256(p) for p in sorted(paths)}


def smi():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True)


def environment():
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
    return dict(python=platform.python_version(), executable=os.sys.executable,
                versions={name: importlib.metadata.version(name) for name in
                          ("torch", "transformers", "accelerate", "huggingface-hub", "tokenizers", "Pillow", "numpy")},
                torch_cuda=torch.version.cuda, official_model_source=str(inspect.getfile(modeling)),
                official_model_source_sha256=sha256(inspect.getfile(modeling)))


def freeze():
    from transformers import AutoProcessor
    assert not (OUT / "protocol.json").exists(), "Frozen protocol already exists; do not resample"
    OUT.mkdir(parents=True, exist_ok=True)
    raw = read(SOURCE)
    assert len(raw["samples"]) == 3000
    assert all(sum(r["task"] == task for r in raw["samples"]) == 500 for task in {r["task"] for r in raw["samples"]})
    # Reuse the prior scorer function exactly, without importing LLaVA dependencies.
    tree = ast.parse(HISTORICAL.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gqa_prediction")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(HISTORICAL), "exec"), scope)
    for text in ("Dog.", "two cats", "yes...", "A red bus", "red!", ""):
        assert scope["gqa_prediction"](text) == gqa_prediction(text)
    scorer_files = [HISTORICAL,
        Path("/data/lk/ATE-resources/paper_rq1/annotations/gqa_eval.py"),
        Path("/data/lk/ATE-resources/benchmarks/phase4_0/sources/llava/scripts/convert_gqa_for_eval.py")]
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    samples = []
    for n, row in enumerate(select(raw["samples"]), 1):
        sample = {k: row[k] for k in ("task", "question_id", "image_id", "image_path", "image_sha256", "question", "answer")}
        inputs, layout = prepare(processor, sample)
        sample.update(layout=layout, input_binding=digest(layout))
        samples.append(sample)
        if n % 40 == 0:
            print(f"Prepared {n}/240 inputs", flush=True)
    manifest = dict(seed=SEED, source_manifest=str(SOURCE), source_manifest_sha256=sha256(SOURCE),
        source_commit="221cdde56dde53908aaa953fe26672fd33334f84", selection="Task-sorted round robin, 40 rounds; lowest SHA256(JSON[seed,task,qid,imageSHA]) not previously selected; prefer globally unused imageSHA. No model output used.",
        count=len(samples), unique_images=len({s["image_sha256"] for s in samples}), samples=samples)
    assert len(samples) == 240 and len({(s["task"], s["question_id"]) for s in samples}) == 240
    write(OUT / "sample_manifest.json", manifest)
    old = OUT.parent / "qwen3vl_8b_single_gpu_smoke/model_manifest.json"
    model_manifest = read(old)
    for file in model_manifest["files"]:
        path = MODEL / file["path"]
        assert path.stat().st_size == file["size"]
        assert sha256(path) == file["sha256"], f"Model content changed: {path}"
    write(OUT / "model_manifest.json", model_manifest)
    write(OUT / "environment.json", environment())
    protocol = dict(base_commit="96988a7bd8b25cedf0e55acf073604835e7c4329",
        remote_main_preflight="e89c8acce6e674ecfbd2c8acad57faf71bf4d547", seed=SEED,
        model_repo="Qwen/Qwen3-VL-8B-Instruct", model_revision=REVISION, model_path=str(MODEL),
        dtype="bfloat16", attention="eager", language_layers=36, batch_size=1,
        physical_gpu=1, gpu_selection="User opening explicit 'use GPU1' overrides copied section5 exclusion; never use another GPU",
        conditions=CONDITIONS, max_new_tokens=32, do_sample=False, num_beams=1, use_cache=True,
        intervention="Post-language-self_attn.o_proj output zero at original P-1 and every real incremental decode query only; QKV/cache/MLP/residual/other prefill/DeepStack untouched",
        image_budget="Unmodified official processor defaults; no silent downsampling",
        scoring="Native decoded text.strip(), then exact prior official-GQA conversion text.rstrip('.').lower(), equals frozen answer; no new aliases",
        scorer_bindings={str(p): sha256(p) for p in scorer_files},
        sample_manifest_sha256=sha256(OUT / "sample_manifest.json"),
        model_manifest_sha256=sha256(OUT / "model_manifest.json"), environment_sha256=sha256(OUT / "environment.json"),
        code_sha256=sources(), processor_chat_template=processor.chat_template,
        bootstrap=dict(seed=SEED, repetitions=200000, cluster="Whole Image SHA256", weighting="Question-weighted",
                       comparisons=6, ordinary_quantiles=[.025, .975], bonferroni_quantiles=[.025/6, 1-.025/6]),
        interpretation="240-question exploratory comparison only. -1pp is a reference, not independent noninferiority confirmation. No physical skipping/speedup claim.")
    write(OUT / "protocol.json", protocol)
    write(OUT / "resource_preflight.json", dict(gpus=smi(), time=time.time(),
          disk=subprocess.check_output(["df", "-h", "/data", "/data1"], text=True)))
    print(f"Frozen 240 questions / {manifest['unique_images']} images; protocol {sha256(OUT / 'protocol.json')}", flush=True)


def verify_binding():
    p = read(OUT / "protocol.json")
    assert p["code_sha256"] == sources(), "Frozen source changed; stop rather than mix code versions"
    assert p["sample_manifest_sha256"] == sha256(OUT / "sample_manifest.json")
    assert p["environment_sha256"] == sha256(OUT / "environment.json")
    assert environment() == read(OUT / "environment.json")
    for path, expected in p["scorer_bindings"].items():
        assert sha256(path) == expected
    return p


def load():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and torch.cuda.device_count() == 1
    assert importlib.metadata.version("transformers") == "4.57.6"
    query = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True).strip()
    assert not query, f"GPU1 has a compute process, do not interfere: {query}"
    write(OUT / f"gpu_preload_{time.time_ns()}.json", dict(gpus=smi(), physical_gpu=1,
        uuid=subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip(),
        visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], process_device="cuda:0", pid=os.getpid()))
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", device_map={"": "cuda:0"}, use_safetensors=True, low_cpu_mem_usage=True).eval()
    assert {str(p.device) for p in model.parameters()} == {"cuda:0"}
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert len(model.model.language_model.layers) == 36
    assert model.config.text_config._attn_implementation == model.config.vision_config._attn_implementation == "eager"
    return model, processor


def checked_inputs(processor, sample):
    inputs, layout = prepare(processor, sample)
    assert layout == sample["layout"] and digest(layout) == sample["input_binding"]
    return inputs.to("cuda:0")


def native_generate(model, inputs):
    with torch.inference_mode():
        return model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=32, use_cache=True,
                              return_dict_in_generate=True, output_scores=True)


def smoke():
    verify_binding()
    assert read(OUT / "cpu_tests.json")["passed"]
    model, processor = load()
    weights_before = [(name, p._version, p.data_ptr()) for name, p in model.named_parameters()]
    hooks_before = hook_counts(model)
    samples = read(OUT / "sample_manifest.json")["samples"]
    cases = sorted(samples, key=lambda s: s["layout"]["input_length"])
    cases = [cases[0], cases[-1]]
    audit = dict(passed=False, cases=[], residual_check="Exact native BF16 additions, no adjustable tolerance",
                 code_sha256=sources(), protocol_sha256=sha256(OUT / "protocol.json"))
    started = time.perf_counter()
    try:
        for sample in cases:
            inputs = checked_inputs(processor, sample)
            inputs_before = {k: tensor_digest(v) for k, v in inputs.items()}
            n = inputs["input_ids"].shape[1]
            native = native_generate(model, inputs)
            full, full_score, full_trace = generate(model, processor, inputs, trace=True)
            repeat, repeat_score, _ = generate(model, processor, inputs)
            noop, noop_score, _ = generate(model, processor, inputs, CONDITIONS["W0"], noop=True)
            assert native.sequences[0, n:].tolist() == full["token_ids"] == repeat["token_ids"] == noop["token_ids"]
            assert torch.equal(native.scores[0][0].float().cpu(), full_score)
            assert torch.equal(full_score, repeat_score) and torch.equal(full_score, noop_score)
            del native
            # Compare every valid prompt logit, not only the last-token slice.
            with torch.inference_mode():
                positions = torch.arange(n, device="cuda:0")
                plain = model(**inputs, use_cache=True, logits_to_keep=0, cache_position=positions).logits.detach().cpu()
                with AttentionBypass(model, CONDITIONS["W0"], n, noop=True):
                    no_op = model(**inputs, use_cache=True, logits_to_keep=0, cache_position=positions).logits.detach().cpu()
            assert torch.equal(plain, no_op)
            max_logit_difference = float((plain.float()-no_op.float()).abs().max())
            del plain, no_op
            changes = []
            for condition in ("W0", "W5"):
                changed, _, trace = generate(model, processor, inputs, CONDITIONS[condition], trace=True)
                for i in range(36):
                    current, reference = trace.records[i], full_trace.records[i]
                    assert torch.equal(current["attention"][:, :-1], reference["attention"][:, :-1])
                    assert torch.equal(current["output"][:, :-1], reference["output"][:, :-1])
                    if i in CONDITIONS[condition]:
                        assert torch.count_nonzero(current["attention"][:, -1]) == 0
                    else:
                        assert torch.count_nonzero(current["attention"][:, -1]) > 0
                    if i < 3:
                        assert torch.equal(trace.features[i], full_trace.features[i])
                changes.append(dict(condition=condition, token_ids=changed["token_ids"],
                                    hook_events=changed["hook_events"], cache_lengths=changed["cache_lengths"],
                                    residual_and_deepstack_exact=True, non_p1_prefill_exact=True,
                                    all_mlp_calls_native=True))
                del trace
            # The exception path must also unregister without executing the model.
            try:
                with AttentionBypass(model, CONDITIONS["W0"], n):
                    raise RuntimeError("intentional cleanup fixture")
            except RuntimeError as exc:
                assert str(exc) == "intentional cleanup fixture"
            assert hook_counts(model) == hooks_before
            assert inputs_before == {k: tensor_digest(v) for k, v in inputs.items()}
            audit["cases"].append(dict(task=sample["task"], question_id=sample["question_id"],
                input_length=n, image_tokens=sample["layout"]["image_tokens"], full_token_ids=full["token_ids"],
                native_repeat_and_noop_tokens_exact=True, all_valid_logits_noop_max_abs=max_logit_difference,
                current_query_and_cache_correct=True, intervention_checks=changes))
            print(f"Smoke passed {sample['task']} {sample['question_id']} length={n}", flush=True)
            del full_trace, inputs
        assert weights_before == [(name, p._version, p.data_ptr()) for name, p in model.named_parameters()]
        assert hook_counts(model) == hooks_before
        audit.update(passed=True, weights_versions_and_storage_unchanged=True, hooks_clean=True,
                     inputs_unchanged=True, no_nan_inf_oom=True)
    except BaseException:
        audit["traceback"] = traceback.format_exc()
        raise
    finally:
        audit.update(seconds=time.perf_counter()-started, cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                     cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        write(OUT / "validation.json", audit)


def formal():
    protocol = verify_binding()
    validation = read(OUT / "validation.json")
    assert validation["passed"] and validation["protocol_sha256"] == sha256(OUT / "protocol.json")
    # The unchanged prior smoke runner uses a script-local `common` import.
    sys.path.insert(0, str(Path(__file__).parents[1] / "qwen3vl_smoke"))
    from ..qwen3vl_smoke.run import MemoryMonitor
    model, processor = load()
    weights_before = [(name, p._version, p.data_ptr()) for name, p in model.named_parameters()]
    hooks_before = hook_counts(model)
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    binding = sha256(OUT / "protocol.json")
    started, generated_count, skipped = time.perf_counter(), 0, 0
    manifest = read(OUT / "sample_manifest.json")
    with MemoryMonitor(1, OUT / "gpu_memory_report.csv", "formal-eager-bf16") as monitor:
        try:
            for number, sample in enumerate(manifest["samples"], 1):
                path = checkpoint_dir / f"{sample['task']}__{sample['question_id']}.json"
                if path.exists():
                    record = read(path)
                    assert record["protocol_sha256"] == binding and record["input_binding"] == sample["input_binding"]
                    assert list(record["conditions"]) == list(CONDITIONS)
                    skipped += 1
                    continue
                inputs = checked_inputs(processor, sample)
                record = {k: sample[k] for k in ("task", "question_id", "image_id", "image_sha256", "question", "answer", "input_binding")}
                record.update(protocol_sha256=binding, conditions={}, pid=os.getpid(), physical_gpu=1,
                    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
                full_score = None
                for condition, layers in CONDITIONS.items():
                    before = time.perf_counter()
                    result, first, _ = generate(model, processor, inputs, layers)
                    if condition == "Full":
                        full_score = first
                    full_token = int(full_score.argmax())
                    result.update(correct=int(result["prediction"] == sample["answer"]),
                        full_first_token_id=full_token, first_top1_flip=int(first.argmax()) != full_token,
                        full_first_token_logp=float(full_score.log_softmax(-1)[full_token]),
                        changed_full_token_logp=float(first.log_softmax(-1)[full_token]),
                        first_logp_loss=float(full_score.log_softmax(-1)[full_token]-first.log_softmax(-1)[full_token]),
                        seconds=time.perf_counter()-before)
                    record["conditions"][condition] = result
                    generated_count += 1
                assert hook_counts(model) == hooks_before
                assert sample["layout"]["input_tensor_hashes"] == {k: tensor_digest(v) for k, v in inputs.items()}
                write(path, record)
                del inputs
                elapsed = time.perf_counter()-started
                print(f"DONE {number}/240 {sample['task']} {sample['question_id']} generated={generated_count} elapsed={elapsed:.1f}s", flush=True)
                write(OUT / "progress.json", dict(completed_questions=number, total_questions=240,
                    generated_this_run=generated_count, reused_questions=skipped, elapsed_seconds=elapsed,
                    time=time.time(), status="running", protocol_sha256=binding))
            assert weights_before == [(name, p._version, p.data_ptr()) for name, p in model.named_parameters()]
            assert hook_counts(model) == hooks_before
            write(OUT / "execution.json", dict(status="completed", count=240, conditions=7,
                generated_this_run=generated_count, reused_questions=skipped, seconds=time.perf_counter()-started,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                sampled_process_peak_mib=monitor.peak, monitor_errors=monitor.errors,
                weights_versions_and_storage_unchanged=True, hooks_clean=True, protocol_sha256=binding,
                code_sha256=protocol["code_sha256"]))
        except BaseException:
            write(OUT / f"error_{time.time_ns()}.json", dict(traceback=traceback.format_exc(),
                generated_this_run=generated_count, seconds=time.perf_counter()-started, protocol_sha256=binding))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "smoke", "formal"])
    globals()[parser.parse_args().stage]()

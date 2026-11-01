"""Freeze 120 historical inputs, smoke on GPU1, then 120 + 4320 prefills only."""
import argparse
import collections
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from ..qwen3vl_early_windows.experiment import MODEL, REVISION, digest, hook_counts, prepare, sha256, tensor_digest, write
from ..qwen3vl_early_windows.run import environment
from .engine import MASS_TOLERANCE, SOURCES, MassObserver, partition, prefill, sensitivity

OUT = Path(__file__).resolve().parents[1] / "results/qwen3vl_four_stage_pilot_120"
OLD = OUT.parent / "qwen3vl_early_attention_windows_240"
RAW = Path("/data/lk/ATE-resources/experiments/qwen3vl_four_stage_pilot_120")
SEED = 20261101
LLAVA_MASS = Path("/data/lk/ATE-worktrees/first-answer-attention-mass/vlm_research/results/first_answer_attention_mass/layer_summary.csv")
LLAVA_SENS = Path("/data/lk/ATE-worktrees/single-layer-gqa6/vlm_research/results/paper_rq1_single_layer_gqa6/first_token_layer_summary.csv")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    root = Path(__file__).parents[2]
    paths = list(Path(__file__).parent.glob("*.py"))
    for name in ("qwen3vl_early_windows/experiment.py", "qwen3vl_early_windows/run.py",
                 "qwen3vl_smoke/common.py", "qwen3vl_smoke/run.py"):
        paths.append(root / "vlm_research" / name)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}


def choose(samples):
    selected = []
    for task in sorted({r["task"] for r in samples}):
        candidates = [r for r in samples if r["task"] == task]
        assert len(candidates) == 40
        candidates.sort(key=lambda r:digest([SEED, r["input_binding"], task, r["question_id"]]))
        selected.extend(candidates[:20])
    assert len(selected) == 120 and len({r["image_sha256"] for r in selected}) == 120
    return selected


def freeze():
    from transformers import AutoConfig, AutoProcessor
    assert not (OUT / "protocol.json").exists(), "Do not overwrite frozen inputs"
    OUT.mkdir(parents=True,exist_ok=True)
    RAW.mkdir(parents=True,exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL,local_files_only=True)
    original = read(OLD / "sample_manifest.json")
    samples = []
    for row in choose(original["samples"]):
        inputs, layout = prepare(processor,row)
        assert layout == row["layout"] and digest(layout) == row["input_binding"]
        sample = dict(row)
        sample["source_partition"] = partition(processor.tokenizer,layout,config.text_config)
        samples.append(sample)
    write(OUT / "sample_manifest.json",dict(seed=SEED,source_commit="e5d6ab1935dd9305ccbcf33110b9693e66ba71fe",
        source_manifest_sha256=sha256(OLD / "sample_manifest.json"),count=120,unique_images=120,
        selection="Lowest SHA256(JSON[seed,input_binding,task,qid]) within each original 40-question task; 20/task; no correctness or model output used",
        samples=samples))
    source_audit = [{"task":s["task"],"question_id":s["question_id"],**s["source_partition"]} for s in samples]
    write(OUT / "source_partition_audit.json",dict(passed=True,mutually_exclusive_complete=True,
        prompt_definition="Question + Template; excludes Start Anchor and P-1 Self",samples=source_audit))
    write(OUT / "environment.json",environment())
    model_manifest = read(OLD / "model_manifest.json")
    for file in model_manifest["files"]:
        path = MODEL / file["path"]
        assert path.stat().st_size == file["size"] and sha256(path) == file["sha256"]
    write(OUT / "model_manifest.json",model_manifest)
    protocol = dict(base_commit="e5d6ab1935dd9305ccbcf33110b9693e66ba71fe",seed=SEED,
        scope="120-question exploratory native P-1 mass and 36 single-layer P-1-only Attention output ablations; NO generation, decode, W2 confirmation or source mask",
        samples=120,per_task=20,layers=list(range(36)),formal_native_prefills=120,formal_ablated_prefills=4320,
        model_revision=REVISION,model_path=str(MODEL),dtype="bfloat16",attention="eager",batch_size=1,
        physical_gpu=1,visible_devices="1",process_device="cuda:0",offload=False,quantization=False,
        source_partition="First real <|im_start|> is Start Anchor (user role opener), Image, real Question, Template/Other, Self; Prompt=Question+Template",
        observer="Wrap unchanged official eager_attention_forward; copy only current layer P-1 softmax row to CPU, reduce, return original output and original probabilities; never request output_attentions",
        mass_tolerance=MASS_TOLERANCE,mass_tolerance_basis="BF16 nonnegative probability relative roundoff <= 2^-8 plus 2e-6 summation allowance",
        noop_tolerance=0.0,first_token_target="Original Full Top-1, not gold; logP_Full(t*) - logP_OFF(t*) in nat; negative values retained",
        bootstrap=dict(repetitions=10000,seed=SEED,cluster="Image SHA256",weighting="questions",interval="Descriptive ordinary percentile 95%; no confirmatory significance or NI claim"),
        stage_assessment=dict(changepoint_detection=False,smoothing=False,forced_boundaries=False,
            majority_anchor_threshold=.5,min_run=3,
            early_probe=[2,12],middle_probe=[13,24],late_probe=[25,35],
            note="Probe ranges only, not fitted or claimed stage boundaries. Report raw L0/L1 separately; enumerate actual majority runs and task maxima; all findings descriptive."),
        code_sha256=source_hashes(),sample_manifest_sha256=sha256(OUT / "sample_manifest.json"),
        environment_sha256=sha256(OUT / "environment.json"),model_manifest_sha256=sha256(OUT / "model_manifest.json"),
        llava_bindings={str(p):sha256(p) for p in (LLAVA_MASS,LLAVA_SENS)},
        llava_direction="Historical delta_logp=logP_OFF-logP_Full; multiply by -1 for signed loss; task means equally weighted because each original task has 500 questions",
        llava_comparison_limit="LLaVA 3000 questions/32 layers/FP16 vs Qwen 120-question subset/36 layers/BF16; different templates; BOS is not the same token as Start Anchor",
        raw_tensor_directory=str(RAW),historical_windows="Read old 240-question CSV only; never regenerate windows; no independent noninferiority test")
    write(OUT / "protocol.json",protocol)
    write(OUT / "gpu1_resource_preflight.json",dict(time=time.time(),
        snapshot=subprocess.check_output(["nvidia-smi"],text=True),
        gpu1=subprocess.check_output(["nvidia-smi","-i","1","--query-gpu=index,uuid,memory.used,memory.free","--format=csv,noheader,nounits"],text=True),
        compute_processes=subprocess.check_output(["nvidia-smi","-i","1","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True),
        disk=subprocess.check_output(["df","-h","/data"],text=True)))
    print("Frozen 120 questions / 120 images, input lengths",min(s["layout"]["input_length"] for s in samples),max(s["layout"]["input_length"] for s in samples),flush=True)


def verify():
    p = read(OUT / "protocol.json")
    assert p["code_sha256"] == source_hashes(), "Frozen code changed"
    assert p["sample_manifest_sha256"] == sha256(OUT / "sample_manifest.json")
    assert p["environment_sha256"] == sha256(OUT / "environment.json") and environment() == read(OUT / "environment.json")
    for path,expected in p["llava_bindings"].items():
        assert sha256(path) == expected
    return p


def load():
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and torch.cuda.device_count() == 1
    processes = subprocess.check_output(["nvidia-smi","-i","1","--query-compute-apps=pid,used_gpu_memory","--format=csv,noheader,nounits"],text=True).strip()
    assert not processes, f"GPU1 busy: {processes}; do not switch GPU or terminate other processes"
    write(OUT / f"gpu1_load_{time.time_ns()}.json",dict(pid=os.getpid(),visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        process_device="cuda:0",physical_gpu=1,uuid=subprocess.check_output(["nvidia-smi","-i","1","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()))
    processor = AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="eager",device_map={"":"cuda:0"},use_safetensors=True,low_cpu_mem_usage=True).eval()
    assert {str(p.device) for p in model.parameters()} == {"cuda:0"}
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert {str(b.device) for b in model.buffers()} == {"cuda:0"}
    assert len(model.model.language_model.layers) == 36
    assert model.config.text_config._attn_implementation == model.config.vision_config._attn_implementation == "eager"
    return model,processor


def inputs_for(processor,sample):
    inputs,layout = prepare(processor,sample)
    assert layout == sample["layout"] and digest(layout) == sample["input_binding"]
    return inputs.to("cuda:0")


def historical(sample):
    return read(OLD / "checkpoints" / f"{sample['task']}__{sample['question_id']}.json")["conditions"]["Full"]


def smoke():
    verify()
    assert read(OUT / "cpu_tests.json")["passed"]
    model,processor = load()
    before = hook_counts(model)
    versions = [(n,p._version,p.data_ptr()) for n,p in model.named_parameters()]
    ordered = sorted(read(OUT / "sample_manifest.json")["samples"],key=lambda r:r["layout"]["input_length"])
    cases = [ordered[0],ordered[len(ordered)//2],ordered[-1]]
    audit = dict(passed=False,cases=[],protocol_sha256=sha256(OUT / "protocol.json"))
    started = time.perf_counter()
    try:
        for sample in cases:
            inputs = inputs_for(processor,sample)
            hashes = {k:tensor_digest(v) for k,v in inputs.items()}
            full,_,_,native_trace = prefill(model,inputs,trace=True)
            noop,_,_,_ = prefill(model,inputs,[0,14,35],noop=True)
            observed,heads,error,_ = prefill(model,inputs,layout=sample["layout"],groups=sample["source_partition"])
            assert torch.equal(full,noop) and torch.equal(full,observed)
            assert int(full.argmax()) == historical(sample)["first_token_id"]
            assert heads.shape == (36,32,5)
            for layer in (0,14,35):
                changed,_,_,trace = prefill(model,inputs,[layer],trace=True)
                for i in range(36):
                    current,reference = trace.records[i],native_trace.records[i]
                    assert torch.equal(current["attention"][:,:-1],reference["attention"][:,:-1])
                    assert torch.equal(current["output"][:,:-1],reference["output"][:,:-1])
                    if i == layer:
                        assert torch.count_nonzero(current["attention"][:,-1]) == 0
                    else:
                        assert torch.count_nonzero(current["attention"][:,-1]) > 0
                assert sensitivity(full,changed)["full_first_token"] == int(full.argmax())
                del trace
            import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
            original = modeling.eager_attention_forward
            try:
                with MassObserver(model,sample["layout"],sample["source_partition"]):
                    raise RuntimeError("intentional observer cleanup")
            except RuntimeError as exc:
                assert str(exc) == "intentional observer cleanup"
            assert modeling.eager_attention_forward is original and hook_counts(model) == before
            assert hashes == {k:tensor_digest(v) for k,v in inputs.items()}
            audit["cases"].append(dict(task=sample["task"],question_id=sample["question_id"],
                input_length=sample["layout"]["input_length"],image_tokens=sample["layout"]["image_tokens"],
                valid_first_answer_vocab_logits_noop_max_abs=0.,mass_observer_logits_max_abs=0.,
                historical_full_top1_exact=True,mass_row_max_error=error,
                target_only_zero=True,non_p1_prefill_exact=True,residual_bf16_additions_exact=True,
                deepstack_three_injections_exact=True,all_36_mlp_called=True,kv_lengths_correct=True))
            print(f"SMOKE passed {sample['task']} {sample['question_id']} length={sample['layout']['input_length']}",flush=True)
            del inputs,native_trace
        assert versions == [(n,p._version,p.data_ptr()) for n,p in model.named_parameters()]
        audit.update(passed=True,weights_unchanged=True,hook_and_function_cleanup=True,no_nan_inf_oom=True)
    except BaseException:
        audit["traceback"] = traceback.format_exc()
        raise
    finally:
        audit.update(seconds=time.perf_counter()-started,cuda_allocated_peak_mib=torch.cuda.max_memory_allocated()/1024**2,
                     cuda_reserved_peak_mib=torch.cuda.max_memory_reserved()/1024**2)
        write(OUT / "validation.json",audit)


def save_tensors(path,values):
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(".tmp")
    save_file({k:v.contiguous() for k,v in values.items()},str(temporary))
    os.replace(temporary,path)


def formal():
    from safetensors.torch import load_file
    verify()
    assert read(OUT / "validation.json")["passed"]
    assert read(OUT / "validation.json")["protocol_sha256"] == sha256(OUT / "protocol.json")
    sys.path.insert(0,str(Path(__file__).parents[1] / "qwen3vl_smoke"))
    from ..qwen3vl_smoke.run import MemoryMonitor
    model,processor = load()
    samples = read(OUT / "sample_manifest.json")["samples"]
    binding = sha256(OUT / "protocol.json")
    commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    before = hook_counts(model)
    versions = [(n,p._version,p.data_ptr()) for n,p in model.named_parameters()]
    native_dir,scan_dir = OUT / "native_checkpoints",OUT / "scan_checkpoints"
    native_dir.mkdir(exist_ok=True)
    scan_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    counts = dict(native_prefills=0,ablated_prefills=0)
    with MemoryMonitor(1,OUT / "gpu1_memory.csv","120-prefill-only") as monitor:
        try:
            for index,sample in enumerate(samples,1):
                key = f"{sample['task']}__{sample['question_id']}"
                path = native_dir / f"{key}.json"
                if path.exists():
                    checkpoint = read(path)
                    assert checkpoint["protocol_sha256"] == binding and checkpoint["input_binding"] == sample["input_binding"]
                    assert sha256(checkpoint["raw_path"]) == checkpoint["raw_sha256"]
                    continue
                inputs = inputs_for(processor,sample)
                logits,heads,error,_ = prefill(model,inputs,layout=sample["layout"],groups=sample["source_partition"])
                assert int(logits.argmax()) == historical(sample)["first_token_id"]
                raw_path = RAW / "native" / f"{key}.safetensors"
                save_tensors(raw_path,dict(logits=logits,head_mass=torch.from_numpy(heads)))
                write(path,dict(task=sample["task"],question_id=sample["question_id"],image_sha256=sample["image_sha256"],
                    input_binding=sample["input_binding"],protocol_sha256=binding,code_commit=commit,
                    raw_path=str(raw_path),raw_sha256=sha256(raw_path),full_first_token=int(logits.argmax()),
                    full_logits_sha256=tensor_digest(logits),native_mass=heads.mean(1).tolist(),head_std=heads.std(1).tolist(),
                    mass_row_max_error=error,historical_full_top1_exact=True,
                    historical_full_token_logp_difference=float(logits.float().log_softmax(-1)[int(logits.argmax())])-historical(sample)["full_first_token_logp"]))
                counts["native_prefills"] += 1
                assert hook_counts(model) == before
                del inputs
                print(f"MASS {index}/120 elapsed={time.perf_counter()-started:.1f}s",flush=True)
            write(OUT / "mass_completion.json",dict(completed=120,protocol_sha256=binding,seconds=time.perf_counter()-started))
            for index,sample in enumerate(samples,1):
                key = f"{sample['task']}__{sample['question_id']}"
                path = scan_dir / f"{key}.json"
                if path.exists():
                    saved = read(path)
                    assert saved["protocol_sha256"] == binding and saved["input_binding"] == sample["input_binding"]
                    assert len(saved["rows"]) == 36 and sha256(saved["raw_path"]) == saved["raw_sha256"]
                    continue
                native = read(native_dir / f"{key}.json")
                full = load_file(native["raw_path"])["logits"]
                inputs = inputs_for(processor,sample)
                changed_logits,rows = [],[]
                for layer in range(36):
                    changed,_,_,_ = prefill(model,inputs,[layer])
                    rows.append(dict(layer=layer,**sensitivity(full,changed)))
                    changed_logits.append(changed)
                    counts["ablated_prefills"] += 1
                raw_path = RAW / "single_layer" / f"{key}.safetensors"
                save_tensors(raw_path,dict(logits=torch.stack(changed_logits)))
                write(path,dict(task=sample["task"],question_id=sample["question_id"],image_sha256=sample["image_sha256"],
                    input_binding=sample["input_binding"],protocol_sha256=binding,code_commit=commit,
                    native_checkpoint_sha256=sha256(native_dir / f"{key}.json"),raw_path=str(raw_path),
                    raw_sha256=sha256(raw_path),rows=rows))
                assert hook_counts(model) == before
                assert sample["layout"]["input_tensor_hashes"] == {k:tensor_digest(v) for k,v in inputs.items()}
                del inputs,changed_logits
                elapsed = time.perf_counter()-started
                print(f"SCAN {index}/120 layers=36 elapsed={elapsed:.1f}s",flush=True)
                write(OUT / "progress.json",dict(stage="single_layer_scan",completed_questions=index,total=120,
                    elapsed_seconds=elapsed,counts=counts,status="running",time=time.time()))
            assert versions == [(n,p._version,p.data_ptr()) for n,p in model.named_parameters()]
            assert hook_counts(model) == before
            write(OUT / "execution.json",dict(status="completed",questions=120,layers=36,formal_counts=counts,
                seconds=time.perf_counter()-started,code_commit=commit,protocol_sha256=binding,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                sampled_process_peak_mib=monitor.peak,monitor_errors=monitor.errors,
                weights_versions_and_storage_unchanged=True,hooks_clean=True,generation_calls=0))
        except BaseException:
            write(OUT / f"failure_{time.time_ns()}.json",dict(traceback=traceback.format_exc(),counts=counts,
                elapsed_seconds=time.perf_counter()-started,protocol_sha256=binding))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["freeze","smoke","formal"])
    globals()[parser.parse_args().stage]()

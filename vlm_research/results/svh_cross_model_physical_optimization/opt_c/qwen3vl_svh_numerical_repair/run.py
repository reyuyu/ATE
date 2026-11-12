"""Three fixed inputs; historical reproduction and Plan-B HF numerical gates."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from ..qwen3vl_early_windows.experiment import prepare as historical_prepare, tensor_digest, hook_counts
from .engine import generate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "vlm_research/results/qwen3vl_svh_physical_numerical_repair"
MODEL = Path("/data/lk/ATE-resources/models/qwen3-vl-8b-instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
PARENT = "d89363f17a394de559f76366890551d3c802a1d1"
HISTORY = "fc3f10eabf48a30b05f4e927a00f50bcd20780ad"
UUID = "GPU-96bc853f-ceea-2404-c704-c48a3d1ca185"
GATE = dict(mean_abs=0.02, max_abs=0.5, top1_equal=True, complete_tokens_equal=True)
ORDER = ["layer_input", "input_norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm",
         "rope_q", "rope_k", "mask", "probabilities", "AV", "o_proj", "attention_output",
         "attention_residual", "post_norm", "gate_proj", "up_proj", "down_proj", "mlp_output", "layer_output"]


def sha(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, p)


def table(p, rows):
    if not rows:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)


def metrics(a, b):
    assert a.shape == b.shape
    bad = ~(torch.isfinite(a) & torch.isfinite(b))
    mismatch = int((bad & (a != b)).sum())
    if mismatch:
        # A causal finfo.min and a hard -inf must be recorded, not subtracted.
        # Logits and real activations have independent finite-value assertions.
        return dict(equal=False, max_abs=None, mean_abs=None, nonfinite_mismatch=mismatch)
    d = torch.where(bad, 0., a.float() - b.float()).abs()
    return dict(equal=torch.equal(a, b), max_abs=float(d.max()), mean_abs=float(d.mean()))


def compare(a, av, b, bv):
    rows = []
    for t, (x, y) in enumerate(zip(av, bv)):
        r = dict(step=t, **metrics(x, y), top1_equal=int(x.argmax()) == int(y.argmax()),
                 complete_tokens_equal=a["token_ids"] == b["token_ids"], steps_equal=len(av) == len(bv))
        r["pass"] = r["max_abs"] is not None and r["max_abs"] <= GATE["max_abs"] and r["mean_abs"] <= GATE["mean_abs"] and r["top1_equal"] and r["complete_tokens_equal"] and r["steps_equal"]
        rows.append(r)
    return rows


def source_files():
    dirs = ["qwen3vl_svh_numerical_repair", "qwen3vl_c3v2_physical", "qwen3vl_c3v2_numerical_repair2",
            "qwen3vl_c3v2_numerical_debug", "qwen3vl_c3v2_l22_kernel_diagnosis", "qwen3vl_early_windows", "qwen3vl_smoke"]
    result = {str(p.relative_to(ROOT)): sha(p) for d in dirs for p in (ROOT / "vlm_research" / d).glob("*.py")}
    result["/data/lk/ATE-resources/experiments/qwen3vl_c3v2_l22_kernel_diagnosis/build/liblt_bridge.so"] = sha("/data/lk/ATE-resources/experiments/qwen3vl_c3v2_l22_kernel_diagnosis/build/liblt_bridge.so")
    return result


def prepare(processor, sample, plan_b):
    if not plan_b:
        return historical_prepare(processor, sample)
    from PIL import Image
    assert sha(sample["image_path"]) == sample["image_sha256"]
    body = sample["question"] + "\nAnswer the question using a single word or phrase."
    messages = [{"role": "user", "content": [{"type": "image", "image": sample["image_path"]}, {"type": "text", "text": body}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with Image.open(sample["image_path"]) as im:
        inputs = processor(text=[prompt], images=[im.convert("RGB")], return_tensors="pt",
                           min_pixels=65536, max_pixels=2097152)
    ids = inputs["input_ids"][0].tolist()
    visual_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    positions = [i for i, tid in enumerate(ids) if tid == visual_id]
    grid = inputs["image_grid_thw"][0].tolist()
    assert len(positions) == grid[0] * grid[1] * grid[2] // 4
    assert positions == list(range(positions[0], positions[-1] + 1))
    return inputs, dict(prompt=prompt, input_ids=ids, input_length=len(ids), p_minus_1=len(ids)-1,
                        image_positions=positions, image_tokens=len(positions), image_grid_thw=grid,
                        input_tensor_hashes={k: tensor_digest(v) for k, v in inputs.items()})


def cpu_prepare():
    assert not torch.cuda.is_initialized()
    historical = json.loads(subprocess.check_output(["git", "show", f"{HISTORY}:vlm_research/results/qwen3vl_c3v2_prefill_av_final/sample_manifest.json"], cwd=ROOT))
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    bindings = []
    for s in historical["samples"]:
        for b in [False, True]:
            inputs, layout = prepare(processor, s, b)
            if not b:
                assert layout == historical["input_bindings"][s["id"]], "Historical binding mismatch"
            bindings.append(dict(sample=s, plan_b=b, layout=layout))
    write(OUT / "sample_manifest.json", bindings)
    write(OUT / "protocol.json", dict(parent=PARENT, historical=HISTORY, label="ATE_UNIFIED_HF_NON_AUTHOR_OFFICIAL",
          author_official_status="BLOCKED", gate=GATE, native_noop="torch.equal all valid logits", gpu_uuid=UUID,
          historical_inputs="Frozen original min/median/max; exact tensor and prompt hashes required",
          plan_b="Official model chat; min_pixels65536/max_pixels2097152; GQA single word or phrase suffix",
          revision=MODEL.name, bf16=True, eager=True, max_new_tokens=32, logits_to_keep=1,
          source_files=source_files(), dataset_evaluations_authorized=False))
    write(OUT / "cpu_preparation.json", dict(status="PASS", cuda_initialized=torch.cuda.is_initialized(), bindings=len(bindings)))
    print("CPU preparation PASS", [(r["sample"]["question_id"], r["plan_b"], r["layout"]["input_length"], r["layout"]["image_tokens"]) for r in bindings], flush=True)


class Trace:
    """Capture real computations, retaining only matched text rows in RAM."""
    def __init__(self, model, layout):
        self.model = model
        self.d = model.model.language_model
        self.n = layout["input_length"]
        self.images = set(layout["image_positions"])
        self.text = [i for i in range(self.n) if i not in self.images]
        self.step = 0
        self.layer = -1
        self.records = {}
        self.positions = {}
        self.deepstack = []
        self.masks = []
        self.caches = {}

    def logical(self, length):
        if self.step and length == 1:
            return [self.n + self.step - 1]
        if length == self.n:
            return list(range(self.n))
        if length == self.n - 1:
            return list(range(self.n - 1))
        if length == len(self.text):
            return self.text
        if length == self.n + self.step:
            return list(range(length))
        if length == len(self.text) + self.step:
            return self.text + list(range(self.n, self.n+self.step))
        raise AssertionError(("Unknown position extent", self.step, self.layer, length))

    def put(self, stage, x, axis=1, keys=None):
        pos = self.logical(x.shape[axis])
        ix = [i for i, p in enumerate(pos) if p not in self.images]
        if not ix:
            return
        ix = torch.tensor(ix, device=x.device)
        value = x.detach().index_select(axis, ix).contiguous().cpu()
        self.records[(self.step, self.layer, stage)] = dict(value=value, positions=[p for p in pos if p not in self.images], axis=axis, keys=keys)

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            def begin(m, args, kw):
                cp = kw["cache_position"]
                self.step = 0 if int(cp[0]) == 0 else int(cp[0]) - self.n + 1
            h = self.model.register_forward_pre_hook(begin, with_kwargs=True)
            stack.callback(h.remove)
            for i, layer in enumerate(self.d.layers):
                def start(m, args, kw, i=i):
                    self.layer = i
                    self.put("layer_input", args[0])
                    self.cache = kw["past_key_values"]
                    if self.step == 0:
                        p = kw["cache_position"].detach().cpu().tolist()
                        ix = torch.tensor([j for j, v in enumerate(p) if v not in self.images], device=args[0].device)
                        self.positions[i] = dict(cp=[v for v in p if v not in self.images],
                            position_ids=kw["position_ids"].index_select(-1, ix).detach().cpu(),
                            cos=kw["position_embeddings"][0].index_select(1, ix).detach().cpu(),
                            sin=kw["position_embeddings"][1].index_select(1, ix).detach().cpu())
                h = layer.register_forward_pre_hook(start, with_kwargs=True)
                stack.callback(h.remove)
                modules = [(layer.input_layernorm, "input_norm"), (layer.post_attention_layernorm, "post_norm"),
                           (layer.mlp, "mlp_output"), (layer, "layer_output")]
                modules += [(getattr(layer.self_attn, name), name) for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj")]
                modules += [(getattr(layer.mlp, name), name) for name in ("gate_proj", "up_proj", "down_proj")]
                for module, name in modules:
                    h = module.register_forward_hook(lambda m, a, o, name=name: self.put(name, o))
                    stack.callback(h.remove)
                h = layer.self_attn.register_forward_hook(lambda m, a, o: self.put("attention_output", o[0]))
                stack.callback(h.remove)
                h = layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: self.put("attention_residual", a[0]))
                stack.callback(h.remove)
                def cache_end(m, args, out, i=i):
                    c = self.cache
                    if c is None:
                        return
                    if hasattr(c, "entries"):
                        pair = c.entries[i]
                        pair = None if pair is None else [v[..., :c.used[i], :] for v in pair]
                    else:
                        pair = [c.layers[i].keys, c.layers[i].values]
                    if pair is None:
                        self.caches[(self.step, i)] = None
                        return
                    pos = self.logical(pair[0].shape[-2])
                    ix = [j for j, p in enumerate(pos) if p not in self.images]
                    self.caches[(self.step, i)] = dict(positions=[pos[j] for j in ix],
                        k=pair[0][..., ix, :].detach().cpu().clone(), v=pair[1][..., ix, :].detach().cpu().clone())
                h = layer.register_forward_hook(cache_end)
                stack.callback(h.remove)
            rotary = qm.apply_rotary_pos_emb
            def rope(q, k, *args, **kw):
                out = rotary(q, k, *args, **kw)
                self.put("rope_q", out[0], 2)
                self.put("rope_k", out[1], 2)
                return out
            stack.enter_context(patch.object(qm, "apply_rotary_pos_emb", rope))
            eager = qm.eager_attention_forward
            def observe(module, q, k, v, attention_mask=None, **kw):
                out = eager(module, q, k, v, attention_mask, **kw)
                if not hasattr(module, "layer_idx"):
                    return out
                keys = self.logical(k.shape[-2])
                self.put("probabilities", out[1], 2, keys)
                self.put("AV", out[0])
                if attention_mask is not None:
                    self.put("mask", attention_mask, 2, keys)
                    blocked = torch.isneginf(attention_mask)
                    assert not bool(blocked.all(-1).any()), "All keys blocked"
                    p = out[1].masked_select(blocked.expand_as(out[1]))
                    assert not p.numel() or float(p.abs().max()) == 0
                    self.masks.append(dict(step=self.step, layer=self.layer, blocked_probability=0.,
                                           row_sum_error=float((out[1].float().sum(-1)-1).abs().max())))
                return out
            stack.enter_context(patch.object(qm, "eager_attention_forward", observe))
            native_deep = self.d._deepstack_process
            def deep(hidden, mask, feature):
                out = native_deep(hidden, mask, feature)
                self.deepstack.append(dict(layer=self.layer, feature=tensor_digest(feature), mask=tensor_digest(mask)))
                return out
            stack.enter_context(patch.object(self.d, "_deepstack_process", deep))
            yield self


def trace_diff(a, b):
    rows = []
    for key in sorted(a.records.keys() & b.records.keys(), key=lambda k: (k[0], k[1], ORDER.index(k[2]))):
        x, y = a.records[key], b.records[key]
        pos = sorted(set(x["positions"]) & set(y["positions"]))
        xx = x["value"].index_select(x["axis"], torch.tensor([x["positions"].index(p) for p in pos]))
        yy = y["value"].index_select(y["axis"], torch.tensor([y["positions"].index(p) for p in pos]))
        if x["keys"] is not None:
            keys = sorted(set(x["keys"]) & set(y["keys"]))
            xx = xx[..., [x["keys"].index(p) for p in keys]]
            yy = yy[..., [y["keys"].index(p) for p in keys]]
        r = dict(step=key[0], layer=key[1], component=key[2], **metrics(xx, yy))
        # Pre-zero o_proj output at skipped P-1 is intentionally absent.
        r["scope"] = "shared real logical text rows; keys aligned by original position"
        rows.append(r)
    first = next((r for r in rows if not r["equal"] and r["component"] != "mask"), None)
    return rows, first


def audit_semantics(a, b):
    positions_equal = all(x["cp"] == b.positions[i]["cp"] and all(torch.equal(x[k], b.positions[i][k]) for k in ["position_ids", "cos", "sin"]) for i, x in a.positions.items())
    caches = []
    for key, y in b.caches.items():
        if y is None:
            assert key[1] in range(1, 7)
            continue
        x = a.caches[key]
        assert x["positions"] == y["positions"]
        caches.append(dict(step=key[0], layer=key[1], positions=y["positions"], k=metrics(x["k"], y["k"]), v=metrics(x["v"], y["v"])))
    return dict(mrope_original_positions_exact=positions_equal, deepstack_exact=a.deepstack == b.deepstack,
                cache_comparisons=caches, masks=b.masks)


def live_preflight():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == UUID
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    assert not any(UUID in line for line in raw.splitlines()), "Selected GPU acquired by another process"
    state = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu", "--format=csv"], text=True)
    assert "RTX 3090" in next(line for line in state.splitlines() if UUID in line)
    assert os.statvfs("/data").f_bavail * os.statvfs("/data").f_frsize > 2 * 2**30
    return dict(gpus=state, processes=raw, selected_uuid=UUID)


def gpu_run(verify=False, label=None):
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["source_files"] == source_files(), "Changed execution snapshot"
    assert protocol["gate"] == GATE
    tag = label or ("independent" if verify else "diagnosis")
    assert not (OUT / tag).exists(), "Preserve earlier attempts; select a new label"
    write(OUT / tag / "preflight.json", live_preflight())
    write(OUT / tag / "execution_protocol.json", protocol)
    torch.set_num_threads(1)
    torch.manual_seed(3407)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="eager", device_map={"": "cuda:0"}).eval()
    hooks = hook_counts(model)
    summaries = []
    bindings = json.loads((OUT / "sample_manifest.json").read_text())
    if verify:
        bindings = [r for r in bindings if r["plan_b"]]
    for item in bindings:
        sample = item["sample"]
        inputs, layout = prepare(processor, sample, item["plan_b"])
        assert layout == item["layout"]
        folder = OUT / tag / (("planB_" if item["plan_b"] else "historical_") + sample["question_id"])
        records = {}
        traces = {}
        conditions = ["Native", "Noop", "Dense", "Physical"] if verify else ["Native", "Noop", "Dense", "Original", "Physical"]
        for name in conditions:
            kind = "N" if name in ["Native", "Noop"] else "F" if name == "Dense" else "C"
            trace = Trace(model, layout) if name in ["Dense", "Original", "Physical"] and not verify else None
            with ExitStack() as stack:
                if trace:
                    stack.enter_context(trace.context())
                result, vectors = generate(model, processor, inputs, layout, kind, repaired=name == "Physical",
                                           audit=True, capture=True, noop=name == "Noop", max_tokens=32)
            assert hooks == hook_counts(model), "Hook leak"
            assert {k: tensor_digest(v) for k, v in inputs.items()} == layout["input_tensor_hashes"]
            assert all(torch.isfinite(v).all() for v in vectors)
            # Only small complete valid vocabulary vectors are retained; no full attention arrays.
            write(folder / (name + ".json"), result)
            torch.save(dict(logits=vectors, token_ids=result["token_ids"]), folder / (name + "_logits.pt"))
            records[name] = (result, vectors)
            if trace:
                traces[name] = trace
            print(tag, folder.name, name, result["token_ids"], flush=True)
        native, noop = records["Native"], records["Noop"]
        strict = native[0]["token_ids"] == noop[0]["token_ids"] and len(native[1]) == len(noop[1]) and all(torch.equal(x, y) for x, y in zip(native[1], noop[1]))
        assert strict, "Native/No-op strict logits failed"
        # Independently exercise HF GenerationMixin, including the genuine EOS step.
        with torch.inference_mode():
            hf = model.generate(**{k: v.cuda() for k, v in inputs.items()}, do_sample=False, num_beams=1,
                    max_new_tokens=32, use_cache=True, return_dict_in_generate=True, output_logits=True,
                    logits_to_keep=1, eos_token_id=[151645, 151643], pad_token_id=151643)
        hf_strict = hf.sequences[0, layout["input_length"]:].tolist() == native[0]["token_ids"] and len(hf.logits) == len(native[1]) and all(torch.equal(x, y[0].float().cpu()) for x, y in zip(native[1], hf.logits))
        assert hf_strict, "HF native generation mismatch"
        del hf
        rows = []
        for name in ["Original", "Physical"]:
            if name not in records:
                continue
            comparison = compare(*records["Dense"], *records[name])
            rows += [dict(comparison=name, **r) for r in comparison]
            if not verify:
                differences, first = trace_diff(traces["Dense"], traces[name])
                table(folder / (name + "_layerwise.csv"), differences)
                write(folder / (name + "_first_divergence.json"), first)
                write(folder / (name + "_semantics.json"), audit_semantics(traces["Dense"], traces[name]))
        table(folder / "logits_comparison.csv", rows)
        passed = all(r["pass"] for r in rows if r["comparison"] == "Physical")
        summary = dict(sample=sample["question_id"], plan_b=item["plan_b"], native_noop_strict=strict,
                       hf_native_strict=hf_strict, physical_pass=passed, rows=rows,
                       input_length=layout["input_length"], image_tokens=layout["image_tokens"], gpu_uuid=UUID)
        summaries.append(summary)
        write(folder / "validation.json", summary)
        write(OUT / tag / "summary.json", summaries)
        print("GATE", summary, flush=True)
        del records, traces, result, vectors
        gc.collect()
    write(OUT / tag / "completion.json", dict(status="PASS" if all(s["physical_pass"] for s in summaries) else "FAIL",
          inputs_completed=len(summaries), gpu_uuid=UUID, source_files=source_files(), benchmark_predictions=0))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "diagnose", "verify"])
    parser.add_argument("--label")
    args = parser.parse_args()
    try:
        cpu_prepare() if args.mode == "prepare" else gpu_run(args.mode == "verify", args.label)
    except BaseException as exc:
        write(OUT / ((args.label or args.mode) + "_failure.json"), dict(error=repr(exc), traceback=traceback.format_exc()))
        raise

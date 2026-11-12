"""Frozen sampling, official prompt preparation, and query-local output hooks."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import torch

from ..qwen3vl_smoke.common import question_positions, sha256, write

SEED = 20261101
CONDITIONS = {"Full": [], "W0": list(range(4)), "W1": list(range(1, 5)),
              "W2": list(range(2, 6)), "W3": list(range(3, 7)),
              "W4": list(range(4, 8)), "W5": list(range(16, 20))}
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL = Path("/data/lk/ATE-resources/models/qwen3-vl-8b-instruct") / REVISION
SOURCE = Path("/data/lk/ATE-worktrees/single-layer-gqa6/vlm_research/results/paper_rq1_single_layer_gqa6/sample_manifest.json")
OUT = Path(__file__).resolve().parents[1] / "results/qwen3vl_early_attention_windows_240"
HISTORICAL = Path("/data/lk/ATE-worktrees/first-answer-early-attention/vlm_research/paper_rq1/scorers.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def gqa_prediction(text):
    # Exact existing pinned official-GQA submission conversion, not a new parser.
    return text.rstrip('.').lower()


def select(rows, count=40):
    tasks = sorted({r["task"] for r in rows})
    pools = {t: sorted([r for r in rows if r["task"] == t], key=lambda r:
             digest([SEED, t, r["question_id"], r["image_sha256"]])) for t in tasks}
    selected, used, qids = [], set(), set()
    for _ in range(count):
        for task in tasks:
            candidates = [r for r in pools[task] if (task, r["question_id"]) not in qids]
            row = next((r for r in candidates if r["image_sha256"] not in used), candidates[0])
            selected.append(row)
            used.add(row["image_sha256"])
            qids.add((task, row["question_id"]))
    return selected


def prepare(processor, sample):
    from PIL import Image
    assert sha256(sample["image_path"]) == sample["image_sha256"]
    with Image.open(sample["image_path"]) as handle:
        image = handle.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": sample["image_path"]},
        {"type": "text", "text": sample["question"] + "\nAnswer using a single word or short phrase."}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")[0].tolist()
    ids = inputs["input_ids"][0].tolist()
    visual = [i for i, v in enumerate(ids) if v == processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    grid = inputs["image_grid_thw"][0].tolist()
    count = grid[0] * grid[1] * grid[2] // processor.image_processor.merge_size ** 2
    assert len(visual) == count and visual == list(range(visual[0], visual[-1] + 1))
    expanded = prompt.replace(processor.image_token, processor.image_token * count)
    assert processor.tokenizer(expanded)["input_ids"] == ids
    qpos = question_positions(offsets, expanded, sample["question"])
    assert processor.tokenizer.decode([ids[i] for i in qpos]).strip() == sample["question"].strip()
    layout = dict(prompt=prompt, input_ids=ids, input_length=len(ids), p_minus_1=len(ids)-1,
                  image_positions=visual, image_tokens=count, question_positions=qpos, image_grid_thw=grid,
                  input_tensor_hashes={k: tensor_digest(v) for k, v in inputs.items()})
    assert inputs["attention_mask"].bool().all() and len(inputs["input_ids"]) == 1
    return inputs, layout


class AttentionBypass:
    """Post-o_proj edit only; native Q/K/V and cache updates have already run."""
    def __init__(self, model, layers, prompt_length, noop=False):
        if len(set(layers)) != len(layers) or any(i < 0 or i >= len(model.model.language_model.layers) for i in layers):
            raise ValueError("Invalid language decoder layers")
        self.model, self.layers, self.prompt_length, self.noop = model, list(layers), prompt_length, noop
        self.events, self.calls, self.handles = [], [], []
        self.query_length = self.cache_position = None

    def begin(self, module, args, kwargs):
        embeddings = kwargs.get("inputs_embeds")
        ids = kwargs.get("input_ids")
        self.query_length = (embeddings if embeddings is not None else ids).shape[1]
        cp = kwargs.get("cache_position")
        assert cp is not None
        positions = cp.detach().cpu().tolist()
        expected = 0 if not self.calls else self.prompt_length + len(self.calls) - 1
        assert positions == list(range(expected, expected + self.query_length))
        assert self.query_length == (self.prompt_length if not self.calls else 1)
        self.cache_position = positions[-1]
        self.calls.append(dict(query_length=self.query_length, cache_position=positions,
                               phase="prefill" if len(self.calls) == 0 else "decode"))

    def edit(self, layer, module, args, output):
        assert output.ndim == 3 and output.shape[0] == 1 and output.shape[1] == self.query_length
        row = self.prompt_length - 1 if len(self.calls) == 1 else 0
        if self.noop:
            result = output
        else:
            result = output.clone()
            result[:, row, :] = 0
            assert torch.count_nonzero(result[:, row, :]) == 0
            assert torch.equal(result[:, :row], output[:, :row])
            assert torch.equal(result[:, row+1:], output[:, row+1:])
        self.events.append(dict(layer=layer, phase=self.calls[-1]["phase"], local_query=row,
                                cache_position=self.cache_position, noop=self.noop))
        return result

    def __enter__(self):
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.begin, with_kwargs=True))
            for layer in self.layers:
                module = self.model.model.language_model.layers[layer].self_attn.o_proj
                self.handles.append(module.register_forward_hook(lambda m, a, o, i=layer: self.edit(i, m, a, o)))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class PrefillTrace:
    """Read-only smoke trace: native prefix, residual arithmetic, and DeepStack."""
    def __init__(self, model, length, visual):
        self.model, self.length, self.visual = model, length, visual
        self.handles, self.records, self.features = [], {}, {}
        self.mlp_calls = [0] * 36

    def __enter__(self):
        def attn(i, module, args, output):
            if output.shape[1] == self.length:
                self.records[i]["attention"] = output.detach().cpu().clone()
        def start(i, module, args):
            if args[0].shape[1] == self.length:
                self.records[i] = {"input": args[0].detach().cpu().clone()}
        def mid(i, module, args):
            if args[0].shape[1] == self.length:
                self.records[i]["mid"] = args[0].detach().cpu().clone()
        def mlp(i, module, args, output):
            self.mlp_calls[i] += 1
            if output.shape[1] == self.length:
                self.records[i]["mlp"] = output.detach().cpu().clone()
        def end(i, module, args, output):
            if output.shape[1] == self.length:
                self.records[i]["output"] = output.detach().cpu().clone()
        for i, layer in enumerate(self.model.model.language_model.layers):
            self.handles.extend([
                layer.register_forward_pre_hook(lambda m, a, j=i: start(j, m, a)),
                layer.self_attn.o_proj.register_forward_hook(lambda m, a, o, j=i: attn(j, m, a, o)),
                layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a, j=i: mid(j, m, a)),
                layer.mlp.register_forward_hook(lambda m, a, o, j=i: mlp(j, m, a, o)),
                layer.register_forward_hook(lambda m, a, o, j=i: end(j, m, a, o))])
        for i, merger in enumerate(self.model.model.visual.deepstack_merger_list):
            self.handles.append(merger.register_forward_hook(lambda m, a, o, j=i:
                                self.features.__setitem__(j, o.detach().cpu().clone())))
        return self

    def validate(self):
        assert len(self.records) == 36 and len(self.features) == 3
        for i, r in self.records.items():
            assert all(torch.isfinite(v).all() for v in r.values())
            # Actual BF16 addition, not FP32 approximate residual recomputation.
            assert torch.equal(r["input"] + r["attention"], r["mid"])
            assert torch.equal(r["mid"] + r["mlp"], r["output"])
            if i < 3:
                expected = r["output"][:, self.visual] + self.features[i].unsqueeze(0)
                assert torch.equal(expected, self.records[i+1]["input"][:, self.visual])
                assert torch.count_nonzero(self.features[i]) > 0
        return True

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def hook_counts(model):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


@torch.inference_mode()
def generate(model, processor, inputs, layers=(), noop=False, trace=False):
    length = inputs["input_ids"].shape[1]
    before = hook_counts(model)
    with ExitStack() as stack:
        intervention = stack.enter_context(AttentionBypass(model, layers, length, noop=noop))
        observer = stack.enter_context(PrefillTrace(model, length,
            (inputs["input_ids"][0] == model.config.image_token_id).nonzero().flatten().tolist())) if trace else None
        generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=32,
            use_cache=True, return_dict_in_generate=True, output_scores=True,
            output_attentions=False, output_hidden_states=False)
        ids = generated.sequences[0, length:].tolist()
        assert ids and len(intervention.calls) == len(ids)
        assert len(intervention.events) == len(layers) * len(ids)
        assert all(torch.isfinite(v).all() for v in generated.scores)
        assert all(generated.past_key_values.get_seq_length(i) == length+len(ids)-1 for i in range(36))
        if observer:
            observer.validate()
            assert observer.mlp_calls == [len(ids)] * 36
        first = generated.scores[0][0].detach().float().cpu().clone()
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
        text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        result = dict(text=text, prediction=gqa_prediction(text), token_ids=ids,
                      first_token_id=ids[0], first_top1_id=int(first.argmax()),
                      generated_tokens=len(ids), non_eos_tokens=sum(i not in eos_ids for i in ids),
                      eos=ids[-1] in eos_ids, truncated=ids[-1] not in eos_ids and len(ids) == 32,
                      empty=not bool(text), anomaly=False, decode_steps=len(ids)-1,
                      hook_events=intervention.events,
                      cache_lengths=[generated.past_key_values.get_seq_length(i) for i in range(36)])
        assert result["first_token_id"] == result["first_top1_id"]
    assert hook_counts(model) == before
    return result, first, observer

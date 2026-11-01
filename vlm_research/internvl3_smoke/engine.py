"""Query-local hooks and read-only row collection, separate from model code."""
import ast
from contextlib import ExitStack
import hashlib

import torch

SOURCES = ("anchor", "image", "question", "template", "self")
MASS_TOLERANCE = 2**-8 + 2e-6


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def official_preprocessor(directory):
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    text = (directory / "README.md").read_text(encoding="utf-8")
    text = text[text.index("IMAGENET_MEAN ="):]
    text = text[:text.index("def split_model(model_name):")]
    tree = ast.parse(text)
    names = {"build_transform", "find_closest_aspect_ratio", "dynamic_preprocess", "load_image"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    scope = dict(Image=Image, T=T, InterpolationMode=InterpolationMode, torch=torch,
                 IMAGENET_MEAN=(.485, .456, .406), IMAGENET_STD=(.229, .224, .225))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(directory / "README.md"), "exec"), scope)
    return scope["load_image"], scope["dynamic_preprocess"]


def partitions(tokenizer, fast_tokenizer, prompt, question, ids, image_id):
    encoding = fast_tokenizer(prompt, return_offsets_mapping=True)
    assert encoding["input_ids"] == ids, "Fast offsets must bind exact official slow-tokenizer IDs"
    assert prompt.count(question) == 1
    start = prompt.index(question)
    end = start + len(question)
    question_positions = [i for i, (a, b) in enumerate(encoding["offset_mapping"]) if a < end and b > start]
    anchor_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    anchors = [i for i, value in enumerate(ids) if value == anchor_id]
    assert anchors and question_positions
    groups = dict(anchor=[anchors[0]], image=[i for i, value in enumerate(ids) if value == image_id],
                  question=question_positions, self=[len(ids)-1])
    assigned = [i for positions in groups.values() for i in positions]
    assert len(assigned) == len(set(assigned)) and groups["image"]
    groups["template"] = [i for i in range(len(ids)) if i not in set(assigned)]
    assert sorted(i for positions in groups.values() for i in positions) == list(range(len(ids)))
    return dict(positions=groups, p_minus_1=len(ids)-1, input_length=len(ids),
                anchor_token_id=anchor_id, anchor_name="First actual <|im_start|> system-role opener, not assumed BOS",
                actual_bos_positions=[i for i, value in enumerate(ids) if value == tokenizer.bos_token_id],
                image_token_id=image_id, image_tokens=len(groups["image"]),
                question_character_span=[start, end], offsets_verified_against_official_ids=True)


class P1OutputHook:
    def __init__(self, module, position, noop=False):
        self.module, self.position, self.noop = module, position, noop
        self.calls = 0

    def __enter__(self):
        def hook(module, args, output):
            assert output.ndim == 3 and output.shape[0] == 1 and output.shape[1] == self.position+1
            self.calls += 1
            if self.noop:
                return output
            changed = output.clone()
            changed[:, self.position, :] = 0
            assert torch.equal(changed[:, :self.position], output[:, :self.position])
            assert torch.count_nonzero(changed[:, self.position]) == 0
            return changed
        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        self.handle.remove()


def reduce_mass(row, positions):
    assert row.ndim == 2 and torch.isfinite(row).all() and (row >= 0).all()
    assigned = [i for key in SOURCES for i in positions[key]]
    assert sorted(assigned) == list(range(row.shape[-1]))
    masses = torch.stack([row[:, positions[key]].sum(-1) for key in SOURCES], -1)
    error = float((masses.sum(-1)-1).abs().max())
    assert error <= MASS_TOLERANCE, f"Native BF16 row sum error {error}"
    return masses, error


class MassObserver:
    def __init__(self, layers, layout):
        self.targets = {id(module): layer for layer, module in layers.items()}
        self.layout, self.rows, self.errors = layout, {}, {}

    def __enter__(self):
        import transformers.models.qwen2.modeling_qwen2 as modeling
        self.modeling, self.original = modeling, modeling.eager_attention_forward
        def observer(module, *args, **kwargs):
            output, weights = self.original(module, *args, **kwargs)
            if id(module) in self.targets:
                layer = self.targets[id(module)]
                assert layer not in self.rows and weights.shape[-1] == self.layout["input_length"]
                row = weights[0, :, self.layout["p_minus_1"], :].detach().float().cpu()
                masses, error = reduce_mass(row, self.layout["positions"])
                self.rows[layer], self.errors[layer] = masses.tolist(), error
            return output, weights
        modeling.eager_attention_forward = observer
        return self

    def __exit__(self, *args):
        self.modeling.eager_attention_forward = self.original


class LayerTrace:
    """Keep P-1 vectors and prefix digests, not full hidden/attention matrices."""
    def __init__(self, layers, position):
        self.layers, self.position = layers, position
        self.records = [{} for _ in layers]

    def __enter__(self):
        self.stack = ExitStack()
        def install(module, callback, pre=False):
            method = module.register_forward_pre_hook if pre else module.register_forward_hook
            self.stack.callback(method(callback).remove)
        for index, layer in enumerate(self.layers):
            record = self.records[index]
            def capture(name, pre=False, record=record):
                def callback(module, args, output=None):
                    value = args[0] if pre else output[0] if isinstance(output, tuple) else output
                    assert value.shape[1] == self.position+1
                    assert name not in record
                    record[name] = value[0, self.position].detach().cpu().clone()
                    if name in ("attention", "output"):
                        record[name+"_prefix_sha256"] = tensor_hash(value[:, :self.position])
                return callback
            install(layer, capture("input", True), True)
            install(layer.self_attn.o_proj, capture("attention"))
            install(layer.post_attention_layernorm, capture("mid", True), True)
            install(layer.mlp, capture("mlp"))
            install(layer, capture("output"))
        return self

    def validate(self):
        for record in self.records:
            assert all(name in record for name in ("input", "attention", "mid", "mlp", "output"))
            assert all(torch.isfinite(record[name]).all() for name in ("input", "attention", "mid", "mlp", "output"))
            assert torch.equal(record["input"]+record["attention"], record["mid"])
            assert torch.equal(record["mid"]+record["mlp"], record["output"])

    def __exit__(self, *args):
        self.stack.close()


def hooks_count(model):
    return sum(len(module._forward_hooks)+len(module._forward_pre_hooks) for module in model.modules())


def cache_audit(cache, layers, expected_length):
    lengths = [int(cache.get_seq_length(i)) for i in range(layers)]
    assert lengths == [expected_length]*layers
    assert all(torch.isfinite(key).all() and torch.isfinite(value).all()
               for key, value in zip(cache.key_cache, cache.value_cache))
    return dict(class_name=type(cache).__name__, sequence_lengths=lengths,
                key_shapes=[list(key.shape) for key in cache.key_cache],
                value_shapes=[list(value.shape) for value in cache.value_cache])


@torch.inference_mode()
def replay(model, saved, layer=None, noop=False, layout=None):
    from transformers.cache_utils import DynamicCache
    inputs = {key: value.to("cuda:0") if isinstance(value, torch.Tensor) else value
              for key, value in saved.items() if key != "past_key_values"}
    assert saved["past_key_values"] == "DynamicCache(empty)"
    inputs["past_key_values"] = DynamicCache()
    layers = model.language_model.model.layers
    with ExitStack() as stack:
        intervention = stack.enter_context(P1OutputHook(layers[layer].self_attn.o_proj,
            inputs["inputs_embeds"].shape[1]-1, noop)) if layer is not None else None
        observer = stack.enter_context(MassObserver({i: layers[i].self_attn for i in (0, len(layers)-1)}, layout)) if layout else None
        trace = stack.enter_context(LayerTrace(layers, inputs["inputs_embeds"].shape[1]-1))
        output = model.language_model(**inputs)
        logits = output.logits[0, -1].detach().cpu().clone()
        assert torch.isfinite(logits).all()
        state = cache_audit(output.past_key_values, len(layers), inputs["inputs_embeds"].shape[1])
        trace.validate()
        if intervention:
            assert intervention.calls == 1
    return logits, trace.records, state, observer

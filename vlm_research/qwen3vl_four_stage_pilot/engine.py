"""Read actual native softmax rows and reuse the verified query-local bypass."""
from contextlib import ExitStack
import math

import numpy as np
import torch

from ..qwen3vl_early_windows.experiment import AttentionBypass, PrefillTrace, tensor_digest

SOURCES = ("anchor", "image", "question", "template", "self")
# BF16 unit roundoff bounds the sum of nonnegative rounded probabilities.
MASS_TOLERANCE = 2**-8 + 2e-6


def partition(tokenizer, layout, text_config):
    ids = layout["input_ids"]
    opener = tokenizer.convert_tokens_to_ids("<|im_start|>")
    anchors = [i for i, value in enumerate(ids) if value == opener]
    assert anchors, "No real chat opener; never invent a BOS or anchor"
    sets = dict(anchor=[anchors[0]], image=layout["image_positions"], question=layout["question_positions"],
                self=[layout["p_minus_1"]])
    assigned = [p for values in sets.values() for p in values]
    assert len(assigned) == len(set(assigned))
    sets["template"] = [p for p in range(len(ids)) if p not in set(assigned)]
    assert sorted(p for values in sets.values() for p in values) == list(range(len(ids)))
    return dict(positions={key: sets[key] for key in SOURCES}, anchor_token_id=opener,
        anchor_position=anchors[0], anchor_role="First user role opener <|im_start|>, not BOS",
        configured_bos_id=text_config.bos_token_id,
        actual_bos_positions=[i for i,t in enumerate(ids) if t == text_config.bos_token_id],
        special_token_positions={tokenizer.convert_ids_to_tokens(t): [i for i,x in enumerate(ids) if x == t]
            for t in tokenizer.all_special_ids if t in ids})


def reduce_row(row, positions):
    assert row.ndim == 2 and row.shape[0] == 32
    assert torch.isfinite(row).all() and (row >= 0).all()
    masses = torch.stack([row[:, positions[key]].sum(-1) for key in SOURCES], -1)
    error = float((masses.sum(-1)-1).abs().max())
    assert error <= MASS_TOLERANCE, f"Native BF16 row mass error {error}"
    return masses, error


class MassObserver:
    """No arithmetic changes: call official eager, copy current P-1 row, return originals."""
    def __init__(self, model, layout, groups):
        self.model, self.layout, self.groups = model, layout, groups
        self.rows, self.errors = {}, {}
        self.text_modules = {id(m.self_attn): i for i,m in enumerate(model.model.language_model.layers)}

    def __enter__(self):
        import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
        self.modeling, self.original = modeling, modeling.eager_attention_forward
        def observe(module, *args, **kwargs):
            output, probabilities = self.original(module, *args, **kwargs)
            if id(module) in self.text_modules:
                layer = self.text_modules[id(module)]
                assert layer not in self.rows, "Exactly one full prefill; no decode"
                assert probabilities.shape[0] == 1 and probabilities.shape[-1] == self.layout["input_length"]
                row = probabilities[0, :, self.layout["p_minus_1"], :].detach().float().cpu()
                masses, error = reduce_row(row, self.groups["positions"])
                self.rows[layer], self.errors[layer] = masses.numpy(), error
            return output, probabilities
        self.wrapper = observe
        modeling.eager_attention_forward = observe
        return self

    def __exit__(self, *_):
        assert self.modeling.eager_attention_forward is self.wrapper
        self.modeling.eager_attention_forward = self.original

    def array(self):
        assert sorted(self.rows) == list(range(36))
        return np.stack([self.rows[i] for i in range(36)])


@torch.inference_mode()
def prefill(model, inputs, layers=(), noop=False, layout=None, groups=None, trace=False):
    n = inputs["input_ids"].shape[1]
    with ExitStack() as stack:
        intervention = stack.enter_context(AttentionBypass(model, layers, n, noop=noop)) if layers else None
        observer = stack.enter_context(MassObserver(model, layout, groups)) if layout is not None else None
        tracer = stack.enter_context(PrefillTrace(model, n,
            (inputs["input_ids"][0] == model.config.image_token_id).nonzero().flatten().tolist())) if trace else None
        output = model(**inputs, use_cache=True, cache_position=torch.arange(n, device=inputs["input_ids"].device),
                       logits_to_keep=1, output_attentions=False, output_hidden_states=False)
        logits = output.logits[0, -1].detach().cpu().clone()
        assert torch.isfinite(logits).all()
        assert all(output.past_key_values.get_seq_length(i) == n for i in range(36))
        if intervention:
            assert len(intervention.calls) == 1 and len(intervention.events) == len(layers)
        if tracer:
            tracer.validate()
            assert tracer.mlp_calls == [1] * 36
        masses = observer.array() if observer else None
        error = max(observer.errors.values()) if observer else None
    return logits, masses, error, tracer


def sensitivity(full, changed):
    full, changed = full.float(), changed.float()
    assert full.ndim == 1 and full.shape == changed.shape
    assert torch.isfinite(full).all() and torch.isfinite(changed).all()
    token = int(full.argmax())
    p, q = full.log_softmax(-1), changed.log_softmax(-1)
    mixture = torch.logaddexp(p,q)-math.log(2)
    js = float(((p.exp()*(p-mixture)).sum()+(q.exp()*(q-mixture)).sum())/2)
    assert math.isfinite(js) and js >= -1e-6
    loss = float(p[token]-q[token])
    return dict(full_first_token=token, changed_first_token=int(changed.argmax()),
        full_top1_logp=float(p[token]), ablated_full_top1_logp=float(q[token]),
        signed_logp_loss=loss, absolute_logp_change=abs(loss), top1_flip=int(token != int(changed.argmax())),
        js_divergence=js, changed_logits_sha256=tensor_digest(changed.to(torch.bfloat16)))

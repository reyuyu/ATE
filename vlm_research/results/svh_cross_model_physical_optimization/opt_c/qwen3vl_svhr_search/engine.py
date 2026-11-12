from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

from ..qwen3vl_c3v2_physical import engine as original
from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest

FrozenExecution = original.Execution
SKIP = set(original.SKIP)


def restore_answer_access(mask, image_positions, allowed):
    assert mask is not None
    if not allowed:
        assert bool(torch.isneginf(mask[..., -1, image_positions]).all())
        return mask
    out = mask.clone()
    out[..., -1, image_positions] = 0
    return out


class Functional(FrozenExecution):
    def __init__(self, *args, layers=(), **kwargs):
        super().__init__(*args, **kwargs)
        assert self.c == 'F' and not self.physical and self.bypass and self.late
        self.read_layers = frozenset(layers)
        assert not self.read_layers & SKIP, 'NOT_APPLICABLE_FROZEN_SKIP'
        assert all(0 <= i < 36 for i in self.read_layers)
        self.read_audit = []
        self.prefill_hashes = []
        self.permission_audit = []

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            if self.audit_enabled:
                eager = qm.eager_attention_forward
                def observed(module, q, k, v, attention_mask=None, **kwargs):
                    out, weights = eager(module, q, k, v, attention_mask, **kwargs)
                    if hasattr(module, 'layer_idx'):
                        i = module.layer_idx
                        allowed = i in self.read_layers
                        visual = weights[..., -1, self.I]
                        assert bool(torch.isfinite(weights).all())
                        if not allowed:
                            assert float(visual.abs().max()) == 0
                        else:
                            assert bool((attention_mask[..., -1, self.I] == 0).all())
                        if self.step == 0 and i >= 22:
                            relay = list(range(self.I[-1]+1, self.n-1))
                            if relay:
                                assert float(weights[0, :, relay][:, :, self.I].abs().max()) == 0
                        self.permission_audit.append(dict(layer=i, step=self.step, allowed=allowed,
                            visual_mass=float(visual.float().sum(-1).mean()), masked_probability_zero=not allowed))
                    return out, weights
                stack.enter_context(patch.object(qm, 'eager_attention_forward', observed))
            for i, block in enumerate(self.d.layers):
                def before(module, args, kw, i=i):
                    mask = kw['attention_mask']
                    if self.step == 0:
                        assert kw['hidden_states'].shape[1] == self.n
                    else:
                        assert kw['hidden_states'].shape[1] == 1
                    allowed = i in self.read_layers
                    changed = restore_answer_access(mask, self.I, allowed)
                    if self.step == 0:
                        assert torch.equal(changed[..., :-1, :], mask[..., :-1, :])
                    # All non-image keys of the current query retain the frozen definition.
                    text_keys = [k for k in range(mask.shape[-1]) if k not in set(self.I)]
                    assert torch.equal(changed[..., -1, text_keys], mask[..., -1, text_keys])
                    kw['attention_mask'] = changed
                    self.read_audit.append(dict(layer=i, step=self.step, allowed=allowed, visual_keys=len(self.I)))
                    return args, kw
                h = block.self_attn.register_forward_pre_hook(before, with_kwargs=True)
                stack.callback(h.remove)
                if self.audit_enabled:
                    def after(module, args, out, i=i):
                        if self.step == 0:
                            value = out[0] if isinstance(out, tuple) else out
                            self.prefill_hashes.append(dict(layer=i, non_P1_hidden_sha256=tensor_digest(value[:, :-1])))
                    stack.callback(block.register_forward_hook(after).remove)
            stack.enter_context(super().context())
            yield self

    def audit(self):
        out = super().audit()
        out.update(reaccess=self.read_audit, method='FUNCTIONAL_REACCESS_ONLY', real_physical_visual_KV_compaction=False)
        out.update(permission_audit=self.permission_audit, prefill_non_target_hashes=self.prefill_hashes)
        return out


def generate(model, processor, inputs, layout, layers, audit=False, capture=False, cap=128):
    hooks = hook_counts(model)
    versions = [(p.data_ptr(), p._version) for p in model.parameters()]
    def execution(*args, **kwargs):
        return Functional(*args, layers=layers, **kwargs)
    try:
        with patch.object(original, 'Execution', execution):
            result, vectors = original.generate(model, processor, inputs, layout, 'F', audit=audit, capture=capture, max_tokens=cap)
        assert versions == [(p.data_ptr(), p._version) for p in model.parameters()]
        assert {k:tensor_digest(v) for k,v in inputs.items()} == layout['input_tensor_hashes']
        assert result['audit']['deepstack'] == [0,1,2]
        for row in result['audit']['kv']['layers']:
            assert row['live_tokens'] == layout['input_length']+len(result['token_ids'])-1
        result['functional_image_KV_live_bytes'] = 36 * 2 * 8 * layout['image_tokens'] * 128 * 2
        result['physical_svhr'] = False
        return result, vectors
    finally:
        assert hooks == hook_counts(model)

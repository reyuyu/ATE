from contextlib import ExitStack
from unittest.mock import patch

import torch

from ..qwen3vl_early_windows.experiment import PrefillTrace, hook_counts, tensor_digest


def image_mask(mask, length, query, visual):
    assert mask is not None and mask.ndim == 4 and mask.shape[-2:] == (length, length)
    assert mask.dtype.is_floating_point and 0 <= query < length
    assert visual and len(set(visual)) == len(visual) and query not in visual
    assert all(0 <= key <= query for key in visual)
    changed = mask.clone()
    changed[:, :, query, visual] = float('-inf')
    assert torch.isfinite(changed[:, :, query]).any(-1).all()
    return changed


class ImageOff:
    """Change only an attention module's additive mask, before native softmax."""
    def __init__(self, model, layer, layout, noop=False):
        self.module = model.model.language_model.layers[layer].self_attn
        self.layout, self.noop = layout, noop
        self.calls, self.audit, self.handles = 0, {}, []

    def before(self, module, args, kwargs):
        self.calls += 1
        assert self.calls == 1, 'One independent prefill, never decode'
        n, q = self.layout['input_length'], self.layout['p_minus_1']
        assert kwargs['hidden_states'].shape[:2] == (1, n) and q == n - 1
        original = kwargs['attention_mask']
        changed = original if self.noop else image_mask(original, n, q, self.layout['image_positions'])
        self.audit['mask_changed_entries'] = int((original != changed).sum())
        self.audit['expected_mask_entries'] = 0 if self.noop else len(self.layout['image_positions'])
        assert self.audit['mask_changed_entries'] == self.audit['expected_mask_entries']
        return args, dict(kwargs, attention_mask=changed)

    def after(self, module, args, output):
        weights = output[1]
        assert weights is not None and weights.shape[0] == 1
        row = weights[0, :, self.layout['p_minus_1'], :]
        assert torch.isfinite(row).all() and (row >= 0).all()
        visual = row[:, self.layout['image_positions']]
        maximum = float(visual.max())
        if not self.noop:
            assert maximum == 0, 'Mask failed: native post-softmax Image probability is nonzero'
        row_error = float((row.float().sum(-1) - 1).abs().max())
        assert row_error <= 2**-8 + 2e-6, 'BF16 native probability mass error'
        self.audit.update(image_probability_max=maximum, row_sum_max_error=row_error,
                          heads=row.shape[0], observed_native_probabilities=True)

    def __enter__(self):
        self.handles = [self.module.register_forward_pre_hook(self.before, with_kwargs=True),
                        self.module.register_forward_hook(self.after)]
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return type(value)(clone_tree(x) for x in value)


def tree_hash(value):
    if isinstance(value, torch.Tensor):
        return tensor_digest(value)
    return [tree_hash(x) for x in value]


class Prefill:
    def __init__(self, model, inputs, layout):
        self.model, self.inputs, self.layout = model, inputs, layout
        self.visual = None
        self.hooks = hook_counts(model)
        self.input_hashes = {k: tensor_digest(v) for k, v in inputs.items()}

    @torch.inference_mode()
    def forward(self, layer=None, noop=False, trace=False, native=False, all_logits=False):
        n = self.layout['input_length']
        original = self.model.model.get_image_features
        def features(*args, **kwargs):
            if native or self.visual is None:
                result = original(*args, **kwargs)
                if self.visual is None:
                    self.visual = clone_tree(result)
                else:
                    assert tree_hash(result) == tree_hash(self.visual)
                return result
            return clone_tree(self.visual)
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.model.model, 'get_image_features', features))
            intervention = stack.enter_context(ImageOff(self.model, layer, self.layout, noop)) if layer is not None else None
            tracer = stack.enter_context(PrefillTrace(self.model, n, self.layout['image_positions'])) if trace else None
            result = self.model(**self.inputs, use_cache=False, past_key_values=None,
                cache_position=torch.arange(n, device=self.inputs['input_ids'].device),
                logits_to_keep=0 if all_logits else 1, output_attentions=False, output_hidden_states=False)
            assert result.past_key_values is None
            logits = result.logits[0].detach().cpu().clone() if all_logits else result.logits[0, -1].detach().cpu().clone()
            assert torch.isfinite(logits).all()
            if intervention:
                assert intervention.calls == 1 and intervention.audit['observed_native_probabilities']
            if tracer:
                tracer.validate()
                assert tracer.mlp_calls == [1] * 36
        assert hook_counts(self.model) == self.hooks
        return logits, intervention.audit if intervention else {}, tracer

    def verify(self):
        assert self.input_hashes == {k: tensor_digest(v) for k, v in self.inputs.items()}
        assert hook_counts(self.model) == self.hooks


def metrics(full, changed):
    assert full.ndim == 1 and full.shape == changed.shape
    p, q = full.float().log_softmax(-1), changed.float().log_softmax(-1)
    target, predicted = int(full.argmax()), int(changed.argmax())
    loss = float(p[target] - q[target])
    return dict(full_first_token=target, changed_first_token=predicted,
                full_top1_logp=float(p[target]), imageoff_full_top1_logp=float(q[target]),
                signed_logp_loss=loss, absolute_logp_change=abs(loss), top1_flip=int(target != predicted))

"""Read-only frozen physical implementation with lightweight structural guards."""
from contextlib import ExitStack, contextmanager

import torch

from ..qwen3vl_svh_sqa7908_repair.engine import generate as physical
from ..qwen3vl_svh_numerical_repair.engine import generate as reference
from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest


class Guard:
    def __init__(self, model, layout, condition):
        self.model, self.layout, self.condition = model, layout, condition
        self.step = -1
        self.layers = self.mlps = 0
        self.last = None

    @contextmanager
    def context(self):
        n = self.layout['input_length']
        images = set(self.layout['image_positions'])
        text = [i for i in range(n) if i not in images]
        with ExitStack() as stack:
            def begin(module, args, kw):
                self.step += 1
                assert kw['past_key_values'] is None if self.step == 0 else kw['past_key_values'] is not None
                expected = list(range(n)) if self.step == 0 else [n + self.step - 1]
                assert kw['cache_position'].tolist() == expected
            h = self.model.register_forward_pre_hook(begin, with_kwargs=True)
            stack.callback(h.remove)
            for i, block in enumerate(self.model.model.language_model.layers):
                def start(module, args, kw, i=i):
                    cp = kw['cache_position'].tolist()
                    expected = (text if self.condition == 'SVH' and i >= 22 else list(range(n))) if self.step == 0 else [n+self.step-1]
                    assert cp == expected and args[0].shape[1] == len(cp)
                    cos, sin = kw['position_embeddings']
                    assert cos.shape[1] == sin.shape[1] == len(cp)
                    assert bool(torch.isfinite(cos).all()) and bool(torch.isfinite(sin).all())
                    mask = kw['attention_mask']
                    if self.condition == 'SVH' and self.step == 0 and i < 22 and i not in range(1, 7):
                        assert bool(torch.isneginf(mask[..., -1, self.layout['image_positions']]).all())
                    self.last = dict(step=self.step, layer=i, hidden=list(args[0].shape), logical_positions=cp)
                    self.layers += 1
                h = block.register_forward_pre_hook(start, with_kwargs=True)
                stack.callback(h.remove)
                def mlp(module, args, out):
                    self.mlps += 1
                h = block.mlp.register_forward_hook(mlp)
                stack.callback(h.remove)
            yield self
        assert self.layers == self.mlps == 36 * (self.step+1)


def run(model, processor, inputs, layout, condition, audit=False, capture=False, noop=False, cap=None):
    hooks = hook_counts(model)
    pointers = [(p.data_ptr(), p._version) for p in model.parameters()]
    guard = Guard(model, layout, condition)
    try:
        with guard.context():
            fn = physical if condition == 'SVH' else reference
            result, vectors = fn(model, processor, inputs, layout, 'C' if condition == 'SVH' else 'F' if condition == 'Dense' else 'N',
                audit=audit, capture=capture, max_tokens=cap or 128, **({'noop': noop} if condition != 'SVH' else {}))
        assert hooks == hook_counts(model) and pointers == [(p.data_ptr(), p._version) for p in model.parameters()]
        assert {k: tensor_digest(v) for k, v in inputs.items()} == layout['input_tensor_hashes']
        if condition == 'SVH':
            audit_record = result['audit']
            assert len(audit_record['early_aligned_linears']) == 24
            assert audit_record['deepstack'] == [0, 1, 2]
            for row in audit_record['kv']['layers']:
                expected = 0 if row['layer'] in range(1, 7) else layout['input_length']-layout['image_tokens']+len(result['token_ids'])-1
                assert row['live_tokens'] == expected
        return result, vectors, dict(status='PASS', layers=guard.layers, mlps=guard.mlps,
            independent_initial_cache=True, hooks_restored=True, immutable_weights=True, last=guard.last)
    finally:
        assert hooks == hook_counts(model), 'Hook leak even after exception'

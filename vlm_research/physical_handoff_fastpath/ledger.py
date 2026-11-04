"""Executed shape census; this is NOT a GPU hardware floating-point counter."""
from collections import defaultdict
from contextlib import contextmanager
import math
from unittest.mock import patch

import torch


class Census:
    def __init__(self, adapter):
        self.adapter = adapter
        self.phase, self.vision, self.calls = 'prefill', False, 0
        self.totals, self.shapes = defaultdict(int), defaultdict(set)

    def add(self, component, flops, shape):
        phase = 'vision' if self.vision else self.phase
        k = phase, component
        self.totals[k] += flops
        self.shapes[k].add(str(shape))

    @contextmanager
    def context(self):
        handles = []
        model = self.adapter.model
        target = model if self.adapter.kind == 'llava' else model.model
        original_visual, original_matmul = target.get_image_features, torch.matmul
        def visual(*a, **kw):
            self.vision = True
            try:
                return original_visual(*a, **kw)
            finally:
                self.vision = False
        def matmul(a, b, *args, **kwargs):
            value = original_matmul(a, b, *args, **kwargs)
            self.add('attention_qk_av', 2 * value.numel() * a.shape[-1],
                     (tuple(a.shape), tuple(b.shape), tuple(value.shape)))
            return value
        def start(m, a, kw):
            self.phase = 'prefill' if self.calls == 0 else 'decode'
            self.calls += 1
        handles.append(model.register_forward_pre_hook(start, with_kwargs=True))
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if module is model.get_output_embeddings():
                    component = 'lm_head'
                elif any(x in name for x in ('.q_proj', '.k_proj', '.v_proj', '.qkv')):
                    component = 'qkv'
                elif '.mlp.' in name:
                    component = 'mlp'
                elif '.o_proj' in name or '.out_proj' in name:
                    component = 'attention_output_projection'
                else:
                    component = 'other_linear'
                def linear(m, a, out, component=component):
                    self.add(component, 2 * out.numel() * m.in_features,
                             (tuple(a[0].shape), tuple(m.weight.shape), tuple(out.shape)))
                handles.append(module.register_forward_hook(linear))
            elif isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                def conv(m, a, out):
                    self.add('patch_convolution', 2 * out.numel() * m.in_channels // m.groups * math.prod(m.kernel_size),
                             (tuple(a[0].shape), tuple(m.weight.shape), tuple(out.shape)))
                handles.append(module.register_forward_hook(conv))
        try:
            with patch.object(target, 'get_image_features', visual), patch.object(torch, 'matmul', matmul):
                yield self
        finally:
            for handle in handles:
                handle.remove()

    def rows(self):
        return [dict(phase=p, component=c, estimated_flops=v, observed_shapes=sorted(self.shapes[p, c]),
                     provenance='executed linear/conv/QK/AV shape census; 2 FLOPs/MAC; excludes nonlinear/norm/softmax/rotary')
                for (p, c), v in sorted(self.totals.items())]

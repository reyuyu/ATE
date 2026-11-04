"""Temporary, matched numerical kernels; no persistent FP32 weights or extra visual rows."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import importlib

import torch
from torch.nn import functional as F


def linear_fp32(module, x):
    return F.linear(x.float(), module.weight.float(),
                    None if module.bias is None else module.bias.float()).to(x.dtype)


@contextmanager
def numerical_kernels(adapter, candidate):
    assert candidate in ('native', 'fp32_linears', 'fp32_linears_attention')
    cutoff = 20 if adapter.kind == 'llava' else 22
    with ExitStack() as stack:
        if candidate != 'native':
            modules = [m for layer in adapter.layers[cutoff:] for m in layer.modules()
                       if isinstance(m, torch.nn.Linear)]
            modules.append(adapter.model.lm_head)
            for module in modules:
                stack.enter_context(patch.object(module, 'forward',
                    lambda x, module=module: linear_fp32(module, x)))
        if candidate == 'fp32_linears_attention':
            source = importlib.import_module(adapter.layers[-1].self_attn.__class__.__module__)
            native = source.eager_attention_forward

            def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
                if module.layer_idx < cutoff:
                    return native(module, query, key, value, attention_mask,
                                  scaling=scaling, dropout=dropout, **kwargs)
                assert dropout == 0.
                k = source.repeat_kv(key, module.num_key_value_groups)
                v = source.repeat_kv(value, module.num_key_value_groups)
                # Keep the frozen low-precision boundaries, changing GEMM accumulation only.
                scores = (query.float() @ k.float().transpose(2, 3)).to(query.dtype) * scaling
                if attention_mask is not None:
                    scores = scores + attention_mask[..., :k.shape[-2]]
                weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
                output = (weights.float() @ v.float()).to(query.dtype).transpose(1, 2).contiguous()
                return output, weights

            stack.enter_context(patch.object(source, 'eager_attention_forward', attention))
        yield

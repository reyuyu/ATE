"""Retain eager QK/AV shapes; bound the FP32 softmax temporary by query rows."""
from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm

RUNTIME = 'VISION_EAGER_ROW_SOFTMAX128_TEXT_EAGER'


def row_attention(module, query, key, value, attention_mask, scaling, dropout=0., rows=128, **kwargs):
    assert not module.training and dropout == 0 and rows > 0
    key = qm.repeat_kv(key, module.num_key_value_groups)
    value = qm.repeat_kv(value, module.num_key_value_groups)
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[..., :key.shape[-2]]
    # Rows are independent. Keep the full QK/AV GEMM geometry and BF16 probability buffer.
    for start in range(0, weights.shape[-2], rows):
        chunk = weights[..., start:start+rows, :]
        chunk.copy_(torch.nn.functional.softmax(chunk, dim=-1, dtype=torch.float32).to(query.dtype))
    out = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return out, weights


@contextmanager
def vision_runtime(audit=None):
    native = qm.eager_attention_forward
    def attention(module, query, key, value, attention_mask=None, **kwargs):
        if not isinstance(module, qm.Qwen3VLVisionAttention):
            return native(module, query, key, value, attention_mask, **kwargs)
        if audit is not None:
            audit.append(dict(query=list(query.shape),key=list(key.shape),value=list(value.shape),
                dtype=str(query.dtype),rows=128,runtime=RUNTIME))
        return row_attention(module, query, key, value, attention_mask, **kwargs)
    with patch.object(qm, 'eager_attention_forward', attention):
        yield

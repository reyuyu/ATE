"""Match QK query geometry without restoring visual hidden states or KV."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from ..qwen3vl_svh_l28_repair.engine import AVAlignedExecution, restore_query_rows
from ..qwen3vl_svh_numerical_repair.engine import original

class QKAlignedExecution(AVAlignedExecution):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qk_row_alignment = []

    @contextmanager
    def context(self):
        eager = qm.eager_attention_forward

        def aligned(module, q, k, v, attention_mask=None, **kwargs):
            if not (self.physical and self.step == 0 and (getattr(module, 'layer_idx', -1) >= 22)):
                return eager(module, q, k, v, attention_mask, **kwargs)
            native_matmul = torch.matmul
            calls = 0

            def matmul(a, b, *args, **kw):
                nonlocal calls
                calls += 1
                if calls != 1:
                    return native_matmul(a, b, *args, **kw)
                assert a.dtype == b.dtype == torch.bfloat16
                assert a.shape[-2] == len(self.text) and b.shape[-1] == self.n
                full, ix = restore_query_rows(a, self._text_cpu, self.n)
                out = native_matmul(full, b, *args, **kw)
                if self.audit_enabled:
                    self.qk_row_alignment.append(dict(layer=module.layer_idx, step=0, real_queries=a.shape[-2], temporary_QK_queries=full.shape[-2], heads=a.shape[-3], keys=b.shape[-1], head_dim=a.shape[-1], zero_query_rows=self.n - len(self.text), image_hidden_rows=0, persistent_dummy_KV=0, dtype=str(a.dtype), query_buffer_bytes=full.numel() * full.element_size(), output_buffer_bytes=out.numel() * out.element_size(), extra_QK_flops=2 * a.shape[0] * a.shape[-3] * (self.n - len(self.text)) * a.shape[-1] * b.shape[-1]))
                return out.index_select(-2, ix)
            with patch.object(torch, 'matmul', matmul):
                out = eager(module, q, k, v, attention_mask, **kwargs)
            assert calls == 2, 'Pinned eager must execute exactly QK and AV'
            return out
        with patch.object(qm, 'eager_attention_forward', aligned), super().context():
            yield self

    def audit(self):
        out = super().audit()
        out['QK_query_row_alignment'] = self.qk_row_alignment
        out['extra_QK_query_alignment_flops'] = sum((r['extra_QK_flops'] for r in self.qk_row_alignment))
        out['actual_language_matrix_flops'] += out['extra_QK_query_alignment_flops']
        return out

def generate(model, processor, inputs, layout, condition='C', **kwargs):
    assert condition == 'C'
    with patch.object(original, 'Execution', QKAlignedExecution):
        return original.generate(model, processor, inputs, layout, condition, **kwargs)

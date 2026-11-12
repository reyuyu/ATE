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
            return eager(module, q, k, v, attention_mask, **kwargs)
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

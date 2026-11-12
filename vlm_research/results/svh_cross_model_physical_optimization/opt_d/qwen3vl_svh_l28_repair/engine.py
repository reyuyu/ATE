"""Align only the post-exit AV batch geometry; keep real hidden/KV compact."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from ..qwen3vl_svh_numerical_repair.engine import AlignedExecution, original

def restore_query_rows(probabilities, text_positions, original_length):
    if probabilities.shape[-2] != len(text_positions):
        raise ValueError('Query position count mismatch')
    if len(set(text_positions)) != len(text_positions) or any((p < 0 or p >= original_length for p in text_positions)):
        raise ValueError('Invalid original query mapping')
    full = probabilities.new_zeros((*probabilities.shape[:-2], original_length, probabilities.shape[-1]))
    indices = torch.as_tensor(text_positions, device=probabilities.device)
    full.index_copy_(-2, indices, probabilities)
    return (full, indices)

class AVAlignedExecution(AlignedExecution):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.av_row_alignment = []

    @contextmanager
    def context(self):
        eager = qm.eager_attention_forward

        def aligned(module, q, k, v, attention_mask=None, **kwargs):
            return eager(module, q, k, v, attention_mask, **kwargs)
        with patch.object(qm, 'eager_attention_forward', aligned), super().context():
            yield self

    def audit(self):
        out = super().audit()
        out['AV_query_row_alignment'] = self.av_row_alignment
        out['extra_AV_query_alignment_flops'] = sum((r['extra_AV_flops'] for r in self.av_row_alignment))
        out['actual_language_matrix_flops'] += out['extra_AV_query_alignment_flops']
        return out

def generate(model, processor, inputs, layout, condition='C', **kwargs):
    assert condition == 'C'
    with patch.object(original, 'Execution', AVAlignedExecution):
        return original.generate(model, processor, inputs, layout, condition, **kwargs)

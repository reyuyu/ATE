"""Targeted numerical repairs; real hidden-row exit and six-layer skip retained."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import time, torch
from transformers.models.qwen3_vl import modeling_qwen3_vl as qm
from ..qwen3vl_c3v2_physical import engine as original
from ..qwen3vl_c3v2_numerical_debug.repair import RepairExecution

def restore_dead_image_columns(k, v, text_positions, n, step):
    positions = list(text_positions) + list(range(n, n + step))
    ix = torch.tensor(positions, device=k.device)
    assert len(positions) == k.shape[-2]
    full_k = k.new_zeros((*k.shape[:-2], n + step, k.shape[-1]))
    full_v = v.new_zeros((*v.shape[:-2], n + step, v.shape[-1]))
    full_k.index_copy_(-2, ix, k)
    full_v.index_copy_(-2, ix, v)
    mask = k.new_full((1, 1, 1, n + step), -torch.inf)
    mask.index_fill_(-1, ix, 0.0)
    return (full_k, full_v, mask, ix)

class Repair2Execution(RepairExecution):

    def __init__(self, *args, variant='decode-zero-key', **kw):
        super().__init__(*args, **kw)
        self.variant = variant
        self.decode_padding = []
        self.high_precision = []

    @contextmanager
    def context(self):
        eager = qm.eager_attention_forward

        def aligned(module, q, k, v, attention_mask=None, **kw):
            return eager(module, q, k, v, attention_mask, **kw)
        with ExitStack() as st:
            st.enter_context(patch.object(qm, 'eager_attention_forward', aligned))
            layers = [22] if self.variant == 'FP32-L22+decode' else list(range(22, 36)) if self.variant == 'FP32-late+decode' else []
            for i in layers:
                block = self.d.layers[i]
                for name in ['k_proj', 'v_proj', 'gate_proj', 'up_proj']:
                    mod = getattr(block.self_attn if name in ['k_proj', 'v_proj'] else block.mlp, name)
                    fn = mod.forward

                    def local_fp32(x, mod=mod, fn=fn, i=i, name=name):
                        if self.physical and self.step == 0:
                            assert x.shape[1] == len(self.text) and x.dtype == torch.bfloat16
                            begin = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            begin.record()
                            y = torch.nn.functional.linear(x.float(), mod.weight.float(), None if mod.bias is None else mod.bias.float()).to(x.dtype)
                            end.record()
                            self.high_precision.append(dict(layer=i, step=0, operator=name, input_shape=list(x.shape), weight_shape=list(mod.weight.shape), output_shape=list(y.shape), input_dtype=str(x.dtype), compute_dtype='float32', output_dtype=str(y.dtype), fp32_weight_conversion_bytes=mod.weight.numel() * 4, start=begin, end=end))
                            return y
                        return fn(x)
                    st.enter_context(patch.object(mod, 'forward', local_fp32))
            st.enter_context(super().context())
            yield self

    def audit(self):
        d = super().audit()
        d['variant'] = self.variant
        d['decode_temporary_zero_keys'] = self.decode_padding
        d['local_fp32'] = [{**{k: v for k, v in r.items() if k not in ['start', 'end']}, 'single_call_gpu_ms': r['start'].elapsed_time(r['end'])} for r in self.high_precision]
        d['actual_language_matrix_flops'] += sum((r['extra_QK_AV_flops'] for r in self.decode_padding))
        d['temporary_full_key_extent_note'] = 'Restore only transient masked zero-image columns for reduction shape; no image KV projection/storage or skipped L1-L6 Decode attention restored. Decode QK/AV extra work is explicit.'
        return d

def generate(*args, variant='decode-zero-key', **kw):

    class Bound(Repair2Execution):

        def __init__(self, *a, **k):
            super().__init__(*a, variant=variant, **k)
    with patch.object(original, 'Execution', Bound):
        return original.generate(*args, **kw)

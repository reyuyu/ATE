"""Extend the existing row observer, restricted to six layers and original P-1."""
from contextlib import ExitStack

import numpy as np
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from knockout_probe import model_layers
from paper_rq1.engine import AttentionStep
from paper_rq1_early_joint_attention.engine import JointAttention


class EarlyRows(AttentionStep):
    def __init__(self, model, query, keys):
        super().__init__(model, query, observe=keys)
        self.arrays, self.rotary = {}, {}

    def __enter__(self):
        try:
            for index, block in enumerate(model_layers(self.model)[:6]):
                def residual(module, args, kwargs, index=index):
                    hidden = args[0] if args else kwargs['hidden_states']
                    assert hidden.shape[0] == 1 and hidden.shape[1] == self.query + 1
                    value = hidden[0, self.query].detach()
                    self.residuals[index] = float(value.float().norm())
                    self.arrays[f'hidden_in_{index}'] = value.cpu().clone().numpy()

                def before(module, args, kwargs, index=index):
                    hidden = args[0] if args else kwargs['hidden_states']
                    self.arrays[f'attention_input_{index}'] = hidden[0, self.query].detach().cpu().clone().numpy()
                    cos, sin = kwargs['position_embeddings']
                    self.rotary[index] = (cos[:, self.query:self.query+1], sin[:, self.query:self.query+1])

                def query(module, args, output, index=index, attention=block.self_attn):
                    head_dim = attention.head_dim
                    selected = output[:, self.query:self.query+1].view(1, 1, -1, head_dim).transpose(1, 2)
                    rotated, _ = apply_rotary_pos_emb(selected, selected, *self.rotary.pop(index))
                    self.arrays[f'query_raw_{index}'] = selected[0, :, 0].detach().cpu().clone().numpy()
                    self.arrays[f'query_rope_{index}'] = rotated[0, :, 0].detach().cpu().clone().numpy()

                def after(module, args, output, index=index):
                    self.arrays[f'hidden_out_{index}'] = output[0][0, self.query].detach().cpu().clone().numpy()

                base_mass = super().mass_hook(index)

                def mass(module, args, output, index=index, base_mass=base_mass):
                    self.arrays[f'attention_row_{index}'] = output[1][0, :, self.query].detach().cpu().clone().numpy()
                    return base_mass(module, args, output)

                self.handles += [block.register_forward_pre_hook(residual, with_kwargs=True),
                    block.self_attn.register_forward_pre_hook(before, with_kwargs=True),
                    block.self_attn.q_proj.register_forward_hook(query),
                    block.self_attn.o_proj.register_forward_hook(self.output_hook(index)),
                    block.self_attn.register_forward_hook(mass), block.register_forward_hook(after)]
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise


@torch.inference_mode()
def prefill_capture(model, prompt, keys, off=False):
    query = prompt['input_ids'].shape[1] - 1
    with ExitStack() as stack:
        trace = stack.enter_context(JointAttention(model, [0, 1] if off else [],
            'P-1-only' if off else 'Full', query + 1, audit=True))
        rows = stack.enter_context(EarlyRows(model, query, keys))
        output = model(**prompt, past_key_values=None, use_cache=True, output_attentions=False, return_dict=True)
        logits = output.logits[0, -1].detach().float().cpu().clone()
        cache = output.past_key_values
        assert cache is not None and all(cache.get_seq_length(k) == query + 1 for k in range(32))
        assert len(rows.stats) == 6 and trace.edits == (2 if off else 0)
        for layer in range(6):
            # These are the actual cached post-RoPE keys and native values, not re-projections.
            rows.arrays[f'bos_key_{layer}'] = cache.key_cache[layer][0, :, keys['bos']].detach().cpu().clone().numpy()
            rows.arrays[f'bos_value_{layer}'] = cache.value_cache[layer][0, :, keys['bos']].detach().cpu().clone().numpy()
        assert torch.isfinite(logits).all() and all(np.isfinite(a).all() for a in rows.arrays.values())
        return logits, rows.arrays, rows.stats


def vector_change(full, changed):
    a, b = np.asarray(full, dtype=np.float64), np.asarray(changed, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = np.linalg.norm(b - a)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dict(l2_change=float(delta), relative_l2_change=float(delta / na) if na else None,
        cosine=float(np.vdot(a.ravel(), b.ravel()) / (na * nb)) if na and nb else None,
        full_norm=float(na), off_norm=float(nb), exact=bool(np.array_equal(a, b)))


def head_mass(row, keys):
    a = np.asarray(row, dtype=np.float64)
    return {name: a[:, positions].sum(-1) for name, positions in keys.items()}

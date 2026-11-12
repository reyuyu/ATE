"""Fast-P2 with answer-only physical Attention bypass; no legacy A2 execution."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv

from ..physical_handoff_fastpath import engine as base

LAYERS = (2, 3, 4, 5)
CONDITIONS = ('C0', 'C1', 'C2', 'C3-V2', 'C4-V2')
FrozenHandoff = base.FastHandoff


class AnswerKV(base.InplaceKV):
    def __init__(self, *args, absent=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.absent = absent

    def audit(self):
        rows = []
        for i, pair in enumerate(self.entries):
            if i in self.absent:
                assert pair is None and self.used[i] == 0
                rows.append(dict(layer=i, used_tokens=0, capacity_tokens=0,
                                 live_bytes=0, storage_bytes=0))
                continue
            k, v = pair
            positions = list(self.layout.text) + list(range(self.layout.length, self.logical_length))
            assert len(positions) == self.used[i]
            rows.append(dict(layer=i, used_tokens=self.used[i], capacity_tokens=k.shape[-2],
                live_bytes=sum(t.shape[0]*t.shape[1]*self.used[i]*t.shape[-1]*t.element_size() for t in pair),
                storage_bytes=sum(t.untyped_storage().nbytes() for t in pair),
                original_positions=positions))
        return dict(layers=rows, live_bytes=sum(r['live_bytes'] for r in rows),
                    storage_bytes=sum(r['storage_bytes'] for r in rows), allocations=self.allocations,
                    write_calls=self.write_calls, direct_gather_calls=self.gather_calls,
                    historical_cache_copies=0, append_concatenations=0)


class AnswerWriter(FrozenHandoff):
    def __init__(self, adapter, layout, name, writers, max_tokens, audit=False):
        super().__init__(adapter, layout, 'Fast-P2', max_tokens, False)
        self.name, self.collect, self.audit_enabled = name, audit, audit
        self.events = []
        self.physical_skip = name in ('C3-V2', 'C4-V2')
        self.zero_reference = name.endswith('-ZeroRef')
        self.cache = AnswerKV(len(adapter.layers), layout, 'Fast-P2', max_tokens, self.device,
                              absent=LAYERS if self.physical_skip else ())
        if name.startswith('C4-V2') or name == 'C2':
            assert set(writers) <= set(layout.relay)
            blocked = sorted(set(layout.relay) - set(writers))
            if blocked:
                self.prefill_early[0, 0, torch.tensor(blocked, device=self.device)[:, None],
                                   torch.tensor(layout.image, device=self.device)[None, :]] = -torch.inf
        self.active_mask = self.prefill_early[..., :-1, :]

    def record(self, out, weights, positions, shape, mask):
        if not self.collect:
            return
        assert torch.isfinite(out).all()
        blocked_max, row_error = 0., 0.
        if weights is not None:
            assert torch.isfinite(weights).all()
            row_error = float((weights.float().sum(-1) - 1).abs().max())
            assert row_error < .005
            if mask is not None:
                blocked = torch.isneginf(mask[0, 0])
                assert (mask == 0).any(-1).all()
                if blocked.any():
                    blocked_max = float(weights[0, :, blocked].abs().max())
                    assert blocked_max == 0.
        self.events.append(dict(layer=self.current[0], prefill=self.cache.prefill, step=self.step,
            input_queries=shape[1], computed_queries=len(positions), query_positions=list(positions),
            blocked_probability_max=blocked_max, row_sum_error=row_error, output_finite=True))

    def answer_rows(self, module, hidden_states, position_embeddings, **kwargs):
        if not self.cache.prefill:
            out = torch.zeros_like(hidden_states)
            self.record(out, None, (), hidden_states.shape, None)
            return out, None
        n, hd = hidden_states.shape[1], module.head_dim
        # Retain every visual/prefix/receiver query. Only the causal last query is absent.
        q = module.q_proj(hidden_states[:, :-1]).view(1, n-1, -1, hd).transpose(1, 2)
        k = module.k_proj(hidden_states).view(1, n, -1, hd).transpose(1, 2)
        v = module.v_proj(hidden_states).view(1, n, -1, hd).transpose(1, 2)
        cos, sin = position_embeddings
        q = q*cos[:, None, :-1] + rotate_half(q)*sin[:, None, :-1]
        k = k*cos[:, None] + rotate_half(k)*sin[:, None]
        k, v = repeat_kv(k, module.num_key_value_groups), repeat_kv(v, module.num_key_value_groups)
        weights = torch.matmul(q, k.transpose(2, 3))*module.scaling + self.active_mask
        weights = torch.nn.functional.softmax(weights, -1, dtype=torch.float32).to(q.dtype)
        attended = torch.matmul(weights, v)
        projected = module.o_proj(attended.transpose(1, 2).reshape(1, n-1, -1).contiguous())
        out = torch.zeros_like(hidden_states)
        out[:, :-1] = projected
        # No persistent KV here: all future queries at this layer also bypass Attention.
        self.record(out, weights, range(n-1), hidden_states.shape, self.active_mask)
        return out, None

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.decoder, 'forward', self.forward))
            for i, layer in enumerate(self.a.layers):
                module, original = layer.self_attn, layer.self_attn.forward

                def forward(*args, original=original, module=module, i=i, **kwargs):
                    if self.physical_skip and i in LAYERS:
                        return self.answer_rows(module, *args, **kwargs)
                    out, weights = original(*args, **kwargs)
                    if self.zero_reference and i in LAYERS:
                        out = out.clone()
                        if self.cache.prefill:
                            out[:, -1] = 0
                        else:
                            out.zero_()
                    hidden = kwargs.get('hidden_states', args[0] if args else None)
                    self.record(out, weights, range(hidden.shape[1]), hidden.shape,
                                kwargs.get('attention_mask'))
                    return out, weights

                stack.enter_context(patch.object(module, 'forward', forward))
            yield self
        self.audit_rows = self.events


def generate(adapter, layout, name, writers, max_tokens=32, capture=False, audit=False):
    if name in ('C0', 'C1', 'Native-Noop'):
        label = {'C0': 'Native Full', 'C1': 'Fast-P2', 'Native-Noop': 'Native-Noop'}[name]
        return base.generate(adapter, layout, label, max_tokens, False, capture, audit)

    def factory(a, l, condition, length, collect):
        return AnswerWriter(a, l, name, writers, length, collect)

    with patch.object(base, 'FastHandoff', factory):
        result = base.generate(adapter, layout, 'Fast-P2', max_tokens, False, capture, audit)
    result[0]['condition'] = name
    return result

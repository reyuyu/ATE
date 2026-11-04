"""Physical branch/row bypass and split-key eager attention, around frozen Fast-P2."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv

from ..physical_handoff_fastpath import engine as base

LAYERS = (2, 3, 4, 5)
CONDITIONS = ('A0', 'A1', 'A2', 'B1', 'B2-20261103', 'B2-20261107', 'B2-20261111', 'B3')
REFERENCES = ('A3', 'A2-Reference')


class SparseKV(base.InplaceKV):
    def __init__(self, *args, absent=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.absent = tuple(absent)

    def audit(self):
        # The base auditor intentionally assumes KV exists in every layer.
        # Bypassed layers never consume future KV, so do not allocate dummy buffers.
        rows = []
        for i, pair in enumerate(self.entries):
            if i in self.absent:
                assert pair is None and self.used[i] == 0
                rows.append(dict(layer=i, used_tokens=0, capacity_tokens=0, live_bytes=0,
                                 storage_bytes=0, bypassed_attention=True))
                continue
            k, v = pair
            positions = list(self.layout.text) + list(range(self.layout.length, self.logical_length))
            assert len(positions) == self.used[i]
            rows.append(dict(layer=i, used_tokens=self.used[i], capacity_tokens=k.shape[-2],
                live_bytes=sum(t.shape[0]*t.shape[1]*self.used[i]*t.shape[-1]*t.element_size() for t in pair),
                storage_bytes=sum(t.untyped_storage().nbytes() for t in pair),
                logical_length=self.logical_length, original_positions=positions))
        return dict(layers=rows, live_bytes=sum(r['live_bytes'] for r in rows),
                    storage_bytes=sum(r['storage_bytes'] for r in rows), allocations=self.allocations,
                    write_calls=self.write_calls, direct_gather_calls=self.gather_calls,
                    historical_cache_copies=0, append_concatenations=0)


class BypassWriter(base.FastHandoff):
    def __init__(self, adapter, layout, name, selection, max_tokens, audit=False):
        super().__init__(adapter, layout, 'Fast-P2', max_tokens, False)
        self.name, self.collect = name, audit
        self.audit_enabled = audit  # Captures retained Hidden, not full Attention matrices.
        self.events = []
        self.absent = LAYERS if name in ('A1', 'A2') else ()
        self.cache = SparseKV(len(adapter.layers), layout, 'Fast-P2', max_tokens, self.device,
                              absent=self.absent)
        all_positions = set(range(layout.length))
        self.ia = tuple(sorted(set(layout.image) | {layout.length-1}))
        self.active = tuple(sorted(all_positions - set(self.ia)))
        self.active_index = torch.tensor(self.active, device=self.device)
        self.ia_index = torch.tensor(self.ia, device=self.device)
        self.active_mask = self.prefill_early.index_select(-2, self.active_index)
        self.writer = name.startswith('B')
        if self.writer:
            chosen = selection['writers']['Semantic-k' if name in ('B1', 'B3') else
                                          'Random-k-' + name.split('-', 1)[1]]
            blocked = sorted(set(layout.relay) - set(chosen))
            self.prefill_early = self.prefill_early.clone()
            if blocked:
                qi = torch.tensor(blocked, device=self.device)
                ki = torch.tensor(layout.image, device=self.device)
                self.prefill_early[0, 0, qi[:, None], ki[None, :]] = -torch.inf
            self.no_image_queries = tuple(blocked + [layout.length-1])
            self.image_queries = tuple(sorted(all_positions - set(self.no_image_queries)))
            self.no_image_index = torch.tensor(self.no_image_queries, device=self.device)
            self.image_index = torch.tensor(self.image_queries, device=self.device)
            self.full_group_mask = self.prefill_early.index_select(-2, self.image_index)
            self.text_group_mask = self.prefill_early.index_select(-2, self.no_image_index).index_select(-1, self.text_index)

    def record(self, module, output, weights, query_positions, original_shape, mask):
        if not self.collect:
            return
        i, prefill, _, positions, _ = self.current
        if weights is not None:
            assert torch.isfinite(weights).all()
            assert float((weights.float().sum(-1)-1).abs().max()) < .005
            if mask is not None:
                blocked = torch.isneginf(mask[0, 0])
                if blocked.any():
                    assert float(weights[0, :, blocked].abs().max()) == 0.
        self.events.append(dict(layer=i, prefill=prefill, step=self.step,
            input_queries=original_shape[1], computed_queries=len(query_positions),
            query_positions=list(query_positions), output_finite=bool(torch.isfinite(output).all()),
            qkv_skipped=self.name == 'A1' and i in LAYERS or
                        self.name == 'A2' and i in LAYERS and not prefill,
            cache_logical_positions=positions.tolist()))

    def physical_rows(self, module, hidden_states, position_embeddings, attention_mask, **kwargs):
        prefill = self.cache.prefill
        if self.name == 'A1' or not prefill:
            out = torch.zeros_like(hidden_states)
            self.record(module, out, None, (), hidden_states.shape, None)
            return out, None
        # R (and the original prefix text) needs K/V for all original Tokens.
        n, hd = hidden_states.shape[1], module.head_dim
        x = hidden_states.index_select(1, self.active_index)
        q = module.q_proj(x).view(1, len(self.active), -1, hd).transpose(1, 2)
        k = module.k_proj(hidden_states).view(1, n, -1, hd).transpose(1, 2)
        v = module.v_proj(hidden_states).view(1, n, -1, hd).transpose(1, 2)
        cos, sin = position_embeddings
        qc, qs = cos.index_select(1, self.active_index).unsqueeze(1), sin.index_select(1, self.active_index).unsqueeze(1)
        q = q * qc + rotate_half(q) * qs
        k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
        attended, weights = self.attend(module, q, k, v, self.active_mask)
        projected = module.o_proj(attended.transpose(1, 2).reshape(1, len(self.active), -1).contiguous())
        out = torch.zeros_like(hidden_states)
        out.index_copy_(1, self.active_index, projected)
        self.record(module, out, weights, self.active, hidden_states.shape, self.active_mask)
        return out, None

    @staticmethod
    def attend(module, q, k, v, mask):
        k, v = repeat_kv(k, module.num_key_value_groups), repeat_kv(v, module.num_key_value_groups)
        weights = torch.matmul(q, k.transpose(2, 3)) * module.scaling
        if mask is not None:
            weights = weights + mask
        weights = torch.nn.functional.softmax(weights, -1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(weights, v), weights

    def split_writer(self, module, hidden_states, position_embeddings, attention_mask,
                     past_key_value=None, **kwargs):
        n, hd = hidden_states.shape[1], module.head_dim
        q = module.q_proj(hidden_states).view(1, n, -1, hd).transpose(1, 2)
        k = module.k_proj(hidden_states).view(1, n, -1, hd).transpose(1, 2)
        v = module.v_proj(hidden_states).view(1, n, -1, hd).transpose(1, 2)
        cos, sin = (t.unsqueeze(1) for t in position_embeddings)
        q, k = q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin
        # Store exactly the same text-only persistent KV as the frozen baseline.
        self.cache.update(k, v, module.layer_idx)
        output = torch.empty_like(q)
        aq, bq = q.index_select(-2, self.image_index), q.index_select(-2, self.no_image_index)
        a, aw = self.attend(module, aq, k, v, self.full_group_mask)
        tk, tv = k.index_select(-2, self.text_index), v.index_select(-2, self.text_index)
        b, bw = self.attend(module, bq, tk, tv, self.text_group_mask)
        output.index_copy_(-2, self.image_index, a)
        output.index_copy_(-2, self.no_image_index, b)
        out = module.o_proj(output.transpose(1, 2).reshape(1, n, -1).contiguous())
        self.record(module, out, aw, self.image_queries, hidden_states.shape, self.full_group_mask)
        self.record(module, out, bw, self.no_image_queries, hidden_states.shape, self.text_group_mask)
        return out, None

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.decoder, 'forward', self.forward))
            for i, layer in enumerate(self.a.layers):
                module, original = layer.self_attn, layer.self_attn.forward
                def forward(*args, module=module, original=original, i=i, **kw):
                    if self.name in ('A1', 'A2') and i in LAYERS:
                        return self.physical_rows(module, *args, **kw)
                    if self.name == 'B3' and self.cache.prefill and i < self.layout.cutoff:
                        return self.split_writer(module, *args, **kw)
                    out, weights = original(*args, **kw)
                    if self.name in REFERENCES and i in LAYERS:
                        out = out.clone()
                        if self.name == 'A3' or not self.cache.prefill:
                            out.zero_()
                        else:
                            out.index_fill_(1, self.ia_index, 0)
                    x = kw.get('hidden_states', args[0] if args else None)
                    qpos = range(x.shape[1])
                    self.record(module, out, weights, qpos, x.shape, kw.get('attention_mask'))
                    return out, weights
                # A0 invokes no wrapper: this preserves the exact frozen baseline hot path.
                if self.name != 'A0':
                    stack.enter_context(patch.object(module, 'forward', forward))
            yield self
        self.audit_rows = self.events


def generate(adapter, layout, name, selection, max_tokens=32, fixed=False,
             capture=False, audit=False, profile=False):
    if name == 'A0':
        return base.generate(adapter, layout, 'Fast-P2', max_tokens, fixed, capture, audit, profile)
    def factory(a, l, condition, length, audit):
        return BypassWriter(a, l, name, selection, length, audit)
    with patch.object(base, 'FastHandoff', factory):
        result = base.generate(adapter, layout, 'Fast-P2', max_tokens, fixed, capture, audit, profile)
    result[0]['condition'] = name
    return result

"""One decoder coordinator: FastV at L3, SVH handoff at L20, text-only KV.

The L2 selector observes the actual, normalized SVH layer input. Its independent
attention probability computation never contributes an answer residual update.
"""
from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
from ..ate_fulljoint_v2.engine import AnswerWriter, LAYERS
from fastv_hf_adapter import selector
OFFICIAL = Path(__file__).resolve().parents[1] / 'external_baselines/fastv/official_source/src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py'

def indices_sha(indices):
    return hashlib.sha256(indices.detach().cpu().long().numpy().tobytes()).hexdigest()

class TextOnlyKV:
    """Transient physical Prefill keys; persistent text-only, in-place buffers."""

    def __init__(self, layout, layers, max_tokens):
        self.layout = layout
        self.entries = [None] * layers
        self.used = [0] * layers
        self.logical_length = 0
        self.prefill = True
        self.text_index = None
        self.max_tokens = max_tokens
        self.allocations = self.write_calls = self.gather_calls = 0

    def get_seq_length(self, layer_idx=0):
        return self.logical_length

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        assert layer_idx not in LAYERS, 'Answer-bypassed layers must not cache KV'
        if self.prefill:
            assert self.entries[layer_idx] is None and self.text_index is not None
            count = len(self.layout.text)
            assert self.text_index.numel() == count
            shape = (*keys.shape[:-2], count + self.max_tokens - 1, keys.shape[-1])
            kbuf = torch.empty(shape, dtype=keys.dtype, device=keys.device)
            vbuf = torch.empty_like(kbuf)
            torch.index_select(keys, -2, self.text_index, out=kbuf[..., :count, :])
            torch.index_select(values, -2, self.text_index, out=vbuf[..., :count, :])
            self.entries[layer_idx] = (kbuf, vbuf)
            self.used[layer_idx] = count
            self.allocations += 2
            self.gather_calls += 2
            return (keys, values)
        kbuf, vbuf = self.entries[layer_idx]
        count = self.used[layer_idx]
        end = count + keys.shape[-2]
        assert end <= kbuf.shape[-2], 'KV capacity exceeded'
        kbuf[..., count:end, :].copy_(keys)
        vbuf[..., count:end, :].copy_(values)
        self.used[layer_idx] = end
        self.write_calls += 2
        return (kbuf[..., :end, :], vbuf[..., :end, :])

    def audit(self):
        positions = list(self.layout.text) + list(range(self.layout.length, self.logical_length))
        rows = []
        for i, pair in enumerate(self.entries):
            if i in LAYERS:
                assert pair is None and self.used[i] == 0
                rows.append(dict(layer=i, used_tokens=0, capacity_tokens=0, image_tokens=0, live_bytes=0, storage_bytes=0))
                continue
            assert pair is not None and self.used[i] == len(positions)
            k, v = pair
            live = sum((t.shape[0] * t.shape[1] * self.used[i] * t.shape[-1] * t.element_size() for t in pair))
            rows.append(dict(layer=i, used_tokens=self.used[i], capacity_tokens=k.shape[-2], image_tokens=0, live_bytes=live, storage_bytes=sum((t.untyped_storage().nbytes() for t in pair)), original_positions_sha256=indices_sha(torch.tensor(positions))))
        return dict(layers=rows, live_bytes=sum((r['live_bytes'] for r in rows)), storage_bytes=sum((r['storage_bytes'] for r in rows)), persistent_original_visual_tokens=0, allocations=self.allocations, write_calls=self.write_calls, direct_gather_calls=self.gather_calls, historical_cache_copies=0, append_concatenations=0)

class Fusion(AnswerWriter):

    def __init__(self, adapter, layout, max_tokens=32, *, prune=True, audit=False, verify_selector=False):
        assert adapter.kind == 'llava' and layout.cutoff == 20
        assert len(adapter.layers) == 32 and len(layout.image) == 576
        assert layout.image == tuple(range(layout.image[0], layout.image[0] + 576))
        assert layout.text[-1] == layout.length - 1
        super().__init__(adapter, layout, 'C3-V2', layout.relay, max_tokens, audit)
        self.prune, self.verify_selector = (prune, verify_selector)
        self.cache = TextOnlyKV(layout, len(adapter.layers), max_tokens)
        self.select, self.selector_provenance = selector(str(OFFICIAL))
        self.keep = self.top = None
        self.selection = None
        self.structure = []
        self.stage_positions = self.positions[:layout.length]
        self.native_causal = torch.full((layout.length, layout.length), torch.finfo(self.dtype).min, device=self.device, dtype=self.dtype).triu(1)[None, None]
        self.original_attention = None

    def rank_l2(self, module, hidden, rotary, keys, begin=None):
        n, hd = (hidden.shape[1], module.head_dim)
        q = module.q_proj(hidden).view(1, n, -1, hd).transpose(1, 2)
        cos, sin = rotary
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        probabilities = torch.matmul(q, keys.transpose(2, 3)) * module.scaling + self.native_causal
        probabilities = torch.nn.functional.softmax(probabilities, -1, dtype=torch.float32).to(q.dtype)
        if self.audit_enabled:
            assert torch.isfinite(probabilities).all()
        keep, top, scores = self.select((None, probabilities), self.layout.image[0], 576, 2 / 3, self.layout.length, hidden.device)
        end = torch.cuda.Event(enable_timing=True) if begin is not None else None
        if end is not None:
            end.record()
        assert keep.numel() == n - 384 and top.numel() == 192
        if self.audit_enabled:
            assert len(set(top.tolist())) == 192 and set(top.tolist()) <= set(self.layout.image)
        if self.audit_enabled:
            assert [v for v in keep.tolist() if v not in self.layout.image] == list(self.layout.text)
        info = dict(ranking_layer=2, prune_before_layer=3, selected_visual_indices=top.tolist() if self.audit_enabled else None, selected_visual_indices_sha256=indices_sha(top) if self.audit_enabled else None, physical_keep_positions_sha256=indices_sha(keep) if self.audit_enabled else None, score_only=True, residual_injection=False, extra_q_shape=list(q.shape), extra_qk_shape=list(probabilities.shape), reused_actual_k=True, extra_v_projection=False, extra_o_projection=False, extra_matrix_flops=2 * n * module.q_proj.in_features * module.q_proj.out_features + 2 * q.shape[1] * n * n * hd, score_min=float(scores.min()), score_max=float(scores.max()))
        if end is not None:
            end.synchronize() if self.audit_enabled else None
            info['selector_gpu_ms'] = begin.elapsed_time(end)
            info['selector_gpu_ms_scope'] = 'Extra K/Q/RoPE/QK/softmax/official-topk only; excludes reference and residual'
        if self.verify_selector:
            _, reference = self.original_attention[2](hidden_states=hidden, position_embeddings=rotary, attention_mask=self.native_causal, past_key_value=None, output_attentions=True)
            rk, rt, _ = self.select((None, reference), self.layout.image[0], 576, 2 / 3, self.layout.length, hidden.device)
            exact = torch.equal(probabilities, reference)
            assert torch.allclose(probabilities, reference, atol=0.1, rtol=0.003)
            if self.audit_enabled:
                assert torch.equal(top, rt) and torch.equal(keep, rk), 'Official Top-192 ranking mismatch'
            info.update(reference_top192_exact=True, reference_probability_exact=exact, reference_probability_max_abs=float((probabilities.float() - reference.float()).abs().max()))
        self.keep, self.top, self.selection = (keep, top, info)

    def answer_rows(self, module, hidden_states, position_embeddings, **kwargs):
        if self.cache.prefill and self.current[0] == 2 and self.prune:
            n, hd = (hidden_states.shape[1], module.head_dim)
            begin = torch.cuda.Event(enable_timing=True) if hidden_states.is_cuda else None
            if begin is not None:
                begin.record()
            k = module.k_proj(hidden_states).view(1, n, -1, hd).transpose(1, 2)
            cos, sin = position_embeddings
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            self.rank_l2(module, hidden_states, position_embeddings, repeat_kv(k, module.num_key_value_groups), begin)
            self.selection['reused_actual_k'] = False
            self.selection['extra_k_projection'] = True
            self.selection['extra_matrix_flops'] += 2 * n * module.k_proj.in_features * module.k_proj.out_features
        return super().answer_rows(module, hidden_states, position_embeddings, **kwargs)

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, cache_position=None, **kwargs):
        h = inputs_embeds if inputs_embeds is not None else self.decoder.embed_tokens(input_ids)
        prefill = self.step == 0
        if prefill:
            assert h.shape[1] == self.layout.length
            positions = self.positions[:self.layout.length]
        else:
            assert h.shape[1] == 1
            positions = self.positions[self.layout.length + self.step - 1:self.layout.length + self.step]
        rotary = self.decoder.rotary_emb(h, positions.unsqueeze(0))
        mask = self.prefill_early if prefill else None
        self.cache.prefill = prefill
        for i, layer in enumerate(self.a.layers):
            if prefill and i == 3 and self.prune:
                assert self.keep is not None, 'Missing same-Prefill L2 selector'
                h = h.index_select(1, self.keep)
                positions = positions.index_select(0, self.keep)
                rotary = tuple((t.index_select(1, self.keep) for t in rotary))
                mask = mask.index_select(-2, self.keep).index_select(-1, self.keep)
            if prefill and i == 20:
                text = torch.tensor([j for j, p in enumerate(positions.tolist()) if p not in self.layout.image], device=h.device, dtype=torch.long)
                h = h.index_select(1, text)
                positions = positions.index_select(0, text)
                rotary = tuple((t.index_select(1, text) for t in rotary))
                mask = mask.index_select(-2, text).index_select(-1, text)
                if self.audit_enabled:
                    assert positions.tolist() == list(self.layout.text)
            if prefill and i in (0, 3, 20):
                text = torch.tensor([j for j, p in enumerate(positions.tolist()) if p not in self.layout.image], device=h.device, dtype=torch.long)
                if self.audit_enabled:
                    assert positions.index_select(0, text).tolist() == list(self.layout.text)
                self.cache.text_index = text
            self.active_mask = mask[..., :-1, :] if prefill else None
            self.stage_positions = positions
            self.current = (i, prefill, mask, positions, h.shape[1] if prefill else self.cache.used[i] + 1)
            if prefill:
                if self.audit_enabled:
                    self.structure.append(dict(layer=i, hidden_rows=h.shape[1], visual_rows=h.shape[1] - len(self.layout.text), original_positions_sha256=indices_sha(positions)))
            out = layer(h, attention_mask=mask, position_ids=positions.unsqueeze(0), cache_position=positions, position_embeddings=rotary, past_key_value=self.cache, use_cache=True)
            h = out[0]
            if self.collect:
                if self.audit_enabled:
                    assert torch.isfinite(h).all()
        h = self.decoder.norm(h)
        self.cache.logical_length = self.layout.length + self.step
        self.step += 1
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=self.cache)

    @contextmanager
    def context(self):
        assert self.original_attention is None, 'Do not nest fusion contexts'
        self.original_attention = [layer.self_attn.forward for layer in self.a.layers]
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.decoder, 'forward', self.forward))
                for i, layer in enumerate(self.a.layers):
                    module, original = (layer.self_attn, self.original_attention[i])

                    def run(*args, module=module, original=original, i=i, **kw):
                        if i in LAYERS:
                            return self.answer_rows(module, *args, **kw)
                        out, weights = original(*args, **kw)
                        hidden = kw.get('hidden_states', args[0] if args else None)
                        self.record(out, weights, range(hidden.shape[1]), hidden.shape, kw.get('attention_mask'))
                        return (out, weights)
                    stack.enter_context(patch.object(module, 'forward', run))
                yield self
        finally:
            self.original_attention = None

@torch.inference_mode()
def generate(adapter, layout, cap, *, prune=True, capture=False, audit=False, verify_selector=False):
    from ..information_horizon_alignment.common import hook_counts, tensor_sha
    from ..information_horizon_alignment.engine import result
    ctl = Fusion(adapter, layout, cap, prune=prune, audit=audit, verify_selector=verify_selector)
    hooks = hook_counts(adapter.model)
    device, n = (adapter.inputs['input_ids'].device, layout.length)
    mask = torch.ones((1, n + cap - 1), dtype=torch.long, device=device)
    positions = torch.arange(n + cap - 1, device=device)
    cache = token = None
    ids, logits = ([], [])
    try:
        with ctl.context():
            for step in range(cap):
                kw = dict(adapter.inputs) if step == 0 else dict(input_ids=token.reshape(1, 1), attention_mask=mask[:, :n + step])
                kw.update(past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1, cache_position=positions[:n] if step == 0 else positions[n + step - 1:n + step])
                out = adapter.model(**kw)
                assert out.logits.shape[:2] == (1, 1) and torch.isfinite(out.logits).all()
                v = out.logits[:, -1]
                if step == 0:
                    first = tensor_sha(v)
                if capture:
                    logits.append(v.detach().float().cpu().clone())
                token = v.argmax(-1)
                ids.append(int(token))
                cache = out.past_key_values
                del out
                eos = adapter.model.generation_config.eos_token_id
                if ids[-1] in ([eos] if isinstance(eos, int) else eos):
                    break
        record = result(adapter, ids, cap)
        record.update(condition='FastV-K3 + llava-SVH' if prune else 'SVH-pruning-disabled', first_logits_sha256=first, logits_to_keep=1, selector=ctl.selection, physical_rows=ctl.structure, kv=ctl.cache.audit())
        if audit:
            record['attention_audit'] = ctl.events
        return (record, logits)
    finally:
        del ctl, cache
        assert hook_counts(adapter.model) == hooks

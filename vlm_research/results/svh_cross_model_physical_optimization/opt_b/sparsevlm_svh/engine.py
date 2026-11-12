"""SparseVLM-prefix + SVH with original text positions and no visual KV persistence."""
from contextlib import ExitStack, contextmanager
import hashlib
from unittest.mock import patch
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
from ..ate_fulljoint_v2.engine import AnswerWriter, LAYERS
from ..fastv_svh.engine import TextOnlyKV, indices_sha
from .common import GATE
from .official import kernels, prune_reference, attention_reference
STAGES = {2: 309, 6: 204, 15: 113}

class Fusion(AnswerWriter):

    def __init__(self, adapter, layout, cap, *, prune=True, dense=False, audit=False, verify_reference=False, capture=False):
        assert adapter.kind == 'llava' and layout.cutoff == 20
        assert len(adapter.layers) == 32 and len(layout.image) == 576
        assert layout.image == tuple(range(layout.image[0], layout.image[0] + 576))
        super().__init__(adapter, layout, 'C3-V2', layout.relay, cap, audit)
        self.prune, self.dense = (prune, dense)
        self.verify_reference, self.capture = (verify_reference, capture)
        self.cache = TextOnlyKV(layout, 32, cap)
        self.ns, self.official_prune, self.provenance = kernels()
        self.left, self.visual = (layout.image[0], 576)
        self.raters = self.relation = None
        self.selection, self.structure, self.hidden = ([], [], [])
        self.original_attention = None
        self.stage_positions = self.positions[:layout.length]
        self.visual_positions = self.stage_positions[self.left:self.left + 576].clone()
        self.visual_labels = [f'patch:{i}' for i in range(576)]
        self.mask = self.prefill_early
        self.active = torch.arange(layout.length, device=self.device)
        from ..information_horizon_alignment.common import tensor_sha
        seed_material = tensor_sha(adapter.inputs['input_ids']) + tensor_sha(adapter.inputs['pixel_values'])
        self.rng_seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:15], 16)

    def score_only(self, module, hidden, rotary, mask):
        """Official query set/softmax, same SVH mask, no score AV/O or residual."""
        n, hd = (hidden.shape[1], module.head_dim)
        start = torch.cuda.Event(enable_timing=True) if hidden.is_cuda else None
        end = torch.cuda.Event(enable_timing=True) if hidden.is_cuda else None
        if start is not None:
            start.record()
        q = module.q_proj(hidden).view(1, n, -1, hd).transpose(1, 2)
        k = module.k_proj(hidden).view(1, n, -1, hd).transpose(1, 2)
        cos, sin = rotary
        q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        k = repeat_kv(k, module.num_key_value_groups)
        probabilities = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * module.scaling + mask, -1)
        if self.audit_enabled:
            assert torch.isfinite(probabilities).all()
        if self.audit_enabled:
            assert torch.count_nonzero(probabilities[..., -1, self.left:self.left + self.visual]) == 0
        if end is not None:
            end.record()
            end.synchronize() if self.audit_enabled else None
        event = dict(layer=module.layer_idx, score_only=True, residual_injection=False, score_uses_actual_svh_source_mask=True, pminus1_image_probability_max=0.0, selected_text_offsets=self.raters[1].tolist(), includes_pminus1=bool((self.raters[1] == len(self.layout.relay)).any()), extra_q_shape=list(q.shape), extra_k_shape=list(k.shape), extra_matrix_flops=2 * n * module.q_proj.in_features * (module.q_proj.out_features + module.k_proj.out_features) + 2 * q.shape[1] * n * n * hd, score_only_gpu_ms=start.elapsed_time(end) if start is not None and self.audit_enabled else None)
        if self.verify_reference:
            v = module.v_proj(hidden).view(1, n, -1, hd).transpose(1, 2)
            v = repeat_kv(v, module.num_key_value_groups)
            _, reference = attention_reference(q, k, v, mask[0, 0])
            if self.audit_enabled:
                assert torch.equal(probabilities, reference), 'Official scoring probability mismatch'
            a = self.ns['attn_postprocess_topk'](probabilities, self.left, self.visual, self.left + self.visual, self.raters, module.layer_idx, 192)
            b = self.ns['attn_postprocess_topk'](reference, self.left, self.visual, self.left + self.visual, self.raters, module.layer_idx, 192)
            if self.audit_enabled:
                assert all((torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y for x, y in zip(a, b)))
            event['official_score_and_topk_exact'] = True
        self.relation = probabilities
        self.score_event = event

    def compact(self, h):
        return h.index_select(1, self.active) if self.dense else h

    def update_geometry(self, h):
        if self.visual:
            compact_positions = torch.cat((self.positions[:self.left], self.visual_positions, self.positions[self.layout.image[-1] + 1:self.layout.length]))
        else:
            compact_positions = self.positions[torch.tensor(self.layout.text, device=h.device)]
        if self.dense and self.visual:
            filler = 576 - self.visual
            self.stage_positions = torch.cat((compact_positions[:self.left + self.visual], self.positions[self.left:self.left + filler], compact_positions[self.left + self.visual:]))
            self.active = torch.cat((torch.arange(self.left + self.visual, device=h.device), torch.arange(self.left + 576, self.layout.length, device=h.device)))
        else:
            self.stage_positions = compact_positions
            self.active = torch.arange(h.shape[1], device=h.device)
        n = h.shape[1]
        self.mask = torch.full((n, n), torch.finfo(self.dtype).min, device=h.device, dtype=self.dtype).triu(1)[None, None]
        if self.visual:
            self.mask[..., -1, self.left:self.left + (576 if self.dense else self.visual)] = -torch.inf
        if self.dense and self.visual:
            self.mask[..., self.left + self.visual:self.left + 576] = -torch.inf
        if self.audit_enabled:
            assert (self.mask == 0).any(-1).all()
        self.cache.text_index = torch.tensor(self.layout.text, device=h.device) if self.dense and self.visual else torch.cat((torch.arange(self.left, device=h.device), torch.arange(self.left + self.visual, n, device=h.device))) if self.visual else torch.arange(n, device=h.device)
        if self.audit_enabled:
            assert self.stage_positions[self.cache.text_index].tolist() == list(self.layout.text)

    def recycle(self, h, layer):
        compact = self.compact(h)
        relation = self.relation
        assert relation is not None
        selected, _, scores = self.ns['attn_postprocess_topk'](relation, self.left, self.visual, self.left + self.visual, self.raters, layer, 192)
        kept = torch.where(selected[0])[0]
        dropped = torch.where(~selected[0])[0]
        seed = self.rng_seed + layer
        device = compact.device.index
        begin = torch.cuda.Event(enable_timing=True) if compact.is_cuda else None
        end = torch.cuda.Event(enable_timing=True) if compact.is_cuda else None
        if begin is not None:
            begin.record()
        with torch.random.fork_rng(devices=[device] if compact.is_cuda else []):
            if compact.is_cuda:
                torch.cuda.manual_seed(seed)
            else:
                torch.random.default_generator.manual_seed(seed)
            new_h, visual_count, _ = self.official_prune(compact, relation, self.left, self.visual, self.left + self.visual, self.raters, layer, self.stage_positions.index_select(0, self.active).unsqueeze(0))
        if end is not None:
            end.record()
            end.synchronize() if self.audit_enabled else None
        count = int(visual_count)
        clusters = count - len(kept)
        assert count == STAGES[layer] and new_h.shape[1] == len(self.layout.text) + count
        if self.audit_enabled:
            assert torch.equal(new_h[:, :self.left], compact[:, :self.left])
        if self.audit_enabled:
            assert torch.equal(new_h[:, self.left + count:], compact[:, self.left + self.visual:])
        event = dict(**self.score_event, visual_before=self.visual, visual_after=count, topk_retained=len(kept), recycled_pool=len(dropped), merged_tokens=clusters, kept_stage_indices=kept.tolist(), kept_stage_indices_sha256=indices_sha(kept), recycling_rng_seed=seed, recycling_gpu_ms=begin.elapsed_time(end) if begin is not None and self.audit_enabled else None, official_recycling_ast_unmodified=True, original_text_hidden_preserved=True, original_text_positions_preserved=True)
        if self.verify_reference:
            with torch.random.fork_rng(devices=[device] if compact.is_cuda else []):
                if compact.is_cuda:
                    torch.cuda.manual_seed(seed)
                else:
                    torch.random.default_generator.manual_seed(seed)
                reference = prune_reference(compact, relation, self.left, self.visual, self.raters, layer)
            x, y = (new_h.float(), reference.float())
            relative = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y))
            cosine = float((1 - torch.nn.functional.cosine_similarity(x.double(), y.double(), dim=-1)).abs().max())
            assert relative <= GATE['hidden_relative_rmse'] and cosine <= GATE['hidden_cosine_distance'], ('Official recycling/reference mismatch', relative, cosine)
            event.update(independent_recycling_reference_exact=torch.equal(reference, new_h), independent_recycling_reference_passed=True, recycling_reference_relative_rmse=relative, recycling_reference_cosine_distance=cosine, recycling_reference_max_abs=float((x - y).abs().max()))
        previous_labels = self.visual_labels
        self.visual_labels = [previous_labels[i] for i in kept.tolist()] + [f'merged:L{layer}:{i}' for i in range(clusters)]
        self.visual_positions = torch.cat((self.visual_positions[kept], self.positions[self.left:self.left + clusters]))
        event['merged_rope_anchors'] = self.visual_positions[-clusters:].tolist()
        event['visual_labels_sha256'] = hashlib.sha256('|'.join(self.visual_labels).encode()).hexdigest()
        self.visual = count
        if self.dense:
            filler = compact.new_zeros((1, 576 - count, compact.shape[-1]))
            h = torch.cat((new_h[:, :self.left + count], filler, new_h[:, self.left + count:]), 1)
        else:
            h = new_h
        self.selection.append(event)
        self.relation = None
        self.update_geometry(h)
        return h

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        h = inputs_embeds if inputs_embeds is not None else self.decoder.embed_tokens(input_ids)
        prefill = self.step == 0
        self.cache.prefill = prefill
        if prefill:
            self.cache.text_index = torch.tensor(self.layout.text, device=h.device)
            if self.prune:
                self.raters = self.ns['rate_text'](h, self.left, self.left + 576)
                assert self.raters[1].numel(), 'Official text-rater set is empty; stop without fallback'
            self.update_geometry(h)
            positions = self.stage_positions
        else:
            positions = self.positions[self.layout.length + self.step - 1:self.layout.length + self.step]
        rotary = self.decoder.rotary_emb(h, positions.unsqueeze(0))
        for i, layer in enumerate(self.a.layers):
            if prefill and i == 20:
                h = h.index_select(1, self.cache.text_index)
                self.visual = 0
                self.update_geometry(h)
                positions = self.stage_positions
                rotary = self.decoder.rotary_emb(h, positions.unsqueeze(0))
            mask = self.mask if prefill else None
            self.active_mask = mask[..., :-1, :] if prefill else None
            self.current = (i, prefill, mask, positions, h.shape[1] if prefill else self.cache.used[i] + 1)
            if prefill:
                if self.audit_enabled:
                    self.structure.append(dict(layer=i, hidden_rows=h.shape[1], visual_rows=self.visual, physically_allocated_visual_rows=h.shape[1] - len(self.layout.text), original_positions_sha256=indices_sha(positions)))
            out = layer(h, attention_mask=mask, position_ids=positions.unsqueeze(0), cache_position=positions, position_embeddings=rotary, past_key_value=self.cache, use_cache=True)
            h = out[0]
            if self.collect:
                if self.audit_enabled:
                    assert torch.isfinite(h).all()
            if prefill and self.capture:
                self.hidden.append(h.index_select(1, self.cache.text_index).float().cpu().clone())
            if prefill and self.prune and (i in STAGES):
                h = self.recycle(h, i)
                positions = self.stage_positions
                rotary = self.decoder.rotary_emb(h, positions.unsqueeze(0))
        self.cache.logical_length = self.layout.length + self.step
        self.step += 1
        return BaseModelOutputWithPast(last_hidden_state=self.decoder.norm(h), past_key_values=self.cache)

    @contextmanager
    def context(self):
        assert self.original_attention is None
        self.original_attention = [layer.self_attn.forward for layer in self.a.layers]
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.decoder, 'forward', self.forward))
                for i, layer in enumerate(self.a.layers):
                    module, original = (layer.self_attn, self.original_attention[i])

                    def forward(*args, module=module, original=original, i=i, **kw):
                        hidden = kw.get('hidden_states', args[0] if args else None)
                        if self.cache.prefill and self.prune and (i in STAGES):
                            compact = self.compact(hidden)
                            rotary = tuple((t.index_select(1, self.active) for t in kw['position_embeddings']))
                            mask = kw['attention_mask'].index_select(-2, self.active).index_select(-1, self.active)
                            self.score_only(module, compact, rotary, mask)
                        if i in LAYERS:
                            if not self.dense:
                                return self.answer_rows(module, *args, **kw)
                            if not self.cache.prefill:
                                return (torch.zeros_like(hidden), None)
                            ref_kw = dict(kw, past_key_value=None)
                            result, weights = original(*args, **ref_kw)
                            result = result.clone()
                            result[:, -1] = 0
                        else:
                            result, weights = original(*args, **kw)
                        self.record(result, weights, range(hidden.shape[1]), hidden.shape, kw.get('attention_mask'))
                        return (result, weights)
                    stack.enter_context(patch.object(module, 'forward', forward))
                yield self
        finally:
            self.original_attention = None

@torch.inference_mode()
def generate(adapter, layout, cap, *, prune=True, dense=False, audit=False, verify_reference=False, capture=False):
    from ..information_horizon_alignment.common import hook_counts, tensor_sha
    from ..information_horizon_alignment.engine import result
    ctl = Fusion(adapter, layout, cap, prune=prune, dense=dense, audit=audit, verify_reference=verify_reference, capture=capture)
    hooks = hook_counts(adapter.model)
    n, device = (layout.length, next(adapter.model.parameters()).device)
    masks = torch.ones((1, n + cap - 1), device=device, dtype=torch.long)
    positions = torch.arange(n + cap - 1, device=device)
    cache = token = None
    ids, logits = ([], [])
    try:
        with ctl.context():
            for step in range(cap):
                kw = dict(adapter.inputs) if step == 0 else dict(input_ids=token.reshape(1, 1), attention_mask=masks[:, :n + step])
                kw.update(past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1, cache_position=positions[:n] if step == 0 else positions[n + step - 1:n + step])
                out = adapter.model(**kw)
                assert out.logits.shape[:2] == (1, 1) and torch.isfinite(out.logits).all()
                values = out.logits[:, -1]
                if step == 0:
                    first = tensor_sha(values)
                if capture:
                    logits.append(values.float().cpu().clone())
                token = values.argmax(-1)
                ids.append(int(token))
                cache = out.past_key_values
                del out
                eos = adapter.model.generation_config.eos_token_id
                if ids[-1] in ([eos] if isinstance(eos, int) else eos):
                    break
        record = result(adapter, ids, cap)
        record.update(condition='SparseVLM-192-prefix + llava-SVH' if prune else 'SVH-pruning-disabled', first_logits_sha256=first, logits_to_keep=1, stages=ctl.selection, physical_rows=ctl.structure, kv=ctl.cache.audit(), official_source_provenance=ctl.provenance, mean_visual_rows=sum((s['visual_rows'] for s in ctl.structure)) / 32, original_text_rope=True, dense_reference=dense)
        if audit:
            record['attention_audit'] = ctl.events
        return (record, dict(logits=logits, hidden=ctl.hidden))
    finally:
        del ctl, cache
        assert hook_counts(adapter.model) == hooks

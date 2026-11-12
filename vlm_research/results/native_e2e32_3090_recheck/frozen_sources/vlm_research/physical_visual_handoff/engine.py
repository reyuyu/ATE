"""Native visual front end and native decoder blocks, with explicit logical KV positions."""
from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import patch
import time
import weakref

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..p1_layerwise_accuracy.engine import Adapter, footprint, tensor_hash
from .common import TOLERANCES


@dataclass
class Plan:
    condition: str
    cutoff: int
    physical: bool = False
    compact: bool = False
    immediate: bool = False
    question_only: bool = False
    skip: tuple = ()
    physical_skip: bool = False


def plan(kind, condition, immediate=False):
    p = Plan(condition, 20 if kind == 'llava' else 22)
    p.physical = condition in ('P1', 'P2', 'P3', 'P4', 'P4-six')
    p.compact = condition in ('P2', 'P3', 'P4', 'P4-six')
    p.immediate = p.compact and immediate
    p.question_only = condition in ('D2', 'D3', 'P3', 'P4', 'D3-six', 'P4-six')
    p.skip = tuple(range(1, 7)) if condition.endswith('-six') else tuple(range(2, 6)) if condition in ('D3', 'P4') else ()
    p.physical_skip = condition in ('P4', 'P4-six')
    return p


class CompactKV:
    """Physical tensors and original sequence positions are separate at every layer."""
    def __init__(self, layers):
        self.entries = [None] * layers
        self.logical_length = 0
        self.current_positions = None
        self.released = []

    def get_seq_length(self, layer_idx=0):
        return self.logical_length

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        pos = self.current_positions
        assert pos.numel() == keys.shape[-2]
        previous = self.entries[layer_idx]
        if previous is not None:
            keys = torch.cat([previous[0], keys], dim=-2)
            values = torch.cat([previous[1], values], dim=-2)
            pos = torch.cat([previous[2], pos])
        self.entries[layer_idx] = (keys, values, pos)
        return keys, values

    def remove_visual(self, i, image):
        entry = self.entries[i]
        if entry is None:
            return
        k, v, pos = entry
        keep = torch.where(~torch.isin(pos, image))[0]
        if len(keep) == len(pos):
            return
        self.released += [weakref.ref(k), weakref.ref(v)]
        self.entries[i] = (k.index_select(-2, keep), v.index_select(-2, keep), pos.index_select(0, keep))

    def drop_layer(self, i):
        if self.entries[i] is not None:
            self.released += [weakref.ref(v) for v in self.entries[i][:2]]
            self.entries[i] = None

    def audit(self):
        rows = []
        for i, e in enumerate(self.entries):
            tensors = [] if e is None else list(e[:2])
            rows.append(dict(layer=i, shapes=[list(v.shape) for v in tensors],
                storage_bytes=sum(v.untyped_storage().nbytes() for v in tensors),
                logical_length=self.logical_length, physical_length=0 if e is None else e[0].shape[-2],
                original_key_positions=[] if e is None else e[2].tolist()))
        return dict(layers=rows, live_kv_bytes=sum(r['storage_bytes'] for r in rows),
                    old_tensor_references_alive=sum(ref() is not None for ref in self.released))


def source_mask(positions, keys, image, relay, question, p, dtype):
    legal = keys[None, :] <= positions[:, None]
    mask = torch.zeros((1, 1, len(positions), len(keys)), device=positions.device, dtype=dtype)
    mask[0, 0].masked_fill_(~legal, torch.finfo(dtype).min)
    if p.condition == 'N0':
        return mask, torch.zeros_like(legal)
    answer = positions >= p.length - 1
    active = positions.new_tensor(question if p.question_only else relay)
    writer = torch.isin(positions, active) if p.layer < p.cutoff or p.condition == 'D0' else torch.zeros_like(answer)
    disabled_r = torch.isin(positions, positions.new_tensor(relay)) & ~writer
    blocked = (answer | disabled_r)[:, None] & torch.isin(keys, image)[None, :]
    mask[0, 0].masked_fill_(blocked, -torch.inf)
    assert (mask[0, 0] == 0).any(-1).all()
    return mask, blocked


class Handoff:
    def __init__(self, adapter, p, sample, spec, audit=False):
        self.a, self.p, self.audit_enabled = adapter, p, audit
        self.n = adapter.inputs['input_ids'].shape[1]
        self.p.length = self.n
        self.decoder = adapter.model.language_model.model if adapter.kind == 'llava' else adapter.model.model.language_model
        ids = adapter.inputs['input_ids'][0]
        token = adapter.model.config.image_token_index if adapter.kind == 'llava' else adapter.model.config.image_token_id
        self.image = torch.where(ids == token)[0]
        self.relay = list(range(int(self.image[-1]) + 1, self.n - 1))
        if adapter.kind == 'qwen':
            self.question = spec['layout']['question_positions']
        else:
            # Pin the question span using the exact historical rendered prompt and tokenizer offsets.
            prompt = adapter.processor.tokenizer.decode(ids.tolist(), skip_special_tokens=False)
            # Fast-tokenizer offsets are derived from the original chat rendering, not decoded approximations.
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': sample['prompt_text']}]}]
            rendered = adapter.processor.apply_chat_template(messages, add_generation_prompt=True)
            expanded = rendered.replace(adapter.processor.image_token, adapter.processor.image_token * len(self.image))
            encoded = adapter.processor.tokenizer(expanded, return_offsets_mapping=True)
            assert encoded['input_ids'] == ids.tolist()
            start = expanded.index(sample['question']); end = start + len(sample['question'])
            self.question = [j for j, (lo, hi) in enumerate(encoded['offset_mapping']) if hi > start and lo < end]
        assert self.question and set(self.question) <= set(self.relay)
        self.cache = CompactKV(len(adapter.layers))
        self.events, self.hidden, self.mask_events = [], [], []
        self.step = 0
        self.initial_positions = None
        self.rotary_hashes = []

    def attention_without_answer(self, module, hidden_states, position_embeddings, attention_mask, **kwargs):
        # All non-answer Prefill Q/K/V are required; the final P-1 cannot be a Key for earlier causal Queries.
        if hidden_states.shape[1] == 1:
            self.events.append(dict(layer=self.p.layer, phase='decode', qkv_rows=0, attention_query_rows=0,
                                    o_proj_rows=0, physical_skip=True))
            return torch.zeros_like(hidden_states), None
        h = hidden_states[:, :-1]
        rotary = tuple(v[:, :-1] for v in position_embeddings)
        kw = dict(kwargs, past_key_value=None, past_key_values=None)
        kw.pop('past_key_value' if self.a.kind == 'qwen' else 'past_key_values')
        kw['cache_position'] = kwargs['cache_position'][:-1]
        if kw.get('position_ids') is not None:
            kw['position_ids'] = kw['position_ids'][..., :-1]
        original = self.original_attention[self.p.layer]
        output = original(hidden_states=h, position_embeddings=rotary, attention_mask=attention_mask[:, :, :-1, :-1], **kw)[0]
        self.events.append(dict(layer=self.p.layer, phase='prefill', qkv_rows=h.shape[1],
                                attention_query_rows=h.shape[1], o_proj_rows=h.shape[1], physical_skip=True))
        return torch.cat([output, torch.zeros_like(hidden_states[:, -1:])], dim=1), None

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, cache_position=None,
                visual_pos_masks=None, deepstack_visual_embeds=None, **kwargs):
        h = inputs_embeds if inputs_embeds is not None else self.decoder.embed_tokens(input_ids)
        prefill = self.step == 0
        positions = cache_position
        if positions is None:
            positions = torch.arange(self.cache.logical_length, self.cache.logical_length + h.shape[1], device=h.device)
        if position_ids is None:
            position_ids = positions.unsqueeze(0)
        if self.a.kind == 'qwen' and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        rotary = self.decoder.rotary_emb(h, position_ids)
        if prefill:
            self.initial_positions = position_ids.clone()
        if deepstack_visual_embeds is not None:
            assert len(deepstack_visual_embeds) < self.p.cutoff
        self.rotary_hashes.append(tensor_hash(position_ids) if self.audit_enabled else '')
        for i, layer in enumerate(self.a.layers):
            self.p.layer = i
            if prefill and self.p.physical and i == self.p.cutoff:
                keep = torch.where(~torch.isin(positions, self.image))[0]
                h = h.index_select(1, keep)
                positions = positions.index_select(0, keep)
                position_ids = position_ids.index_select(-1, keep)
                rotary = tuple(v.index_select(1, keep) for v in rotary)
            old = self.cache.entries[i]
            keys = positions if old is None else torch.cat([old[2], positions])
            mask, blocked = source_mask(positions, keys, self.image, self.relay, self.question, self.p, h.dtype)
            self.cache.current_positions = positions
            kw = dict(attention_mask=mask, position_ids=position_ids[0] if self.a.kind == 'qwen' else position_ids,
                      cache_position=positions, position_embeddings=rotary, use_cache=True)
            kw['past_key_values' if self.a.kind == 'qwen' else 'past_key_value'] = self.cache
            self.events.append(dict(layer=i, phase='prefill' if prefill else 'decode', token_rows=h.shape[1],
                key_rows=len(keys), mlp_rows=h.shape[1], original_positions=positions.tolist() if self.audit_enabled else [],
                blocked_connections=int(blocked.sum()) if self.audit_enabled else -1))
            if self.audit_enabled:
                self.pending_blocked = blocked
            out = layer(h, **kw)
            h = out if self.a.kind == 'qwen' else out[0]
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                h = self.decoder._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[i])
            if prefill and self.audit_enabled:
                text = ~torch.isin(positions, self.image)
                self.hidden.append(h[:, text].detach().float().cpu())
            if prefill and self.p.immediate:
                self.cache.remove_visual(i, self.image)
                if self.p.physical_skip and i in self.p.skip:
                    self.cache.drop_layer(i)
        h = self.decoder.norm(h)
        self.cache.logical_length = self.n + self.step
        if prefill and self.p.compact and not self.p.immediate:
            for i in range(len(self.a.layers)):
                self.cache.remove_visual(i, self.image)
                if self.p.physical_skip and i in self.p.skip:
                    self.cache.drop_layer(i)
        self.step += 1
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=self.cache)

    @contextmanager
    def context(self):
        handles = []
        self.original_attention = {i: v.self_attn.forward for i, v in enumerate(self.a.layers)}
        from contextlib import ExitStack
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.decoder, 'forward', self.forward))
                for i in self.p.skip:
                    attn = self.a.layers[i].self_attn
                    if self.p.physical_skip:
                        stack.enter_context(patch.object(attn, 'forward', lambda *a, _attn=attn, **kw:
                                                        self.attention_without_answer(_attn, *a, **kw)))
                    else:
                        def zero(m, args, out):
                            changed = out.clone()
                            changed[:, -1 if self.step == 0 else 0] = 0
                            return changed
                        handles.append(attn.o_proj.register_forward_hook(zero))
                if self.audit_enabled:
                    for i, layer in enumerate(self.a.layers):
                        def check(m, args, out, i=i):
                            w = out[1]
                            b = self.pending_blocked
                            if w is not None and w.shape[-2:] == b.shape and b.any():
                                assert float(w[0, :, b].abs().max()) == 0
                                self.mask_events.append(dict(layer=i, step=self.step, blocked_probability_max=0.0))
                        handles.append(layer.self_attn.register_forward_hook(check))
                yield self
        finally:
            for handle in handles:
                handle.remove()


@torch.inference_mode()
def run(adapter, sample, spec, condition, max_tokens=32, fixed=False, audit=False, capture=False, immediate=False,
        native=False):
    before = footprint(adapter.model)
    p = plan(adapter.kind, condition, immediate)
    control = Handoff(adapter, p, sample, spec, audit)
    inputs = adapter.inputs
    n = inputs['input_ids'].shape[1]
    tokens, scores, cache = [], [], None
    valid_hashes, vision_events = [], []
    eos = adapter.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    timings = []
    peak_start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    from contextlib import nullcontext, ExitStack
    target = adapter.model if adapter.kind == 'llava' else adapter.model.model
    original_visual = target.get_image_features
    def visual(*args, **kwargs):
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record(); value = original_visual(*args, **kwargs); b.record()
        vision_events.append((a,b))
        return value
    with ExitStack() as stack:
        stack.enter_context(nullcontext() if native else control.context())
        stack.enter_context(patch.object(target,'get_image_features',visual))
        for step in range(max_tokens):
            if step == 0:
                kw = dict(inputs)
            else:
                kw = dict(input_ids=torch.tensor([[tokens[-1]]], device='cuda:0'),
                          attention_mask=torch.ones((1, n + step), dtype=torch.long, device='cuda:0'))
            kw.update(use_cache=True, return_dict=True, past_key_values=cache,
                      cache_position=torch.arange(n, device='cuda:0') if step == 0 else torch.tensor([n + step - 1], device='cuda:0'),
                      logits_to_keep=0 if adapter.kind == 'llava' else 1)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            out = adapter.model(**kw)
            end.record(); end.synchronize()
            timings.append(begin.elapsed_time(end))
            cache = out.past_key_values
            logits = out.logits[:, -1].detach()
            assert torch.isfinite(logits).all()
            if capture:
                scores.append(logits.float().cpu())
                valid_hashes.append(tensor_hash(out.logits))
            tokens.append(int(logits.argmax(-1)))
            del out
            if not fixed and tokens[-1] in eos:
                break
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    text = adapter.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    if native:
        entries = [(v.keys,v.values) for v in cache.layers] if hasattr(cache,'layers') else list(zip(cache.key_cache,cache.value_cache))
        cache_info=dict(layers=[dict(layer=i,shapes=[list(v.shape) for v in pair],
                                    storage_bytes=sum(v.untyped_storage().nbytes() for v in pair)) for i,pair in enumerate(entries)],
                        live_kv_bytes=sum(v.untyped_storage().nbytes() for pair in entries for v in pair),
                        old_tensor_references_alive=0)
    else:
        cache_info=control.cache.audit()
    vision_ms=sum(a.elapsed_time(b) for a,b in vision_events)
    result = dict(condition=condition, token_ids=tokens, text=text, correct=text.rstrip('.').lower() == sample['answer'].lower(),
        eos=tokens[-1] in eos, truncated=len(tokens) == max_tokens and tokens[-1] not in eos, empty=not text,
        latency_ms=elapsed, ttft_ms=timings[0], decode_ms=timings[1:], peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(), initial_allocated=peak_start, fixed_length_benchmark=fixed,
        vision_ms=vision_ms,language_prefill_ms=timings[0]-vision_ms,all_valid_logits_sha256=valid_hashes,
        cache=cache_info,
        execution=control.events if audit else [], mask_events=control.mask_events,
        inputs_sha256={k: tensor_hash(v) for k, v in inputs.items()}, immediate_release=immediate)
    assert footprint(adapter.model) == before
    del cache
    return result, scores, control.hidden


def equivalence(kind, a, b, ash, bsh):
    tol = TOLERANCES[kind]
    assert len(a) == len(b)
    errors = [(x - y).abs() for x, y in zip(a, b)]
    max_abs = max(float(e.max()) for e in errors)
    relative = [e / x.abs().clamp_min(1e-6) for e, x in zip(errors, a)]
    hidden_rmse, hidden_cosine = [], []
    for x, y in zip(ash, bsh):
        assert x.shape == y.shape
        hidden_rmse.append(float(((x-y).square().mean(-1).sqrt() / x.square().mean(-1).sqrt().clamp_min(1e-12)).max()))
        hidden_cosine.append(float((1-torch.nn.functional.cosine_similarity(x,y,dim=-1)).clamp_min(0).max()))
    passed = all(torch.allclose(x,y,atol=tol['logit_atol'],rtol=tol['logit_rtol']) for x,y in zip(a,b))
    passed &= max(hidden_rmse, default=0) <= tol['hidden_relative_rmse'] and max(hidden_cosine, default=0) <= tol['hidden_cosine']
    return dict(passed=bool(passed), max_abs_logits=max_abs, mean_relative_error=sum(float(e.mean()) for e in relative)/len(relative),
        max_relative_error=max(float(e.max()) for e in relative), hidden_relative_rmse=max(hidden_rmse, default=0),
        hidden_cosine_distance=max(hidden_cosine, default=0), all_argmax_equal=all(int(x.argmax())==int(y.argmax()) for x,y in zip(a,b)),
        tolerances=tol)

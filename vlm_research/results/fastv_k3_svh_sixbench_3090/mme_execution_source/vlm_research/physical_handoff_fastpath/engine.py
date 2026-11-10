"""Precomputed masks and persistent, in-place KV around unmodified native blocks."""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import time
import os
import subprocess
from unittest.mock import patch

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast


@dataclass
class Layout:
    length: int
    image: tuple
    text: tuple
    relay: tuple
    question: tuple
    cutoff: int

    @classmethod
    def from_adapter(cls, adapter, sample, spec):
        ids = adapter.inputs['input_ids'][0].tolist()
        image_id = (adapter.model.config.image_token_index if adapter.kind == 'llava'
                    else adapter.model.config.image_token_id)
        image = tuple(i for i, v in enumerate(ids) if v == image_id)
        assert image and len(image) == len(set(image))
        image_set = set(image)
        text = tuple(i for i in range(len(ids)) if i not in image_set)
        relay = tuple(range(image[-1] + 1, len(ids) - 1))
        if adapter.kind == 'qwen':
            question = tuple(spec['layout']['question_positions'])
            assert len(image) == spec['layout']['image_tokens']
        else:
            assert len(image) == 576
            prompt = adapter.processor.apply_chat_template([
                {'role': 'user', 'content': [{'type': 'image'},
                 {'type': 'text', 'text': sample['prompt_text']}]}], add_generation_prompt=True)
            prompt = prompt.replace(adapter.processor.image_token,
                                    adapter.processor.image_token * len(image))
            encoded = adapter.processor.tokenizer(prompt, return_offsets_mapping=True)
            assert encoded['input_ids'] == ids
            begin = prompt.index(sample['question'])
            end = begin + len(sample['question'])
            question = tuple(i for i, (a, b) in enumerate(encoded['offset_mapping'])
                             if b > begin and a < end)
        assert question and set(question) <= set(relay)
        assert len(text) + len(image) == len(ids) and text[-1] == len(ids) - 1
        return cls(len(ids), image, text, relay, question, 20 if adapter.kind == 'llava' else 22)


class InplaceKV:
    """Return transient full Prefill KV, but persist only what Decode can access.

    Logical length is the original untrimmed sequence length. Buffer used lengths
    are physical lengths; no position renumbering and no historical concatenation.
    """
    def __init__(self, layers, layout, condition, max_tokens, device):
        self.layout, self.condition, self.max_tokens = layout, condition, max_tokens
        self.entries = [None] * layers
        self.used = [0] * layers
        self.logical_length = 0
        self.prefill = True
        self.text_index = torch.tensor(layout.text, device=device, dtype=torch.long)
        self.allocations = self.write_calls = self.gather_calls = 0
        self.profile = False
        self.management_events = []

    def get_seq_length(self, layer_idx=0):
        return self.logical_length

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        region = torch.profiler.record_function('handoff::kv_update') if self.profile else nullcontext()
        with region:
            begin = torch.cuda.Event(enable_timing=True) if self.profile and keys.is_cuda else None
            if begin:
                begin.record()
            if self.prefill:
                compact = self.condition == 'Fast-P2' and keys.shape[-2] == self.layout.length
                count = len(self.layout.text) if compact else keys.shape[-2]
                capacity = count + self.max_tokens - 1
                shape = (*keys.shape[:-2], capacity, keys.shape[-1])
                kbuf = torch.empty(shape, device=keys.device, dtype=keys.dtype)
                vbuf = torch.empty_like(kbuf)
                self.allocations += 2
                if compact:
                    # Gather directly into final buffers; never allocate full persistent visual KV.
                    torch.index_select(keys, -2, self.text_index, out=kbuf[..., :count, :])
                    torch.index_select(values, -2, self.text_index, out=vbuf[..., :count, :])
                    self.gather_calls += 2
                else:
                    kbuf[..., :count, :].copy_(keys)
                    vbuf[..., :count, :].copy_(values)
                self.entries[layer_idx] = kbuf, vbuf
                self.used[layer_idx] = count
                result = keys, values
            else:
                kbuf, vbuf = self.entries[layer_idx]
                count = self.used[layer_idx]
                new_count = count + keys.shape[-2]
                if new_count > kbuf.shape[-2]:
                    raise RuntimeError('KV capacity exceeded')
                kbuf[..., count:new_count, :].copy_(keys)
                vbuf[..., count:new_count, :].copy_(values)
                self.used[layer_idx] = new_count
                result = kbuf[..., :new_count, :], vbuf[..., :new_count, :]
            self.write_calls += 2
            if begin:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self.management_events.append(('prefill' if self.prefill else 'decode', begin, end))
            return result

    def audit(self):
        rows = []
        for i, pair in enumerate(self.entries):
            k, v = pair
            element_bytes = k.element_size()
            live = sum(t.shape[0] * t.shape[1] * self.used[i] * t.shape[-1] * element_bytes for t in pair)
            storage = sum(t.untyped_storage().nbytes() for t in pair)
            compact = self.condition == 'Fast-P2' or (
                self.condition == 'Fast-P1' and i >= self.layout.cutoff)
            positions = list(self.layout.text) if compact else list(range(self.layout.length))
            positions += list(range(self.layout.length, self.logical_length))
            assert len(positions) == self.used[i]
            rows.append(dict(layer=i, used_tokens=self.used[i], capacity_tokens=k.shape[-2],
                             logical_length=self.logical_length, live_bytes=live,
                             storage_bytes=storage, original_positions=positions,
                             key_stride=list(k.stride()), value_stride=list(v.stride())))
        return dict(layers=rows, live_bytes=sum(r['live_bytes'] for r in rows),
                    storage_bytes=sum(r['storage_bytes'] for r in rows), allocations=self.allocations,
                    write_calls=self.write_calls, direct_gather_calls=self.gather_calls,
                    historical_cache_copies=0, append_concatenations=0)


class FastHandoff:
    def __init__(self, adapter, layout, condition, max_tokens=32, audit=False):
        assert condition in ('Native-Noop', 'D1 Dense', 'Fast-P1', 'Fast-P2')
        self.a, self.layout, self.condition = adapter, layout, condition
        self.decoder = (adapter.model.language_model.model if adapter.kind == 'llava'
                        else adapter.model.model.language_model)
        self.physical = condition in ('Fast-P1', 'Fast-P2')
        self.device, self.dtype = adapter.inputs['input_ids'].device, next(adapter.layers[0].parameters()).dtype
        self.positions = torch.arange(layout.length + max_tokens - 1, device=self.device)
        self.text_index = torch.tensor(layout.text, device=self.device, dtype=torch.long)
        self.text_positions = self.positions[:layout.length].index_select(0, self.text_index)
        self.cache = InplaceKV(len(adapter.layers), layout, condition, max_tokens, self.device)
        self.step, self.audit_enabled = 0, audit
        self.audit_rows, self.hidden, self.rotary, self.valid_logits = [], [], [], []
        self.current = None
        n = layout.length
        causal = torch.full((n, n), torch.finfo(self.dtype).min, device=self.device,
                            dtype=self.dtype).triu(diagonal=1)
        early = causal.clone()
        late = causal.clone()
        if condition != 'Native-Noop':
            image = torch.tensor(layout.image, device=self.device)
            early[n - 1, image] = -torch.inf
            blocked = torch.tensor((*layout.relay, n - 1), device=self.device)
            late[blocked[:, None], image[None, :]] = -torch.inf
        self.prefill_early = early[None, None]
        self.prefill_late = (late.index_select(0, self.text_index).index_select(1, self.text_index)
                             if self.physical else late)[None, None]
        # Decode sees no future Keys; only dense variants need a source mask.
        self.decode_mask = torch.zeros((1, 1, 1, n + max_tokens - 1), device=self.device, dtype=self.dtype)
        if condition != 'Native-Noop':
            self.decode_mask[..., list(layout.image)] = -torch.inf

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, cache_position=None,
                visual_pos_masks=None, deepstack_visual_embeds=None, **kwargs):
        h = inputs_embeds if inputs_embeds is not None else self.decoder.embed_tokens(input_ids)
        prefill = self.step == 0
        positions = self.positions[:self.layout.length] if prefill else self.positions[
            self.layout.length + self.step - 1:self.layout.length + self.step]
        if position_ids is None:
            position_ids = positions.unsqueeze(0)
        if self.a.kind == 'qwen' and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        rotary = self.decoder.rotary_emb(h, position_ids)
        if self.audit_enabled:
            self.rotary.append(dict(position_ids=position_ids.detach().cpu().clone(),
                                    cos=rotary[0].detach().cpu().clone(), sin=rotary[1].detach().cpu().clone()))
        if deepstack_visual_embeds is not None and len(deepstack_visual_embeds) >= self.layout.cutoff:
            raise RuntimeError('Unexpected DeepStack beyond physical cutoff')
        self.cache.prefill = prefill
        for i, layer in enumerate(self.a.layers):
            if prefill and self.physical and i == self.layout.cutoff:
                h = h.index_select(1, self.text_index)
                positions = self.text_positions
                position_ids = position_ids.index_select(-1, self.text_index)
                rotary = tuple(v.index_select(1, self.text_index) for v in rotary)
            if prefill:
                mask = self.prefill_early if i < self.layout.cutoff else self.prefill_late
                key_count = h.shape[1]
            else:
                key_count = self.cache.used[i] + 1
                compact = self.condition == 'Fast-P2' or (self.physical and i >= self.layout.cutoff)
                mask = None if compact else self.decode_mask[..., :key_count]
            self.current = (i, prefill, mask, positions, key_count)
            kw = dict(attention_mask=mask, position_ids=position_ids[0] if self.a.kind == 'qwen' else position_ids,
                      cache_position=positions, position_embeddings=rotary, use_cache=True)
            kw['past_key_values' if self.a.kind == 'qwen' else 'past_key_value'] = self.cache
            out = layer(h, **kw)
            h = out if self.a.kind == 'qwen' else out[0]
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                h = self.decoder._deepstack_process(h, visual_pos_masks, deepstack_visual_embeds[i])
            if self.audit_enabled and prefill:
                kept = self.text_index if h.shape[1] == self.layout.length else torch.arange(
                    h.shape[1], device=h.device)
                self.hidden.append(h.index_select(1, kept).detach().float().cpu())
        h = self.decoder.norm(h)
        self.cache.logical_length = self.layout.length + self.step
        self.step += 1
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=self.cache)

    @contextmanager
    def context(self):
        handles = []
        try:
            with patch.object(self.decoder, 'forward', self.forward):
                if self.audit_enabled:
                    for layer in self.a.layers:
                        def observe(module, args, out):
                            i, prefill, mask, pos, key_count = self.current
                            weights = out[1]
                            assert weights is not None and torch.isfinite(weights).all()
                            blocked_max = 0.0
                            if mask is not None:
                                blocked = torch.isneginf(mask[0, 0])
                                assert (mask[0, 0] == 0).any(-1).all()
                                if blocked.any():
                                    blocked_max = float(weights[0, :, blocked].abs().max())
                                    assert blocked_max == 0.0
                            error = float((weights.float().sum(-1) - 1).abs().max())
                            assert error < .005
                            self.audit_rows.append(dict(layer=i, step=self.step, prefill=prefill,
                                query_count=weights.shape[-2], key_count=key_count,
                                original_query_positions=pos.tolist(), blocked_probability_max=blocked_max,
                                row_sum_error=error))
                        handles.append(layer.self_attn.register_forward_hook(observe))
                yield self
        finally:
            for handle in handles:
                handle.remove()


def native_cache_info(cache):
    pairs = ([(v.keys, v.values) for v in cache.layers] if hasattr(cache, 'layers')
             else list(zip(cache.key_cache, cache.value_cache)))
    rows = [dict(layer=i, used_tokens=k.shape[-2], capacity_tokens=k.shape[-2],
                 live_bytes=k.numel() * k.element_size() + v.numel() * v.element_size(),
                 storage_bytes=k.untyped_storage().nbytes() + v.untyped_storage().nbytes())
            for i, (k, v) in enumerate(pairs)]
    return dict(layers=rows, live_bytes=sum(v['live_bytes'] for v in rows),
                storage_bytes=sum(v['storage_bytes'] for v in rows),
                allocations='native DynamicCache', historical_cache_copies='native torch.cat')


@torch.inference_mode()
def generate(adapter, layout, condition, max_tokens=32, fixed=False, capture=False, audit=False,
             profile=False):
    """Cold per-request cache/setup included in E2E; no layer-wise sync in fast mode."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    control = (None if condition == 'Native Full' else
               FastHandoff(adapter, layout, condition, max_tokens, audit))
    if control:
        control.cache.profile = profile
    n, device = layout.length, adapter.inputs['input_ids'].device
    all_positions = torch.arange(n + max_tokens - 1, device=device)
    attention_mask = torch.ones((1, n + max_tokens - 1), device=device, dtype=torch.long)
    eos = adapter.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    ids, gpu_ids, logits_capture, valid, cache = [], [], [], [], None
    timings, vision_events = [], []
    target = adapter.model if adapter.kind == 'llava' else adapter.model.model
    original_vision = target.get_image_features
    def vision(*a, **kw):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.profiler.record_function('handoff::vision') if profile else nullcontext():
            result = original_vision(*a, **kw)
        end.record()
        vision_events.append((begin, end))
        return result
    with (control.context() if control else nullcontext()), patch.object(target, 'get_image_features', vision):
        for step in range(max_tokens):
            if step == 0:
                kw = dict(adapter.inputs)
            else:
                kw = dict(input_ids=gpu_ids[-1].reshape(1, 1), attention_mask=attention_mask[:, :n + step])
            kw.update(use_cache=True, return_dict=True, past_key_values=cache,
                      cache_position=all_positions[:n] if step == 0 else all_positions[n + step - 1:n + step],
                      logits_to_keep=0 if adapter.kind == 'llava' else 1)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.profiler.record_function('handoff::prefill' if step == 0 else 'handoff::decode') if profile else nullcontext():
                out = adapter.model(**kw)
            logits = out.logits[:, -1]
            token = logits.argmax(-1)
            gpu_ids.append(token)
            end.record()
            timings.append((begin, end))
            cache = out.past_key_values
            if capture:
                logits_capture.append(logits.detach().float().cpu())
                if audit:
                    valid.append(out.logits.detach().cpu().clone())
                assert torch.isfinite(logits).all()
            del out
            if not fixed:
                current = int(token)
                ids.append(current)
                if current in eos:
                    break
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    if fixed:
        ids = torch.cat(gpu_ids).tolist()
    times = [a.elapsed_time(b) for a, b in timings]
    vision_ms = sum(a.elapsed_time(b) for a, b in vision_events)
    info = control.cache.audit() if control else native_cache_info(cache)
    text = adapter.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
    kv_times = {}
    if profile and control:
        for phase, a, b in control.cache.management_events:
            kv_times[phase] = kv_times.get(phase, 0) + a.elapsed_time(b)
    result = dict(condition=condition, token_ids=ids, text=text, eos=ids[-1] in eos,
                  truncated=len(ids) == max_tokens and ids[-1] not in eos, empty=not bool(text),
                  prediction_positions=len(ids), decode_steps=len(ids) - 1,
                  content_tokens=sum(t not in eos for t in ids), latency_ms=elapsed,
                  ttft_ms=times[0], vision_ms=vision_ms, language_prefill_ms=times[0] - vision_ms,
                  decode_ms=sum(times[1:]), step_ms=times, gpu_model_ms=sum(times),
                  cpu_setup_and_gaps_ms=elapsed - sum(times),
                  peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
                  initial_allocated=initial_memory, peak_increment=torch.cuda.max_memory_allocated() - initial_memory,
                  kv=info, kv_management_ms=kv_times, fixed=fixed,
                  audit=[] if control is None else control.audit_rows)
    if capture:
        # Outside the measured region, while this request's KV is still live.
        snapshot = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                            '--format=csv,noheader,nounits'], text=True)
        result['process_memory_snapshot'] = [line for line in snapshot.splitlines()
                                             if f', {os.getpid()},' in line]
    hidden = [] if control is None else control.hidden
    rotary = [] if control is None else control.rotary
    del cache, control
    return result, logits_capture, valid, hidden, rotary

"""Selective visual KV persistence around the unchanged C3-V2 native blocks."""
from contextlib import nullcontext
from dataclasses import replace
import time
from unittest.mock import patch

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from .common import CFG, KEEP, CUTOFF
from ..ate_fulljoint_v2.engine import AnswerWriter, LAYERS
from ..accuracy_autoresearch.engine import ResearchWriter
from ..physical_handoff_fastpath import engine as base


class SelectiveKV(base.InplaceKV):
    def __init__(self, *args, keep=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.keep = frozenset(keep)

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        # Reuse the frozen direct-gather/in-place append implementation. Only
        # designated layers retain full original Prefill keys in final buffers.
        original = self.condition
        if self.prefill and layer_idx in self.keep:
            assert keys.shape[-2] == self.layout.length
            self.condition = 'selective-retained-image'
        try:
            return super().update(keys, values, layer_idx, cache_kwargs)
        finally:
            self.condition = original

    def audit(self):
        rows = []
        for layer, pair in enumerate(self.entries):
            if layer in LAYERS:
                assert pair is None and self.used[layer] == 0
                rows.append(dict(layer=layer, used_tokens=0, capacity_tokens=0,
                    live_bytes=0, storage_bytes=0, original_positions=[]))
                continue
            keys, values = pair
            positions = list(range(self.layout.length)) if layer in self.keep else list(self.layout.text)
            positions += list(range(self.layout.length, self.logical_length))
            assert len(positions) == self.used[layer]
            rows.append(dict(layer=layer, used_tokens=self.used[layer], capacity_tokens=keys.shape[-2],
                live_bytes=sum(t.shape[0]*t.shape[1]*self.used[layer]*t.shape[-1]*t.element_size() for t in pair),
                storage_bytes=sum(t.untyped_storage().nbytes() for t in pair),
                original_positions=positions))
        return dict(layers=rows, live_bytes=sum(r['live_bytes'] for r in rows),
            storage_bytes=sum(r['storage_bytes'] for r in rows), allocations=self.allocations,
            write_calls=self.write_calls, direct_gather_calls=self.gather_calls,
            historical_cache_copies=0, append_concatenations=0,
            retained_image_layers=sorted(self.keep))


class MiddleWriter(AnswerWriter):
    def __init__(self, adapter, layout, keep, max_tokens=32, audit=False, dense=False):
        self.keep, self.dense = frozenset(keep), dense
        assert self.keep.isdisjoint(LAYERS)
        assert all(i < layout.cutoff for i in self.keep)
        super().__init__(adapter, layout, 'C3-V2-ZeroRef' if dense else 'C3-V2',
                         list(layout.relay), max_tokens, audit)
        self.physical = not dense
        self.cache = (base.InplaceKV(len(adapter.layers), layout, 'D1 Dense', max_tokens, self.device)
                      if dense else SelectiveKV(len(adapter.layers), layout, 'Fast-P2', max_tokens,
                                                self.device, keep=keep))
        n = layout.length
        self.full_late = self.prefill_early.clone()
        relay = torch.tensor(layout.relay, device=self.device, dtype=torch.long)
        image = torch.tensor(layout.image, device=self.device, dtype=torch.long)
        self.full_late[0, 0, relay[:, None], image[None, :]] = -torch.inf
        self.middle = self.prefill_early.clone()
        self.middle[0, 0, n-1, image] = 0
        self.middle_late = self.full_late.clone()
        self.middle_late[0, 0, n-1, image] = 0
        self.visual_events = []

    def prefill_mask(self, layer):
        if self.physical and layer >= self.layout.cutoff:
            return self.prefill_late
        if layer < 20:
            return self.middle if layer in self.keep else self.prefill_early
        return self.middle_late if layer in self.keep else self.full_late

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None,
                cache_position=None, **kwargs):
        h = inputs_embeds if inputs_embeds is not None else self.decoder.embed_tokens(input_ids)
        prefill = self.step == 0
        positions = (self.positions[:self.layout.length] if prefill else
                     self.positions[self.layout.length+self.step-1:self.layout.length+self.step])
        if position_ids is None:
            position_ids = positions.unsqueeze(0)
        rotary = self.decoder.rotary_emb(h, position_ids)
        if self.audit_enabled:
            self.rotary.append(dict(position_ids=position_ids.detach().cpu().clone(),
                cos=rotary[0].detach().cpu().clone(), sin=rotary[1].detach().cpu().clone()))
        self.cache.prefill = prefill
        for i, layer in enumerate(self.a.layers):
            if prefill and self.physical and i == self.layout.cutoff:
                h = h.index_select(1, self.text_index)
                positions = self.text_positions
                position_ids = position_ids.index_select(-1, self.text_index)
                rotary = tuple(v.index_select(1, self.text_index) for v in rotary)
            if prefill:
                mask = self.prefill_mask(i)
                key_count = h.shape[1]
            else:
                key_count = self.cache.used[i] + 1
                mask = None if self.physical or i in self.keep else self.decode_mask[..., :key_count]
            self.current = (i, prefill, mask, positions, key_count)
            h = layer(h, attention_mask=mask, position_ids=position_ids,
                cache_position=positions, position_embeddings=rotary,
                past_key_value=self.cache, use_cache=True)[0]
            if self.audit_enabled and prefill:
                kept = self.text_index if h.shape[1] == self.layout.length else torch.arange(h.shape[1], device=h.device)
                self.hidden.append(h.index_select(1, kept).detach().float().cpu())
        h = self.decoder.norm(h)
        self.cache.logical_length = self.layout.length + self.step
        self.step += 1
        return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=self.cache)

    def record(self, out, weights, positions, shape, mask):
        super().record(out, weights, positions, shape, mask)
        if not self.collect:
            return
        i, prefill, _, original, _ = self.current
        if weights is None:
            assert i in LAYERS and self.physical_skip
            return
        if i in LAYERS and self.physical_skip:
            assert prefill and weights.shape[-2] == self.layout.length-1
            # The P-1 row was physically absent, not a masked probability row.
            return
        image_present = self.dense or (prefill and shape[1] == self.layout.length) or (not prefill and i in self.keep)
        mass = relay_mass = 0.
        if image_present:
            query = self.layout.length-1 if prefill else 0
            visual = weights[0, :, query, list(self.layout.image)]
            mass = float(visual.float().sum(-1).mean())
            if i not in self.keep:
                assert torch.count_nonzero(visual) == 0
            else:
                assert mass > 0
            if prefill:
                relay_visual = weights[0, :, list(self.layout.relay), :][..., list(self.layout.image)]
                relay_mass = float(relay_visual.float().sum(-1).mean())
                if i >= 20:
                    assert torch.count_nonzero(relay_visual) == 0
        self.visual_events.append(dict(layer=i, step=self.step, prefill=prefill,
            answer_image_mass=mass, relay_image_mass=relay_mass,
            image_keys_present=image_present, image_keys_permitted=i in self.keep,
            original_query_positions=original.detach().cpu().tolist()))


def prepare(adapter, sample):
    from ..reverse_writer.engine import prepare as frozen_prepare
    return frozen_prepare(adapter, sample, 'C3')


def kv_sources(info, layout, label, token_count, dense=False):
    for row in info['layers']:
        image_count = (len(layout.image) if row['used_tokens'] and
            (label == 'Native' or dense or row['layer'] in KEEP[label]) else 0)
        unit = row['live_bytes'] / row['used_tokens'] if row['used_tokens'] else 0
        row.update(image_tokens=image_count, text_tokens=row['used_tokens']-image_count,
            image_live_bytes=int(unit*image_count), text_live_bytes=int(unit*(row['used_tokens']-image_count)),
            image_capacity_bytes=int(unit*image_count), text_capacity_bytes=int(row['storage_bytes']-unit*image_count))
    for name in ('image_live_bytes', 'text_live_bytes', 'image_capacity_bytes', 'text_capacity_bytes'):
        info[name] = sum(r[name] for r in info['layers'])
    info['retained_image_layers'] = [r['layer'] for r in info['layers'] if r['image_tokens']]
    if not dense:
        assert info['retained_image_layers'] == list(KEEP[label])
    assert info['live_bytes'] == info['image_live_bytes'] + info['text_live_bytes']
    assert info['storage_bytes'] == info['image_capacity_bytes'] + info['text_capacity_bytes']
    return info


@torch.inference_mode()
def generate(adapter, layout, label, max_tokens=32, fixed=False, capture=False,
             audit=False, dense=False, noop=False):
    """Same cold-request boundaries for all four conditions, including Vision.

    This is the frozen greedy loop with H1 for every model call. Wall TTFT is
    captured after the first token becomes available; GPU Prefill is separate.
    Fixed-length control deliberately ignores EOS and never supplies gold tokens.
    """
    assert label in KEEP and not (dense and label == 'Native')
    layout = replace(layout, cutoff=CUTOFF[label])
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    control = None
    if noop:
        control = base.FastHandoff(adapter, layout, 'Native-Noop', max_tokens, audit)
    elif label == 'C3-V2' and not dense:
        control = ResearchWriter(adapter, layout, CFG, list(layout.relay), max_tokens, audit)
    elif label != 'Native':
        control = MiddleWriter(adapter, layout, KEEP[label], max_tokens, audit, dense)
    n, device = layout.length, adapter.inputs['input_ids'].device
    positions = torch.arange(n+max_tokens-1, device=device)
    attention = torch.ones((1, n+max_tokens-1), dtype=torch.long, device=device)
    eos = adapter.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    ids, gpu_tokens, logits, valid, times, vision_times = [], [], [], [], [], []
    cache, ttft_wall = None, None
    original_vision = adapter.model.get_image_features

    def vision(*args, **kwargs):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original_vision(*args, **kwargs)
        end.record()
        vision_times.append((begin, end))
        return result

    try:
        with (control.context() if control else nullcontext()), patch.object(adapter.model, 'get_image_features', vision):
            for step in range(max_tokens):
                kw = dict(adapter.inputs) if step == 0 else dict(
                    input_ids=gpu_tokens[-1].reshape(1, 1), attention_mask=attention[:, :n+step])
                kw.update(use_cache=True, return_dict=True, past_key_values=cache,
                    cache_position=positions[:n] if step == 0 else positions[n+step-1:n+step],
                    logits_to_keep=1)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                output = adapter.model(**kw)
                last = output.logits[:, -1]
                token = last.argmax(-1)
                gpu_tokens.append(token)
                end.record()
                times.append((begin, end))
                cache = output.past_key_values
                current = int(token)
                ids.append(current)
                if step == 0:
                    ttft_wall = (time.perf_counter()-start)*1000
                if capture:
                    assert output.logits.shape[1] == 1 and torch.isfinite(output.logits).all()
                    if capture == 'all' or step == 0:
                        logits.append(last.detach().float().cpu().clone())
                    if capture == 'all':
                        valid.append(output.logits.detach().cpu().clone())
                del output
                if not fixed and current in eos:
                    break
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)*1000
        durations = [a.elapsed_time(b) for a, b in times]
        vision_ms = sum(a.elapsed_time(b) for a, b in vision_times)
        assert len(vision_times) == 1, 'Vision must be recomputed exactly once per request'
        info = control.cache.audit() if control else base.native_cache_info(cache)
        info = kv_sources(info, layout, label, len(ids), dense or noop)
        text = adapter.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        result = dict(condition=label, text=text, token_ids=ids, eos=ids[-1] in eos,
            truncated=not fixed and len(ids)==max_tokens and ids[-1] not in eos,
            empty=not bool(text), repeated=any(ids[i:i+3]==ids[j:j+3] for i in range(len(ids)-2) for j in range(i+3, len(ids)-2)),
            content_tokens=sum(t not in eos for t in ids), decode_steps=len(ids)-1,
            prefill_ms=durations[0], language_prefill_ms=durations[0]-vision_ms,
            vision_ms=vision_ms, ttft_ms=ttft_wall, gpu_ttft_ms=durations[0],
            e2e_ms=elapsed, decode_ms=sum(durations[1:]), step_ms=durations,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            initial_allocated=initial, kv=info, fixed=fixed, vision_calls=1,
            audit=[] if control is None else control.audit_rows,
            visual_audit=getattr(control, 'visual_events', []))
        hidden = [] if control is None else control.hidden
        rotary = [] if control is None else control.rotary
        return result, logits, valid, hidden, rotary
    finally:
        del cache, control

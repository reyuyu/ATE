"""Parameterized physical MiddleWriter reuse; historical engines remain read-only."""
from contextlib import nullcontext
from dataclasses import replace
import torch

from .common import MIDDLE16
from ..c3v2_middle_reaccess.engine import MiddleWriter
from ..physical_handoff_fastpath.engine import FastHandoff
from ..information_horizon_alignment.engine import result
from ..information_horizon_alignment.common import hook_counts, tensor_sha


def summarize_cache(cache, layout, algorithm, dense):
    info = cache.audit()
    expected = set(range(32)) if dense else set(algorithm.reaccess)
    actual = []
    for row in info['layers']:
        positions = row.pop('original_positions', [])
        present = set(layout.image).issubset(positions)
        if present:
            actual.append(row['layer'])
        assert present == (row['layer'] in expected)
        row['original_positions_sha256'] = __import__('hashlib').sha256(str(positions).encode()).hexdigest()
        row['image_tokens'] = len(layout.image) if present else 0
    assert actual == sorted(expected)
    assert info['historical_cache_copies'] == info['append_concatenations'] == 0
    return info


@torch.inference_mode()
def generate(adapter, layout, algorithm=MIDDLE16, cap=32, *, native=False,
             noop=False, dense=False, audit=False, capture=False, fixed=False):
    assert not (native and dense)
    layout = replace(layout, cutoff=32 if native else algorithm.exit_before)
    control = (FastHandoff(adapter, layout, 'Native-Noop', cap, audit) if noop else
               None if native else MiddleWriter(adapter, layout, algorithm.reaccess, cap, audit, dense))
    before = hook_counts(adapter.model)
    device = adapter.inputs['input_ids'].device
    n = layout.length
    positions = torch.arange(n + cap - 1, device=device)
    mask = torch.ones((1, n + cap - 1), dtype=torch.long, device=device)
    cache = token = None
    ids, logits = [], []
    try:
        with control.context() if control else nullcontext():
            for t in range(cap):
                kw = dict(adapter.inputs) if t == 0 else dict(
                    input_ids=token.reshape(1, 1), attention_mask=mask[:, :n+t])
                kw.update(use_cache=True, return_dict=True, past_key_values=cache,
                          cache_position=positions[:n] if t == 0 else positions[n+t-1:n+t],
                          logits_to_keep=1)
                output = adapter.model(**kw)
                assert output.logits.shape[:2] == (1, 1) and torch.isfinite(output.logits).all()
                v = output.logits[:, -1]
                if t == 0:
                    first = tensor_sha(v)
                if capture:
                    logits.append(v.detach().float().cpu().clone())
                token = v.argmax(-1)
                ids.append(int(token))
                cache = output.past_key_values
                del output
                eos = adapter.model.generation_config.eos_token_id
                if not fixed and ids[-1] in ([eos] if isinstance(eos, int) else eos):
                    break
        record = result(adapter, ids, cap)
        record.update(condition='Native' if native else algorithm.name,
                      first_logits_sha256=first, logits_to_keep=1,
                      frozen_reaccess=list(algorithm.reaccess), physical_exit_before=layout.cutoff)
        if control and not native:
            record['kv_audit'] = summarize_cache(control.cache, layout, algorithm, dense)
            if audit:
                record.update(attention_audit=control.audit_rows,
                              visual_audit=control.visual_events,
                              _hidden=control.hidden, _rotary=control.rotary)
        return record, logits
    finally:
        del control, cache
        assert hook_counts(adapter.model) == before


def prepare_input(adapter, sample, original_tokenizer):
    from ..information_horizon_alignment.engine import image_ids, paper_pixels
    from ..physical_handoff_fastpath.engine import Layout
    tok = original_tokenizer if sample['benchmark'] == 'GQA' else adapter.processor.tokenizer
    raw = image_ids(sample['prompt'], tok, -200)
    assert raw[0] == 1 and raw.count(-200) == 1
    ids = [v for t in raw for v in ([32000] * 576 if t == -200 else [t])]
    visual = tuple(i for i, t in enumerate(ids) if t == 32000)
    text = tuple(i for i, t in enumerate(ids) if t != 32000)
    relay = tuple(range(visual[-1] + 1, len(ids) - 1))
    assert len(visual) == 576 and relay and len(ids) - 1 not in relay
    pixels = paper_pixels(sample['image_path'], adapter.image_processor).half()
    cpu_ids = torch.tensor([ids], dtype=torch.long)
    cpu_mask = torch.ones_like(cpu_ids)
    binding = dict(prompt=sample['prompt'], input_ids=ids, raw_image_placeholder_ids=raw,
                   pixel_values_sha256=tensor_sha(pixels), expanded_length=len(ids),
                   Pminus1=len(ids)-1, R=list(relay), image_positions=list(visual),
                   input_hashes=dict(input_ids=tensor_sha(cpu_ids), attention_mask=tensor_sha(cpu_mask),
                                     pixel_values=tensor_sha(pixels)),
                   tokenizer_path=str(original_tokenizer.name_or_path if sample['benchmark']=='GQA'
                                      else adapter.processor.tokenizer.name_or_path))
    adapter.inputs = dict(input_ids=cpu_ids.cuda(), attention_mask=cpu_mask.cuda(), pixel_values=pixels.cuda())
    return Layout(len(ids), visual, text, relay, relay, 20), binding

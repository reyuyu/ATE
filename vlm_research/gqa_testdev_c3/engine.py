"""Reuse the audited physical selective-KV writer; change only its fixed keep set."""
from contextlib import nullcontext
from dataclasses import replace

import torch

from .common import CUTOFF, KEEP, LABELS, digest, expand
from ..c3v2_middle_reaccess.engine import MiddleWriter
from ..p1_layerwise_accuracy.engine import tensor_hash
from ..physical_handoff_fastpath.engine import FastHandoff, Layout


def prepare(adapter, sample):
    ids = expand(sample)
    assert digest(ids) == sample['expanded_prompt_ids_sha256']
    adapter.prepare(dict(sample, input_mapping=dict(prompt_ids=ids)), {})
    image = tuple(i for i, value in enumerate(ids) if value == 32000)
    text = tuple(i for i, value in enumerate(ids) if value != 32000)
    relay = tuple(range(image[-1]+1, len(ids)-1))
    question = tuple(sample['question_positions'])
    assert len(image) == 576 and question and set(question) <= set(relay)
    layout = Layout(len(ids), image, text, relay, question, 20)
    return layout, dict(input_hashes=adapter.input_hashes, length=len(ids),
        prompt_ids_sha256=digest(ids), image_positions_sha256=digest(image),
        relay_positions_sha256=digest(relay), Pminus1=len(ids)-1)


@torch.inference_mode()
def generate(adapter, layout, label, max_tokens=32, capture=False, audit=False,
             dense=False, noop=False, fixed=False):
    assert label in LABELS or label == 'Native'
    if label != 'Native':
        layout = replace(layout, cutoff=CUTOFF[label])
        ctl = MiddleWriter(adapter, layout, KEEP[label], max_tokens, audit, dense)
    else:
        ctl = FastHandoff(adapter, layout, 'Native-Noop', max_tokens, audit) if noop else None
    n, device = layout.length, adapter.inputs['input_ids'].device
    positions = torch.arange(n+max_tokens-1, device=device)
    mask = torch.ones((1, n+max_tokens-1), dtype=torch.long, device=device)
    eos = adapter.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    ids, logits, valid, token, cache = [], [], [], None, None
    first_sha = None
    try:
        with ctl.context() if ctl is not None else nullcontext():
            for step in range(max_tokens):
                kw = dict(adapter.inputs) if step == 0 else dict(
                    input_ids=token.reshape(1, 1), attention_mask=mask[:, :n+step])
                kw.update(use_cache=True, return_dict=True, past_key_values=cache,
                    cache_position=positions[:n] if step == 0 else positions[n+step-1:n+step],
                    logits_to_keep=1)
                output = adapter.model(**kw)
                assert output.logits.shape[:2] == (1, 1)
                last = output.logits[:, -1]
                assert torch.isfinite(last).all(), 'Nonfinite effective logits'
                token = last.argmax(-1)
                ids.append(int(token))
                cache = output.past_key_values
                if step == 0:
                    first_sha = tensor_hash(last)
                if capture:
                    logits.append(last.detach().float().cpu().clone())
                    valid.append(output.logits.detach().cpu().clone())
                del output
                if not fixed and ids[-1] in eos:
                    break
        text = adapter.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        result = dict(condition=label, text=text, token_ids=ids, eos=ids[-1] in eos,
            truncated=len(ids) == max_tokens and ids[-1] not in eos, empty=not bool(text),
            repeated=any(ids[i:i+3] == ids[j:j+3] for i in range(len(ids)-2)
                for j in range(i+3, len(ids)-2)), first_logits_sha256=first_sha,
            first_token_id=ids[0], all_effective_logits_finite=True,
            independent_cache=True, logits_to_keep=1)
        if audit and ctl is not None:
            result['attention_audit'] = ctl.audit_rows
            result['visual_audit'] = getattr(ctl, 'visual_events', [])
            result['kv_audit'] = ctl.cache.audit()
        return result, logits, valid, ([] if ctl is None else ctl.hidden), ([] if ctl is None else ctl.rotary)
    finally:
        del ctl, cache

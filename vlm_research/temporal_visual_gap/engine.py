"""Reuse audited masks and cache copies; never carry a counterfactual cache forward."""
from contextlib import nullcontext

import numpy as np
import torch

from benchmarks.dataset_registry import digest
from knockout_probe import model_layers
from paper_rq1.engine import clone_cache, cache_fingerprint, step_inputs
from rq2_decode_visual.engine import DecodeImageKnockout, DecodeProbabilities
from rq2_source_knockout.engine import Observer
from rq2_visual_sufficiency.engine import MultiImageKnockout
from .common import gate


@torch.inference_mode()
def forward(model, inputs, tokens, cache, blocked=False, noop=False, audit=False):
    t, n = len(tokens), inputs['input_ids'].shape[1]
    visual = int((inputs['input_ids'] == model.config.image_token_index).sum())
    mask = (DecodeImageKnockout(model, inputs, list(range(32)) if blocked else [], t, noop, visual)
            if t else MultiImageKnockout(model, inputs, 1, list(range(32)) if blocked else [], noop, visual))
    obs = DecodeProbabilities(model, mask) if t else Observer(model, inputs, 1, audit=audit, expected_visual=visual)
    with mask, obs:
        result = model(**step_inputs(inputs, tokens), past_key_values=cache,
                       use_cache=True, output_attentions=True, return_dict=True)
    mask.assert_one_forward()
    assert not mask.handles and not obs.handles
    active = blocked and not noop
    assert [r['layer'] for r in mask.records if r['applied']] == (list(range(32)) if active else [])
    assert [r['layer'] for r in obs.rows] == list(range(32))
    rows = []
    for r in obs.rows:
        maximum = r['image_probability_max'] if t else r['Image_max']
        assert maximum == 0. if active else maximum > 0.
        rows.append(dict(layer=r['layer'], blocked=bool(active), image_probability_max=maximum,
                         image_mass=r['image_mass'] if t else r['Image'],
                         row_error=r['row_sum_max_error'] if t else r['row_error']))
    assert all(result.past_key_values.get_seq_length(i) == n+t for i in range(32))
    logits = result.logits[0, -1].detach().float().cpu().clone()
    assert torch.isfinite(logits).all()
    debug = dict(full_logits=result.logits.detach().float().cpu().clone(),
                 states=obs.states, rows=obs.rows) if audit and not t else {}
    return logits, result.past_key_values, rows, debug


def generation(tokenizer, tokens, limit):
    return dict(token_ids=tokens, text=tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                eos=tokens[-1] == 2, truncated=len(tokens) == limit and tokens[-1] != 2,
                decode_steps=len(tokens)-1, empty=not tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                own_cache=True, own_prefix=True)


def compare(full, off):
    p, q = full.double().log_softmax(-1), off.double().log_softmax(-1)
    token, changed = int(full.argmax()), int(off.argmax())
    mid = torch.logaddexp(p, q)-np.log(2.)
    return dict(full_token=token, off_token=changed, top1_flip=token != changed,
                full_logp=float(p[token]), off_logp=float(q[token]),
                signed_logp_loss=float(p[token]-q[token]), absolute_logp_change=float(abs(p[token]-q[token])),
                full_entropy=float(-(p.exp()*p).sum()),
                js_divergence=max(0., float(.5*((p.exp()*(p-mid)).sum()+(q.exp()*(q-mid)).sum()))))


@torch.inference_mode()
def native_counterfactual(model, tokenizer, inputs, limit, audit=False, noop=False):
    tokens, native, full_logits, off_logits, stats, audits = [], None, [], [], [], []
    while len(tokens) < limit and (not tokens or tokens[-1] != 2):
        t = len(tokens)
        # Both branches own cloned tensors. The native pre-step cache is never passed to a forward.
        fingerprint = cache_fingerprint(native) if audit and native is not None else None
        off, off_cache, off_rows, od = forward(model, inputs, tokens, clone_cache(native),
                                              blocked=True, noop=noop, audit=audit)
        del off_cache
        if fingerprint is not None:
            assert cache_fingerprint(native) == fingerprint
        full, next_native, full_rows, fd = forward(model, inputs, tokens, clone_cache(native), audit=audit)
        if fingerprint is not None:
            assert cache_fingerprint(native) == fingerprint
        if audit and not t:
            assert torch.equal(fd['full_logits'][:, :-1], od['full_logits'][:, :-1])
            assert fd['states'] == od['states']
            assert [r['other_query_attention_sha256'] for r in fd['rows']] == [r['other_query_attention_sha256'] for r in od['rows']]
        row = dict(t=t, consumed_token=tokens[-1] if t else None, is_eos=int(full.argmax()) == 2,
                   absolute_query=inputs['input_ids'].shape[1]+t-1,
                   cache_before=0 if not t else inputs['input_ids'].shape[1]+t-1,
                   cache_after=inputs['input_ids'].shape[1]+t, **compare(full, off))
        stats.append(row)
        audits.append(dict(t=t, Full=full_rows, ImageOFF=off_rows))
        full_logits.append(full.numpy()); off_logits.append(off.numpy())
        tokens.append(int(full.argmax())); native = next_native
    return generation(tokenizer, tokens, limit), stats, np.stack(full_logits), np.stack(off_logits), audits


@torch.inference_mode()
def temporal_generate(model, tokenizer, inputs, image, condition, limit=96, audit=False, noop=False):
    tokens, cache, gates, trace, debug = [], None, [], [], {}
    while len(tokens) < limit and (not tokens or tokens[-1] != 2):
        t = len(tokens)
        on = not t or gate(condition, image, t)
        logits, cache, rows, details = forward(model, inputs, tokens, cache,
                                               blocked=not on, noop=noop, audit=audit)
        if not t:
            first = logits
            debug = details
        else:
            gates.append(dict(t=t, on=on, consumed_token=tokens[-1], predicted_token=int(logits.argmax()),
                absolute_query=inputs['input_ids'].shape[1]+t-1, probabilities=rows))
        trace.append(logits)
        tokens.append(int(logits.argmax()))
    result = generation(tokenizer, tokens, limit)
    result.update(visual_open_steps=sum(g['on'] for g in gates),
                  visual_open_fraction=sum(g['on'] for g in gates)/len(gates) if gates else None)
    return result, first, gates, trace, debug

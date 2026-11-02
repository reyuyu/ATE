"""Compose historical hooks at distinct modules; maintain each condition's own cache."""
from contextlib import ExitStack

import torch

from knockout_probe import model_layers, tensor_hash
from paper_rq1.engine import step_inputs, cache_fingerprint
from paper_rq1_early_joint_attention.engine import JointAttention
from rq2_decode_visual.engine import DecodeImageKnockout, DecodeProbabilities
from rq2_source_knockout.engine import Observer
from rq2_visual_sufficiency.engine import MultiImageKnockout
from .common import CONDITIONS, LAYERS


@torch.inference_mode()
def generate(model, prompt, condition, bos_id=1, limit=32, noop=False,
             audit=False, collect=False, expected_visual=576):
    if condition not in CONDITIONS or not 1 <= limit <= 32:
        raise ValueError('Only the four frozen conditions and max32 are permitted')
    if len(model_layers(model)) != 32 or not bool(prompt['attention_mask'].all()):
        raise ValueError('Require original unpadded prompt and32 decoder layers')
    skip, off = CONDITIONS[condition]
    n = prompt['input_ids'].shape[1]
    tokens, history, events, logits_trace, debug = [], None, [], [], {}
    blocked = list(range(32)) if off else []
    with ExitStack() as outer:
        # Combined mode in the historical hook means ALL Prefill queries. Compose
        # P-1-only and Decode-only instead; their active temporal scopes are disjoint.
        early = outer.enter_context(JointAttention(model, LAYERS if skip else [],
            'P-1-only', n, noop=noop, audit=True))
        decode = outer.enter_context(JointAttention(model, LAYERS if skip else [],
            'Decode-only', n, noop=noop, audit=True))
        while len(tokens) < limit and (not tokens or tokens[-1] != 2):
            t = len(tokens)
            if history is not None:
                assert all(history.get_seq_length(k) == n + t - 1 for k in range(32))
            mask = (DecodeImageKnockout(model, prompt, blocked, t, noop, expected_visual)
                    if t else MultiImageKnockout(model, prompt, bos_id, blocked, noop, expected_visual))
            obs = (DecodeProbabilities(model, mask) if t else
                   Observer(model, prompt, bos_id, audit=audit, expected_visual=expected_visual))
            with mask, obs:
                output = model(**step_inputs(prompt, tokens), past_key_values=history,
                               use_cache=True, output_attentions=True, return_dict=True)
            mask.assert_one_forward()
            assert not mask.handles and not obs.handles
            active = blocked if not noop else []
            assert [r['layer'] for r in mask.records if r['applied']] == active
            assert [r['layer'] for r in obs.rows] == list(range(32))
            probabilities = []
            for row in obs.rows:
                maximum = row['image_probability_max'] if t else row['Image_max']
                if off and not noop:
                    assert maximum == 0., 'Actual post-softmax image probability is not zero'
                probabilities.append(dict(layer=row['layer'], image_probability_max=maximum,
                    image_mass=row['image_mass'] if t else row['Image'],
                    row_sum_max_error=row['row_sum_max_error'] if t else row['row_error']))
            assert torch.isfinite(output.logits).all(), 'Nonfinite effective logits'
            history = output.past_key_values
            assert all(history.get_seq_length(k) == n + t for k in range(32))
            logits = output.logits[0, -1].detach().float().cpu().clone()
            if collect:
                logits_trace.append(logits)
            if not t:
                first = logits
                if audit:
                    debug = dict(prefill_logits=output.logits.detach().float().cpu().clone(),
                        other_query_states=obs.states,
                        other_query_attention=[r['other_query_attention_sha256'] for r in obs.rows],
                        initial_cache_sha256=cache_fingerprint(history))
            events.append(dict(t=t, absolute_query=n+t-1, local_query=0 if t else n-1,
                consumed_token=tokens[-1] if t else None, predicted_token=int(logits.argmax()),
                cache_before=n+t-1 if t else 0, cache_after=n+t,
                masks=mask.records, probabilities=probabilities))
            tokens.append(int(logits.argmax()))
    expected = 4 if skip else 0
    assert early.edits == expected and decode.edits == expected * (len(tokens)-1)
    assert not early.handles and not decode.handles
    edits = [r for r in early.events + decode.events if r['active']]
    assert all(r['layer'] in LAYERS for r in edits)
    assert all(r['target'] == 'original_P-1' if not r['decode'] else r['queries'] == 1 for r in edits)
    return dict(token_ids=tokens, length=len(tokens), eos=tokens[-1] == 2,
        truncated=len(tokens) == limit and tokens[-1] != 2, decode_steps=len(tokens)-1,
        original_p_minus_one=n-1, image_keys=mask.layout['visual_indices'].tolist(),
        own_prefill=True, own_prefix=True, own_cache=True, initial_cache_empty=True,
        image_off=off and not noop, attention_skip=skip and not noop,
        attention_output_edits=0 if noop else early.edits+decode.edits,
        noop=noop, steps=events, attention_output_events=edits), first, logits_trace, debug

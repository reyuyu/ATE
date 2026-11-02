"""Only o_proj output rows change; no attention masks, weights, or cache edits."""
from contextlib import AbstractContextManager

import torch

from benchmarks.dataset_registry import digest
from knockout_probe import layout_from_inputs, model_layers
from paper_rq1.engine import step_inputs
from paper_rq1_decode_layerwise_quality.engine import is_decode
from paper_rq1.scorers import gqa_prediction
from .common import LAYERS, CONDITIONS


def roles(inputs, image_token_id=32000):
    layout = layout_from_inputs(inputs, image_token_id, expected_visual=576)
    if not bool(layout['valid'].all()):
        raise ValueError('Require an unpadded single-sample prompt')
    n = layout['sequence_length']
    image = layout['visual_indices'].tolist()
    p1 = n-1
    if image[-1] >= p1-1:
        raise ValueError('Missing relay or answer query')
    relay = list(range(image[-1]+1, p1))
    other = list(range(image[0]))
    if sorted(image+relay+[p1]+other) != list(range(n)):
        raise AssertionError('Roles must be disjoint and cover the prompt')
    return dict(prompt_length=n, I=image, R=relay, A=[p1], other=other,
                counts=dict(I=len(image), R=len(relay), A=1, other=len(other)))


def target_positions(role, layout, decode):
    if role not in CONDITIONS:
        raise ValueError('Unknown frozen condition')
    if role == 'Full':
        return []
    if decode:
        return [0] if role == 'A-Bypass' else []
    return layout[role[0]]


class RoleBypass(AbstractContextManager):
    def __init__(self, model, layout, condition, noop=False):
        if condition not in CONDITIONS:
            raise ValueError('Unfrozen intervention')
        self.model, self.layout, self.condition, self.noop = model, layout, condition, noop
        self.handles, self.phases, self.events = [], {}, []

    def __enter__(self):
        try:
            blocks = model_layers(self.model)
            if len(blocks) != 32:
                raise ValueError('Require frozen 32-layer language decoder')
            for k in LAYERS:
                def before(module, args, kwargs, k=k):
                    hidden = args[0] if args else kwargs['hidden_states']
                    cache = kwargs.get('past_key_value')
                    past = cache.get_seq_length(k) if cache is not None else 0
                    cp = kwargs.get('cache_position')
                    decode = is_decode(cp, self.layout['prompt_length'], past, hidden.shape[1])
                    expected = torch.arange(past, past+hidden.shape[1], device=cp.device)
                    if not torch.equal(cp.reshape(-1), expected):
                        raise AssertionError('Non-contiguous true absolute query positions')
                    self.phases[k] = (decode, past, hidden.shape[1])

                def output(module, args, out, k=k):
                    decode, absolute_start, length = self.phases[k]
                    if out.shape[:2] != (1, length) or not torch.isfinite(out).all():
                        raise AssertionError('Invalid native attention output')
                    target = target_positions(self.condition, self.layout, decode)
                    changed = out
                    if target and not self.noop:
                        changed = out.clone()
                        changed[:, target] = 0
                        keep = [i for i in range(length) if i not in set(target)]
                        if torch.count_nonzero(changed[:, target]):
                            raise AssertionError('Target output is not exactly zero')
                        if not torch.equal(changed[:, keep], out[:, keep]):
                            raise AssertionError('Directly changed a non-target output row')
                    elif not torch.equal(changed, out):
                        raise AssertionError('Inactive/no-op changed output')
                    self.events.append(dict(layer=k, phase='decode' if decode else 'prefill',
                        absolute_start=absolute_start, query_length=length,
                        target_count=len(target), local_positions_sha256=digest(target),
                        absolute_first=absolute_start+target[0] if target else None,
                        absolute_last=absolute_start+target[-1] if target else None,
                        applied=bool(target and not self.noop), target_exact_zero=bool(target and not self.noop),
                        non_target_rows_exact=True, noop=self.noop))
                    return changed

                self.handles.append(blocks[k].self_attn.register_forward_pre_hook(before, with_kwargs=True))
                self.handles.append(blocks[k].self_attn.o_proj.register_forward_hook(output))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


@torch.inference_mode()
def generate(model, tokenizer, prompt, condition, noop=False, capture=False, retain_cache=False,
             logits_to_keep=1):
    if type(logits_to_keep) is not int or logits_to_keep not in (0, 1):
        raise ValueError('Use 1 for native generation parity or 0 for all-position diagnostics')
    layout = roles(prompt, model.config.image_token_index)
    tokens, history, logits_trace, cache_lengths = [], None, [], []
    n = layout['prompt_length']
    with RoleBypass(model, layout, condition, noop) as controller:
        while len(tokens) < 32 and (not tokens or tokens[-1] != 2):
            t = len(tokens)
            expected_before = n+t-1 if t else 0
            if history is not None and any(history.get_seq_length(k) != expected_before for k in range(32)):
                raise AssertionError('Condition cache has wrong prefix length')
            output = model(**step_inputs(prompt, tokens), past_key_values=history,
                           use_cache=True, output_attentions=False, return_dict=True,
                           logits_to_keep=logits_to_keep)
            history = output.past_key_values
            if any(history.get_seq_length(k) != n+t for k in range(32)):
                raise AssertionError('Cache lengths differ from real absolute input positions')
            if not torch.isfinite(output.logits).all():
                raise AssertionError('Nonfinite logits')
            current = output.logits[0, -1].detach().float().cpu().clone()
            if not tokens:
                first = current
            if capture:
                logits_trace.append(output.logits[0].detach().float().cpu().clone())
            cache_lengths.append(dict(t=t, before=expected_before, after=n+t,
                                      consumed_token=tokens[-1] if tokens else None,
                                      absolute_query=n+t-1))
            tokens.append(int(current.argmax()))
    assert not controller.handles
    if len(controller.events) != len(LAYERS)*len(tokens):
        raise AssertionError('Not all target layers/steps observed')
    for t in range(len(tokens)):
        events = controller.events[t*4:(t+1)*4]
        assert [e['layer'] for e in events] == LAYERS
        count = len(target_positions(condition, layout, bool(t)))
        assert all(e['target_count'] == count for e in events)
    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    result = dict(token_ids=tokens, text=text, prediction=gqa_prediction(text), length=len(tokens),
        eos=tokens[-1] == 2, truncated=len(tokens) == 32 and tokens[-1] != 2, empty=not bool(text),
        decode_steps=len(tokens)-1, own_cache=True, own_prefix=True, cache_lengths=cache_lengths,
        query_layout=layout, hook_events=controller.events,
        intervened_query_layer_updates=sum(e['target_count'] for e in controller.events if e['applied']))
    return result, first, logits_trace, history if retain_cache else None

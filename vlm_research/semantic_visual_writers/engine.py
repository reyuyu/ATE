"""Pre-softmax hard masks only; no attention-output or cache edits."""
from contextlib import AbstractContextManager

import torch
from knockout_probe import model_layers, ROW_ATOL
from query_role_attention.engine import roles
from paper_rq1.engine import step_inputs
from paper_rq1.scorers import gqa_prediction
from query_role_formal.common import repeated_trigram


def blocked_edges(layout, writers, past, device):
    n = layout['prompt_length']
    assert (past == 0 or past >= n) and set(writers) <= set(layout['R'])
    blocked = torch.zeros((1 if past else n, past+1 if past else n), dtype=torch.bool, device=device)
    queries = [0] if past else [q for q in layout['R'] if q not in set(writers)] + [n-1]
    q = torch.tensor(queries, device=device)
    image = torch.tensor(layout['I'], device=device)
    blocked[q[:, None], image[None, :]] = True
    return blocked


def apply_mask(old, layout, writers, past, noop=False):
    blocked = blocked_edges(layout, writers, past, old.device)
    assert old.shape == (1, 1, *blocked.shape) and old.is_floating_point()
    assert bool(((old == 0) | (old == torch.finfo(old.dtype).min) | torch.isneginf(old)).all())
    legal = old[0, 0] == 0
    assert not (blocked & ~legal).any()
    if not past:
        assert not legal.triu(1).any()
        other = [q for q in range(layout['prompt_length']) if q not in layout['R'] and q != layout['prompt_length']-1]
        assert not blocked[other].any()
    else:
        assert legal.all()
    changed = old
    if not noop:
        changed = old.clone()
        changed[0, 0].masked_fill_(blocked, -torch.inf)
        assert torch.equal(changed[0, 0][~blocked], old[0, 0][~blocked])
    assert (changed[0, 0] == 0).any(-1).all()
    return changed, blocked


class WriterMask(AbstractContextManager):
    def __init__(self, model, inputs, writers, noop=False):
        self.model, self.writers, self.noop = model, list(writers), noop
        self.layout = roles(inputs, model.config.image_token_index)
        self.bos = torch.where(inputs['input_ids'][0] == model.config.text_config.bos_token_id)[0].tolist()
        assert len(self.bos) == 1 and self.bos[0] < self.layout['I'][0]
        self.handles, self.state, self.events = [], {}, []

    def __enter__(self):
        try:
            for k, layer in enumerate(model_layers(self.model)):
                def before(module, args, kw, k=k):
                    hidden, cp, cache = kw['hidden_states'], kw['cache_position'], kw.get('past_key_value')
                    past = cache.get_seq_length(k) if cache is not None else 0
                    n = self.layout['prompt_length']
                    assert hidden.shape[:2] == (1, 1 if past else n)
                    assert torch.equal(cp, torch.arange(past, past+hidden.shape[1], device=hidden.device))
                    changed, blocked = apply_mask(kw['attention_mask'], self.layout, self.writers, past, self.noop)
                    assert (changed[0, 0, :, self.bos[0]] == 0).all()
                    assert (changed[0, 0, 0, past] == 0) if past else (changed[0, 0].diagonal() == 0).all()
                    self.state[k] = (past, blocked)
                    return args, dict(kw, attention_mask=changed, output_attentions=True)

                def after(module, args, output, k=k):
                    past, blocked = self.state[k]
                    weights = output[1]
                    assert weights is not None and weights.shape[2:] == blocked.shape
                    idx = [0] if past else self.layout['R'] + [self.layout['prompt_length']-1]
                    rows = weights[0, :, idx, :].float()
                    assert torch.isfinite(rows).all() and (rows >= 0).all()
                    error = float((rows.sum(-1)-1).abs().max())
                    assert error <= ROW_ATOL
                    maximum = float(weights[0, :, blocked].max())
                    if not self.noop:
                        assert maximum == 0, 'A prohibited softmax probability is nonzero'
                    self.events.append(dict(layer=k, phase='decode' if past else 'prefill', absolute_start=past,
                        query_length=weights.shape[2], key_length=weights.shape[3], answer_image_blocked=not self.noop,
                        blocked_connections_per_head=int(blocked.sum()), writer_queries=len(self.writers),
                        blocked_probability_max=maximum, row_sum_max_error=error,
                        non_target_mask_unchanged=True, BOS_self_legal=True))
                    return (output[0], None) + tuple(output[2:])

                self.handles.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
                self.handles.append(layer.self_attn.register_forward_hook(after))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles = []


@torch.inference_mode()
def generate(model, tokenizer, prompt, writers, noop=False, capture=False, retain_cache=False):
    n = prompt['input_ids'].shape[1]
    tokens, trace, lengths, cache = [], [], [], None
    with WriterMask(model, prompt, writers, noop) as ctl:
        while len(tokens) < 32 and (not tokens or tokens[-1] != 2):
            t = len(tokens)
            before = n+t-1 if t else 0
            if cache is not None:
                assert all(cache.get_seq_length(k) == before for k in range(32))
            output = model(**step_inputs(prompt, tokens), past_key_values=cache, use_cache=True,
                output_attentions=False, return_dict=True, logits_to_keep=1)
            cache = output.past_key_values
            assert all(cache.get_seq_length(k) == n+t for k in range(32))
            assert output.logits.shape[:2] == (1, 1) and torch.isfinite(output.logits).all()
            current = output.logits[0, -1].detach().float().cpu().clone()
            if not tokens:
                first = current
            if capture:
                trace.append(current)
            lengths.append(dict(t=t, before=before, after=n+t,
                consumed_token=tokens[-1] if tokens else None, absolute_query=n+t-1))
            tokens.append(int(current.argmax()))
    assert not ctl.handles and len(ctl.events) == 32*len(tokens)
    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    result = dict(token_ids=tokens, text=text, prediction=gqa_prediction(text), length=len(tokens),
        eos=tokens[-1] == 2, truncated=len(tokens) == 32 and tokens[-1] != 2, empty=not bool(text),
        repeated_trigram=repeated_trigram(tokens), own_cache=True, own_prefix=True, logits_to_keep=1,
        query_layout=ctl.layout, cache_lengths=lengths, source_events=ctl.events,
        writer_positions=list(writers), writer_count=len(writers), answer_image_blocked=not noop,
        noop=noop, all_logits_finite=True)
    return result, first, trace, cache if retain_cache else None

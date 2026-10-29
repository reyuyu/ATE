"""Native cached generation and isolated single-Query attention-output bypass."""
from contextlib import AbstractContextManager
import hashlib

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from knockout_probe import model_layers, ROW_ATOL


def clone_cache(cache):
    if cache is None:
        return DynamicCache()
    legacy = cache.to_legacy_cache() if isinstance(cache, DynamicCache) else cache
    return DynamicCache.from_legacy_cache(tuple(tuple(t.clone() for t in pair) for pair in legacy))


def cache_fingerprint(cache):
    legacy = cache.to_legacy_cache() if isinstance(cache, DynamicCache) else cache
    h = hashlib.sha256()
    for pair in legacy:
        for tensor in pair:
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class AttentionStep(AbstractContextManager):
    """Only o_proj output may change; observe native attention rows without retaining matrices."""
    def __init__(self, model, query, start=None, observe=None, noop=False):
        self.model, self.query, self.start = model, query, start
        self.observe, self.noop = observe, noop
        self.handles, self.edits, self.stats = [], [], []
        self.residuals = {}
        if start is not None and not 0 <= start <= 29:
            raise ValueError('Invalid three-layer start')

    def __enter__(self):
        try:
            for index, layer in enumerate(model_layers(self.model)):
                if self.observe is not None:
                    def residual(module, args, kwargs, index=index):
                        hidden = args[0] if args else kwargs['hidden_states']
                        self.residuals[index] = float(hidden[0, self.query].float().norm())
                    self.handles.append(layer.register_forward_pre_hook(residual, with_kwargs=True))
                    self.handles.append(layer.self_attn.register_forward_hook(self.mass_hook(index)))
                if self.observe is not None or self.start is not None and self.start <= index <= self.start+2:
                    self.handles.append(layer.self_attn.o_proj.register_forward_hook(self.output_hook(index)))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def output_hook(self, index):
        def hook(module, args, output):
            if not 0 <= self.query < output.shape[1] or output.shape[0] != 1:
                raise AssertionError('Wrong current Query')
            if self.observe is not None:
                self.residuals[index] = (self.residuals[index], float(output[0, self.query].float().norm()))
            if self.start is not None and self.start <= index <= self.start+2:
                self.edits.append(dict(layer=index, query=self.query, sequence_length=output.shape[1]))
                if not self.noop:
                    changed = output.clone()
                    changed[0, self.query] = 0
                    return changed
            return output
        return hook

    def mass_hook(self, index):
        def hook(module, args, output):
            attention = output[1]
            if attention is None:
                raise AssertionError('Eager attention required')
            row = attention[0, :, self.query].float()
            if not torch.isfinite(row).all() or float((row.sum(-1)-1).abs().max()) > ROW_ATOL:
                raise AssertionError('Invalid native attention row')
            stats = dict(layer=index)
            for name, keys in self.observe.items():
                stats[name+'_mass'] = float(row[:, keys].sum(-1).mean()) if keys else 0.
            residual, update = self.residuals[index]
            stats.update(attention_output_norm=update, residual_norm=residual,
                         attention_residual_ratio=update/residual if residual else None)
            self.stats.append(stats)
            return (output[0], None)+tuple(output[2:])
        return hook

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def step_inputs(prompt, generated):
    """t0 complete image prompt; t>0 one consumed generated token with original positions."""
    if not generated:
        return dict(prompt)
    n = prompt['input_ids'].shape[1]
    t = len(generated)
    device = prompt['input_ids'].device
    return dict(input_ids=torch.tensor([[generated[-1]]], device=device),
                attention_mask=torch.cat([prompt['attention_mask'], torch.ones((1, t), dtype=prompt['attention_mask'].dtype, device=device)], dim=-1),
                position_ids=torch.tensor([[n+t-1]], device=device), cache_position=torch.tensor([n+t-1], device=device))


def native_keys(prompt, question_positions, t, image_token_id=32000):
    ids = prompt['input_ids'][0].tolist()
    if ids[0] != 1 or len([x for x in ids if x == image_token_id]) != 576:
        raise ValueError('Unexpected frozen BOS/image layout')
    return dict(bos=[0], image=[i for i, x in enumerate(ids) if x == image_token_id],
                question=list(question_positions), history=list(range(len(ids), len(ids)+t)))


def forward_step(model, inputs, history, query, start=None, observe=None, noop=False):
    # Every call owns a fresh cache. No window may modify another window's native history.
    with torch.inference_mode(), AttentionStep(model, query, start, observe, noop) as trace:
        output = model(**inputs, past_key_values=clone_cache(history), use_cache=True,
                       output_attentions=observe is not None, return_dict=True)
    logits = output.logits[0, query].detach()
    if not torch.isfinite(logits).all():
        raise AssertionError('Nonfinite prediction')
    if start is not None and [e['layer'] for e in trace.edits] != list(range(start, start+3)):
        raise AssertionError('Wrong intervention layers')
    return logits, output.past_key_values, trace.stats


def sensitivity(full, changed):
    logp, logq = full.double().log_softmax(-1), changed.double().log_softmax(-1)
    midpoint = torch.logaddexp(logp, logq)-np.log(2.)
    js = .5*((logp.exp()*(logp-midpoint)).sum()+(logq.exp()*(logq-midpoint)).sum())
    token, altered = int(full.argmax()), int(changed.argmax())
    return dict(js=max(0., float(js)), full_token_delta_logp=float(logq[token]-logp[token]),
                top1_changed=token != altered, full_token=token, changed_token=altered,
                full_confidence=float(logp[token].exp()))


def generate(model, prompt, limit, start=None, mode='Full', eos=2):
    if mode not in ('Full', 'Prefill-only', 'Decode-only', 'Combined'):
        raise ValueError('Unknown frozen confirmation mode')
    generated, history = [], None
    for t in range(limit):
        inputs = step_inputs(prompt, generated)
        query = prompt['input_ids'].shape[1]-1 if t == 0 else 0
        active = mode == 'Combined' or mode == 'Prefill-only' and t == 0 or mode == 'Decode-only' and t > 0
        logits, history, _ = forward_step(model, inputs, history, query, start if active else None)
        token = int(logits.argmax())
        generated.append(token)
        if token == eos:
            break
    return dict(token_ids=generated, length=len(generated), eos=generated[-1] == eos,
                truncated=len(generated) == limit and generated[-1] != eos)


def discovery(model, prompt, question_positions, limit, times=(0, 1, 4, 8, 16), eos=2):
    generated, history, rows, stats, probabilities = [], None, [], [], []
    for t in range(limit):
        inputs = step_inputs(prompt, generated)
        query = prompt['input_ids'].shape[1]-1 if t == 0 else 0
        observe = native_keys(prompt, question_positions, t) if t in times else None
        full, next_history, masses = forward_step(model, inputs, history, query, observe=observe)
        probabilities.append(full.float().softmax(-1).cpu().numpy())
        if t in times:
            stats.extend(dict(time=t, **s) for s in masses)
            for start in range(30):
                altered, _, _ = forward_step(model, inputs, history, query, start=start)
                rows.append(dict(time=t, window_start=start, current_full_is_eos=int(full.argmax()) == eos,
                                 **sensitivity(full, altered)))
        token = int(full.argmax())
        generated.append(token)
        history = next_history
        if token == eos:
            break
    return dict(token_ids=generated, length=len(generated), eos=generated[-1] == eos,
                truncated=len(generated) == limit and generated[-1] != eos,
                sensitivity=rows, mechanism=stats), np.stack(probabilities)

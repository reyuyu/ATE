"""Instance-local additive attention masks; no weights, positions or tokens change."""
from contextlib import contextmanager
import math

import torch

PATHS = ('image_to_last', 'question_to_last', 'image_to_question')


def window(center):
    if not 0 <= center < 32:
        raise ValueError('Center must be L0..L31')
    return tuple(range(max(0, center - 4), min(32, center + 5)))


def edges(path, mapping):
    image = mapping['visual_positions']
    question = mapping['question_positions']
    last = [mapping['p_minus_1']]
    if not image or not question or set(image) & set(question) or last[0] in image + question:
        raise ValueError('Image, true Question and Last must be nonempty and disjoint')
    if max(image) >= min(question) or max(question) >= last[0]:
        raise ValueError('Expected causal image-before-question-before-Last ordering')
    if path == 'image_to_last':
        return last, image
    if path == 'question_to_last':
        return last, question
    if path == 'image_to_question':
        return question, image
    raise ValueError(path)


def block(mask, batch_rows, queries, keys, validate=False):
    if mask is None or mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError('Expected eager additive [batch,1,query,key] causal mask')
    changed = mask.clone()
    b = torch.tensor(batch_rows, device=mask.device)[:, None, None]
    q = torch.tensor(queries, device=mask.device)[None, :, None]
    k = torch.tensor(keys, device=mask.device)[None, None, :]
    if not bool((mask[b, 0, q, k] == 0).all()):
        raise ValueError('Requested edges must be originally accessible')
    changed[b, 0, q, k] = -torch.inf
    if validate:
        expected = torch.zeros_like(mask, dtype=torch.bool)
        expected[b, 0, q, k] = True
        if not torch.equal(mask[~expected], changed[~expected]):
            raise AssertionError('Non-target mask entries changed')
        if not bool(torch.isneginf(changed[expected]).all()):
            raise AssertionError('Target mask entries not blocked')
        if not bool((changed[b, 0, q, 0] == 0).all()):
            raise AssertionError('BOS must remain accessible')
    return changed


@contextmanager
def intervention(layers, path, mapping, centers, validate=False):
    """Batch rows are replicas of one prompt, each with its own window center."""
    if len(layers) != 32:
        raise ValueError('This frozen experiment requires 32 LLM layers')
    queries, keys = edges(path, mapping)
    handles = []
    stats = {'layers': {}, 'target_weights_checked': 0}
    try:
        for layer_id, layer in enumerate(layers):
            active = [b for b, center in enumerate(centers) if layer_id in window(center)]
            if not active:
                continue

            def pre(module, args, kwargs, active=active, layer_id=layer_id):
                if kwargs.get('past_key_value') is not None:
                    raise ValueError('KV cache is forbidden')
                mask = kwargs['attention_mask']
                if mask.shape[0] != len(centers):
                    raise ValueError('Unexpected mask batch size')
                kwargs = dict(kwargs)
                kwargs['attention_mask'] = block(mask, active, queries, keys, validate)
                stats['layers'][str(layer_id)] = active
                return args, kwargs

            handles.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
            if validate:
                def post(module, args, output, active=active):
                    weights = output[1]
                    b = torch.tensor(active, device=weights.device)[:, None, None, None]
                    h = torch.arange(weights.shape[1], device=weights.device)[None, :, None, None]
                    q = torch.tensor(queries, device=weights.device)[None, None, :, None]
                    k = torch.tensor(keys, device=weights.device)[None, None, None, :]
                    if not bool((weights[b, h, q, k] == 0).all()):
                        raise AssertionError('Blocked softmax attention must be exactly zero')
                    if not bool(torch.isfinite(weights).all()):
                        raise AssertionError('Nonfinite attention weights')
                    stats['target_weights_checked'] += 1
                handles.append(layer.self_attn.register_forward_hook(post))
        yield stats
    finally:
        for handle in handles:
            handle.remove()


def probability_change(full_logp, masked_logp):
    if not math.isfinite(full_logp) or not math.isfinite(masked_logp):
        raise ValueError('Nonfinite log probability')
    return 100.0 * math.expm1(masked_logp - full_logp)

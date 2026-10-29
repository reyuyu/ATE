"""Stream small per-head Native mass summaries; never retain full attention maps."""
from contextlib import contextmanager

import numpy as np
import torch

FIELDS = ('last_bos_mass', 'last_image_mass', 'last_sink_mass',
          'question_image_mass', 'question_sink_mass')


def mass_heads(weights, mapping, sink_indices):
    """Return [head, five raw masses], mean over true Question queries."""
    visual = mapping['visual_positions']
    question = mapping['question_positions']
    last = mapping['p_minus_1']
    if len(visual) != 576 or any(not 0 <= p < 576 for p in sink_indices):
        raise ValueError('Expected 576 visual tokens and valid frozen patch indices')
    if len(set(sink_indices)) != len(sink_indices):
        raise ValueError('Duplicate Sink indices')
    a = weights[0, :, last, :].float()
    q = weights[0, :, question, :].float().mean(1)
    sinks = [visual[p] for p in sink_indices]
    result = torch.stack((a[:, 0], a[:, visual].sum(-1), a[:, sinks].sum(-1),
                          q[:, visual].sum(-1), q[:, sinks].sum(-1)), dim=-1)
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise AssertionError('Invalid Native mass')
    if bool((result[:, 2] > result[:, 1]).any()) or bool((result[:, 4] > result[:, 3]).any()):
        raise AssertionError('Sink must be a subset of Image')
    return result.cpu().numpy()


@contextmanager
def observe_mass(layers, mapping, sink_indices):
    records, handles = {}, []
    try:
        for index, layer in enumerate(layers):
            def post(module, args, output, index=index):
                if output[1] is None:
                    raise ValueError('Eager attention weights required')
                records[index] = mass_heads(output[1], mapping, sink_indices)
                # The decoder does not need these weights with output_attentions=False.
                return (output[0], None) + tuple(output[2:])
            handles.append(layer.self_attn.register_forward_hook(post))
        yield records
        if set(records) != set(range(32)):
            raise ValueError('Missing Native layers')
    finally:
        for handle in handles:
            handle.remove()


def add_shares(row):
    row = dict(row)
    for group in ('last', 'question'):
        total = row[f'{group}_image_mass']
        row[f'{group}_sink_over_image'] = row[f'{group}_sink_mass'] / total if total else None
    return row


def layer_rows(heads):
    values = np.asarray(heads, dtype=np.float64)
    if values.shape != (32, 32, 5):
        raise ValueError('Expected [32 layers,32 heads,5 masses]')
    return [add_shares({'layer': layer, **dict(zip(FIELDS, values[layer].mean(0).tolist()))})
            for layer in range(32)]


def window_rows(rows):
    from information_flow_window9.masks import window
    if [r['layer'] for r in rows] != list(range(32)):
        raise ValueError('Missing or unsorted layers')
    return [add_shares({'center': center, 'window_start': window(center)[0],
                       'window_end': window(center)[-1], 'window_layers': len(window(center)),
                       **{field: float(np.mean([rows[l][field] for l in window(center)])) for field in FIELDS}})
            for center in range(32)]

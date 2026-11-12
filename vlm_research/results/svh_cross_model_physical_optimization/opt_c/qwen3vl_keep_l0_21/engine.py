"""Reuse the audited layer mask unchanged, selecting exactly L0 through L21."""
import torch

from .common import OPEN_LAYERS, CONDITION
from ..qwen3vl_visual_writing_necessity.engine import LayerWriterMask
from ..qwen3vl_query_role.engine import generate as native_generate
from ..qwen3vl_early_windows.experiment import hook_counts


def controller(model, spec, noop=False):
    ctl = LayerWriterMask(model, spec, 'All-Writer', noop=noop)
    ctl.condition = CONDITION
    ctl.open_layers = set(OPEN_LAYERS)
    return ctl


@torch.inference_mode()
def generate(model, processor, inputs, spec, noop=False, audit=False, trace=False):
    before = hook_counts(model)
    try:
        with controller(model, spec, noop) as ctl:
            g, scores, observer, cache = native_generate(model, processor, inputs, spec['roles'],
                native=True, audit=audit, trace=trace)
        n, count = spec['layout']['input_length'], len(spec['roles']['I'])
        prefill = [count*(1+(0 if i in OPEN_LAYERS else len(spec['roles']['R']))) for i in range(36)]
        assert len(ctl.positions) == len(g['token_ids']) and len(ctl.events) == 36*len(g['token_ids'])
        for e in ctl.events:
            assert e['past'] == (n+e['step']-1 if e['step'] else 0)
            assert e['blocked_connections_per_head'] == (count if e['step'] else prefill[e['layer']])
        g.update(condition=CONDITION, writer_layers=OPEN_LAYERS, P1_Decode_image_off=not noop,
            independent_native_cache=True, logits_to_keep=1,
            source_audit=dict(layer_events=len(ctl.events), steps=len(ctl.positions),
                blocked_probability_max=max(e['blocked_probability_max'] for e in ctl.events),
                row_sum_max_error=max(e['row_sum_max_error'] for e in ctl.events),
                prefill_blocked_connections_per_layer=prefill,
                decode_blocked_connections_per_layer=[count]*36, actual_positions=ctl.positions,
                open_R_image_probability_max={str(e['layer']): e['open_R_image_probability_max']
                    for e in ctl.events if not e['step'] and e['layer'] in OPEN_LAYERS},
                non_target_mask_exact=True, BOS_Self_legal=True, Image_Query_native=True,
                native_mrope_DeepStack_MLP=True, hook_cleanup=True, noop=noop))
        return g, scores, observer, cache
    finally:
        assert hook_counts(model) == before

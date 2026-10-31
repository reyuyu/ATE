"""Minimal multi-layer specialization; remove every mask before native Decode."""
from contextlib import nullcontext

import torch

from knockout_probe import model_layers, tensor_hash
from paper_rq1.engine import step_inputs
from rq2_source_knockout.engine import SourceKnockout, Observer

CONDITIONS = {
    'A': list(range(32)),
    'B': [k for k in range(32) if k != 14],
    'C': [k for k in range(32) if k not in (14, 19)],
    'D': [14, 19],
}
LABELS = dict(Full='Full', A='All Image OFF', B='Keep L14',
              C='Keep L14+L19', D='Drop L14+L19')


class MultiImageKnockout(SourceKnockout):
    def __init__(self, model, inputs, bos_id, blocked, noop=False, expected_visual=576):
        blocks = model_layers(model)
        if len(blocked) != len(set(blocked)) or any(type(k) is not int or not 0 <= k < len(blocks) for k in blocked):
            raise ValueError('Invalid multi-layer selection')
        super().__init__(blocks, inputs, model.config.image_token_index, bos_id,
                         0, 'Image', noop, expected_visual)
        self.active = set() if noop else set(blocked)

    def _hook(self, index):
        historical = super()._hook(index)

        def cached_prefill(module, args, kwargs):
            cache = kwargs.get('past_key_value')
            if cache is not None and cache.get_seq_length(index) != 0:
                raise ValueError('Only empty per-layer cache at original Prefill is allowed')
            # Reuse the exact historical mask checks and transformation. Only its
            # no-cache guard is adapted; the real cache and all other arguments
            # are passed through untouched to native attention.
            _, changed = historical(module, args, dict(kwargs,past_key_value=None,use_cache=False))
            return args, dict(kwargs,attention_mask=changed['attention_mask'])
        return cached_prefill


@torch.inference_mode()
def generate(model, inputs, bos_id, blocked, limit=32, noop=False,
             audit=False, collect=False, expected_visual=576):
    if limit < 1 or not bool(inputs['attention_mask'].all()):
        raise ValueError('Require positive limit and unpadded single prompt')
    n = inputs['input_ids'].shape[1]
    tokens, history, logits_trace = [], None, []
    mask = MultiImageKnockout(model, inputs, bos_id, blocked, noop, expected_visual)
    observer = Observer(model, inputs, bos_id, audit=True, expected_visual=expected_visual) if audit else None
    debug = {}
    first = None
    while len(tokens) < limit and (not tokens or tokens[-1] != 2):
        if history is not None and any(history.get_seq_length(k) != n + len(tokens) - 1
                                       for k in range(len(model_layers(model)))):
            raise AssertionError('Independent cache prefix length mismatch')
        # The historical mask supports exactly one original full Prefill.
        # It is never installed during any Decode call.
        with mask if not tokens else nullcontext():
            with observer if not tokens and observer is not None else nullcontext():
                output = model(**step_inputs(inputs, tokens), past_key_values=history,
                               use_cache=True, output_attentions=not tokens and audit,
                               return_dict=True)
        if not tokens:
            mask.assert_one_forward()
            active = [r for r in mask.records if r['applied']]
            if [r['layer'] for r in active] != ([] if noop else sorted(blocked)):
                raise AssertionError('Wrong multi-layer scope')
            if any(r['blocked_connections_per_head'] != expected_visual for r in active):
                raise AssertionError('Wrong original P-1 -> image scope')
            first = output.logits[0, n-1].detach().float().cpu().clone()
            if audit:
                debug = dict(prefill_logits=output.logits.detach().float().cpu().clone(),
                             observer=observer, cache_ids=[id(output.past_key_values)],
                             first_cache_key_sha256=tensor_hash(output.past_key_values.key_cache[0]))
        elif audit:
            debug['cache_ids'].append(id(output.past_key_values))
        history = output.past_key_values
        logits = output.logits[0, -1]
        if not torch.isfinite(logits).all():
            raise AssertionError('Nonfinite effective prediction')
        if collect:
            logits_trace.append(logits.detach().float().cpu().clone())
        tokens.append(int(logits.argmax()))
    if mask.handles or len(mask.records) != len(model_layers(model)):
        raise AssertionError('Mask hook leaked into Decode')
    return dict(token_ids=tokens, length=len(tokens), eos=tokens[-1] == 2,
                truncated=len(tokens) == limit and tokens[-1] != 2,
                decode_steps=len(tokens)-1, masked_layers=[] if noop else sorted(blocked),
                mask_calls=len(mask.records), own_prefill=True, own_prefix=True,
                decode_native=True), first, logits_trace, debug

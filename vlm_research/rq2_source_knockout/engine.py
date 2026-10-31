"""Specialize the historical audited hard-mask implementation, never edit it."""
from contextlib import AbstractContextManager, nullcontext

import torch

from knockout_probe import (VisualAttentionKnockout, layout_from_inputs,
                            model_layers, tensor_hash, ROW_ATOL)

SOURCES = ('Image', 'Prompt', 'BOS')


def partition(inputs, image_id, bos_id, expected_visual=576):
    layout = layout_from_inputs(inputs, image_id, expected_visual)
    ids, valid = inputs['input_ids'][0], layout['valid']
    query = int(valid.nonzero().flatten()[-1])
    bos = (ids == bos_id) & valid
    if int(bos.sum()) != 1 or query <= int(layout['visual_indices'][-1]):
        raise ValueError('Require unique BOS and a real post-image P-1')
    own = torch.zeros_like(valid)
    own[query] = True
    groups = dict(Image=layout['visual'], BOS=bos, Self=own)
    groups['Prompt'] = valid & ~groups['Image'] & ~bos & ~own
    if any(not mask.any() for mask in groups.values()):
        raise ValueError('Empty source partition')
    if not torch.equal(sum(mask.int() for mask in groups.values()), valid.int()):
        raise AssertionError('Source partition overlaps or misses real keys')
    return query, groups


class SourceKnockout(VisualAttentionKnockout):
    def __init__(self, layers, inputs, image_id, bos_id, layer, source,
                 noop=False, expected_visual=576):
        if source not in SOURCES:
            raise ValueError('Only Image/Prompt/BOS are allowed')
        super().__init__(layers, inputs, image_id, 0, expected_visual,
                         only_layers=[] if noop else [layer])
        self.query, self.groups = partition(inputs, image_id, bos_id, expected_visual)
        target = torch.zeros_like(self.layout['valid'])
        target[self.query] = True
        # The existing generic mask helper uses these two fields as Query/Key sets.
        self.layout = dict(self.layout, text_queries=target, visual=self.groups[source])
        self.source, self.layer = source, layer


class Observer(AbstractContextManager):
    """Read-only rows and optional smoke fingerprints; drop returned matrices only."""
    def __init__(self, model, inputs, bos_id, audit=False, expected_visual=576):
        self.model, self.audit = model, audit
        self.query, self.groups = partition(inputs, model.config.image_token_index,
                                             bos_id, expected_visual)
        self.rows, self.states, self.handles = [], [], []

    def __enter__(self):
        try:
            for index, block in enumerate(model_layers(self.model)):
                def observe(module, args, output, index=index):
                    weights = output[1]
                    row = weights[0, :, self.query].float()
                    if not torch.isfinite(row).all() or (row < 0).any():
                        raise AssertionError('Invalid attention')
                    error = float((row.sum(-1) - 1).abs().max())
                    if error > ROW_ATOL:
                        raise AssertionError('Historical row-sum bound exceeded')
                    data = dict(layer=index, row_error=error)
                    for key, mask in self.groups.items():
                        data[key] = float(row[:, mask].sum(-1).mean())
                        data[key + '_max'] = float(row[:, mask].max())
                    if abs(sum(data[k] for k in self.groups) - 1) > ROW_ATOL:
                        raise AssertionError('Partition mass sum failed')
                    if self.audit:
                        data['other_query_attention_sha256'] = tensor_hash(weights[:, :, :self.query])
                    self.rows.append(data)
                    return (output[0], None) + tuple(output[2:])
                self.handles.append(block.self_attn.register_forward_hook(observe))
                if self.audit:
                    def state(module, args, output, index=index):
                        h = output[0]
                        if not torch.isfinite(h).all():
                            raise AssertionError('Nonfinite decoder state')
                        self.states.append(dict(layer=index,
                            other_queries_sha256=tensor_hash(h[:, :self.query])))
                    self.handles.append(block.register_forward_hook(state))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def forward(model, inputs, bos_id, layer=None, source='Image', noop=False,
            observe=False, audit=False, expected_visual=576):
    layers = model_layers(model)
    query, groups = partition(inputs, model.config.image_token_index, bos_id, expected_visual)
    intervention = (SourceKnockout(layers, inputs, model.config.image_token_index,
        bos_id, layer, source, noop, expected_visual) if layer is not None else None)
    observer = Observer(model, inputs, bos_id, audit, expected_visual) if observe else None
    with intervention if intervention is not None else nullcontext():
        with observer if observer is not None else nullcontext():
            output = model(**inputs, use_cache=False, output_attentions=observe, return_dict=True)
    if output.past_key_values is not None or not torch.isfinite(output.logits).all():
        raise AssertionError('Cache or nonfinite output')
    if intervention is not None:
        intervention.assert_one_forward()
        active = [r for r in intervention.records if r['applied']]
        if len(active) != (0 if noop else 1):
            raise AssertionError('Wrong layer scope')
        if active and active[0]['blocked_connections_per_head'] != int(groups[source].sum()):
            raise AssertionError('Wrong query/source scope')
    if observer is not None:
        if [r['layer'] for r in observer.rows] != list(range(len(layers))):
            raise AssertionError('Missing attention layers')
        if intervention is not None and not noop and observer.rows[layer][source + '_max'] != 0:
            raise AssertionError('Masked source attention is nonzero')
    logits = output.logits.detach().float().cpu() if audit else output.logits[0, query].detach().float().cpu()
    return logits, observer, intervention

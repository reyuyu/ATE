"""Local clone-and-zero hooks; no attention mask, weights or positions changed."""
from contextlib import AbstractContextManager

import torch

from knockout_probe import model_layers

KINDS = ('A-Skip', 'M-Skip', 'Joint-Skip')


class WindowTrace(AbstractContextManager):
    def __init__(self, model, query, kind='Full', start=0, noop=False, strict_zero=None):
        if kind not in ('Full',)+KINDS or not 0 <= start <= 29 or query < 0:
            raise ValueError('Frozen three-layer window and actual P-1 required')
        self.model, self.p, self.kind, self.start = model, query, kind, start
        self.noop, self.strict_zero = noop, strict_zero
        self.handles, self.edits, self.states = [], [], {}
        self.forwards = 0

    def __enter__(self):
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.guard, with_kwargs=True))
            for i, layer in enumerate(model_layers(self.model)):
                if self.kind == 'Full' or i in (self.start+2, 31):
                    self.handles.append(layer.register_forward_hook(self.capture(i)))
                if self.start <= i < self.start+3 and self.kind != 'Full':
                    if self.kind in ('A-Skip', 'Joint-Skip'):
                        self.handles.append(layer.self_attn.o_proj.register_forward_hook(self.zero(i, 'attention')))
                    if self.kind in ('M-Skip', 'Joint-Skip'):
                        self.handles.append(layer.mlp.register_forward_hook(self.zero(i, 'mlp')))
            self.handles.append(self.model.language_model.model.norm.register_forward_hook(self.norm))
            return self
        except BaseException:
            self.cleanup()
            raise

    def guard(self, module, args, kwargs):
        ids = kwargs['input_ids']
        if kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None:
            raise ValueError('Only no-cache complete-prefix prefill is permitted')
        if ids.shape[0] != 1 or not self.p < ids.shape[1]:
            raise ValueError('Expected batch-one actual original P-1')
        self.forwards += 1

    def zero(self, layer, kind):
        def hook(module, args, output):
            if self.noop:
                return output
            if output.ndim != 3 or not self.p < output.shape[1]:
                raise AssertionError('Unexpected sublayer layout')
            if self.strict_zero:
                result, audit = self.strict_zero(output, self.p)
            else:
                # Exact original Phase3.8 clone-and-zero operation; expensive
                # non-target hashing is restricted to the small smoke checks.
                result = output.clone()
                result[:, self.p] = 0
                audit = {}
            self.edits.append(dict(audit, layer=layer, kind=kind, query=self.p))
            return result
        return hook

    def capture(self, index):
        def hook(module, args, output):
            self.states[index] = output[0][0, self.p].detach().clone()
        return hook

    def norm(self, module, args, output):
        self.normalized = output[0, self.p].detach().clone()

    def cleanup(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

    def __exit__(self, *exc):
        self.cleanup()
        if exc[0] is None:
            if self.forwards != 1:
                raise AssertionError('Exactly one prefill required')
            kinds = ('attention', 'mlp') if self.kind == 'Joint-Skip' else ('attention',) if self.kind == 'A-Skip' else ('mlp',)
            expected = [] if self.kind == 'Full' or self.noop else [(i, k) for i in range(self.start, self.start+3) for k in kinds]
            if [(r['layer'], r['kind']) for r in self.edits] != expected:
                raise AssertionError('Intervention coverage mismatch')

    def arrays(self):
        return {'layer_indices': sorted(self.states),
                'layer_states': torch.stack([self.states[i] for i in sorted(self.states)]).cpu().numpy(),
                'normalized_final': self.normalized.cpu().numpy()}

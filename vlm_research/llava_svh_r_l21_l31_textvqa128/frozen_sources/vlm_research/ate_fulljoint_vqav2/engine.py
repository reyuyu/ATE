"""Compose the frozen partial-query bypass with the final receiver mask."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch

from ..physical_handoff_fastpath import engine as base
from ..fastp2_bypass_writer.engine import BypassWriter, LAYERS

CONDITIONS = ('C0', 'C1', 'C2', 'C3', 'C4')
FrozenHandoff = base.FastHandoff


class JointWriter(BypassWriter):
    def __init__(self, adapter, layout, name, writers, max_tokens, audit=False):
        self.joint_name = name
        self.partial = name in ('C3', 'C4')
        if name == 'C4-DenseRef':
            FrozenHandoff.__init__(self, adapter, layout, 'D1 Dense', max_tokens, False)
            self.name, self.collect, self.events = name, audit, []
            self.ia = tuple(sorted((*layout.image, layout.length - 1)))
            self.ia_index = torch.tensor(self.ia, device=self.device)
        else:
            super().__init__(adapter, layout, 'A2' if self.partial else 'A0', {}, max_tokens, audit)
        self.audit_enabled = audit
        self.collect = audit
        if name in ('C2', 'C4', 'C4-DenseRef', 'C4-ZeroRef'):
            assert set(writers) <= set(layout.relay)
            blocked = sorted(set(layout.relay) - set(writers))
            self.prefill_early = self.prefill_early.clone()
            if blocked:
                qi = torch.tensor(blocked, device=self.device)
                ki = torch.tensor(layout.image, device=self.device)
                self.prefill_early[0, 0, qi[:, None], ki[None, :]] = -torch.inf
        # The parent computes this before receiver selection: rebuild from the FINAL mask.
        if hasattr(self, 'active_index'):
            self.active_mask = self.prefill_early.index_select(-2, self.active_index)

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.decoder, 'forward', self.forward))
            for i, layer in enumerate(self.a.layers):
                module, original = layer.self_attn, layer.self_attn.forward

                def forward(*args, module=module, original=original, i=i, **kwargs):
                    if self.partial and i in LAYERS:
                        return self.physical_rows(module, *args, **kwargs)
                    out, weights = original(*args, **kwargs)
                    if self.joint_name in ('C4-DenseRef', 'C4-ZeroRef') and i in LAYERS:
                        out = out.clone()
                        if self.cache.prefill:
                            out.index_fill_(1, self.ia_index, 0)
                        else:
                            out.zero_()
                    hidden = kwargs.get('hidden_states', args[0] if args else None)
                    self.record(module, out, weights, range(hidden.shape[1]), hidden.shape,
                                kwargs.get('attention_mask'))
                    return out, weights

                stack.enter_context(patch.object(module, 'forward', forward))
            yield self
        self.audit_rows = self.events


def generate(adapter, layout, name, writers, max_tokens=32, capture=False, audit=False):
    if name in ('C0', 'C1', 'Native-Noop'):
        label = {'C0': 'Native Full', 'C1': 'Fast-P2', 'Native-Noop': 'Native-Noop'}[name]
        return base.generate(adapter, layout, label, max_tokens, False, capture, audit)

    def factory(a, l, condition, length, collect):
        return JointWriter(a, l, name, writers, length, collect)

    with patch.object(base, 'FastHandoff', factory):
        result = base.generate(adapter, layout, 'Fast-P2', max_tokens, False, capture, audit)
    result[0]['condition'] = name
    return result

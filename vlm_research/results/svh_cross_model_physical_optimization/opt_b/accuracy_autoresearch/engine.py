"""Writer/depth/answer-stage controls around the frozen V2 implementation."""
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from unittest.mock import patch

import torch

from ..ate_fulljoint_v2.engine import AnswerWriter, LAYERS
from ..physical_handoff_fastpath import engine as base


class ResearchWriter(AnswerWriter):
    def __init__(self, adapter, layout, config, writers, max_tokens, audit=False, reference=False):
        mode = config['answer_attention']
        name = 'C4-V2' if mode == 'both' and not reference else 'C4-V2-ZeroRef' if mode == 'both' else 'C2'
        super().__init__(adapter, layout, name, writers, max_tokens, audit)
        self.mode, self.reference = mode, reference
        if mode == 'decode' and not reference:
            self.cache.absent = LAYERS

    @contextmanager
    def context(self):
        if self.mode != 'decode':
            with super().context():
                yield self
            return
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.decoder, 'forward', self.forward))
            for i, layer in enumerate(self.a.layers):
                original = layer.self_attn.forward
                def forward(*args, original=original, i=i, **kw):
                    hidden = kw.get('hidden_states', args[0] if args else None)
                    if i in LAYERS and not self.reference:
                        if not self.cache.prefill:
                            out = torch.zeros_like(hidden)
                            self.record(out, None, (), hidden.shape, None)
                            return out, None
                        # Prefill P-1 still attends text. Full transient KV is available,
                        # but this layer never reads historical KV during Decode.
                        kw['past_key_value'] = None
                    out, weights = original(*args, **kw)
                    if i in LAYERS and self.reference and not self.cache.prefill:
                        out = torch.zeros_like(out)
                    self.record(out, weights, range(hidden.shape[1]), hidden.shape, kw.get('attention_mask'))
                    return out, weights
                stack.enter_context(patch.object(layer.self_attn, 'forward', forward))
            yield self
        self.audit_rows = self.events


def generate(adapter, layout, config, writers, max_tokens=32, capture=False, audit=False, reference=False):
    layout = replace(layout, cutoff=config.get('cutoff', 20))
    native = config['writer'] == 'native'
    head = config.get('logits_to_keep', 0)
    original_model = adapter.model.forward
    def model_forward(*args, **kw):
        kw['logits_to_keep'] = head
        return original_model(*args, **kw)
    def factory(a, l, condition, length, collect):
        return ResearchWriter(a, l, config, writers, length, collect, reference)
    with patch.object(adapter.model, 'forward', model_forward):
        if native:
            label = 'Native-Noop' if reference else 'Native Full'
            return base.generate(adapter, layout, label, max_tokens, False, capture, audit)
        # Original All-Writer/no-skip hot path remains identical to Fast-P2.
        if config['writer'] == 'all' and config['answer_attention'] == 'none':
            return base.generate(adapter, layout, 'Fast-P2', max_tokens, False, capture, audit)
        with patch.object(base, 'FastHandoff', factory):
            return base.generate(adapter, layout, 'Fast-P2', max_tokens, False, capture, audit)

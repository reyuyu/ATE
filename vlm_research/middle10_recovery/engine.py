"""Small mask extensions around the frozen Middle10 physical execution path."""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch

from .common import KEEP, CUTOFF, PREFILL_KEEP, WRITER_LAST
from ..c3v2_middle_reaccess import engine as frozen
from ..ate_fulljoint_v2.engine import AnswerWriter, LAYERS


class RecoveryWriter(frozen.MiddleWriter):
    def __init__(self, adapter, layout, label, max_tokens=32, audit=False, dense=False):
        self.label = label
        super().__init__(adapter, layout, KEEP[label], max_tokens, audit, dense)
        variants = {(True, False): self.prefill_early, (False, False): self.full_late,
                    (True, True): self.middle, (False, True): self.middle_late}
        self.masks = [variants[layer <= WRITER_LAST[label], layer in PREFILL_KEEP[label]]
                      for layer in range(32)]

    def prefill_mask(self, layer):
        if self.physical and layer >= self.layout.cutoff:
            return self.prefill_late
        return self.masks[layer]

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.decoder, 'forward', self.forward))
            for i, layer in enumerate(self.a.layers):
                module, original = layer.self_attn, layer.self_attn.forward

                def forward(*args, original=original, module=module, i=i, **kw):
                    hidden = kw.get('hidden_states', args[0] if args else None)
                    restore = self.label == 'E4' and i in LAYERS and self.cache.prefill
                    if self.physical_skip and i in LAYERS and not restore:
                        return self.answer_rows(module, *args, **kw)
                    if restore and self.physical_skip:
                        # All Prefill rows are now required, but Decode never reads
                        # this layer. Do not create persistent KV for these four layers.
                        kw['past_key_value'] = None
                    out, weights = original(*args, **kw)
                    if self.zero_reference and i in LAYERS and not restore:
                        out = out.clone()
                        if self.cache.prefill:
                            out[:, -1] = 0
                        else:
                            out.zero_()
                    self.record(out, weights, range(hidden.shape[1]), hidden.shape,
                                kw.get('attention_mask'))
                    return out, weights

                stack.enter_context(patch.object(module, 'forward', forward))
            yield self
        self.audit_rows = self.events

    def record(self, out, weights, positions, shape, mask):
        AnswerWriter.record(self, out, weights, positions, shape, mask)
        if not self.collect:
            return
        layer, prefill, _, original, _ = self.current
        skipped = layer in LAYERS and self.physical_skip
        p1_computed = not skipped or (self.label == 'E4' and prefill)
        if weights is None:
            assert skipped and not prefill
            return
        image_present = self.dense or (prefill and shape[1] == self.layout.length) or (
            not prefill and layer in self.keep)
        mass = relay_mass = 0.0
        allowed = layer in (PREFILL_KEEP[self.label] if prefill else KEEP[self.label])
        if image_present:
            if p1_computed:
                q = self.layout.length - 1 if prefill else 0
                visual = weights[0, :, q, list(self.layout.image)]
                mass = float(visual.float().sum(-1).mean())
                if allowed:
                    assert mass > 0
                else:
                    assert torch.count_nonzero(visual) == 0
            if prefill:
                visual = weights[0, :, list(self.layout.relay), :][..., list(self.layout.image)]
                relay_mass = float(visual.float().sum(-1).mean())
                if layer > WRITER_LAST[self.label]:
                    assert torch.count_nonzero(visual) == 0
                else:
                    assert relay_mass > 0
        self.visual_events.append(dict(layer=layer, step=self.step, prefill=prefill,
            answer_image_mass=mass, relay_image_mass=relay_mass, p1_computed=p1_computed,
            image_keys_present=image_present, image_keys_permitted=allowed,
            original_query_positions=original.detach().cpu().tolist()))


def generate(adapter, layout, label, max_tokens=32, fixed=False, capture=False,
             audit=False, dense=False, noop=False):
    if label not in ('E1', 'E2', 'E3', 'E4', 'E5'):
        return frozen.generate(adapter, layout, label, max_tokens, fixed, capture, audit, dense, noop)

    def factory(a, l, keep, length, collect, reference):
        return RecoveryWriter(a, l, label, length, collect, reference)

    with patch.object(frozen, 'KEEP', KEEP), patch.object(frozen, 'CUTOFF', CUTOFF), \
            patch.object(frozen, 'MiddleWriter', factory):
        return frozen.generate(adapter, layout, label, max_tokens, fixed, capture, audit, dense, noop)


prepare = frozen.prepare

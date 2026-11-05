"""Dense functional paths only. No timing events, FLOPs or performance loops."""
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import replace
from unittest.mock import patch

import torch

from .common import DECODE_IMAGE, policy
from ..ate_fulljoint_v2.engine import AnswerWriter, LAYERS
from ..c3v2_middle_reaccess.engine import MiddleWriter, prepare
from ..middle10_recovery.engine import RecoveryWriter
from ..physical_handoff_fastpath.engine import FastHandoff, native_cache_info
from ..p1_layerwise_accuracy.engine import tensor_hash


class AttributionWriter(MiddleWriter):
    def __init__(self, adapter, layout, label, max_tokens=32, audit=False):
        self.label, self.policy = label, policy(label)
        super().__init__(adapter, replace(layout, cutoff=32), DECODE_IMAGE,
                         max_tokens, audit, dense=True)
        variants = {(True, False): self.prefill_early, (False, False): self.full_late,
                    (True, True): self.middle, (False, True): self.middle_late}
        self.masks = [variants[i in self.policy['receiver_image_layers'],
                               i in self.policy['p1_image_layers']] for i in range(32)]

    def prefill_mask(self, layer):
        return self.masks[layer]

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.decoder, 'forward', self.forward))
            for i, block in enumerate(self.a.layers):
                module, original = block.self_attn, block.self_attn.forward

                def forward(*args, original=original, i=i, **kwargs):
                    h = kwargs.get('hidden_states', args[0] if args else None)
                    out, weights = original(*args, **kwargs)
                    skip = i in self.policy['prefill_p1_attention_skip' if self.cache.prefill
                                              else 'decode_attention_skip']
                    if skip:
                        out = out.clone()
                        if self.cache.prefill:
                            out[:, -1] = 0
                        else:
                            out.zero_()
                    self.record(out, weights, range(h.shape[1]), h.shape, kwargs.get('attention_mask'))
                    return out, weights

                stack.enter_context(patch.object(module, 'forward', forward))
            yield self
        self.audit_rows = self.events

    def record(self, out, weights, positions, shape, mask):
        AnswerWriter.record(self, out, weights, positions, shape, mask)
        if not self.collect:
            return
        layer, prefill, _, original, _ = self.current
        assert weights is not None
        q = self.layout.length-1 if prefill else 0
        visual = weights[0, :, q, list(self.layout.image)]
        answer_mass = float(visual.float().sum(-1).mean())
        allowed = layer in self.policy['p1_image_layers' if prefill else 'decode_image_layers']
        assert answer_mass > 0 if allowed else torch.count_nonzero(visual) == 0
        relay_mass = None
        if prefill:
            visual = weights[0, :, list(self.layout.relay), :][..., list(self.layout.image)]
            relay_mass = float(visual.float().sum(-1).mean())
            allowed_r = layer in self.policy['receiver_image_layers']
            assert relay_mass > 0 if allowed_r else torch.count_nonzero(visual) == 0
        self.visual_events.append(dict(layer=layer, prefill=prefill, step=self.step,
            answer_image_mass=answer_mass, relay_image_mass=relay_mass,
            image_keys_permitted=allowed, original_query_positions=original.detach().cpu().tolist()))


class NativeMaskReference:
    """Independent native decoder/DynamicCache reference using scoped Hooks."""
    def __init__(self, adapter, layout, label):
        self.a, self.layout, self.policy = adapter, layout, policy(label)
        self.prefill = True

    @contextmanager
    def context(self):
        handles = []
        try:
            for i, layer in enumerate(self.a.layers):
                def before(module, args, kw, i=i):
                    h = kw.get('hidden_states', args[0] if args else None)
                    prefill = h.shape[1] == self.layout.length
                    self.prefill = prefill
                    mask = kw.get('attention_mask')
                    if mask is None:
                        assert not prefill
                        count = int(kw['cache_position'][-1])+1
                        mask = h.new_zeros((1, 1, 1, count))
                    else:
                        mask = mask.clone()
                    if prefill:
                        rows = [] if i in self.policy['receiver_image_layers'] else list(self.layout.relay)
                        if i not in self.policy['p1_image_layers']:
                            rows.append(self.layout.length-1)
                        for q in rows:
                            mask[0, 0, q, list(self.layout.image)] = -torch.inf
                    elif i not in self.policy['decode_image_layers']:
                        mask[..., list(self.layout.image)] = -torch.inf
                    assert (mask == 0).any(-1).all()
                    kw['attention_mask'] = mask
                    return args, kw

                def output(module, args, out, i=i):
                    if i in self.policy['prefill_p1_attention_skip' if self.prefill else 'decode_attention_skip']:
                        out = out.clone()
                        if self.prefill:
                            out[:, -1] = 0
                        else:
                            out.zero_()
                    return out

                handles.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(layer.self_attn.o_proj.register_forward_hook(output))
            yield self
        finally:
            for handle in handles:
                handle.remove()


@torch.inference_mode()
def generate(adapter, layout, label, max_tokens=32, fixed=False, capture=False, audit=False,
             reference=False):
    """One independent real cached greedy answer, with no latency instrumentation."""
    control = None
    if reference:
        control = NativeMaskReference(adapter, layout, label)
    elif label == 'Native-Noop':
        control = FastHandoff(adapter, layout, 'Native-Noop', max_tokens, audit)
    elif label == 'E3-Dense':
        control = RecoveryWriter(adapter, replace(layout, cutoff=21), 'E3', max_tokens, audit, dense=True)
    elif label == 'E3-Physical':
        control = RecoveryWriter(adapter, replace(layout, cutoff=21), 'E3', max_tokens, audit, dense=False)
    elif label != 'Native':
        control = AttributionWriter(adapter, layout, label, max_tokens, audit)
    n, device = layout.length, adapter.inputs['input_ids'].device
    positions = torch.arange(n+max_tokens-1, device=device)
    attention = torch.ones((1, n+max_tokens-1), dtype=torch.long, device=device)
    eos = adapter.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    cache, ids, logits, valid = None, [], [], []
    try:
        with (control.context() if control else nullcontext()):
            for step in range(max_tokens):
                kwargs = dict(adapter.inputs) if step == 0 else dict(
                    input_ids=torch.tensor([[ids[-1]]], device=device),
                    attention_mask=attention[:, :n+step])
                kwargs.update(use_cache=True, return_dict=True, past_key_values=cache,
                    cache_position=positions[:n] if step == 0 else positions[n+step-1:n+step],
                    logits_to_keep=1)
                output = adapter.model(**kwargs)
                assert output.logits.shape[:2] == (1, 1) and torch.isfinite(output.logits).all()
                last = output.logits[:, -1]
                token = int(last.argmax(-1))
                if capture == 'all' or step == 0:
                    logits.append(last.detach().float().cpu().clone())
                    if capture:
                        valid.append(output.logits.detach().cpu().clone())
                cache = output.past_key_values
                ids.append(token)
                del output
                if not fixed and token in eos:
                    break
        text = adapter.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        info = (control.cache.audit() if control is not None and not reference
                else native_cache_info(cache))
        for row in info['layers']:
            if label != 'E3-Physical':
                assert row['used_tokens'] == n+len(ids)-1
        g = dict(condition=label, text=text, token_ids=ids, eos=ids[-1] in eos,
            empty=not bool(text), truncated=not fixed and len(ids)==max_tokens and ids[-1] not in eos,
            repeated=any(ids[i:i+3]==ids[j:j+3] for i in range(len(ids)-2) for j in range(i+3, len(ids)-2)),
            first_top1_token=ids[0], first_logits_sha256=tensor_hash(logits[0]),
            first_top1_logp=float(logits[0].log_softmax(-1)[0, ids[0]]),
            content_tokens=sum(t not in eos for t in ids), decode_steps=len(ids)-1,
            kv=info, audit=[] if control is None or reference else control.audit_rows,
            visual_audit=getattr(control, 'visual_events', []))
        return g, logits, valid, ([] if control is None or reference else control.hidden)
    finally:
        del cache, control

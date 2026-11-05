"""Read-only attention observation around the unchanged frozen physical engine."""
from unittest.mock import patch

import torch

from .common import *
from ..accuracy_autoresearch import engine as research
from ..accuracy_autoresearch.engine import ResearchWriter
from ..p1_layerwise_accuracy.engine import tensor_hash


class VisionMappingAudit:
    def __init__(self, a):
        self.a = a
        self.handles, self.events = [], []
        self.patch = self.embedded = self.vision = self.projector = None

    def __enter__(self):
        vision = self.a.model.vision_tower
        embeddings = vision.vision_model.embeddings
        assert embeddings.patch_size == 14 and embeddings.image_size == 336
        def patch_hook(m,args,out): self.patch = out.detach()
        def embedding_hook(m,args,out):
            assert tuple(self.patch.shape[-2:]) == (24,24)
            native = torch.cat([embeddings.class_embedding.expand(1,1,-1),
                self.patch.flatten(2).transpose(1,2)],dim=1)
            native = native + embeddings.position_embedding(embeddings.position_ids)
            assert torch.equal(native,out), 'CLIP embedding order not exact row-major + CLS'
            self.embedded = True
        def vision_hook(m,args,out): self.vision = out.hidden_states
        def projector_hook(m,args):
            selected = self.vision[self.a.model.config.vision_feature_layer][:,1:]
            assert selected.shape[1]==576 and torch.equal(selected,args[0])
            self.projector = True
        self.handles=[embeddings.patch_embedding.register_forward_hook(patch_hook),
            embeddings.register_forward_hook(embedding_hook),vision.register_forward_hook(vision_hook),
            self.a.model.multi_modal_projector.register_forward_pre_hook(projector_hook)]
        return self

    def __exit__(self,*exc):
        for h in self.handles: h.remove()
        self.handles.clear()
        self.patch = self.vision = None
        if exc[0] is None:
            assert self.embedded and self.projector


class ObservedWriter(ResearchWriter):
    def __init__(self, *args, observe=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.observe = observe
        self.rows, self.observation_audit = [], []

    def record(self, out, weights, positions, shape, mask):
        super().record(out, weights, positions, shape, mask)
        layer = self.current[0]
        if not self.observe or not self.cache.prefill or layer >= 20:
            return
        assert layer == len(self.rows) and weights is not None
        assert shape[1] == self.layout.length
        assert all(i in positions for i in self.layout.relay)
        # Copy only actual R-query rows and actual visual-key columns. No recomputed QK.
        selected = weights[0, :, list(self.layout.relay), :]
        visual = selected[..., list(self.layout.image)]
        assert visual.shape == (32, len(self.layout.relay), 576)
        assert torch.isfinite(selected).all() and bool((selected >= 0).all())
        row_error = float((selected.float().sum(-1)-1).abs().max())
        assert row_error < .005
        blocked = torch.isneginf(mask[0, 0, list(self.layout.relay)])
        zero_max = float(selected[:, blocked].abs().max()) if blocked.any() else 0.
        assert zero_max == 0.
        self.rows.append(visual.detach().cpu().numpy().copy())
        self.observation_audit.append(dict(layer=layer, row_sum_error=row_error,
            blocked_probability_max=zero_max, original_queries=list(self.layout.relay),
            original_image_keys=list(self.layout.image), output_unchanged=True))


def generate(a, layout, w, observe=False):
    controls = []
    def factory(*args, **kwargs):
        c = ObservedWriter(*args, observe=observe, **kwargs)
        controls.append(c)
        return c
    with patch.object(research, 'ResearchWriter', factory):
        result = research.generate(a, layout, CFG, w, 32, capture=True, audit=False)
    c = controls.pop()
    rows = np.stack(c.rows) if observe else None
    audit = c.observation_audit
    assert not observe or len(audit) == 20
    a.verify()
    return result, rows, audit


@torch.inference_mode()
def forced_logp(a, layout, w, token_ids, first):
    if len(token_ids) == 1:
        return float(first.float().log_softmax(-1)[0, token_ids[0]])
    control = ResearchWriter(a, layout, CFG, w, len(token_ids), False)
    n = layout.length
    cache, value = None, 0.
    device = a.inputs['input_ids'].device
    with control.context():
        for step, target in enumerate(token_ids):
            kw = dict(a.inputs) if step == 0 else dict(
                input_ids=torch.tensor([[token_ids[step-1]]], device=device),
                attention_mask=torch.ones((1,n+step),dtype=torch.long,device=device))
            kw.update(past_key_values=cache, use_cache=True, return_dict=True,
                cache_position=torch.arange(n,device=device) if step == 0 else torch.tensor([n+step-1],device=device),
                logits_to_keep=1)
            out = a.model(**kw)
            logits = out.logits[:, -1].float()
            assert torch.isfinite(logits).all()
            if step == 0:
                assert torch.equal(logits.cpu(), first), 'Teacher-forced prefill must exactly match free generation'
            value += float(logits.log_softmax(-1)[0,target])
            cache = out.past_key_values
    del cache, control
    a.verify()
    return value


def describe_generation(result):
    g, logits = result[0], result[1]
    assert g['token_ids']
    assert all(torch.isfinite(x).all() for x in logits)
    return dict(text=g['text'], token_ids=g['token_ids'], eos=g['eos'], truncated=g['truncated'],
        empty=g['empty'], first_logits_sha256=tensor_hash(logits[0]),
        all_step_logits_sha256=[tensor_hash(x) for x in logits], kv=g['kv'],
        language_prefill_ms=g['language_prefill_ms'], e2e_ms=g['latency_ms'])

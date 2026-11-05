"""Reuse the frozen E loop; only true Decode Image permissions can differ."""
from unittest.mock import patch

from .common import KEEP, LABELS
from ..middle10_attribution import engine as frozen
from ..middle10_attribution.engine import AttributionWriter, NativeMaskReference

prepare = frozen.prepare


class RegionWriter(AttributionWriter):
    def __init__(self, adapter, layout, label, max_tokens=32, audit=False):
        assert label in KEEP
        # E already has fully native Prefill, all MLP and all Attention updates.
        super().__init__(adapter, layout, 'E', max_tokens, audit)
        self.label = label
        self.keep = frozenset(KEEP[label])
        self.policy['decode_image_layers'] = KEEP[label]
        assert not self.physical and self.layout.cutoff == 32
        assert not self.policy['prefill_p1_attention_skip']
        assert not self.policy['decode_attention_skip']


class RegionReference(NativeMaskReference):
    def __init__(self, adapter, layout, label):
        super().__init__(adapter, layout, 'E')
        self.policy['decode_image_layers'] = KEEP[label]


def generate(adapter, layout, label, **kwargs):
    if label not in LABELS:
        return frozen.generate(adapter, layout, label, **kwargs)
    # Scoped factories preserve the exact historical quality-only greedy loop.
    # Each worker is single-threaded; factories and model Hooks both unwind.
    controls = []

    def writer(a, l, ignored, max_tokens=32, audit=False):
        control = RegionWriter(a, l, label, max_tokens, audit)
        controls.append(control)
        return control

    def reference(a, l, ignored):
        return RegionReference(a, l, label)

    with patch.object(frozen, 'AttributionWriter', writer), \
            patch.object(frozen, 'NativeMaskReference', reference):
        result = frozen.generate(adapter, layout, 'E', **kwargs)
    result[0]['condition'] = label
    if kwargs.get('audit') and not kwargs.get('reference'):
        assert len(controls) == 1
        result[0]['rotary_audit'] = [dict(position_ids=r['position_ids'].flatten().tolist(),
            cos_sha256=frozen.tensor_hash(r['cos']), sin_sha256=frozen.tensor_hash(r['sin']))
            for r in controls[0].rotary]
    return result

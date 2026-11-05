"""Frozen quality-only greedy loop and Dense Native-Prefill implementation."""
from unittest.mock import patch

from .common import KEEP, LABELS
from ..middle10_decode_regions import engine as previous
from ..middle10_decode_regions.engine import RegionWriter, RegionReference

frozen = previous.frozen
prepare = previous.prepare


class LayerWriter(RegionWriter):
    def __init__(self, adapter, layout, label, max_tokens=32, audit=False):
        super().__init__(adapter, layout, 'E', max_tokens, audit)
        self.label, self.keep = label, frozenset(KEEP[label])
        self.policy['decode_image_layers'] = KEEP[label]


class LayerReference(RegionReference):
    def __init__(self, adapter, layout, label):
        super().__init__(adapter, layout, 'E')
        self.policy['decode_image_layers'] = KEEP[label]


def generate(adapter, layout, label, **kwargs):
    if label not in LABELS:
        return previous.generate(adapter, layout, label, **kwargs)
    controls = []

    def writer(a, l, ignored, max_tokens=32, audit=False):
        control = LayerWriter(a, l, label, max_tokens, audit)
        controls.append(control)
        return control

    def reference(a, l, ignored):
        return LayerReference(a, l, label)

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

"""Scoped reuse of the audited single-condition Dense Mask path."""
from unittest.mock import patch

from .common import CONDITION, KEEP
from ..textvqa_deep_add_drop import engine as inherited

prepare = inherited.prepare


def generate(adapter, layout, label, **kwargs):
    if label != CONDITION:
        return inherited.generate(adapter, layout, label, **kwargs)
    with patch.object(inherited, 'KEEP', {CONDITION: KEEP}), \
            patch.object(inherited, 'LABELS', (CONDITION,)):
        return inherited.generate(adapter, layout, label, **kwargs)


def writer(adapter, layout, audit=False):
    with patch.object(inherited, 'KEEP', {CONDITION: KEEP}):
        return inherited.LayerWriter(adapter, layout, CONDITION, 4, audit)

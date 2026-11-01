"""Preserve historical HF generation's logits_to_keep=1 head GEMM shape.

The original frozen Qwen collector is untouched. This InternVL-only addendum
restores the native generation prefill head selection before its formal run.
"""
import argparse
from contextlib import contextmanager
from functools import wraps
import sys
from unittest.mock import patch

import torch

from .common import OUT, ROOT, read, write, sha256, verify_sources
from .engine import Adapter


@contextmanager
def native_head(language_model):
    original = language_model.forward
    @wraps(original)
    def forward(*args, **kwargs):
        assert kwargs.get('logits_to_keep', 1) == 1
        kwargs['logits_to_keep'] = 1
        return original(*args, **kwargs)
    with patch.object(language_model, 'forward', forward):
        yield


class NativeInternAdapter(Adapter):
    def __init__(self, name):
        assert name == 'internvl'
        super().__init__(name)

    def forward(self, *args, **kwargs):
        with native_head(self.model.language_model):
            return super().forward(*args, **kwargs)


def cpu():
    class FakeLM(torch.nn.Module):
        def forward(self, x, logits_to_keep=0):
            return x[:, -logits_to_keep:] if logits_to_keep else x
    model = FakeLM()
    x = torch.randn(1, 8, 4)
    original = model.forward
    with native_head(model):
        assert torch.equal(model(x), x[:, -1:])
    assert model.forward == original and torch.equal(model(x), x)
    try:
        with native_head(model):
            raise RuntimeError('intentional')
    except RuntimeError:
        pass
    assert model.forward == original
    write(OUT / 'internvl_native_head_cpu.json', dict(passed=True, default_selects_last_query=True,
        cleanup_normal_and_exception=True, no_attention_or_input_changes=True))


def freeze():
    verify_sources()
    assert read(OUT / 'internvl_native_head_cpu.json')['passed']
    assert read(OUT / 'qwen_completion.json')['complete']
    path = OUT / 'internvl_native_addendum.json'
    assert not path.exists()
    write(path, dict(original_protocol_sha256=sha256(OUT / 'protocol.json'),
        source_sha256={str(__file__): sha256(__file__)}, qwen_completion_sha256=sha256(OUT / 'qwen_completion.json'),
        prior_failed_smoke='internvl_smoke.log; historical native BF16 logits max difference 0.0625',
        correction='Native generation sets logits_to_keep=1; preserve this exact LM-head input matrix shape. Base adapter omitted it. No numerical tolerance changes.',
        evidence='Installed Transformers 4.49.0 generation/utils.py _supports_logits_to_keep and generate() set logits_to_keep=1; Qwen2ForCausalLM.forward slices hidden states before lm_head.',
        scope='InternVL only; same data, weights, visual budget, P-1 output ablation, fresh caches, metrics and exact legacy/noop requirements. Qwen frozen sources and all completed checkpoints unchanged.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['cpu', 'freeze', 'smoke', 'run'])
    mode = parser.parse_args().mode
    if mode == 'cpu':
        cpu()
    elif mode == 'freeze':
        freeze()
    else:
        addon = read(OUT / 'internvl_native_addendum.json')
        for p, h in addon['source_sha256'].items():
            assert sha256(p) == h
        from . import run
        run.Adapter = NativeInternAdapter
        sys.argv = [sys.argv[0], 'internvl', mode]
        run.main()

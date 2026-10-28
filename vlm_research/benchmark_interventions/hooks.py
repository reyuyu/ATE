"""Prefill-only adapter of the unchanged, hash-pinned B4 and C2 primitives."""
from contextlib import AbstractContextManager
import hashlib
import importlib.util
from pathlib import Path

import torch

from knockout_probe import model_layers, tensor_hash

ORIGINALS = {
    'b4': Path('/data/lk/ATE-worktrees/phase37-l0l1/vlm_research/phase3_7/pathways.py'),
    'c2': Path('/data/lk/ATE-worktrees/phase38b-c2-confirm240/vlm_research/phase38_sink_hooks.py'),
}


def original_modules(expected):
    modules = {}
    for key, path in ORIGINALS.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[key]:
            raise ValueError(f'Frozen original changed: {path}')
        spec = importlib.util.spec_from_file_location(f'original_{key}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[key] = module
    return modules


def footprint(model):
    return {n: (len(m._forward_pre_hooks), len(m._forward_hooks)) for n, m in model.named_modules()}


class PrefillIntervention(AbstractContextManager):
    """Never retarget decode queries: remove all edits after first full forward.

    Legacy B4 rejects caches even when empty. Its full-prefill mask builder sees
    cache-free validation kwargs, then the ORIGINAL cache object/flag is restored
    before calling attention. Position, shape, causality and mask checks stay on.
    """
    def __init__(self, model, inputs, mapping, condition, originals, noop=False, observe=True):
        if condition not in ('Full', 'B4', 'C2'):
            raise ValueError('Only frozen Full/B4/C2 conditions')
        self.model, self.inputs, self.mapping = model, inputs, mapping
        self.condition, self.originals, self.noop, self.observe = condition, originals, noop, observe
        self.handles, self.records, self.prefix, self.masses = [], [], {}, {}
        self.completed = False
        self.patch = None
        self.n = inputs['input_ids'].shape[1]
        self.p = mapping['p_minus_1']
        self.q = mapping['question_positions']
        if not self.q or len(set(self.q)) != len(self.q) or not all(0 <= q < self.p < self.n for q in self.q):
            raise ValueError('Invalid actual Question/P-1')
        if inputs['input_ids'][0, :mapping['prompt_length']].tolist() != mapping['prompt_ids']:
            raise ValueError('Frozen prompt input mismatch')

    def __enter__(self):
        self.before = footprint(self.model)
        try:
            layers = model_layers(self.model)
            if self.condition == 'B4':
                base = self.originals['b4'].PathwayMask

                class CachePrefillMask(base):
                    def _hook(patch, index):
                        legacy = super()._hook(index)

                        def hook(module, args, kwargs):
                            cache = kwargs.get('past_key_value')
                            if cache is not None and cache.get_seq_length(index) != 0:
                                raise AssertionError('Only empty-cache prefill may be edited')
                            clean = dict(kwargs, past_key_value=None, use_cache=False)
                            args, changed = legacy(module, args, clean)
                            changed = dict(changed)
                            for key in ('past_key_value', 'use_cache'):
                                if key in kwargs:
                                    changed[key] = kwargs[key]
                                else:
                                    changed.pop(key, None)
                            return args, changed
                        return hook

                self.patch = CachePrefillMask(self.model, self.inputs, self.q, self.p, 'B4', noop=self.noop)
                self.patch.__enter__()
            for index, layer in enumerate(layers):
                if self.observe:
                    self.handles.append(layer.register_forward_hook(self.hidden(index)))
                if index < 5 and self.observe:
                    self.handles.append(layer.self_attn.register_forward_pre_hook(self.want_weights, with_kwargs=True))
                    self.handles.append(layer.self_attn.register_forward_hook(self.attention(index)))
                if self.condition == 'C2' and index in (2, 3, 4):
                    self.handles.append(layer.mlp.register_forward_hook(self.mlp(index)))
            self.handles.append(self.model.register_forward_hook(self.finish))
            return self
        except BaseException:
            self.cleanup()
            raise

    def hidden(self, index):
        def hook(module, args, out):
            if not torch.isfinite(out[0]).all():
                raise AssertionError('Nonfinite hidden')
            self.prefix[index] = tensor_hash(out[0][:, :self.p])
        return hook

    @staticmethod
    def want_weights(module, args, kwargs):
        return args, dict(kwargs, output_attentions=True)

    def attention(self, index):
        def hook(module, args, out):
            weights = out[1]
            if weights is None or not torch.isfinite(weights).all():
                raise AssertionError('Missing/nonfinite local attention')
            row = weights[0, :, self.p].float()
            if float((row.sum(-1)-1).abs().max()) > .002:
                raise AssertionError('Original row normalization tolerance exceeded')
            if torch.count_nonzero(row[:, self.p+1:]):
                raise AssertionError('Causal attention violation')
            if self.condition == 'B4' and not self.noop and index < 2 and torch.count_nonzero(row[:, self.q]):
                raise AssertionError('B4 target probabilities not exactly zero')
            visual = self.mapping['visual_positions']
            bos = int(torch.where(self.inputs['input_ids'][0] == self.model.config.text_config.bos_token_id)[0][0])
            groups = {'image': visual, 'question': self.q, 'bos': [bos]}
            other = [i for i in range(self.p+1) if i not in set(visual+self.q+[bos])]
            groups['other_text'] = other
            self.masses[index] = {k: float(row[:, v].sum(-1).mean()) for k, v in groups.items()}
            return (out[0], None)+tuple(out[2:])
        return hook

    def mlp(self, index):
        def hook(module, args, out):
            if self.noop:
                return out
            result, audit = self.originals['c2'].zero_query(out, self.p)
            self.records.append(dict(layer=index, **audit))
            return result
        return hook

    def finish(self, module, args, out):
        if self.patch:
            self.patch.assert_one_forward()
            self.records = list(self.patch.records)
        if self.condition == 'C2' and not self.noop and [r['layer'] for r in self.records] != [2, 3, 4]:
            raise AssertionError('C2 edit coverage')
        self.completed = True
        self.cleanup()

    def cleanup(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        if self.patch:
            self.patch.__exit__(None, None, None)

    def __exit__(self, *exc):
        self.cleanup()
        if footprint(self.model) != self.before:
            raise AssertionError('Intervention hook leak')
        if exc[0] is None and not self.completed:
            raise AssertionError('No prefill observed')

"""Instance-local, uncached residual interventions; original zero_query reused."""
from contextlib import AbstractContextManager
from types import MethodType

import torch

from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers, tensor_hash

CONDITIONS = ('Full', 'P1-Joint', 'Whole-Skip')
LAYERS = (2, 3, 4)


class ResidualOnly(AbstractContextManager):
    def __init__(self, model, prompt_ids, condition, zero_query, noop=False, observe=False):
        if condition not in CONDITIONS or not prompt_ids:
            raise ValueError('Frozen condition and nonempty original prompt required')
        self.model, self.ids, self.condition = model, list(prompt_ids), condition
        self.zero_query, self.noop, self.observe = zero_query, noop, observe
        self.p = len(prompt_ids)-1
        self.handles, self.overrides, self.edits, self.identities = [], [], [], []
        self.prefix, self.calls, self.layer_calls = {}, {}, []
        self.forwards = 0

    def __enter__(self):
        self.footprint = hook_footprint(self.model)
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.guard, with_kwargs=True))
            for index, layer in enumerate(model_layers(self.model)):
                self.handles.append(layer.register_forward_hook(self.layer_output(index)))
                if index not in LAYERS:
                    continue
                self.handles.append(layer.register_forward_pre_hook(self.layer_input(index), with_kwargs=True))
                for kind, module in (('attention', layer.self_attn), ('mlp', layer.mlp)):
                    self.calls[index, kind] = 0
                    self.handles.append(module.register_forward_pre_hook(self.count(index, kind)))
                if self.condition == 'P1-Joint' and not self.noop:
                    for kind, module in (('attention', layer.self_attn.o_proj), ('mlp', layer.mlp)):
                        self.handles.append(module.register_forward_hook(self.zero(index, kind)))
                if self.condition == 'Whole-Skip' and not self.noop:
                    self.overrides.append((layer, 'forward' in layer.__dict__, layer.__dict__.get('forward')))
                    layer.forward = MethodType(self.skip(index), layer)
            return self
        except BaseException:
            self.cleanup()
            raise

    def guard(self, module, args, kwargs):
        if kwargs.get('use_cache') is not False or kwargs.get('past_key_values') is not None:
            raise ValueError('Residual pilot requires explicit no-cache complete-prefix forwards')
        ids = kwargs['input_ids']
        if ids.shape[0] != 1 or ids[0, :len(self.ids)].tolist() != self.ids:
            raise ValueError('Original prompt prefix/P-1 changed')
        self.forwards += 1

    def count(self, index, kind):
        def hook(module, args):
            self.calls[index, kind] += 1
        return hook

    def zero(self, index, kind):
        def hook(module, args, output):
            result, audit = self.zero_query(output, self.p)
            self.edits.append(dict(layer=index, kind=kind, **audit))
            return result
        return hook

    def skip(self, index):
        def forward(layer, hidden_states, *args, **kwargs):
            if args or kwargs.get('use_cache', False) or kwargs.get('past_key_value') is not None:
                raise ValueError('Whole-Skip requires native no-cache layer call')
            if kwargs.get('output_attentions', False):
                raise ValueError('Skipped layer has no attention weights')
            if not torch.isfinite(hidden_states).all():
                raise AssertionError('Nonfinite residual')
            return (hidden_states,)
        return forward

    def layer_input(self, index):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs['hidden_states']
            self.current_input = (index, x, x[:, self.p].clone())
        return hook

    def layer_output(self, index):
        def hook(module, args, output):
            self.layer_calls.append(index)
            x = output[0]
            if self.observe:
                self.prefix[index] = tensor_hash(x[:, :self.p])
            if index in LAYERS and self.condition != 'Full' and not self.noop:
                current, original, target = self.current_input
                if current != index or not torch.equal(x[:, self.p], target):
                    raise AssertionError('Target residual not preserved')
                all_tokens = self.condition == 'Whole-Skip'
                if all_tokens and (x is not original or not torch.equal(x, original)):
                    raise AssertionError('Whole-Skip did not return identical input residual')
                self.identities.append(dict(layer=index, p_minus_1=self.p,
                                            target_residual_exact=True, all_tokens_identity=all_tokens))
        return hook

    def cleanup(self):
        for layer, owned, previous in reversed(self.overrides):
            if owned:
                layer.forward = previous
            else:
                delattr(layer, 'forward')
        self.overrides.clear()
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.current_input = None

    def __exit__(self, *exc):
        self.cleanup()
        if hook_footprint(self.model) != self.footprint:
            raise AssertionError('Residual intervention hook leak')
        if exc[0] is None:
            if self.forwards != 1 or self.layer_calls != list(range(len(model_layers(self.model)))):
                raise AssertionError('Expected exactly one complete native decoder traversal')
            skip = self.condition == 'Whole-Skip' and not self.noop
            if any(count != int(not skip) for count in self.calls.values()):
                raise AssertionError('Unexpected attention/MLP execution count')
            if self.condition == 'P1-Joint' and not self.noop:
                if [(r['layer'], r['kind']) for r in self.edits] != [(i, k) for i in LAYERS for k in ('attention', 'mlp')]:
                    raise AssertionError('Joint must edit exactly six fixed sublayer outputs')
            if self.condition != 'Full' and not self.noop and len(self.identities) != 3:
                raise AssertionError('Missing layer residual identity audit')

    def audit(self):
        return {'condition': self.condition, 'original_p_minus_1': self.p, 'edits': self.edits,
                'residual_identity': self.identities,
                'target_sublayer_calls': {f'L{i}_{kind}': value for (i, kind), value in self.calls.items()},
                'all_decoder_layers_visited_once': self.layer_calls == list(range(len(model_layers(self.model))))}

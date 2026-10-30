"""Single Decode Query interventions, preserving the native softmax denominator."""
from contextlib import AbstractContextManager
import time
import numpy as np
import torch
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as llama

from knockout_probe import model_layers, ROW_ATOL
from paper_rq1.engine import clone_cache, cache_fingerprint, step_inputs
from paper_rq1_single_layer_gqa6.engine import metric_rows


def signature(cache):
    if cache is None:
        return None
    legacy = cache.to_legacy_cache() if hasattr(cache, 'to_legacy_cache') else cache
    result = []
    for pair in legacy:
        for value in pair:
            try:
                version = value._version
            except RuntimeError:
                version = None
            result.append((tuple(value.shape), value.data_ptr(), version))
    return getattr(cache, '_seen_tokens', None), result


def owned_clone(history):
    cloned = clone_cache(history)
    if history is not None:
        source = signature(history)[1]; copied = signature(cloned)[1]
        if len(source) != len(copied) or any(a[0] != b[0] or a[1] == b[1] for a, b in zip(source, copied)):
            raise AssertionError('Cache clone aliases native history')
    return cloned


def gamma(n):
    x = n * 2.**-24
    if x >= 1:
        raise ValueError('Invalid FP32 dot-product rounding bound')
    return x / (1 - x)


def decompose(projection, row, values, image_positions, original_input, original_output, audit):
    """Project original image PV in FP32, bias excluded exactly once."""
    keys = torch.as_tensor(image_positions, device=row.device)
    if len(keys) != 576 or len(set(image_positions)) != 576 or int(keys.max()) >= row.shape[-1]:
        raise ValueError('Require576 dynamically located original visual keys')
    p, v = row.float(), values.float()
    image_head = (p.index_select(-1, keys).unsqueeze(-2) @ v.index_select(-2, keys)).squeeze(-2)
    image_vector = image_head.reshape(1, 1, -1)
    weight = projection.weight.float()
    contribution = F.linear(image_vector, weight, bias=None)
    checks = dict(image_mass=float(p[:, keys].sum(-1).mean()),
                  contribution_norm=float(contribution.norm()), original_output_norm=float(original_output.float().norm()))
    if audit:
        if original_input.dtype != torch.float16:
            raise ValueError('Audit requires actual native FP16 projection input')
        mask = torch.ones(p.shape[-1], dtype=torch.bool, device=p.device); mask[keys] = False
        other_head = (p[:, mask].unsqueeze(-2) @ v[:, mask]).squeeze(-2)
        other_vector = other_head.reshape(1, 1, -1)
        reference_head = image_vector + other_vector
        magnitude = (p.abs().unsqueeze(-2) @ v.abs()).squeeze(-2).reshape_as(reference_head)
        head_bound = (gamma(p.shape[-1]) + 2.**-11) * magnitude + 2.**-25
        head_error = (original_input.float() - reference_head).abs()
        if not torch.all(head_error <= head_bound):
            raise AssertionError('Native PV fails frozen FP16/FP32 rounding bound')
        bias = projection.bias.float() if projection.bias is not None else 0.
        nonimage = F.linear(other_vector, weight, bias=None)
        reconstruction = contribution + nonimage + bias
        # Bounds propagate native head rounding plus three FP32 dot accumulations
        # and the native half-output cast. No data-fitted absolute tolerance.
        projection_bound = F.linear(head_bound, weight.abs(), None)
        projection_bound += gamma(weight.shape[-1]) * F.linear(
            original_input.float().abs() + image_vector.abs() + other_vector.abs(), weight.abs(), None)
        projection_bound += 2.**-11 * original_output.float().abs() + 2.**-25
        if projection.bias is not None:
            projection_bound += 2.**-24 * projection.bias.float().abs()
        error = (original_output.float() - reconstruction).abs()
        if not torch.all(error <= projection_bound):
            raise AssertionError('Native Wo decomposition fails frozen rounding bound')
        checks.update(head_max_abs_error=float(head_error.max()), head_max_error_to_bound=float((head_error/head_bound).max()),
            projection_max_abs_error=float(error.max()), projection_max_error_to_bound=float((error/projection_bound).max()),
            bias_present=projection.bias is not None, denominator_unchanged=True,
            formula='A_native approximately Wo(image_PV)+Wo(nonimage_PV)+bias; FP16/FP32 propagated rounding bound')
    if not torch.isfinite(contribution).all():
        raise AssertionError('Nonfinite image contribution')
    return contribution, checks


class DecodeIntervention(AbstractContextManager):
    def __init__(self, model, mode=None, layer=None, image_positions=(), noop=False, audit=False):
        self.model, self.mode, self.layer = model, mode, layer
        self.images, self.noop, self.audit = list(image_positions), noop, audit
        self.handles, self.edits, self.stats = [], [], []
        self.original_eager = llama.eager_attention_forward
        self.pv = None
        if mode not in [None, 'Whole', 'Image', 'All32']:
            raise ValueError('Unknown intervention')
        if mode in ['Whole', 'Image'] and (not isinstance(layer, int) or not 0 <= layer < 32):
            raise ValueError('Invalid layer')

    def __enter__(self):
        try:
            layers = model_layers(self.model)
            if len(layers) != 32:
                raise ValueError('Need native32-layer model')
            active = list(range(32)) if self.mode == 'All32' else [self.layer] if self.mode else []
            for i in active:
                self.handles.append(layers[i].self_attn.o_proj.register_forward_hook(self.output_hook(i)))
            if self.mode == 'Image':
                def tap(module, query, key, value, mask, scaling, **kwargs):
                    output, weights = self.original_eager(module, query, key, value, mask, scaling, **kwargs)
                    if module.layer_idx == self.layer:
                        if query.shape[-2] != 1 or query.shape[0] != 1 or self.pv is not None:
                            raise AssertionError('Image intervention must be single current Decode Query')
                        repeated = llama.repeat_kv(value, module.num_key_value_groups)
                        row = weights[0, :, 0]
                        if not torch.isfinite(row).all() or float((row.float().sum(-1)-1).abs().max()) > ROW_ATOL:
                            raise AssertionError('Invalid original post-softmax row')
                        self.pv = row, repeated[0]
                    return output, weights
                llama.eager_attention_forward = tap
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def output_hook(self, layer):
        def hook(module, args, output):
            if output.shape[0] != 1 or output.shape[1] != 1:
                raise AssertionError('Only true one-token Decode is allowed')
            self.edits.append(dict(layer=layer, query=0, query_length=1, noop=self.noop))
            if self.mode == 'Image':
                if self.pv is None:
                    raise AssertionError('Missing original PV')
                row, values = self.pv
                contribution, stats = decompose(module, row, values, self.images, args[0], output, self.audit)
                self.stats.append(dict(layer=layer, **stats))
                if self.noop:
                    return output
                return (output.float() - contribution).to(output.dtype)
            if self.noop:
                return output
            return torch.zeros_like(output)
        return hook

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.mode == 'Image':
            llama.eager_attention_forward = self.original_eager
        self.pv = None


def step(model, inputs, history, mode=None, layer=None, images=(), noop=False, audit=False):
    parent = signature(history)
    cloned = owned_clone(history)
    query = inputs['input_ids'].shape[1]-1 if history is None else 0
    with torch.inference_mode(), DecodeIntervention(model, mode, layer, images, noop, audit) as trace:
        output = model(**inputs, past_key_values=cloned, use_cache=True, output_attentions=False, return_dict=True)
    logits = output.logits[0, query].detach().float().cpu()
    if not torch.isfinite(logits).all() or signature(history) != parent:
        raise AssertionError('Nonfinite logits or parent cache metadata mutation')
    expected = list(range(32)) if mode == 'All32' else [layer] if mode else []
    if [r['layer'] for r in trace.edits] != expected:
        raise AssertionError('Wrong current-query layers')
    return logits, output.past_key_values, dict(edits=trace.edits, decomposition=trace.stats)


def generate(model, prompt):
    tokens, history, all_logits = [], None, []
    started = time.perf_counter()
    for t in range(64):
        logits, history, _ = step(model, step_inputs(prompt, tokens), history)
        token = int(logits.argmax()); tokens.append(token); all_logits.append(logits.numpy())
        if token == 2:
            break
    return dict(token_ids=tokens, length=len(tokens), eos=tokens[-1] == 2,
        truncated=len(tokens) == 64 and tokens[-1] != 2, seconds=time.perf_counter()-started), np.stack(all_logits)


def metrics(full, changed):
    result = metric_rows(full, torch.as_tensor(changed)[None])[0]
    logp = torch.as_tensor(full).float().log_softmax(-1)
    result.update(loss=-result['delta_logp'], abs_delta_logp=abs(result['delta_logp']),
        relative_probability_change=float(np.expm1(result['delta_logp'])),
        full_token_probability=result['p_full'], full_token_entropy=float(-(logp.exp()*logp).sum()),
        js_divergence=result['js'])
    return result

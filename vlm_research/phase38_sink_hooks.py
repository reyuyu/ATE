"""Instance-local sublayer and pre-softmax interventions for Phase 3.8."""
from contextlib import AbstractContextManager
import hashlib
import numpy as np
import torch

CONDITIONS = {
    'C0': dict(kind='native', layers=[]),
    'C1': dict(kind='attention', layers=[2, 3, 4]),
    'C2': dict(kind='mlp', layers=[2, 3, 4]),
    'C3': dict(kind='bos_only', layers=[2, 3, 4]),
    'C4': dict(kind='attention', layers=[5, 6, 7]),
}


def thash(x):
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(a)).hexdigest()


def zero_query(output, query):
    if output.ndim != 3 or output.shape[0] != 1 or not 0 <= query < output.shape[1]:
        raise ValueError('Expected batch-one sublayer output and actual prediction query')
    if not torch.isfinite(output).all():
        raise AssertionError('Nonfinite sublayer output')
    before = output.clone()
    result = output.clone()
    result[:, query] = 0
    other = torch.arange(output.shape[1], device=output.device) != query
    if not torch.equal(before, output) or not torch.equal(result[:, other], output[:, other]) or torch.count_nonzero(result[:, query]):
        raise AssertionError('Sublayer bypass changed non-targets or original tensor')
    return result, dict(query=query, shape=list(output.shape), target_zero=True, other_tokens_bitwise_exact=True, original_tensor_unchanged=True,
                        original_target_norm=float(output[:, query].double().norm()),
                        non_target_sha256=thash(output[:, other]))


def bos_mask(original, query, bos):
    if original is None or original.ndim != 4 or original.shape[:2] != (1, 1) or original.shape[2] != original.shape[3]:
        raise ValueError('Expected actual batch-one additive eager causal mask')
    n = original.shape[-1]
    if not 0 <= bos <= query < n or original[0, 0, query, bos] != 0:
        raise ValueError('BOS must be a legal causal key')
    if not original.is_floating_point() or torch.isnan(original).any() or torch.isposinf(original).any():
        raise ValueError('Invalid additive mask')
    allowed = original[0, 0] == 0
    if allowed.triu(1).any():
        raise ValueError('Input mask is not causal')
    if not torch.all((original == 0) | (original == torch.finfo(original.dtype).min) | torch.isneginf(original)):
        raise ValueError('Unexpected non-causal additive bias')
    before = original.clone()
    result = original.clone()
    result[0, 0, query, :] = torch.finfo(original.dtype).min
    result[0, 0, query, bos] = 0
    others = torch.arange(n, device=original.device) != query
    if not torch.equal(result[:, :, others], original[:, :, others]) or not torch.equal(before, original):
        raise AssertionError('Mask modification escaped the target query')
    if torch.where(result[0, 0, query] == 0)[0].tolist() != [bos]:
        raise AssertionError('BOS is not the sole legal key')
    return result, dict(other_queries_bitwise_exact=True, original_mask_unchanged=True,
                        sole_legal_key=bos, query=query, key_count=n)


def vector_stats(update, residual):
    a, b = update.double(), residual.double()
    an, bn = float(a.norm()), float(b.norm())
    return dict(norm=an, residual_norm=bn, relative_norm=an/bn if bn else None,
                cosine=float(torch.dot(a, b)/(a.norm()*b.norm())) if an and bn else None)


class SinkStageTrace(AbstractContextManager):
    """Observe true residual locations; edit only the frozen sublayer/query."""
    def __init__(self, layers, final_norm, query, bos, visual, question, prefix_text, sink_patches, condition):
        if condition not in CONDITIONS or len(layers) < 8:
            raise ValueError('Frozen condition and at least eight decoder layers required')
        if len(set(visual)) != len(visual) or any(p < 0 or p >= len(visual) for p in sink_patches):
            raise ValueError('Invalid visual/sink indices')
        if not visual or bos in visual or bos in question or set(visual) & set(question):
            raise ValueError('Overlapping attention groups')
        if any(p >= query or p < 0 for p in prefix_text) or max(visual+question) >= query:
            raise ValueError('Prediction query must follow image and Question')
        self.layers, self.final_norm = layers, final_norm
        self.query, self.bos = query, bos
        self.visual, self.question, self.prefix = list(visual), list(question), list(prefix_text)
        self.sinks = [self.visual[p] for p in sink_patches]
        self.condition, self.plan = condition, CONDITIONS[condition]
        self.handles, self.edits, self.metrics, self.hidden, self.prefix_hashes, self.rows = [], [], {}, {}, {}, {}
        self.inputs, self.mid, self.attention_updates, self.mlp_updates = {}, {}, {}, {}

    def __enter__(self):
        try:
            for i, layer in enumerate(self.layers):
                self.handles.append(layer.register_forward_pre_hook(self.enter_layer(i), with_kwargs=True))
                self.handles.append(layer.self_attn.register_forward_pre_hook(self.mask_hook(i), with_kwargs=True))
                self.handles.append(layer.self_attn.o_proj.register_forward_hook(self.attention_hook(i)))
                self.handles.append(layer.self_attn.register_forward_hook(self.attention_rows(i)))
                self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self.mid_hook(i)))
                self.handles.append(layer.mlp.register_forward_hook(self.mlp_hook(i)))
                self.handles.append(layer.register_forward_hook(self.leave_layer(i)))
            self.handles.append(self.final_norm.register_forward_hook(self.norm_hook))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def enter_layer(self, i):
        def hook(m, args, kwargs):
            if i in self.inputs:
                raise AssertionError('Repeated decoder layer; full-prefill only')
            x = args[0] if args else kwargs['hidden_states']
            self.inputs[i] = x[0, self.query].detach().clone()
        return hook

    def mask_hook(self, i):
        def hook(m, args, kwargs):
            if i in self.plan['layers'] and self.plan['kind'] == 'bos_only':
                result, audit = bos_mask(kwargs.get('attention_mask'), self.query, self.bos)
                self.edits.append(dict(layer=i, kind='bos_mask', **audit))
                return args, dict(kwargs, attention_mask=result)
        return hook

    def attention_hook(self, i):
        def hook(m, args, out):
            self.metrics[i] = dict(layer=i, attention=vector_stats(out[0, self.query], self.inputs[i]))
            result = out
            if i in self.plan['layers'] and self.plan['kind'] == 'attention':
                result, audit = zero_query(out, self.query)
                self.edits.append(dict(layer=i, kind='attention_after_o_proj', **audit))
            self.attention_updates[i] = result[0, self.query].detach().clone()
            return result
        return hook

    def attention_rows(self, i):
        def hook(m, args, out):
            weights = out[1]
            if weights is None or weights.ndim != 4:
                raise AssertionError('Actual eager Attention weights required')
            row = weights[0, :, self.query].float()
            if not torch.isfinite(row).all() or (row < 0).any() or (row.sum(-1)-1).abs().max() > .002:
                raise AssertionError('Invalid actual prediction Attention')
            image = row[:, self.visual].sum(-1)
            sink = row[:, self.sinks].sum(-1) if self.sinks else torch.zeros_like(image)
            rest = [k for k in range(row.shape[-1]) if k not in set(self.visual+self.question+[self.bos])]
            mass = dict(bos_mass=float(row[:, self.bos].mean()), image_mass=float(image.mean()),
                        sink_raw_mass=float(sink.mean()), sink_inside_image=float((sink/image.clamp_min(1e-30)).mean()),
                        sink_ratio_of_head_mean_mass=float(sink.mean()/image.mean()) if image.mean() else None,
                        question_mass=float(row[:, self.question].sum(-1).mean()),
                        other_text_mass=float(row[:, rest].sum(-1).mean()),
                        row_sum_max_error=float((row.sum(-1)-1).abs().max()))
            self.metrics[i].update(mass)
            if i <= 10:
                self.rows[i] = row.detach().cpu().numpy().astype(np.float16)
            if i in self.plan['layers'] and self.plan['kind'] == 'bos_only':
                other = torch.arange(row.shape[-1], device=row.device) != self.bos
                if not torch.all(row[:, self.bos] == 1) or torch.count_nonzero(row[:, other]):
                    raise AssertionError('BOS-only softmax support is not exactly BOS')
                self.edits.append(dict(layer=i, kind='bos_softmax_verified', bos_mass=1., all_other_keys_zero=True))
            # Caller requests output_attentions=False; discard weights only after use.
            return (out[0], None) + tuple(out[2:])
        return hook

    def mid_hook(self, i):
        def hook(m, args):
            value = args[0][0, self.query]
            if not torch.equal(value, self.inputs[i]+self.attention_updates[i]):
                raise AssertionError('Attention hook is not before residual addition')
            if i in self.plan['layers'] and self.plan['kind'] == 'attention' and not torch.equal(value, self.inputs[i]):
                raise AssertionError('Attention bypass failed to preserve entering residual')
            self.mid[i] = value.detach().clone()
        return hook

    def mlp_hook(self, i):
        def hook(m, args, out):
            self.metrics[i]['mlp'] = vector_stats(out[0, self.query], self.mid[i])
            self.metrics[i]['mlp_relative_to_layer_input'] = float(out[0, self.query].double().norm()/self.inputs[i].double().norm())
            result = out
            if i in self.plan['layers'] and self.plan['kind'] == 'mlp':
                result, audit = zero_query(out, self.query)
                self.edits.append(dict(layer=i, kind='mlp_before_residual', **audit))
            self.mlp_updates[i] = result[0, self.query].detach().clone()
            return result
        return hook

    def leave_layer(self, i):
        def hook(m, args, out):
            value = out[0][0, self.query]
            if not torch.equal(value, self.mid[i]+self.mlp_updates[i]) or not torch.isfinite(out[0]).all():
                raise AssertionError('MLP residual location or hidden finiteness failed')
            self.hidden[i] = value.detach().cpu().numpy()
            self.prefix_hashes[i] = thash(out[0][0, self.prefix])
        return hook

    def norm_hook(self, m, args, out):
        self.hidden['final_norm'] = out[0, self.query].detach().cpu().numpy()

    def assert_complete(self):
        if set(self.metrics) != set(range(len(self.layers))) or len(self.hidden) != len(self.layers)+1:
            raise AssertionError('Incomplete layer observation')
        wanted = len(self.plan['layers']) * (2 if self.plan['kind'] == 'bos_only' else 1)
        if len(self.edits) != wanted:
            raise AssertionError('Missing, repeated or out-of-scope intervention')
        if any(e['layer'] not in self.plan['layers'] for e in self.edits):
            raise AssertionError('Intervention escaped frozen window')

    def __exit__(self, *exc):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()

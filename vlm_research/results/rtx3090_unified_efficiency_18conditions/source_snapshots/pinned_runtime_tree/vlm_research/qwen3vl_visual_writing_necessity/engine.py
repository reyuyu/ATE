"""Layer-specific scheduling of the previously audited R-to-Image hard mask."""
import torch

from .common import writer_layers
from ..qwen3vl_semantic_visual_writers.engine import WriterMask, blocked_edges, apply_mask
from ..qwen3vl_early_windows.experiment import hook_counts, tensor_digest
from ..qwen3vl_query_role.engine import generate as native_generate


class LayerWriterMask(WriterMask):
    def __init__(self, model, spec, condition, noop=False):
        super().__init__(model, spec, [], noop)
        self.condition = condition
        self.open_layers = set(writer_layers(condition))

    def __enter__(self):
        try:
            def position_audit(module, args, kwargs):
                position, cp = kwargs['position_ids'], kwargs['cache_position']
                t, n = len(self.positions), self.spec['layout']['input_length']
                if t == 0:
                    assert tensor_digest(position) == self.spec['mrope']['position_ids_sha256']
                    assert cp.tolist() == list(range(n))
                else:
                    expected = n + t - 1 + self.spec['mrope']['delta']
                    assert position.shape == (3, 1, 1) and (position == expected).all()
                    assert cp.tolist() == [n + t - 1]
                self.positions.append(dict(step=t, position_ids_sha256=tensor_digest(position),
                    shape=list(position.shape), cache_position=cp.tolist() if t else [0, n-1]))
            self.handles.append(self.model.model.language_model.register_forward_pre_hook(position_audit, with_kwargs=True))
            for i, layer in enumerate(self.model.model.language_model.layers):
                def before(module, args, kwargs, i=i):
                    hidden, cache, cp = kwargs['hidden_states'], kwargs.get('past_key_values'), kwargs['cache_position']
                    past = cache.get_seq_length(i) if cache is not None else 0
                    n = self.spec['layout']['input_length']
                    assert hidden.shape[:2] == (1, 1 if past else n)
                    assert cp.tolist() == ([past] if past else list(range(n)))
                    writers = self.spec['roles']['R'] if i in self.open_layers else []
                    blocked = blocked_edges(self.spec['roles'], writers, n, past, hidden.device)
                    changed = apply_mask(kwargs['attention_mask'], blocked, past, self.noop, hidden.dtype)
                    self.state[i] = past, blocked
                    return args, dict(kwargs, attention_mask=changed)
                def after(module, args, output, i=i):
                    past, blocked = self.state[i]
                    weights = output[1]
                    assert weights.shape[0] == 1 and weights.shape[2:] == blocked.shape
                    rows = weights[0, :, ([0] if past else self.spec['roles']['R'] + self.spec['roles']['A']), :].float()
                    assert torch.isfinite(rows).all() and (rows >= 0).all()
                    error = float((rows.sum(-1) - 1).abs().max())
                    assert error <= torch.finfo(weights.dtype).eps
                    maximum = float(weights[0, :, blocked].max())
                    if not self.noop:
                        assert maximum == 0, 'Prohibited attention probability is nonzero'
                    allowed_max = None
                    if not past and i in self.open_layers:
                        r = torch.tensor(self.spec['roles']['R'], device=weights.device)
                        image = torch.tensor(self.spec['roles']['I'], device=weights.device)
                        allowed_max = float(weights[0, :, r[:, None], image[None, :]].max())
                        assert allowed_max > 0, 'Open R-to-Image path has no probability'
                    self.events.append(dict(layer=i, step=len(self.positions)-1, past=past,
                        query_length=blocked.shape[0], key_length=blocked.shape[1],
                        blocked_connections_per_head=int(blocked.sum()), blocked_probability_max=maximum,
                        row_sum_max_error=error, open_R_image_probability_max=allowed_max))
                    return output[0], None
                self.handles.append(layer.self_attn.register_forward_pre_hook(before, with_kwargs=True))
                self.handles.append(layer.self_attn.register_forward_hook(after))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise


@torch.inference_mode()
def generate(model, processor, inputs, spec, condition, noop=False, audit=False, trace=False):
    counts = hook_counts(model)
    try:
        with LayerWriterMask(model, spec, condition, noop) as ctl:
            result, scores, observer, cache = native_generate(model, processor, inputs, spec['roles'],
                condition='Full', native=True, audit=audit, trace=trace)
        assert len(ctl.events) == 36 * len(result['token_ids']) and not ctl.handles
        n, image_count = spec['layout']['input_length'], len(spec['roles']['I'])
        expected_prefill = [image_count * (1 + (0 if i in ctl.open_layers else len(spec['roles']['R']))) for i in range(36)]
        for event in ctl.events:
            t, i = event['step'], event['layer']
            assert event['past'] == (n+t-1 if t else 0)
            assert event['blocked_connections_per_head'] == (image_count if t else expected_prefill[i])
        result.update(writer_layers=sorted(ctl.open_layers), logits_to_keep=1,
            P1_Decode_image_off=not noop, independent_native_cache=True,
            source_audit=dict(layer_events=len(ctl.events), steps=len(ctl.positions),
                blocked_probability_max=max(e['blocked_probability_max'] for e in ctl.events),
                row_sum_max_error=max(e['row_sum_max_error'] for e in ctl.events),
                prefill_blocked_connections_per_layer=expected_prefill,
                decode_blocked_connections_per_layer=[image_count]*36,
                actual_positions=ctl.positions, non_target_mask_exact=True, BOS_Self_legal=True,
                native_mrope=True, native_QKV_MLP_DeepStack=True, hook_cleanup=True, noop=noop))
        return result, scores, observer, cache
    finally:
        assert hook_counts(model) == counts, 'Layer mask hook leaked'

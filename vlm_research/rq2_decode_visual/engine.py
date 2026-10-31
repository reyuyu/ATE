"""Reuse the frozen Prefill intervention; extend only the incremental Query mask."""
from contextlib import AbstractContextManager, nullcontext

import torch

from knockout_probe import VisualAttentionKnockout, model_layers, tensor_hash, ROW_ATOL
from paper_rq1.engine import step_inputs
from rq2_source_knockout.engine import Observer
from rq2_visual_sufficiency.engine import MultiImageKnockout

BLOCKED = [k for k in range(32) if k not in (14, 19)]


class DecodeImageKnockout(VisualAttentionKnockout):
    def __init__(self, model, inputs, blocked, step, noop=False, expected_visual=576):
        if step < 1 or len(set(blocked)) != len(blocked):
            raise ValueError('Require positive incremental step and unique layers')
        super().__init__(model_layers(model), inputs, model.config.image_token_index,
                         0, expected_visual, only_layers=[] if noop else blocked)
        self.step = step
        self.key_indices = self.layout['visual_indices']
        self.key_sha256 = tensor_hash(self.key_indices)

    def _hook(self, index):
        def hook(module, args, kwargs):
            hidden = kwargs['hidden_states']
            n = self.layout['sequence_length']
            query = n + self.step - 1
            cache = kwargs.get('past_key_value')
            if hidden.shape[:2] != (1, 1) or not kwargs.get('use_cache'):
                raise ValueError('Require a single true incremental Query and KV Cache')
            if cache is None or cache.get_seq_length(index) != query:
                raise ValueError('Per-layer historical cache length differs from real prefix')
            pos = kwargs.get('cache_position')
            if pos is None or not torch.equal(pos, torch.tensor([query], device=hidden.device)):
                raise ValueError('Wrong absolute incremental Query position')
            ids = kwargs.get('position_ids')
            if ids is not None and not torch.equal(ids, pos[None]):
                raise ValueError('RoPE position differs from the native prefix')
            old = kwargs.get('attention_mask')
            if old is None or old.shape != (1, 1, 1, query + 1):
                raise ValueError('Unexpected incremental additive mask / key length')
            # Unpadded causal Decode can access all keys through the current Query.
            # No future/padding column is admitted, and only real Image keys change.
            if not bool((old == 0).all()) or int(self.key_indices[-1]) >= n:
                raise ValueError('Non-native causal mask or image keys outside original prompt')
            touched = index in self.active
            changed = old
            if touched:
                changed = old.clone()
                changed[0, 0, 0, self.key_indices] = -torch.inf
                other = torch.ones(query + 1, dtype=torch.bool, device=hidden.device)
                other[self.key_indices] = False
                if not torch.equal(changed[..., other], old[..., other]) or not (changed == 0).any():
                    raise AssertionError('Non-image edge changed or fully blocked Query')
            self.records.append(dict(layer=index, decode_step=self.step, local_query=0,
                absolute_query=query, cache_before=query, cache_after=query + 1,
                key_length=query + 1, image_keys_sha256=self.key_sha256,
                image_key_count=len(self.key_indices),
                applied=touched, non_image_mask_unchanged=True,
                original_prefill_p_minus_one=n-1, cloned=touched))
            return args, dict(kwargs, attention_mask=changed)
        return hook


class DecodeProbabilities(AbstractContextManager):
    """Inspect actual post-softmax rows on every Decode, not just hook invocations."""
    def __init__(self, model, mask):
        self.model, self.mask = model, mask
        self.handles, self.rows = [], []

    def __enter__(self):
        try:
            for index, block in enumerate(model_layers(self.model)):
                def observe(module, args, output, index=index):
                    weights = output[1]
                    if weights is None or weights.shape[0] != 1 or weights.shape[2] != 1:
                        raise AssertionError('Missing actual incremental Attention probabilities')
                    expected = self.mask.layout['sequence_length'] + self.mask.step
                    if weights.shape[-1] != expected:
                        raise AssertionError('Actual probabilities disagree with KV key length')
                    row = weights[0, :, 0].float()
                    if not torch.isfinite(row).all() or (row < 0).any():
                        raise AssertionError('Nonfinite or negative Attention probability')
                    error = float((row.sum(-1)-1).abs().max())
                    if error > ROW_ATOL:
                        raise AssertionError('Historical row-sum tolerance exceeded')
                    visual = row[:, self.mask.key_indices]
                    maximum = float(visual.max())
                    if index in self.mask.active and maximum != 0.:
                        raise AssertionError('Decode Image mask did not really produce exact zero')
                    self.rows.append(dict(layer=index, decode_step=self.mask.step,
                        image_mass=float(visual.sum(-1).mean()), image_probability_max=maximum,
                        row_sum_max_error=error, blocked=index in self.mask.active))
                    return (output[0], None) + tuple(output[2:])
                self.handles.append(block.self_attn.register_forward_hook(observe))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def generate(model, inputs, bos_id, blocked=BLOCKED, limit=32, noop=False,
             audit=False, collect=False, expected_visual=576):
    if limit < 1 or not bool(inputs['attention_mask'].all()):
        raise ValueError('Require positive limit, batch1 unpadded original prompt')
    n = inputs['input_ids'].shape[1]
    tokens, cache, steps, logits_trace = [], None, [], []
    prefill = MultiImageKnockout(model, inputs, bos_id, blocked, noop, expected_visual)
    debug = {}
    while len(tokens) < limit and (not tokens or tokens[-1] != 2):
        step = len(tokens)
        if step:
            mask = DecodeImageKnockout(model, inputs, blocked, step, noop, expected_visual)
            observer = DecodeProbabilities(model, mask)
        else:
            mask = prefill
            observer = Observer(model, inputs, bos_id, audit=True,
                                expected_visual=expected_visual) if audit else None
        with mask:
            with observer if observer is not None else nullcontext():
                output = model(**step_inputs(inputs, tokens), past_key_values=cache,
                    use_cache=True, output_attentions=step > 0 or audit, return_dict=True)
        mask.assert_one_forward()
        if mask.handles or observer is not None and observer.handles:
            raise AssertionError('Hook leak')
        expected_length = n + step
        cache = output.past_key_values
        if any(cache.get_seq_length(k) != expected_length for k in range(len(model_layers(model)))):
            raise AssertionError('Native cache did not extend by exactly one Query')
        logits = output.logits[0, -1]
        if not torch.isfinite(logits).all():
            raise AssertionError('Nonfinite effective logits')
        if collect:
            logits_trace.append(logits.detach().float().cpu().clone())
        if not step:
            first = logits.detach().float().cpu().clone()
            if [r['layer'] for r in mask.records if r['applied']] != ([] if noop else sorted(blocked)):
                raise AssertionError('Wrong original P-1 intervention')
            if audit:
                debug.update(prefill_logits=output.logits.detach().float().cpu().clone(), observer=observer)
        else:
            if [r['layer'] for r in observer.rows] != list(range(len(model_layers(model)))):
                raise AssertionError('Missing actual Decode Attention layers')
            if not noop and any(r['image_probability_max'] <= 0 for r in observer.rows if not r['blocked']):
                raise AssertionError('Retained visual layer has no actual Image Attention')
            steps.append(dict(step=step, consumed_token=tokens[-1], predicted_token=int(logits.argmax()),
                masks=mask.records, probabilities=observer.rows))
        tokens.append(int(logits.argmax()))
    return dict(token_ids=tokens, length=len(tokens), eos=tokens[-1] == 2,
        truncated=len(tokens)==limit and tokens[-1] != 2, decode_steps=len(tokens)-1,
        masked_layers=[] if noop else sorted(blocked), original_p_minus_one=n-1,
        image_key_indices=prefill.layout['visual_indices'].tolist(),
        own_prefix=True, own_cache=True, decode_mask_audit=steps), first, logits_trace, debug

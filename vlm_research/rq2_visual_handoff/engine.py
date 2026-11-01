"""Specialize audited visual hard masks; keep native attention, weights and KV."""
from contextlib import AbstractContextManager

import torch

from knockout_probe import model_layers, ROW_ATOL, tensor_hash
from paper_rq1.engine import step_inputs
from rq2_decode_visual.engine import DecodeImageKnockout
from rq2_visual_sufficiency.engine import MultiImageKnockout

CONDITIONS = {'A': (True, True), 'B': (True, False),
              'C': (False, True), 'D': (False, False)}
LABELS = {'A': 'Full', 'B': 'Text Relay Only', 'C': 'Direct Vision Only', 'D': 'Neither'}
CHECKPOINTS = [1, 6, 13, 19, 25, 31]


class PrefillMask(MultiImageKnockout):
    def __init__(self, model, inputs, bos_id, condition, noop=False, expected_visual=576):
        if condition not in CONDITIONS:
            raise ValueError('Only four frozen conditions are permitted')
        super().__init__(model, inputs, bos_id, [], noop, expected_visual)
        if not bool(inputs['attention_mask'].all()):
            raise ValueError('Only unpadded batch-one prompts')
        self.condition = condition
        self.relay = self.layout['valid'].clone()
        positions = torch.arange(self.layout['sequence_length'], device=self.relay.device)
        self.relay &= (positions > self.layout['visual_indices'][-1]) & (positions < self.query)
        self.relay &= ~self.groups['Image']
        if not bool(self.relay.any()):
            raise ValueError('No post-image pre-P-1 relay queries')
        r_on, d_on = CONDITIONS[condition]
        target = torch.zeros_like(self.relay)
        if not r_on:
            target |= self.relay
        if not d_on:
            target[self.query] = True
        self.layout = dict(self.layout, text_queries=target)
        self.active = set(range(len(model_layers(model)))) if bool(target.any()) and not noop else set()


class Monitor(AbstractContextManager):
    """Audit actual probabilities then release matrices; optional read-only states."""
    def __init__(self, model, prefill, step=0, capture=False, checkpoints=CHECKPOINTS):
        self.model, self.prefill, self.step = model, prefill, step
        self.capture, self.checkpoints = capture, list(checkpoints)
        self.handles, self.rows, self.states = [], [], {}
        self.positions = torch.where(prefill.relay | prefill.groups['Self'])[0]

    def __enter__(self):
        try:
            for layer, block in enumerate(model_layers(self.model)):
                def observe(module, args, output, layer=layer):
                    weights = output[1]
                    n = self.prefill.layout['sequence_length']
                    expected_queries = 1 if self.step else n
                    if weights is None or weights.shape[0] != 1 or weights.shape[2:] != (expected_queries, n + self.step):
                        raise AssertionError('Actual attention dimensions differ from real prefix')
                    groups = {'Decode': torch.tensor([0], device=weights.device)} if self.step else {
                        'Relay': torch.where(self.prefill.relay)[0],
                        'P-1': torch.tensor([self.prefill.query], device=weights.device)}
                    r_on, d_on = CONDITIONS[self.prefill.condition]
                    for group, queries in groups.items():
                        row = weights[0, :, queries, :].float()
                        if not torch.isfinite(row).all() or (row < 0).any():
                            raise AssertionError('Invalid actual attention probabilities')
                        error = float((row.sum(-1) - 1).abs().max())
                        if error > ROW_ATOL:
                            raise AssertionError('Historical row-sum bound exceeded')
                        image = row[..., self.prefill.layout['visual_indices']]
                        maximum = float(image.max())
                        blocked = not r_on and bool(self.prefill.active) if group == 'Relay' else not d_on and bool(self.prefill.active)
                        if self.step:
                            blocked = not d_on and bool(self.prefill.active)
                        if blocked and maximum != 0.:
                            raise AssertionError('Blocked Image attention is not exactly zero')
                        self.rows.append(dict(layer=layer, step=self.step, query_group=group,
                            query_count=len(queries), absolute_query=n+self.step-1 if self.step else self.prefill.query if group == 'P-1' else -1,
                            key_length=n+self.step, image_mass=float(image.sum(-1).mean()),
                            image_probability_max=maximum, row_sum_max_error=error, blocked=blocked))
                    return (output[0], None) + tuple(output[2:])
                self.handles.append(block.self_attn.register_forward_hook(observe))
                if self.capture and not self.step and layer in self.checkpoints:
                    def state(module, args, output, layer=layer):
                        vectors = output[0][0, self.positions].detach().float().cpu().clone()
                        if not torch.isfinite(vectors).all():
                            raise AssertionError('Nonfinite captured states')
                        self.states[layer] = vectors
                    self.handles.append(block.register_forward_hook(state))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def generate(model, inputs, bos_id, condition='A', limit=32, noop=False,
             audit=False, capture=False, collect=False, expected_visual=576,
             checkpoints=CHECKPOINTS):
    if limit < 1:
        raise ValueError('Positive generation budget required')
    blocks = model_layers(model)
    prefill = PrefillMask(model, inputs, bos_id, condition, noop, expected_visual)
    n = prefill.layout['sequence_length']
    tokens, cache, rows, masks, trace, debug = [], None, [], [], [], {}
    while len(tokens) < limit and (not tokens or tokens[-1] != 2):
        step = len(tokens)
        direct_on = CONDITIONS[condition][1]
        blocked = [] if direct_on or noop else list(range(len(blocks)))
        mask = DecodeImageKnockout(model, inputs, blocked, step, noop, expected_visual) if step else prefill
        monitor = Monitor(model, prefill, step, capture, checkpoints)
        with mask, monitor:
            output = model(**step_inputs(inputs, tokens), past_key_values=cache,
                use_cache=True, output_attentions=True, return_dict=True)
        mask.assert_one_forward()
        if mask.handles or monitor.handles:
            raise AssertionError('Hook leak')
        cache = output.past_key_values
        if cache is None or any(cache.get_seq_length(k) != n+step for k in range(len(blocks))):
            raise AssertionError('Condition-specific native cache did not extend correctly')
        logits = output.logits[0, -1]
        if not torch.isfinite(logits).all():
            raise AssertionError('Nonfinite effective logits')
        if collect:
            trace.append(logits.detach().float().cpu().clone())
        rows.extend(monitor.rows)
        masks.extend(dict(layer=r['layer'], step=step, applied=r['applied'],
            blocked_connections_per_head=r.get('blocked_connections_per_head', expected_visual if r['applied'] else 0),
            query_count=1 if step else int(prefill.layout['text_queries'].sum()),
            cache_length=n+step) for r in mask.records)
        if not step:
            first = logits.detach().float().cpu().clone()
            debug.update(states=monitor.states, state_positions=monitor.positions.cpu().tolist())
            if capture and set(monitor.states) != set(checkpoints):
                raise AssertionError('Missing fixed hidden checkpoints')
            if audit:
                debug['prefill_logits'] = output.logits.detach().float().cpu().clone()
            expected_count = expected_visual * int(prefill.layout['text_queries'].sum())
            if any(r['blocked_connections_per_head'] != expected_count for r in mask.records if r['applied']):
                raise AssertionError('Incomplete relay or P-1 mask coverage')
        tokens.append(int(logits.argmax()))
    debug['attention_rows'] = rows
    debug['mask_records'] = masks
    return dict(token_ids=tokens, length=len(tokens), eos=tokens[-1] == 2,
        truncated=len(tokens)==limit and tokens[-1]!=2, decode_steps=len(tokens)-1,
        condition=condition, own_cache=True, own_prefix=True,
        image_indices=prefill.layout['visual_indices'].cpu().tolist(),
        relay_indices=torch.where(prefill.relay)[0].cpu().tolist(), p_minus_one=n-1,
        mask_layers=list(range(len(blocks))), noop=noop), first, trace, debug


def distances(a, b):
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    na, nb = a.norm(), b.norm()
    if na <= 0 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Invalid state vector / denominator')
    exact = torch.equal(a, b)
    return dict(exact=exact, norm_a=float(na), norm_b=float(nb),
        relative_l2=float((a-b).norm()/na),
        cosine_distance=0. if exact else float(1-(torch.dot(a,b)/(na*nb)).clamp(-1,1)) if nb>0 else None,
        sha256_a=tensor_hash(a), sha256_b=tensor_hash(b))

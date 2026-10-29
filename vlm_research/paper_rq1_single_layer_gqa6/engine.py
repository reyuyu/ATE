"""Single-layer interface, reusing the validated cache and input helpers only."""
import time

import numpy as np
import torch

from knockout_probe import model_layers, ROW_ATOL
from paper_rq1.engine import AttentionStep, clone_cache, step_inputs


def positions(inputs, bos_token_id):
    if inputs['input_ids'].shape[0] != 1:
        raise ValueError('Single-image unbatched experiment required')
    ids = inputs['input_ids'][0]
    mask = inputs['attention_mask'][0].bool()
    valid = mask.nonzero().flatten().tolist()
    bos = ((ids == bos_token_id) & mask).nonzero().flatten().tolist()
    if not valid or len(bos) != 1:
        raise ValueError('Exactly one real BOS in valid input is required')
    return dict(query=valid[-1], bos=bos, valid_keys=valid)


class SingleLayerStep(AttentionStep):
    """One o_proj output; never change the validated three-layer implementation."""
    def __init__(self, model, query, layer=None, observe=None, noop=False, audit=False):
        super().__init__(model, query, noop=noop)
        if layer is not None and (not isinstance(layer, int) or not 0 <= layer < 32):
            raise ValueError('Single-layer index must be0..31')
        if query < 0:
            raise ValueError('Query must be a real nonnegative position')
        self.layer, self.observation, self.audit = layer, observe, audit

    def __enter__(self):
        try:
            layers = model_layers(self.model)
            if len(layers) != 32:
                raise ValueError('Expected exactly32 native Transformer layers')
            if self.observation is not None:
                for i, layer in enumerate(layers):
                    self.handles.append(layer.self_attn.register_forward_hook(self.bos_hook(i)))
            if self.layer is not None:
                self.handles.append(layers[self.layer].self_attn.o_proj.register_forward_hook(self.single_hook))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def single_hook(self, module, args, output):
        if output.shape[0] != 1 or not 0 <= self.query < output.shape[1]:
            raise AssertionError('Wrong single-layer P-1 query')
        record = dict(layer=self.layer, query=self.query, sequence_length=output.shape[1], noop=self.noop)
        self.edits.append(record)
        if self.noop:
            return output
        changed = output.clone()
        changed[0, self.query] = 0
        if self.audit:
            other = torch.ones(output.shape[1], dtype=torch.bool, device=output.device)
            other[self.query] = False
            record.update(target_zero=bool((changed[0, self.query] == 0).all()),
                          other_queries_exact=torch.equal(output[:, other], changed[:, other]))
        return changed

    def bos_hook(self, layer):
        def hook(module, args, output):
            weights = output[1]
            if weights is None or weights.shape[0] != 1:
                raise AssertionError('Native eager post-softmax attention is missing')
            row = weights[0, :, self.query].float()
            error = float((row.sum(-1)-1).abs().max())
            if not torch.isfinite(row).all() or error > ROW_ATOL:
                raise AssertionError('Invalid native attention row')
            valid = torch.zeros(row.shape[-1], dtype=torch.bool, device=row.device)
            valid[self.observation['valid_keys']] = True
            valid[self.query+1:] = False
            if torch.count_nonzero(row[:, ~valid]):
                raise AssertionError('Native attention violates valid keys/causality')
            heads = row[:, self.observation['bos']].sum(-1)
            if not ((heads >= 0) & (heads <= 1)).all():
                raise AssertionError('BOS head mass outside[0,1]')
            self.stats.append(dict(layer=layer, bos_mass=float(heads.mean()),
                bos_mass_per_head=heads.cpu().tolist(), max_row_sum_error=error,
                query=self.query, bos_positions=self.observation['bos']))
            return (output[0], None)+tuple(output[2:])
        return hook


def forward(model, inputs, query, layer=None, observe=None, noop=False, history=None, cache=False, audit=False):
    with torch.inference_mode(), SingleLayerStep(model, query, layer, observe, noop, audit) as trace:
        output = model(**inputs, use_cache=cache, past_key_values=clone_cache(history) if cache else None,
                       output_attentions=observe is not None, return_dict=True)
    logits = output.logits[0, query].detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise AssertionError('Nonfinite full-vocabulary logits')
    if layer is not None and len(trace.edits) != 1:
        raise AssertionError('Single-layer output hook did not fire exactly once')
    return logits, output.past_key_values if cache else None, trace


def metric_rows(full, changed):
    full, changed = torch.as_tensor(full, dtype=torch.float32), torch.as_tensor(changed, dtype=torch.float32)
    if full.ndim != 1 or changed.ndim != 2 or changed.shape[1] != len(full):
        raise ValueError('Expected one native vector and a matrix of changed full-vocabulary logits')
    if not torch.isfinite(full).all() or not torch.isfinite(changed).all():
        raise ValueError('Invalid logits for stable FP32 metrics')
    lp, lq = full.log_softmax(-1), changed.log_softmax(-1)
    midpoint = torch.logaddexp(lp[None], lq)-np.log(2.)
    js = .5*((lp.exp()[None]*(lp[None]-midpoint)).sum(-1)+(lq.exp()*(lq-midpoint)).sum(-1))
    if float(js.min()) < -1e-6:
        raise ValueError('JS numerical error exceeds predefined rounding bound')
    native = int(full.argmax())
    rows = []
    for i in range(len(changed)):
        delta = lq[i, native]-lp[native]
        token = int(changed[i].argmax())
        rows.append(dict(full_token_id=native, ablated_token_id=token, full_token_logp=float(lp[native]),
            ablated_full_token_logp=float(lq[i, native]), p_full=float(lp[native].exp()),
            p_ablated=float(lq[i, native].exp()), delta_logp=float(delta),
            relative_delta_p=float(100*torch.expm1(delta)), top1_flip=token != native,
            js=float(js[i].clamp_min(0))))
    return rows


def first_token(model, processor, inputs):
    mapping = positions(inputs, processor.tokenizer.bos_token_id)
    started = time.perf_counter()
    full, _, trace = forward(model, inputs, mapping['query'], observe=mapping)
    changed, edits = [], []
    for layer in range(32):
        logits, _, intervention = forward(model, inputs, mapping['query'], layer=layer)
        changed.append(logits)
        edits.extend(intervention.edits)
    matrix = torch.stack(changed)
    rows = [dict(layer=i, **r) for i, r in enumerate(metric_rows(full, matrix))]
    if len(trace.stats) != 32 or [e['layer'] for e in edits] != list(range(32)):
        raise AssertionError('Incomplete single-layer or BOS observations')
    return dict(metrics=rows, bos=trace.stats, mapping=mapping,
                wall_seconds=time.perf_counter()-started), full.numpy(), matrix.numpy()


def generate(model, inputs, layer=None, limit=32):
    tokens, history, edits, first = [], None, [], None
    prompt_query = inputs['attention_mask'][0].nonzero().flatten()[-1].item()
    for t in range(limit):
        current = step_inputs(inputs, tokens)
        logits, history, trace = forward(model, current, prompt_query if t == 0 else 0,
            layer=layer if t == 0 else None, history=history, cache=True, audit=t == 0)
        if t == 0:
            first = logits.numpy()
        edits.extend(dict(time=t, **e) for e in trace.edits)
        token = int(logits.argmax())
        tokens.append(token)
        if token == 2:
            break
    if layer is not None and (len(edits) != 1 or edits[0]['time'] != 0):
        raise AssertionError('Generation hook applied beyond original Prefill P-1')
    return dict(token_ids=tokens, length=len(tokens), eos=tokens[-1] == 2,
                truncated=len(tokens) == limit and tokens[-1] != 2,
                intervention_audit=edits), first

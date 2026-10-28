"""Directed suffix mask knockout. Query rows are targets, visual columns sources.

All causal-mask editing, cloning and hook cleanup use the unchanged Phase 3
VisualAttentionKnockout. No token deletion, model method or library edits.
"""
from contextlib import nullcontext
import time

from vlm_common import gpu_memory
from knockout_probe import VisualAttentionKnockout, model_layers, ROW_ATOL
import torch

MODES = ['question_only', 'answer_only', 'question_answer', 'global']
CUTOFFS = [16, 24, 32]


class DirectedSuffixKnockout(VisualAttentionKnockout):
    def __init__(self, layers, inputs, image_token_id, cutoff, question_positions,
                 answer_positions, expected_visual=576):
        super().__init__(layers, inputs, image_token_id, cutoff, expected_visual)
        q, a = list(question_positions), list(answer_positions)
        if len(q) != len(set(q)) or len(a) != len(set(a)) or set(q) & set(a):
            raise ValueError('Question and answer-prediction queries must be disjoint and unique')
        targets = q + a
        if not targets or any(type(x) is not int or not 0 <= x < self.layout['sequence_length'] for x in targets):
            raise ValueError('Invalid or empty target queries')
        target_mask = torch.zeros_like(self.layout['text_queries'])
        target_mask[targets] = True
        if (target_mask & ~self.layout['text_queries']).any():
            raise ValueError('Directed queries must be valid post-image text positions')
        # The original hook/mask function operates on precisely these query rows.
        self.layout = dict(self.layout, text_queries=target_mask)


def query_mapping(inputs, image_token_id, question_positions, answer_records,
                  prompt_length, expected_visual=576):
    from knockout_probe import layout_from_inputs
    layout = layout_from_inputs(inputs, image_token_id, expected_visual)
    q = list(question_positions)
    a = [r['query_position'] for r in answer_records]
    if not q or len(q) != len(set(q)) or len(a) != 1 or a != [prompt_length - 1]:
        raise ValueError('This study requires one-token candidates and answer Query=P-1')
    if answer_records[0]['target_token_position'] != prompt_length:
        raise ValueError('Candidate target position must be P, not its prediction Query')
    if max(q) >= a[0] or set(q) & set(a):
        raise ValueError('All real question queries must precede and not overlap answer Query')
    for pos in q + a:
        if not 0 <= pos < layout['sequence_length'] or not bool(layout['text_queries'][pos]):
            raise ValueError('Mapped query is not valid post-image text')
    if inputs['input_ids'].shape[1] != prompt_length + 1:
        raise ValueError('Unexpected candidate/padding length for single-token experiment')
    return dict(question_queries=q, answer_prediction_queries=a, answer_token_positions=[prompt_length],
                prompt_length=prompt_length, visual_keys=layout['visual_indices'].cpu().tolist())


def intervention(model, inputs, mode, cutoff, mapping, expected_visual):
    if mode == 'full':
        return nullcontext()
    if mode not in MODES or cutoff not in CUTOFFS:
        raise ValueError('Unknown directed mode/cutoff')
    args = (model_layers(model), inputs, model.config.image_token_index, cutoff)
    if mode == 'global':
        return VisualAttentionKnockout(*args, expected_visual=expected_visual)
    q = mapping['question_queries'] if mode in ('question_only', 'question_answer') else []
    a = mapping['answer_prediction_queries'] if mode in ('answer_only', 'question_answer') else []
    return DirectedSuffixKnockout(*args, q, a, expected_visual=expected_visual)


def audit_attentions(output, layout, layers, heads, active):
    matrices = list(output.attentions or ())
    output['attentions'] = None  # Release both ModelOutput attribute and mapping ownership.
    if len(matrices) != layers:
        raise AssertionError('Unexpected number of LLM attention tensors')
    n = layout['sequence_length']
    targets = layout['text_queries'][:, None] & layout['visual'][None, :]
    future = torch.ones(n, n, dtype=torch.bool, device=targets.device).triu(1)
    records = []
    for i in range(layers):
        attn = matrices[i]; matrices[i] = None
        if attn.shape != (1, heads, n, n) or not torch.isfinite(attn).all() or (attn < 0).any():
            raise AssertionError('Invalid attention tensor')
        row_error = float((attn.float().sum(-1) - 1).abs().max())
        if row_error > ROW_ATOL or torch.count_nonzero(attn[0, :, future]) or torch.count_nonzero(attn[0, :, :, ~layout['valid']]):
            raise AssertionError('Attention normalization/causal/padding violation')
        maximum = float(attn[0, :, targets].max())
        if i in active and maximum != 0:
            raise AssertionError('Nonzero target Query -> Image Key attention')
        records.append(dict(layer=i, shape=list(attn.shape), applied=i in active,
                            target_attention_max=maximum, row_sum_max_error=row_error))
        del attn
    return records


def directed_forward(model, inputs, mode, cutoff, mapping, audit=False, expected_visual=576):
    if model.training:
        raise ValueError('Requires eval mode')
    layers = model_layers(model)
    before = [(len(x._forward_pre_hooks), len(x.self_attn._forward_pre_hooks)) for x in layers]
    cuda = inputs['input_ids'].is_cuda
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    context = intervention(model, inputs, mode, cutoff, mapping, expected_visual)
    try:
        with torch.inference_mode(), context as hook:
            output = model(**inputs, output_attentions=audit, output_hidden_states=True,
                           use_cache=False, return_dict=True)
        if mode != 'full':
            hook.assert_one_forward()
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        measurement = dict(forward_seconds=seconds, gpu_memory=gpu_memory() if cuda else None,
                           output_attentions=audit, use_cache=False)
        if output.past_key_values is not None or not torch.isfinite(output.logits).all() or not torch.isfinite(output.hidden_states[-1]).all():
            raise AssertionError('Non-finite hidden/logits or unexpected KV cache')
        attention_audit = None
        if audit:
            if mode == 'full':
                from knockout_probe import layout_from_inputs
                layout = layout_from_inputs(inputs, model.config.image_token_index, expected_visual)
                active = set()
            else:
                layout, active = hook.layout, hook.active
            attention_audit = audit_attentions(output, layout, len(layers), model.config.text_config.num_attention_heads, active)
        return dict(hidden=output.hidden_states[-1], logits=output.logits, measurement=measurement,
                    hook_audit=hook.records if mode != 'full' else [], attention_audit=attention_audit)
    finally:
        after = [(len(x._forward_pre_hooks), len(x.self_attn._forward_pre_hooks)) for x in layers]
        if before != after:
            raise AssertionError('Directed hooks leaked')

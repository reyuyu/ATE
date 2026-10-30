"""Compose existing disjoint temporal hooks; never ablate all Prefill queries."""
from contextlib import ExitStack
import torch
from paper_rq1_early_joint_attention.engine import JointAttention, generate

CONDITIONS = {
    'A': dict(layers=list(range(5)), decode=False),
    'B': dict(layers=list(range(6)), decode=False),
    'C': dict(layers=list(range(5)), decode=True),
    'D': dict(layers=list(range(6)), decode=True),
    'E': dict(layers=list(range(2, 6)), decode=True),
}


def first_metrics(full, changed):
    full, changed = full.float(), changed.float()
    if full.ndim != 1 or full.shape != changed.shape or not torch.isfinite(full).all() or not torch.isfinite(changed).all():
        raise AssertionError('Invalid first-answer logits')
    token = int(full.argmax())
    native_logp = float(full.log_softmax(-1)[token])
    changed_logp = float(changed.log_softmax(-1)[token])
    return dict(full_first_token=token, first_token=int(changed.argmax()),
        first_top1_flip=int(changed.argmax()) != token,
        full_first_token_logp=native_logp, intervened_full_token_logp=changed_logp,
        first_logp_loss=native_logp - changed_logp)


@torch.inference_mode()
def supplement_generate(model, prompt, condition, noop=False, audit=False, collect=False, limit=32):
    first = []
    def observe(module, args, output):
        if not first:
            first.append(output.logits[0, -1].detach().float().cpu().clone())
    with ExitStack() as stack:
        handle = model.register_forward_hook(observe)
        stack.callback(handle.remove)
        decode = None
        if condition['decode']:
            decode = stack.enter_context(JointAttention(model, condition['layers'], 'Decode-only',
                prompt['input_ids'].shape[1], noop=noop, audit=audit))
        g, logits = generate(model, prompt, dict(layers=condition['layers'], mode='P-1-only'),
            limit=limit, noop=noop, audit=audit, collect=collect)
        if decode is not None:
            assert decode.edits == len(condition['layers']) * g['decode_steps']
            g['edits'] += decode.edits
            g['decode_events'] = decode.events
        assert g['edits'] == len(condition['layers']) * (g['length'] if condition['decode'] else 1)
    assert len(first) == 1
    return g, first[0], logits

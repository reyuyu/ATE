"""Read-only diagnostics around the unmodified Phase 3.7 B4 forward."""
import numpy as np
import torch

from knockout_probe import layout_from_inputs, model_layers, tensor_hash, ROW_ATOL
from phase3_7.pathways import observed_forward, footprint
from .protocol import GROUPS


def group_masks(inputs, qpos, bos_positions, image_token, expected_visual):
    layout = layout_from_inputs(inputs, image_token, expected_visual)
    question = torch.zeros_like(layout['valid'])
    bos = torch.zeros_like(question)
    question[qpos] = True
    bos[bos_positions] = True
    visual = layout['visual']
    if not qpos or not bos_positions or (question & bos).any() or ((question | bos) & visual).any():
        raise ValueError('Invalid overlapping Attention Mass groups')
    if ((question | bos) & ~layout['valid']).any():
        raise ValueError('Group includes padding')
    other = layout['valid'] & ~visual & ~question & ~bos
    groups = [visual, question, bos, other]
    if not torch.equal(torch.stack(groups).sum(0), layout['valid'].long()):
        raise AssertionError('Mass groups do not partition valid keys')
    return groups


def forward(model, inputs, qpos, apos, bos_positions, condition='C0', noop=False, expected_visual=576):
    if condition not in ('C0', 'C1') or (noop and condition != 'C1'):
        raise ValueError('Only Full/B4 and a correctness-only B4 no-op are allowed')
    masks = group_masks(inputs, qpos, bos_positions, model.config.image_token_index, expected_visual)
    before = footprint(model)
    handles = []
    saved = {}
    actual = {}
    local_native = {}
    attention = {}
    question_hashes = {}

    def pre(index):
        def hook(module, args, kwargs):
            saved[index] = (args, dict(kwargs))
        return hook

    def post(index):
        def hook(module, args, output):
            probabilities = output[1]
            if probabilities is None:
                raise AssertionError('Diagnostics must precede the streaming consumer')
            row = probabilities[0, :, apos]
            def masses(value):
                result = torch.stack([value[:, keys].float().sum(-1) for keys in masks], -1)
                if not torch.isfinite(result).all() or (result.sum(-1) - 1).abs().max() > ROW_ATOL:
                    raise AssertionError('Invalid four-way Attention Mass')
                return result.cpu().numpy()
            actual[index] = masses(row)
            attention[index] = row.detach().cpu().numpy()
            args0, kwargs0 = saved.pop(index)
            if condition == 'C1' and not noop:
                native = module.forward(*args0, **kwargs0)[1][0, :, apos]
                local_native[index] = masses(native)
            else:
                local_native[index] = actual[index].copy()
        return hook

    def hidden(index):
        def hook(module, args, output):
            question_hashes[index] = tensor_hash(output[0][0, qpos])
        return hook

    try:
        for index, layer in enumerate(model_layers(model)):
            if index < 2:
                handles.append(layer.self_attn.register_forward_pre_hook(pre(index), with_kwargs=True))
                handles.append(layer.self_attn.register_forward_hook(post(index)))
            handles.append(layer.register_forward_hook(hidden(index)))
        value = observed_forward(model, inputs, qpos, apos, 'B0' if condition == 'C0' else 'B4',
                                 expected_visual=expected_visual, noop=noop)
        if list(sorted(actual)) != [0, 1] or len(question_hashes) != 32:
            raise AssertionError('Incomplete layer diagnostics')
        value['diagnostics'] = dict(group_order=GROUPS, mass=np.stack([actual[i] for i in (0, 1)]),
                                    local_native_mass=np.stack([local_native[i] for i in (0, 1)]),
                                    attention=np.stack([attention[i] for i in (0, 1)]),
                                    question_layer_sha256=[question_hashes[i] for i in range(32)],
                                    final_question_sha256=tensor_hash(torch.from_numpy(value['arrays']['final_hidden'][:-1])))
        return value
    finally:
        for handle in handles:
            handle.remove()
        if footprint(model) != before:
            raise AssertionError('Additional diagnostic hook leaked')

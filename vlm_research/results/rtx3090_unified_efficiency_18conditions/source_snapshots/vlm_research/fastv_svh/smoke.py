"""Mandatory real-image GPU gates. CPU tests never mark these as passed."""
from contextlib import ExitStack

import torch

from .common import GATE
from .engine import generate
from .inputs import prepare
from ..information_horizon_alignment import engine as frozen
from ..information_horizon_alignment.common import tensor_sha, hook_counts


def run(adapter, tokenizer, samples):
    selected = [next(s for s in samples if s['benchmark'] == b)
                for b in ('GQA', 'ScienceQA-IMG', 'TextVQA')]
    reports = []
    for sample in selected:
        layout, binding = prepare(adapter, sample, tokenizer)
        original_inputs = {k: tensor_sha(v) for k, v in adapter.inputs.items()}
        weights = [(p._version, p.data_ptr()) for p in adapter.model.parameters()]
        hooks = hook_counts(adapter.model)
        native, nl = frozen.hf_generate(adapter, layout, 'HF-Native', 3, capture=True)
        noop, ol = frozen.hf_generate(adapter, layout, 'HF-Native', 3, capture=True, noop=True)
        assert native['token_ids'] == noop['token_ids']
        assert len(nl) == len(ol) and all(torch.equal(x, y) for x, y in zip(nl, ol))
        hf = adapter.model.generate(**adapter.inputs, max_new_tokens=3, do_sample=False,
                                    num_beams=1, use_cache=True, logits_to_keep=1)
        assert hf[0, layout.length:].tolist() == native['token_ids']
        svh, sl = frozen.hf_generate(adapter, layout, 'Physical-C3', 3, capture=True)
        disabled, dl = generate(adapter, layout, 3, prune=False, capture=True, audit=True)
        assert svh['token_ids'] == disabled['token_ids']
        assert len(sl) == len(dl) and all(torch.equal(x, y) for x, y in zip(sl, dl))
        counts = {k: [0]*32 for k in ('norm', 'post_norm', 'mlp')}
        positions, skipped = [], []
        before_residual = {}
        with ExitStack() as stack:
            for i, layer in enumerate(adapter.layers):
                def before(m, args, kw, i=i):
                    step = counts['mlp'][i]
                    expected_pos = (list(range(layout.length)) if step==0 and i<3 else
                        None if step==0 and i<20 else list(layout.text) if step==0 else
                        [layout.length+step-1])
                    actual = kw['position_ids'][0].tolist()
                    assert kw['cache_position'].tolist() == actual
                    if expected_pos is not None:
                        assert actual == expected_pos
                    else:
                        assert len(actual) == layout.length-384
                        assert [p for p in actual if p not in layout.image] == list(layout.text)
                    expected_rope = adapter.model.language_model.model.rotary_emb(args[0], kw['position_ids'])
                    assert all(torch.equal(a, b) for a, b in zip(expected_rope, kw['position_embeddings']))
                    positions.append(dict(layer=i, step=step, rows=args[0].shape[1]))
                    if i in (2,3,4,5):
                        before_residual[i] = args[0][:, -1].clone()
                handle = layer.register_forward_pre_hook(before, with_kwargs=True)
                stack.callback(handle.remove)
                for name, module in [('norm', layer.input_layernorm), ('post_norm', layer.post_attention_layernorm),
                                     ('mlp', layer.mlp)]:
                    handle = module.register_forward_hook(lambda m, args, out, name=name, i=i:
                                                           counts[name].__setitem__(i, counts[name][i]+1))
                    stack.callback(handle.remove)
                def mid(m, args, i=i):
                    if i in (2,3,4,5):
                        assert torch.equal(args[0][:, -1], before_residual.pop(i))
                        skipped.append(i)
                handle = layer.post_attention_layernorm.register_forward_pre_hook(mid)
                stack.callback(handle.remove)
            actual, al = generate(adapter, layout, 3, capture=True, audit=True, verify_selector=True)
        assert len(actual['token_ids']) >= 2, 'Smoke requires a real incremental Decode'
        assert all(v == [len(actual['token_ids'])]*32 for v in counts.values())
        assert len(skipped) == 4*len(actual['token_ids'])
        assert actual['selector']['reference_top192_exact']
        assert all(v['visual_rows'] == (576 if v['layer'] < 3 else 192 if v['layer'] < 20 else 0)
                   for v in actual['physical_rows'])
        assert actual['kv']['persistent_original_visual_tokens'] == 0
        assert all(e['blocked_probability_max'] == 0 for e in actual['attention_audit'])
        assert hook_counts(adapter.model) == hooks
        assert [(p._version, p.data_ptr()) for p in adapter.model.parameters()] == weights
        assert {k: tensor_sha(v) for k, v in adapter.inputs.items()} == original_inputs
        reports.append(dict(key=sample['key'], input_binding=binding, passed=True,
            native_noop_all_valid_logits_exact=True, hf_generate_tokens_exact=True,
            pruning_disabled_frozen_svh_tokens_logits_exact=True,
            selector=actual['selector'], physical_rows=actual['physical_rows'], cache=actual['kv'],
            native_tokens=native['token_ids'], svh_tokens=svh['token_ids'], fusion_tokens=actual['token_ids'],
            layer_calls=counts, original_positions_rope_exact=True, answer_residual_checks=len(skipped),
            strict_gate=GATE, all_logits_finite=True, hooks_restored=True, inputs_weights_unchanged=True))
    return dict(passed=True, actual_gpu_images=len(reports), samples=reports)

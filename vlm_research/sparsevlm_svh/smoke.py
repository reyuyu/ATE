"""Six real-image gates; no quality matrix may precede a passing receipt."""
from contextlib import ExitStack
from unittest.mock import patch

import torch

from ..information_horizon_alignment.common import tensor_sha, hook_counts
from ..information_horizon_alignment import engine as frozen
from ..fastv_svh.inputs import prepare
from .common import COUNTS, GATE
from .engine import generate


def run(adapter, tokenizer, samples):
    selected = [next(s for s in samples if s['benchmark'] == b) for b in COUNTS]
    results = []
    for sample in selected:
        layout, binding = prepare(adapter, sample, tokenizer)
        input_hashes = {k: tensor_sha(v) for k, v in adapter.inputs.items()}
        weight_versions = [(p._version, p.data_ptr()) for p in adapter.model.parameters()]
        hooks = hook_counts(adapter.model)
        functions = [l.self_attn.forward for l in adapter.layers]
        native, nl = frozen.hf_generate(adapter, layout, 'HF-Native', 3, capture=True)
        noop, ol = frozen.hf_generate(adapter, layout, 'HF-Native', 3, capture=True, noop=True)
        assert native['token_ids'] == noop['token_ids']
        assert len(nl) == len(ol) and all(torch.equal(x, y) for x, y in zip(nl, ol))
        hf = adapter.model.generate(**adapter.inputs, do_sample=False, num_beams=1, max_new_tokens=3,
                                    use_cache=True, logits_to_keep=1)
        assert hf[0, layout.length:].tolist() == native['token_ids']
        baseline, bl = frozen.hf_generate(adapter, layout, 'Physical-C3', 3, capture=True)
        disabled, dl = generate(adapter, layout, 3, prune=False, capture=True)
        assert baseline['token_ids'] == disabled['token_ids']
        assert len(bl) == len(dl['logits']) and all(torch.equal(x, y) for x, y in zip(bl, dl['logits']))
        calls = {name: [0] * 32 for name in ('mlp', 'norm', 'post_norm')}
        projection_steps = []
        before_residual = {}
        with ExitStack() as stack:
            for i, layer in enumerate(adapter.layers):
                def pre(m, args, kw, i=i):
                    pos = kw['position_ids']
                    assert torch.equal(kw['cache_position'], pos[0])
                    expected = adapter.model.language_model.model.rotary_emb(args[0], pos)
                    assert all(torch.equal(x, y) for x, y in zip(expected, kw['position_embeddings']))
                    if i in (2, 3, 4, 5):
                        before_residual[i] = args[0][:, -1].clone()
                handle = layer.register_forward_pre_hook(pre, with_kwargs=True)
                stack.callback(handle.remove)
                def mid(m, args, i=i):
                    if i in (2, 3, 4, 5):
                        assert torch.equal(args[0][:, -1], before_residual.pop(i))
                handle = layer.post_attention_layernorm.register_forward_pre_hook(mid)
                stack.callback(handle.remove)
                for name, module in (('mlp', layer.mlp), ('norm', layer.input_layernorm),
                                     ('post_norm', layer.post_attention_layernorm)):
                    handle = module.register_forward_hook(lambda m, args, out, i=i, name=name:
                        calls[name].__setitem__(i, calls[name][i] + 1))
                    stack.callback(handle.remove)
                for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                    handle = getattr(layer.self_attn, name).register_forward_hook(
                        lambda m, args, out, i=i, name=name:
                        projection_steps.append((i, name, args[0].shape[1])))
                    stack.callback(handle.remove)
            actual, values = generate(adapter, layout, 3, capture=True, audit=True, verify_reference=True)
        reference, rv = generate(adapter, layout, 3, dense=True, capture=True, audit=True)
        assert actual['token_ids'] == reference['token_ids'], 'Physical/dense reference token mismatch'
        assert len(values['logits']) == len(rv['logits'])
        max_error = 0.
        for x, y in zip(values['logits'], rv['logits']):
            max_error = max(max_error, float((x - y).abs().max()))
            assert torch.allclose(x, y, atol=GATE['logits_atol'], rtol=GATE['logits_rtol'])
        hidden_audits = []
        for i, (x, y) in enumerate(zip(values['hidden'], rv['hidden'])):
            rmse = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y))
            cosine = float((1 - torch.nn.functional.cosine_similarity(x.double(), y.double(), dim=-1)).abs().max())
            assert rmse <= GATE['hidden_relative_rmse'] and cosine <= GATE['hidden_cosine_distance'], (i, rmse, cosine)
            hidden_audits.append(dict(layer=i, relative_rmse=rmse, cosine_distance=cosine))
        assert len(hidden_audits) == 32
        expected_visual = [576]*3 + [309]*4 + [204]*9 + [113]*4 + [0]*12
        assert [s['visual_rows'] for s in actual['physical_rows']] == expected_visual
        assert all(s['physically_allocated_visual_rows'] == s['visual_rows'] for s in actual['physical_rows'])
        assert all(s['official_score_and_topk_exact'] and s['independent_recycling_reference_passed']
                   for s in actual['stages'])
        assert all(e['blocked_probability_max'] == 0 for e in actual['attention_audit'])
        assert len(actual['token_ids']) >= 2, 'Smoke did not reach real incremental Decode'
        assert all(c == [len(actual['token_ids'])] * 32 for c in calls.values())
        assert not any(i in (2, 3, 4, 5) and length == 1 for i, name, length in projection_steps)
        assert actual['kv']['persistent_original_visual_tokens'] == 0
        # Independent requests must reproduce output without retaining prior KV.
        repeat, _ = generate(adapter, layout, 3)
        assert repeat['token_ids'] == actual['token_ids']
        assert repeat['first_logits_sha256'] == actual['first_logits_sha256']
        with patch.object(adapter.layers[4].mlp, 'forward', side_effect=RuntimeError('injected-smoke-cleanup')):
            try:
                generate(adapter, layout, 3)
                raise AssertionError('Injected exception did not execute')
            except RuntimeError as exc:
                assert str(exc) == 'injected-smoke-cleanup'
        assert [l.self_attn.forward for l in adapter.layers] == functions
        assert hook_counts(adapter.model) == hooks
        assert [(p._version, p.data_ptr()) for p in adapter.model.parameters()] == weight_versions
        assert {k: tensor_sha(v) for k, v in adapter.inputs.items()} == input_hashes
        results.append(dict(key=sample['key'], passed=True, input_binding=binding,
            native_noop_all_valid_logits_exact=True, hf_native_tokens_exact=True,
            pruning_disabled_svh_tokens_logits_exact=True, dense_reference_tokens_exact=True,
            physical_dense_logits_max_abs=max_error, hidden_audits=hidden_audits,
            physical_rows=actual['physical_rows'], stages=actual['stages'], cache=actual['kv'],
            original_text_rope_exact=True, pminus1_and_decode_image_access=False,
            mlp_norm_calls=calls, decode_skip_projection_calls_zero=True,
            exception_hooks_restored=True, inputs_weights_unchanged=True,
            native_tokens=native['token_ids'], fusion_tokens=actual['token_ids'], strict_gate=GATE))
    return dict(passed=True, actual_gpu_images=len(results), samples=results)

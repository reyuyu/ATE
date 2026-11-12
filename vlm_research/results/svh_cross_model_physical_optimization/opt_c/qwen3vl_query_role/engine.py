"""Query-local post-o_proj edits, using unmodified native Qwen generation."""
from contextlib import ExitStack

import torch

from ..qwen3vl_early_windows.experiment import (AttentionBypass, PrefillTrace,
    gqa_prediction, hook_counts, tensor_digest)
from .common import LAYERS, target_rows


class RoleBypass(AttentionBypass):
    def __init__(self, model, condition, mapping, prompt_length, noop=False):
        super().__init__(model, [] if condition == 'Full' else LAYERS, prompt_length, noop)
        self.condition, self.mapping = condition, mapping

    def edit(self, layer, module, args, output):
        assert output.ndim == 3 and output.shape[:2] == (1, self.query_length)
        rows = target_rows(self.condition, self.mapping, len(self.calls) == 1)
        assert len(rows) == len(set(rows)) and all(0 <= i < self.query_length for i in rows)
        if self.noop or not rows:
            result = output
        else:
            result = output.clone()
            result[:, rows] = 0
            assert torch.count_nonzero(result[:, rows]) == 0
            keep = torch.ones(self.query_length, dtype=torch.bool, device=output.device)
            keep[rows] = False
            assert torch.equal(result[:, keep], output[:, keep])
        self.events.append(dict(layer=layer, phase=self.calls[-1]['phase'],
            query_count=len(rows), query_positions=rows, noop=self.noop,
            non_target_exact=True, target_zero=bool(rows) and not self.noop))
        return result


@torch.inference_mode()
def generate(model, processor, inputs, mapping, condition='Full', noop=False,
             trace=False, audit=False, native=False):
    length = inputs['input_ids'].shape[1]
    before = hook_counts(model)
    steps, cache_identity = [], []

    def precheck(module, args, kwargs):
        t = len(steps)
        cache = kwargs.get('past_key_values')
        expected = 0 if t == 0 else length + t - 1
        assert cache is None and expected == 0 or cache.get_seq_length() == expected
        cp = kwargs['cache_position'].detach().cpu().tolist()
        assert cp == (list(range(length)) if t == 0 else [length+t-1])

    def observe(module, args, kwargs, output):
        t = len(steps)
        expected_q = length if t == 0 else 1
        assert output.logits.shape[:2] == (1, 1), 'Matched last-logit LM-head path required'
        assert torch.isfinite(output.logits).all()
        cache = output.past_key_values
        assert all(cache.get_seq_length(i) == length + t for i in range(36))
        if t == 0:
            assert kwargs.get('past_key_values') is None or kwargs['past_key_values'].get_seq_length() == length
            cache_identity.append(id(cache))
        ids = kwargs.get('input_ids')
        emb = kwargs.get('inputs_embeds')
        assert (emb if emb is not None else ids).shape[1] == expected_q
        steps.append(dict(logits=output.logits[0, -1].detach().float().cpu().clone(),
                          shape=list(output.logits.shape), cache_length=length + t))

    try:
        with ExitStack() as stack:
            bypass = None if native else stack.enter_context(RoleBypass(model, condition, mapping, length, noop))
            observer = stack.enter_context(PrefillTrace(model, length, mapping['I'])) if trace else None
            handle = model.register_forward_pre_hook(precheck, with_kwargs=True)
            stack.callback(handle.remove)
            handle = model.register_forward_hook(observe, with_kwargs=True)
            stack.callback(handle.remove)
            generated = model.generate(**inputs, do_sample=False, num_beams=1,
                max_new_tokens=32, use_cache=True, logits_to_keep=1,
                return_dict_in_generate=True, output_scores=True,
                output_attentions=False, output_hidden_states=False)
            ids = generated.sequences[0, length:].tolist()
            assert ids and len(ids) == len(steps) == len(generated.scores)
            assert all(torch.isfinite(s).all() for s in generated.scores)
            assert [int(s['logits'].argmax()) for s in steps] == ids
            assert all(generated.past_key_values.get_seq_length(i) == length + len(ids) - 1 for i in range(36))
            if bypass:
                assert len(bypass.calls) == len(ids)
                assert len(bypass.events) == len(bypass.layers) * len(ids)
            if observer:
                observer.validate()
                assert observer.mlp_calls == [len(ids)] * 36
            eos = model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos
            text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
            repeated = any(ids[i:i+3] == ids[j:j+3] for i in range(max(0, len(ids)-2))
                           for j in range(i+3, max(0, len(ids)-2)))
            result = dict(text=text, prediction=gqa_prediction(text), token_ids=ids,
                first_token_id=ids[0], first_top1_id=int(steps[0]['logits'].argmax()),
                eos=ids[-1] in eos, empty=not bool(text),
                truncated=len(ids) == 32 and ids[-1] not in eos, repeated_trigram=repeated,
                generated_tokens=len(ids), decode_queries=len(ids)-1,
                prefill_role_counts={k: len(v) for k, v in mapping.items()},
                target_prefill_queries=len(target_rows(condition, mapping, True)),
                target_decode_queries=(len(ids)-1 if condition in ('A-Bypass', 'I+A-Bypass') else 0),
                hook_events=[] if not bypass else bypass.events,
                cache_lengths=[generated.past_key_values.get_seq_length(i) for i in range(36)],
                cache_initial_empty=True, cache_forward_count=len(steps),
                all_logits_finite=True, logits_shapes=[s['shape'] for s in steps])
            # Holding cache objects in Smoke makes independence test non-vacuous (no id recycling).
            cache = generated.past_key_values if audit else None
            scores = [s['logits'] for s in steps] if audit else [steps[0]['logits']]
    finally:
        assert hook_counts(model) == before, 'Hook leaked on normal or exceptional exit'
    return result, scores, observer, cache


def feature_hashes(observer):
    return {str(i): tensor_digest(v) for i, v in observer.features.items()}

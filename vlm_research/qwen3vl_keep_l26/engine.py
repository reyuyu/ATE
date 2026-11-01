from contextlib import ExitStack

import torch

from ..qwen3vl_image_source.engine import ImageOff
from ..qwen3vl_early_windows.experiment import PrefillTrace, gqa_prediction, hook_counts

CONDITIONS = {'Full': [], 'Keep L26': [i for i in range(36) if i != 26], 'All OFF': list(range(36))}


class PrefillOnly:
    """Install audited source masks for prefill; remove them before first decode."""
    def __init__(self, model, layout, layers, cache, noop=False):
        self.model, self.layout, self.layers, self.cache = model, layout, layers, cache
        self.noop = noop
        self.handles, self.interventions, self.calls = [], {}, []
        self.stack = ExitStack()

    def begin(self, module, args, kwargs):
        n = self.layout['input_length']
        step = len(self.calls)
        start, length = (0, n) if step == 0 else (n + step - 1, 1)
        assert kwargs['inputs_embeds'].shape[1] == length
        assert kwargs['cache_position'].tolist() == list(range(start, start + length))
        assert kwargs['past_key_values'] is self.cache
        assert all(self.cache.get_seq_length(i) == start for i in range(36))
        if not step:
            assert len(kwargs['deepstack_visual_embeds']) == 3
            actual = kwargs['visual_pos_masks'][0].nonzero().flatten().tolist()
            assert actual == self.layout['image_positions']
            for i in self.layers:
                self.interventions[i] = self.stack.enter_context(ImageOff(self.model, i, self.layout, self.noop))
        else:
            assert kwargs.get('deepstack_visual_embeds') is None
            assert all(not intervention.handles for intervention in self.interventions.values())
        self.calls.append(dict(step=step, query_length=length, cache_start=start))

    def end(self, module, args, output):
        if len(self.calls) == 1:
            assert all(x.calls == 1 for x in self.interventions.values())
            self.stack.close()

    def __enter__(self):
        lm = self.model.model.language_model
        self.handles = [lm.register_forward_pre_hook(self.begin, with_kwargs=True), lm.register_forward_hook(self.end)]
        return self

    def __exit__(self, *_):
        self.stack.close()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@torch.inference_mode()
def generate(model, processor, inputs, layout, layers=(), noop=False, trace=False, native=False):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config.text_config)
    assert all(cache.get_seq_length(i) == 0 for i in range(36))
    before = hook_counts(model)
    with ExitStack() as stack:
        controller = stack.enter_context(PrefillOnly(model, layout, layers, cache, noop)) if not native else None
        observer = stack.enter_context(PrefillTrace(model, layout['input_length'], layout['image_positions'])) if trace else None
        output = model.generate(**inputs, past_key_values=cache, use_cache=True, do_sample=False,
            num_beams=1, max_new_tokens=32, logits_to_keep=1, return_dict_in_generate=True,
            output_scores=True, output_attentions=False, output_hidden_states=False)
        ids = output.sequences[0, layout['input_length']:].tolist()
        assert ids and len(output.scores) == len(ids)
        assert output.past_key_values is cache
        expected_length = layout['input_length'] + len(ids) - 1
        assert all(cache.get_seq_length(i) == expected_length for i in range(36))
        assert all(torch.isfinite(score).all() for score in output.scores)
        if controller:
            assert len(controller.calls) == len(ids)
            assert all(x.calls == 1 and not x.handles for x in controller.interventions.values())
            audits = {str(i): x.audit for i, x in controller.interventions.items()}
            assert set(controller.interventions) == set(layers)
        else:
            audits = {}
        if observer:
            observer.validate()
            assert observer.mlp_calls == [len(ids)] * 36
        first = output.scores[0][0].detach().float().cpu().clone()
        assert ids[0] == int(first.argmax())
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        result = dict(text=text, prediction=gqa_prediction(text), token_ids=ids, first_token_id=ids[0],
            generated_tokens=len(ids), eos=ids[-1] in eos, truncated=len(ids) == 32 and ids[-1] not in eos,
            empty=not bool(text), mask_audits=audits, masked_layers=list(layers),
            initial_cache_lengths=[0] * 36, final_cache_lengths=[cache.get_seq_length(i) for i in range(36)],
            forward_calls=controller.calls if controller else [], decode_mask_calls=0,
            independent_fresh_cache=True, no_nan_inf_oom=True)
    assert hook_counts(model) == before
    return result, first, observer

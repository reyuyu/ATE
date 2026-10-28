"""Exact no-op and original-implementation GPU gates, before any formal scores."""
import torch

from benchmarks.protocols import prompt_text
from knockout_probe import model_layers
from vlm_common import prepare_inputs
from .hooks import PrefillIntervention, footprint


def gpu_correctness(model, processor, lookup, originals):
    checks = []
    # Input-defined cases: original-Question plus choices/hint, and multi-token candidates.
    selected = [('gqa_chooseattr', '0074716'), ('scienceqa_img', '11201')]
    for key in selected:
        sample = lookup[key]
        inputs, _ = prepare_inputs(model, processor, sample['image_path'], prompt_text(sample))
        mapping = sample['metadata']['input_mapping']
        p, q = mapping['p_minus_1'], mapping['question_positions']
        hooks = footprint(model)
        with torch.inference_mode():
            full = model(**inputs, use_cache=False).logits.detach().cpu()
            text = torch.where(inputs['input_ids'][0] != model.config.image_token_index)[0].cpu()
            for condition in ('Full', 'B4', 'C2'):
                with PrefillIntervention(model, inputs, mapping, condition, originals, noop=True):
                    out = model(**inputs, use_cache=False)
                if not torch.equal(full[:, text], out.logits.detach().cpu()[:, text]):
                    raise AssertionError(f'No-op is not bitwise native: {condition}')
                del out
            with originals['b4'].PathwayMask(model, inputs, q, p, 'B4') as old:
                legacy = model(**inputs, use_cache=False).logits.detach().cpu()
            old.assert_one_forward()
            with PrefillIntervention(model, inputs, mapping, 'B4', originals):
                adapted = model(**inputs, use_cache=False).logits.detach().cpu()
            if not torch.equal(legacy, adapted):
                raise AssertionError('B4 differs from original successful implementation')
            visual = mapping['visual_positions']
            bos = int(torch.where(inputs['input_ids'][0] == model.config.text_config.bos_token_id)[0][0])
            prefix = [i for i in range(p) if i not in set(visual)]
            with originals['c2'].SinkStageTrace(model_layers(model), model.language_model.model.norm,
                    p, bos, visual, q, prefix, [], 'C2') as old_trace:
                legacy = model(**inputs, use_cache=False, output_attentions=True).logits.detach().cpu()
            old_trace.assert_complete()
            with PrefillIntervention(model, inputs, mapping, 'C2', originals):
                adapted = model(**inputs, use_cache=False).logits.detach().cpu()
            if not torch.equal(legacy, adapted):
                raise AssertionError('C2 differs from original successful implementation')
            # Empty-cache prefill must give exactly the same edited logits as cache-free prefill.
            for condition in ('Full', 'B4', 'C2'):
                with PrefillIntervention(model, inputs, mapping, condition, originals):
                    cachefree = model(**inputs, use_cache=False).logits.detach().cpu()
                with PrefillIntervention(model, inputs, mapping, condition, originals):
                    cached = model(**inputs, use_cache=True)
                if not torch.equal(cachefree, cached.logits.detach().cpu()):
                    raise AssertionError('Cache initialization changed prefill logits')
                if cached.past_key_values.get_seq_length() != inputs['input_ids'].shape[1]:
                    raise AssertionError('Cache not preserved/populated')
                del cached
            generation = dict(do_sample=False, num_beams=1, max_new_tokens=4, use_cache=True,
                              return_dict_in_generate=True, output_scores=True, temperature=1., top_p=1.,
                              pad_token_id=processor.tokenizer.eos_token_id)
            native = model.generate(**inputs, **generation)
            for condition in ('B4', 'C2'):
                with PrefillIntervention(model, inputs, mapping, condition, originals, noop=True):
                    no_op = model.generate(**inputs, **generation)
                if not torch.equal(native.sequences, no_op.sequences) or len(native.scores) != len(no_op.scores):
                    raise AssertionError('Generation no-op sequence mismatch')
                if any(not torch.equal(a, b) for a, b in zip(native.scores, no_op.scores)):
                    raise AssertionError('Generation no-op full-vocabulary logits not bitwise native')
                del no_op
            # Exception cleanup must also remove a partially used context.
            try:
                with PrefillIntervention(model, inputs, mapping, 'B4', originals):
                    raise RuntimeError('expected cleanup probe')
            except RuntimeError as error:
                if str(error) != 'expected cleanup probe':
                    raise
            if footprint(model) != hooks:
                raise AssertionError('Hook cleanup failed')
        checks.append({'unit': key[0], 'question_id': key[1], 'passed': True,
            'native_noop_valid_text_logits_bitwise_exact': True, 'legacy_b4_c2_all_logits_bitwise_exact': True,
            'cache_prefill_all_logits_bitwise_exact': True, 'cache_populated': True,
            'generation_noop_all_logits_bitwise_exact': True, 'normal_exception_cleanup': True})
        del full, legacy, adapted, native
    return {'passed': True, 'checks': checks, 'tolerances_relaxed': False}

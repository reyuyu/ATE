"""Exact original C1, no-op, cache lifecycle and exception-cleanup smoke."""
import torch

from benchmarks.protocols import prompt_text
from knockout_probe import model_layers
from vlm_common import prepare_inputs
from benchmark_interventions.hooks import footprint
from .hooks import AttentionPrefill


def correctness(model, processor, lookup, jobs, originals):
    # First frozen local multi-token ChooseAttr and maximum-choice ScienceQA.
    # Include one question from each of these units even if extras overlap.
    selected = [next(j for j in reversed(jobs) if j['unit'] == unit)
                for unit in ('gqa_chooseattr', 'scienceqa_img')]
    checks = []
    for job in selected:
        sample = lookup[(job['unit'], job['question_id'])]
        inputs, _ = prepare_inputs(model, processor, sample['image_path'], prompt_text(sample))
        mapping = sample['metadata']['input_mapping']
        p, q, visual = mapping['p_minus_1'], mapping['question_positions'], mapping['visual_positions']
        hooks = footprint(model)
        with torch.inference_mode():
            native = model(**inputs, use_cache=False).logits.detach().cpu()
            with AttentionPrefill(model, inputs, mapping, originals, noop=True):
                noop = model(**inputs, use_cache=False).logits.detach().cpu()
            if not torch.equal(native, noop):
                raise AssertionError('C1 no-op all-logits not bitwise native')
            with AttentionPrefill(model, inputs, mapping, originals) as trace:
                current = model(**inputs, use_cache=False).logits.detach().cpu()
            if len(trace.records) != 3 or any(not e['target_zero'] or not e['other_tokens_bitwise_exact'] for e in trace.records):
                raise AssertionError('C1 local edit invalid')
            bos = int(torch.where(inputs['input_ids'][0] == model.config.text_config.bos_token_id)[0][0])
            prefix = [i for i in range(p) if i not in set(visual)]
            with originals['c2'].SinkStageTrace(model_layers(model), model.language_model.model.norm,
                    p, bos, visual, q, prefix, [], 'C1') as legacy:
                old = model(**inputs, use_cache=False, output_attentions=True).logits.detach().cpu()
            legacy.assert_complete()
            if not torch.equal(current, old):
                raise AssertionError('C1 not exactly original Phase3.8')
            with AttentionPrefill(model, inputs, mapping, originals) as cached_trace:
                out = model(**inputs, use_cache=True)
            if not torch.equal(current, out.logits.detach().cpu()) or out.past_key_values.get_seq_length() != inputs['input_ids'].shape[1]:
                raise AssertionError('Cached C1 prefill differs or loses native cache')
            del out
            generation = dict(do_sample=False, num_beams=1, max_new_tokens=4, use_cache=True,
                return_dict_in_generate=True, output_scores=True, temperature=1., top_p=1.,
                pad_token_id=processor.tokenizer.eos_token_id)
            generated = model.generate(**inputs, **generation)
            with AttentionPrefill(model, inputs, mapping, originals, noop=True):
                noop_generated = model.generate(**inputs, **generation)
            if not torch.equal(generated.sequences, noop_generated.sequences) or len(generated.scores) != len(noop_generated.scores):
                raise AssertionError('C1 generation no-op token mismatch')
            if any(not torch.equal(a, b) for a, b in zip(generated.scores, noop_generated.scores)):
                raise AssertionError('C1 generation no-op logits mismatch')
            try:
                with AttentionPrefill(model, inputs, mapping, originals):
                    raise RuntimeError('expected C1 cleanup probe')
            except RuntimeError as error:
                if str(error) != 'expected C1 cleanup probe':
                    raise
            if footprint(model) != hooks:
                raise AssertionError('C1 hooks leak')
        checks.append({'unit': job['unit'], 'question_id': job['question_id'], 'passed': True,
            'original_c1_all_logits_bitwise_exact': True, 'noop_all_logits_bitwise_exact': True,
            'cached_prefill_bitwise_exact': True, 'generation_noop_all_logits_bitwise_exact': True,
            'zeroed_only_L2_L3_L4_o_proj_original_P1': True, 'non_target_tokens_exact': True,
            'cache_preserved': True, 'normal_exception_hook_cleanup': True})
    return {'passed': True, 'checks': checks, 'tolerances_relaxed': False}

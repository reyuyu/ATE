from contextlib import ExitStack
import unittest

import torch

from benchmark_interventions.hooks import ORIGINALS, original_modules
from benchmarks.dataset_registry import sha256
from benchmarks.smoke import hook_footprint
from knockout_probe import model_layers
from residual_only_pilot.hooks import CONDITIONS, LAYERS, ResidualOnly
from residual_only_pilot.run import select
from test_knockout import tiny_model, tiny_inputs


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.zero = staticmethod(original_modules({k: sha256(p) for k, p in ORIGINALS.items()})['c2'].zero_query)
        cls.model = tiny_model()

    def call(self, inputs, condition, prompt=None, **kwargs):
        ids = prompt or inputs['input_ids'][0].tolist()
        with torch.inference_mode(), ResidualOnly(self.model, ids, condition, self.zero, **kwargs) as trace:
            output = self.model(**inputs, use_cache=False).logits
        return output, trace

    def test_noop_exact(self):
        inputs = tiny_inputs()
        base, _ = self.call(inputs, 'Full')
        for condition in CONDITIONS[1:]:
            output, _ = self.call(inputs, condition, noop=True)
            self.assertTrue(torch.equal(output, base))

    def test_joint_fixed_query_locality_and_residual_identity(self):
        inputs = tiny_inputs()
        base, full = self.call(inputs, 'Full', observe=True)
        output, trace = self.call(inputs, 'P1-Joint', observe=True)
        self.assertEqual(full.prefix, trace.prefix)
        self.assertEqual(len(trace.edits), 6)
        self.assertTrue(all(r['other_tokens_bitwise_exact'] and r['target_zero'] for r in trace.edits))
        self.assertTrue(all(r['target_residual_exact'] for r in trace.identities))
        self.assertFalse(torch.equal(output[:, -1], base[:, -1]))

    def test_whole_skip_no_computation_and_restore(self):
        inputs = tiny_inputs()
        before = [dict(layer.__dict__).get('forward') for layer in model_layers(self.model)]
        _, trace = self.call(inputs, 'Whole-Skip')
        self.assertTrue(all(n == 0 for n in trace.calls.values()))
        self.assertTrue(all(r['all_tokens_identity'] for r in trace.identities))
        self.assertEqual(before, [dict(layer.__dict__).get('forward') for layer in model_layers(self.model)])

    def test_both_match_independent_computed_zero_reference(self):
        inputs = tiny_inputs()
        for condition in CONDITIONS[1:]:
            actual, _ = self.call(inputs, condition)
            with ExitStack() as stack, torch.inference_mode():
                for i in LAYERS:
                    layer = model_layers(self.model)[i]
                    for module in (layer.self_attn.o_proj, layer.mlp):
                        def zero(module, args, out, whole=condition == 'Whole-Skip'):
                            result = out.clone()
                            if whole:
                                result.zero_()
                            else:
                                result[:, -1] = 0
                            return result
                        stack.callback(module.register_forward_hook(zero).remove)
                expected = self.model(**inputs, use_cache=False).logits
            self.assertTrue(torch.equal(expected, actual))

    def test_original_query_not_new_decode_position(self):
        inputs = tiny_inputs()
        ids = inputs['input_ids'][0].tolist()
        from attention_position_probe import append_answer_inputs
        for additions in ([], [4], [4, 5]):
            extended = append_answer_inputs(inputs, additions)
            _, trace = self.call(extended, 'P1-Joint', prompt=ids)
            self.assertEqual({r['query'] for r in trace.edits}, {len(ids)-1})

    def test_exception_restores_hooks_and_forward(self):
        inputs = tiny_inputs()
        base, _ = self.call(inputs, 'Full')
        before = hook_footprint(self.model)
        for condition in CONDITIONS[1:]:
            with self.assertRaisesRegex(RuntimeError, 'expected'):
                with ResidualOnly(self.model, inputs['input_ids'][0].tolist(), condition, self.zero):
                    raise RuntimeError('expected')
            self.assertEqual(before, hook_footprint(self.model))
            after, _ = self.call(inputs, 'Full')
            self.assertTrue(torch.equal(base, after))

    def test_actual_forward_exception_cleanup(self):
        inputs = tiny_inputs()
        before = hook_footprint(self.model)
        def fail(module, args):
            raise RuntimeError('forward expected')
        for condition in CONDITIONS[1:]:
            handle = model_layers(self.model)[5].register_forward_pre_hook(fail)
            try:
                with self.assertRaisesRegex(RuntimeError, 'forward expected'):
                    self.call(inputs, condition)
            finally:
                handle.remove()
            self.assertEqual(before, hook_footprint(self.model))

    def test_cache_and_wrong_prompt_rejected(self):
        inputs = tiny_inputs()
        ids = inputs['input_ids'][0].tolist()
        for condition in CONDITIONS:
            with self.assertRaises(ValueError):
                with ResidualOnly(self.model, ids, condition, self.zero):
                    self.model(**inputs, use_cache=True)
            with self.assertRaises(ValueError):
                with ResidualOnly(self.model, [99]+ids[1:], condition, self.zero):
                    self.model(**inputs, use_cache=False)


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [{'question_id': str(i).zfill(5), 'image_sha256': str(i//2),
                 'answer': 'left' if i < 80 else 'right'} for i in range(160)]

    def test_balanced_reproducible_order_independent(self):
        rows = self.rows()
        selected = select(rows, 32, balanced=True)
        self.assertEqual(selected, select(list(reversed(rows)), 32, balanced=True))
        self.assertEqual(sum(r['answer'] == 'left' for r in selected), 16)
        self.assertEqual(len({r['image_sha256'] for r in selected}), 32)

    def test_no_selection_by_predictions(self):
        rows = self.rows()
        ids = [r['question_id'] for r in select(rows, 16)]
        for row in rows:
            row['model_correct'] = row['question_id'] in ids
        self.assertEqual(ids, [r['question_id'] for r in select(rows, 16)])

    def test_insufficient_images_raise(self):
        with self.assertRaises(ValueError):
            select(self.rows()[:4], 16)


if __name__ == '__main__':
    unittest.main()

import copy
import unittest
from unittest.mock import patch

import torch

from visual_exit_cross_dataset.run import generate, select_samples, step_layout, score


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [{'question_id': str(i).zfill(4), 'image_sha256': str(i//2)} for i in range(100)]

    def test_fixed_seed_order_independent_unique_sha(self):
        rows = self.rows()
        selected = select_samples(rows)
        self.assertEqual(selected, select_samples(list(reversed(rows))))
        self.assertEqual(len(selected), 32)
        self.assertEqual(len({r['image_sha256'] for r in selected}), 32)

    def test_not_prediction_based(self):
        rows = self.rows()
        with_predictions = [dict(r, score=0 if i % 3 else 1) for i, r in enumerate(rows)]
        self.assertEqual([r['question_id'] for r in select_samples(rows)],
                         [r['question_id'] for r in select_samples(with_predictions)])

    def test_insufficient_distinct_images_stops(self):
        with self.assertRaises(ValueError):
            select_samples(self.rows()[:20])


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.prompt = {'input_ids': torch.tensor([[1]+[99]*576+[10, 11]]),
                       'attention_mask': torch.ones(1, 579, dtype=torch.long),
                       'pixel_values': torch.ones(1, 3, 2, 2)}
        self.model = type('Model', (), {'config': type('Config', (), {'image_token_index': 99})()})()
        tokenizer = type('Tokenizer', (), {'eos_token_id': 2, 'decode': lambda self, ids, **kw: 'answer'})()
        self.processor = type('Processor', (), {'tokenizer': tokenizer})()

    def output(self, inputs, cutoff, token):
        n = inputs['input_ids'].shape[1]
        positions = torch.arange(n) if cutoff == 32 else torch.where(inputs['input_ids'][0] != 99)[0]
        logits = torch.zeros(1, len(positions), 128)
        logits[0, -1, token] = 4
        trace = [{'layer': i, 'sequence_length': n if i < cutoff else n-576,
                  'mask_shape': [1, 1, n, n] if i < cutoff else [1, 1, n-576, n-576]} for i in range(32)]
        return {'original_positions': positions, 'physical_deletion': cutoff < 32,
                'original_text_position_ids': positions.tolist(), 'layers': trace,
                'logits': logits, 'hidden': torch.ones(1, len(positions), 4), 'measurement': {}}

    def test_full_prefix_recomputed_with_same_cutoff_eos_stops(self):
        for condition in ('Full', 'K16', 'K24'):
            seen = []

            def forward(model, inputs, mode, cutoff):
                seen.append((inputs['input_ids'].clone(), cutoff, mode))
                return self.output(inputs, cutoff, 20 if len(seen) == 1 else 2)

            with patch('visual_exit_cross_dataset.run.measured_forward', forward):
                result = generate(self.model, self.processor, self.prompt, condition)
            self.assertEqual(result['token_ids'], [20, 2])
            self.assertEqual([r[0].shape[1] for r in seen], [579, 580])
            self.assertTrue(torch.equal(seen[1][0][:, :579], self.prompt['input_ids']))
            self.assertEqual(len({r[1] for r in seen}), 1)
            self.assertFalse(result['use_cache'])

    def test_maximum_32_tokens(self):
        with patch('visual_exit_cross_dataset.run.measured_forward',
                   lambda model, inputs, mode, cutoff: self.output(inputs, cutoff, 20)):
            result = generate(self.model, self.processor, self.prompt, 'K24')
        self.assertEqual(len(result['token_ids']), 32)
        self.assertTrue(result['length_limit_reached'])
        self.assertFalse(result['eos_observed'])

    def test_layout_rejects_fake_mask_only_or_renumbering(self):
        output = self.output(self.prompt, 16, 2)
        output['physical_deletion'] = False
        with self.assertRaises(AssertionError):
            step_layout(output, self.prompt, 'K16', 99)
        output = self.output(self.prompt, 24, 2)
        output['original_text_position_ids'] = list(range(3))
        with self.assertRaises(AssertionError):
            step_layout(output, self.prompt, 'K24', 99)

    def test_scienceqa_letter_and_textvqa_soft_score(self):
        science = {'dataset': 'scienceqa_img', 'candidates': ['A', 'B'],
                   'metadata': {'gold_answer_index': 1}}
        self.assertEqual(score(science, {'text': 'B'})[0], 1.)
        self.assertEqual(score(science, {'text': 'wrong'})[0], 0.)
        text = {'dataset': 'textvqa', 'answer': ['yes']*3+['no']*7}
        self.assertAlmostEqual(score(text, {'text': 'yes'})[0], .9)


if __name__ == '__main__':
    unittest.main()

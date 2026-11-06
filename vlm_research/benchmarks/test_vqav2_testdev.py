import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from .dataset_registry import (audit_rows, official_registry, official_benchmark_path,
                               validate_sample, load_prepared_benchmark, sha256)
from .vqav2_submission import convert, question_id
from ..seven_benchmark_quality.common import ORDER, COUNTS, score


class TestDevTests(unittest.TestCase):
    def hidden_row(self):
        return dict(dataset='VQAv2', subset='testdev', split='test-dev2015',
                    question_id='1000', image_id='1', image_path='/tmp/COCO_test2015_000000000001.jpg',
                    question='What?', candidates=None, answer_type='hidden', source_revision='fixed',
                    source_row_sha256='abc', image_sha256='def', metadata={},
                    gold_available=False, official_server_scoring_required=True)

    def test_hidden_schema_without_answer(self):
        row = self.hidden_row()
        self.assertEqual(validate_sample(row), row)
        self.assertEqual(audit_rows([row])['answers'], {})

    def test_hidden_schema_rejects_gold(self):
        for key in ('answer', 'answers', 'gold', 'lecture', 'solution'):
            with self.assertRaises(ValueError):
                validate_sample(dict(self.hidden_row(), **{key: 'not permitted'}))

    def test_public_schema_still_requires_gold(self):
        row = dict(self.hidden_row(), answer_type='open', gold_available=True)
        with self.assertRaises(ValueError):
            validate_sample(row)
        validate_sample(dict(row, answer='red'))

    def test_hidden_metadata_cannot_contain_nested_gold(self):
        with self.assertRaises(ValueError):
            validate_sample(dict(self.hidden_row(), metadata={'original': [{'answers': ['red']}]}))

    def test_hidden_loader_integrity_and_no_answers_file(self):
        with TemporaryDirectory(prefix='vqav2-schema-test-') as directory:
            registry = official_registry()
            registry['data_root'] = directory
            registry['benchmarks']['vqav2_testdev']['expected_questions'] = 1
            root = Path(directory) / 'VQAv2-testdev2015'
            image = root / 'images/test2015/COCO_test2015_000000000001.jpg'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (2, 2), 'red').save(image)
            row = dict(self.hidden_row(), image_path=str(image), image_sha256=sha256(image))
            (root / 'inputs.jsonl').write_text(json.dumps(row) + '\n')
            info = {k: row[k] for k in ('image_id', 'image_path', 'image_sha256')}
            (root / 'image_manifest.json').write_text(json.dumps([info]))
            (root / 'source_manifest.json').write_text('{}')
            audit = dict(status='COMPLETE', split='test-dev2015', actual_questions=1,
                         gold_available=False, official_server_scoring_required=True,
                         inputs_sha256=sha256(root / 'inputs.jsonl'),
                         image_manifest_sha256=sha256(root / 'image_manifest.json'),
                         source_manifest_sha256=sha256(root / 'source_manifest.json'))
            (root / 'dataset_audit.json').write_text(json.dumps(audit))
            with patch('vlm_research.benchmarks.dataset_registry.official_registry', return_value=registry):
                self.assertEqual(load_prepared_benchmark('vqav2_testdev'), [row])
                (root / 'source_manifest.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'source manifest SHA'):
                    load_prepared_benchmark('vqav2_testdev')
                (root / 'source_manifest.json').write_text('{}')
                (root / 'answers.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'must not contain answers'):
                    load_prepared_benchmark('vqav2_testdev')

    def test_hidden_flags_must_be_explicit(self):
        for key in ('gold_available', 'official_server_scoring_required'):
            row = self.hidden_row()
            row.pop(key)
            with self.assertRaises(ValueError):
                validate_sample(row)

    def test_registry_queue_has_only_testdev_vqa(self):
        registry = official_registry()
        self.assertIn('vqav2_testdev', ORDER)
        self.assertEqual(COUNTS['vqav2_testdev'], 107394)
        self.assertNotIn('VQAv2', ORDER)
        self.assertNotIn('vqav2_val2014_diagnostic', ORDER)
        self.assertEqual(official_benchmark_path('vqav2_testdev').name, 'VQAv2-testdev2015')
        with self.assertRaises(ValueError):
            official_benchmark_path('vqav2_val2014_diagnostic')

    def test_hidden_never_gets_local_score(self):
        sample = dict(benchmark='vqav2_testdev', gold_available=False)
        self.assertIsNone(score(sample, 'red')['score'])
        with self.assertRaises(ValueError):
            score(dict(sample, answer='fake'), 'red')

    def test_real_export_requires_all_predictions(self):
        with self.assertRaises(ValueError):
            convert([dict(question_id=1, text='red')], [1, 2])

    def test_full_wrapper_padding_is_not_inference(self):
        output, audit = convert([dict(question_id='1', text='Two'), dict(question_id=2, answer='Red')],
                                [1, 2], [1, 2, 3, 4])
        self.assertEqual(output, [dict(question_id=1, answer='2'), dict(question_id=2, answer='red'),
                                  dict(question_id=3, answer=''), dict(question_id=4, answer='')])
        self.assertEqual(audit['actual_prediction_records'], 2)
        self.assertEqual(audit['official_submission_records'], 4)
        self.assertEqual(audit['padding_records'], 2)

    def test_duplicate_unknown_and_invalid_prediction_rejected(self):
        values = [
            [dict(question_id=1, text='red'), dict(question_id='01', text='red')],
            [dict(question_id=3, text='red')], [dict(question_id=1, text='')],
            [dict(question_id=True, text='red')], [dict(question_id=1.0, text='red')],
            [dict(question_id=1, text='red', answer='red')],
        ]
        for records in values:
            with self.assertRaises(ValueError):
                convert(records, [1])

    def test_synthetic_is_explicit_and_not_uploadable(self):
        fixture = dict(question_id=1, text='test fixture', synthetic_for_structure_only=True)
        with self.assertRaises(ValueError):
            convert([fixture], [1])
        output, audit = convert([fixture], [1, 2], synthetic=True)
        self.assertFalse(audit['upload_allowed'])
        self.assertFalse(audit['submitted'])
        self.assertTrue(audit['synthetic_for_structure_only'])
        self.assertEqual(audit['actual_prediction_records'], 1)


if __name__ == '__main__':
    unittest.main()

"""CPU schema/scorer/sampling regression; no predictions required or selected."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from benchmarks.dataset_registry import (audit_image, audit_rows, digest, freeze_json, load_manifest,
    select_groups, select_unique, sha256, validate_sample)
from benchmarks.gqa_adapter import read_gqa
from benchmarks.pope_adapter import read_pope
from benchmarks.scienceqa_adapter import MATCH_FIELDS, scienceqa_rows, verify_publisher_order
from benchmarks.textvqa_adapter import textvqa_rows
from benchmarks.scoring import (diagnostic_candidate_metrics, generated_score, pope_answer, pope_metrics,
    scienceqa_index, textvqa_score)
from benchmarks.protocols import prompt_text, protocols


def row(q='001', image='im', answer='yes'):
    return dict(dataset='pope', subset='random', split='val', question_id=q, image_id=image,
                image_path='/missing.jpg', question='Is it there?', answer=answer, candidates=['no', 'yes'],
                answer_type='closed', source_revision='abc123', source_row_sha256=digest(q), image_sha256=None,
                metadata={'resource_relative_path': image+'.jpg'})


class SuiteTests(unittest.TestCase):
    def test_required_fields(self):
        r = row()
        del r['question']
        with self.assertRaises(ValueError):
            validate_sample(r)

    def test_gold_mapping(self):
        r = row(answer='maybe')
        with self.assertRaises(ValueError):
            validate_sample(r)

    def test_open_cannot_have_candidates(self):
        r = row()
        r['answer_type'] = 'open'
        with self.assertRaises(ValueError):
            validate_sample(r)

    def test_duplicate_question_ids(self):
        with self.assertRaises(ValueError):
            audit_rows([row(), row()])

    def test_deterministic_freeze(self):
        rows = [row(str(i), str(i)) for i in range(30)]
        self.assertEqual(select_unique(rows, 10), select_unique(list(reversed(rows)), 10))
        self.assertEqual(len({r['image_id'] for r in select_unique(rows, 10)}), 10)

    def test_group_sampling_not_split(self):
        rows = [row(str(i), str(i//6)) for i in range(60)]
        selected = select_groups(rows, 7)
        self.assertEqual(len(selected), 12)
        self.assertEqual({r['question_id'] for r in selected},
                         {r['question_id'] for r in rows if r['image_id'] in {s['image_id'] for s in selected}})

    def test_balance_labels_and_unique_images(self):
        rows = [row(str(i), str(i), 'no' if i % 2 else 'yes') for i in range(30)]
        s = select_unique(rows, 10, balanced=True)
        self.assertEqual(sum(r['answer'] == 'yes' for r in s), 5)
        self.assertEqual(sum(r['answer'] == 'no' for r in s), 5)

    def test_insufficient_not_silently_replaced(self):
        with self.assertRaises(ValueError):
            select_unique([row()], 10)

    def test_historical_exclusion_preference(self):
        rows = [row(str(i), str(i)) for i in range(12)]
        selected = select_unique(rows, 5, excluded={'0', '1', '2', '3'})
        self.assertFalse({r['image_id'] for r in selected} & {'0', '1', '2', '3'})

    def test_missing_corrupt_dimension_and_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'a.jpg'
            self.assertEqual(audit_image(p)['image_status'], 'missing')
            with p.open('wb') as f:
                f.write(b'not a jpeg')
            self.assertEqual(audit_image(p)['image_status'], 'corrupt')
            Image.new('RGB', (16, 12)).save(p)
            self.assertEqual(audit_image(p, [16, 12])['image_status'], 'verified')
            self.assertEqual(audit_image(p, [12, 16])['image_status'], 'corrupt')

    def test_frozen_manifest_and_recovery_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'manifest.json'
            value = {'samples': [row()], 'sources': []}
            value['binding_sha256'] = digest(value)
            freeze_json(p, value)
            freeze_json(p, value)
            self.assertEqual(load_manifest(p, verify_images=False)['samples'][0]['question_id'], '001')
            with self.assertRaises(ValueError):
                freeze_json(p, dict(value, new='resample'))

    def test_source_hash_revision_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.json'
            freeze_json(source, {'version': 1})
            body = {'samples': [row()], 'sources': [{'path': str(source), 'sha256': sha256(source), 'revision': 'v1'}]}
            body['binding_sha256'] = digest(body)
            m = Path(tmp)/'m.json'
            freeze_json(m, body)
            with source.open('w') as f:
                json.dump({'version': 2}, f)
            with self.assertRaises(ValueError):
                load_manifest(m, verify_images=False)

    def test_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = {'samples': [row()], 'sources': []}
            value['binding_sha256'] = digest(value)
            value['samples'][0]['question_id'] = 'changed'
            m = Path(tmp)/'m.json'
            freeze_json(m, value)
            with self.assertRaises(ValueError):
                load_manifest(m, verify_images=False)

    def test_csv_quoted_comma_and_leading_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'q.csv'
            data = {'question_id': '001', 'img_id': '022', 'question': 'Is it red, or blue?',
                    'answer': 'blue', 'true option': 'blue', 'false option': 'red',
                    'operations': "[{'operation': 'choose color', 'argument': 'red|blue'}]",
                    'image_w': '20', 'image_h': '10'}
            with p.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(data))
                writer.writeheader()
                writer.writerow(data)
            parsed = read_gqa(p, 'gqa_chooseattr', 'revision', tmp)[0]
            self.assertEqual(parsed['question_id'], '001')
            self.assertEqual(parsed['question'], data['question'])
            self.assertEqual(parsed['candidates'], ['red', 'blue'])

    def test_queryattr_is_open_even_left_right(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'q.csv'
            data = {'question_id': '0001', 'img_id': '1', 'question': 'Which side?', 'answer': 'left', 'image_w': '1', 'image_h': '1'}
            with p.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(data))
                w.writeheader()
                w.writerow(data)
            r = read_gqa(p, 'gqa_queryattr', 'rev', tmp)[0]
            self.assertIsNone(r['candidates'])
            self.assertEqual(r['answer_type'], 'open')

    def test_pope_jsonl_and_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'q.json'
            good = {'question_id': 1, 'image': 'COCO_val2014_000000000042.jpg', 'text': 'Cat?', 'label': 'yes'}
            with p.open('w') as f:
                f.write(json.dumps(good)+'\n')
            self.assertEqual(read_pope(p, 'random', 'rev', tmp)[0]['answer'], 'yes')
            good['image'] = 'COCO_val2017_000000000042.jpg'
            with p.open('w') as f:
                f.write(json.dumps(good)+'\n')
            with self.assertRaises(ValueError):
                read_pope(p, 'random', 'rev', tmp)

    def test_scienceqa_split_image_and_gold_index(self):
        problems = {'01': {'question': 'Which?', 'image': 'image.png', 'choices': ['same', 'same'], 'answer': 1},
                    '02': {'question': 'Text?', 'image': None}}
        r = scienceqa_rows(problems, {'test': ['01', '02']}, 'rev', '/data')[0]
        self.assertEqual(r['answer'], 'B')
        self.assertEqual(r['metadata']['choices'], ['same', 'same'])
        self.assertEqual(r['split'], 'test')
        self.assertNotIn('solution', prompt_text(r))

    def test_scienceqa_invalid_index(self):
        with self.assertRaises(ValueError):
            scienceqa_rows({'1': {'question': 'Which?', 'image': 'i.png', 'choices': ['x', 'y'], 'answer': 2}},
                          {'test': ['1']}, 'rev', '/data')

    def test_scienceqa_publisher_mapping_checks_entire_order(self):
        a = {k: 'a' for k in MATCH_FIELDS}
        a.update(split='test', image='image.png')
        b = dict(a, question='different')
        mirrors = [dict(a, image={'bytes': b'a'}), dict(b, image={'bytes': b'b'})]
        self.assertEqual(list(verify_publisher_order({'01': a, '02': b}, ['01', '02'], mirrors)), ['01', '02'])
        with self.assertRaises(ValueError):
            verify_publisher_order({'01': a, '02': b}, ['02', '01'], mirrors)
        with self.assertRaises(ValueError):
            verify_publisher_order({'01': a, '02': b}, ['01', '02'], list(reversed(mirrors)))

    def test_textvqa_versions_and_10_votes(self):
        r = {'set_name': 'val', 'question_id': 1, 'image_id': 'ab', 'question': 'Word?',
             'image_width': 20, 'image_height': 10, 'answers': ['red']*10}
        rows = textvqa_rows({'dataset_version': '0.5.1', 'data': [r]}, 'v0.5.1', '/data')
        self.assertEqual(rows[0]['answer'], ['red']*10)
        self.assertIsNone(rows[0]['candidates'])
        with self.assertRaises(ValueError):
            textvqa_rows({'dataset_version': '0.5', 'data': [r]}, 'rev', '/data')

    def test_textvqa_normalization_and_leave_one_out(self):
        self.assertEqual(textvqa_score('The TWO!', ['2']*10), 1.0)
        for count, score in [(1, .3), (2, .6), (3, .9), (4, 1.0)]:
            self.assertAlmostEqual(textvqa_score('cat', ['cat']*count + ['dog']*(10-count)), score)
        with self.assertRaises(ValueError):
            textvqa_score('cat', ['cat'])

    def test_pope_lexical_rule_matches_upstream(self):
        self.assertEqual(pope_answer('No, there is not.'), 'no')
        self.assertEqual(pope_answer('Yes. There is not a dog.'), 'yes')
        self.assertEqual(pope_answer('NO'), 'yes')  # Upstream lexical limitation, not silently fixed.

    def test_all_pope_metrics_and_alignment(self):
        x = pope_metrics(['Yes', 'No', 'Yes', 'No'], ['yes', 'yes', 'no', 'no'])
        self.assertEqual(x['official_generation_accuracy'], .5)
        self.assertEqual(x['official_generation_f1'], .5)
        self.assertEqual(x['official_generation_yes_ratio'], .5)
        with self.assertRaises(ValueError):
            pope_metrics(['yes'], ['yes', 'no'])
        self.assertTrue(pope_metrics(['no'], ['no'])['undefined_denominators']['precision'])

    def test_scienceqa_parser_pinned_formats(self):
        self.assertEqual(scienceqa_index('B', 3), 1)
        self.assertEqual(scienceqa_index('The answer is C.', 3), 2)
        self.assertEqual(scienceqa_index('A. blue', 3), 0)
        self.assertEqual(scienceqa_index('E', 3), -1)
        self.assertEqual(scienceqa_index('unclear', 3), -1)

    def test_multichoice_strongest_wrong_and_tie(self):
        r = row(answer='yes')
        r['candidates'] = ['no', 'yes', 'maybe']
        v = diagnostic_candidate_metrics(r, [-3, -1, -2])
        self.assertEqual(v['diagnostic_candidate_margin'], 1)
        self.assertFalse(diagnostic_candidate_metrics(r, [-1, -1, -2])['diagnostic_candidate_correct'])

    def test_official_diagnostic_isolation(self):
        r = row()
        metrics = diagnostic_candidate_metrics(r, [-2, -1])
        self.assertFalse(any(k.startswith('official') for k in metrics))
        with self.assertRaises(TypeError):
            generated_score(r, [-2, -1])
        r.update(candidates=None, answer_type='open')
        with self.assertRaises(ValueError):
            diagnostic_candidate_metrics(r, [-2, -1])

    def test_complete_sequence_logp_not_first_token(self):
        import torch
        from physical_exit_pilot import sequence_score
        logits = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
        expected = (logits.double().log_softmax(-1)[0, 1] + logits.double().log_softmax(-1)[1, 2]).item()
        self.assertEqual(sequence_score(logits, [1, 2]), expected)
        self.assertNotEqual(sequence_score(logits, [1, 2]), sequence_score(logits[:1], [1]))

    def test_protocols_ocr_and_intervention_gate(self):
        p = protocols()
        self.assertFalse(p['datasets']['gqa_queryattr']['diagnostic_candidate_enabled'])
        self.assertIn('no OCR input', p['datasets']['textvqa']['ocr_policy'])
        self.assertIn('NOT validated', p['common']['future_intervention_gate'])


if __name__ == '__main__':
    unittest.main()

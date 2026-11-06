import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).with_name('prepare.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PreparationTests(unittest.TestCase):
    def test_docvqa_expands_without_losing_answers(self):
        row = dict(image={'path': 'original.png'}, qa=[
            {'question': 'A?', 'answers': ['one', 'ONE']},
            {'question': 'B?', 'answers': ['two']}])
        media = [dict(image_id='abc', image_path='/tmp/abc.png', sha256='abc')]
        pairs = list(p.normalized_questions('DocVQA-val', 'validation', 3, row, media))
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][1], ['one', 'ONE'])
        self.assertNotEqual(pairs[0][0]['question_id'], pairs[1][0]['question_id'])
        self.assertNotIn('answers', pairs[0][0])
        self.assertEqual(pairs[0][0]['image_path'], pairs[1][0]['image_path'])

    def test_info_keeps_all_media_and_id(self):
        row = dict(id='source-12', question_type='layout', answer=['x', 'X'],
                   messages=json.dumps([dict(role='user', content=[
                       {'type': 'image'}, {'type': 'text', 'text': 'Which?'}])]))
        media = [dict(image_id=h, image_path='/tmp/' + h, sha256=h) for h in ('a', 'b')]
        record, answers = next(p.normalized_questions('InfoVQA-val', 'validation', 1, row, media))
        self.assertEqual(len(record['images']), 2)
        self.assertEqual(record['metadata']['original_id'], 'source-12')
        self.assertEqual(answers, ['x', 'X'])
        self.assertNotIn('answers', record)

    def test_invalid_answer_never_silently_filtered(self):
        for value in ([], [''], ['valid', ''], [None], 'text'):
            with self.assertRaises(ValueError):
                p.valid_answers(value)

    def test_unknown_message_content_fails(self):
        with self.assertRaises(ValueError):
            p.question_from_messages([dict(role='user', content=[dict(type='unknown')])])

    def test_actual_infovqa_question_field_schema(self):
        value = json.dumps([dict(role='user', question='How many?', hint='', options={}, choices=[])])
        self.assertEqual(p.question_from_messages(value), 'How many?')
        with self.assertRaises(ValueError):
            p.question_from_messages([dict(role='user', question='A?', content='B?')])

    def test_existing_conflict_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.json'
            p.save(path, {'a': 1})
            p.save(path, {'a': 1})
            with self.assertRaises(FileExistsError):
                p.save(path, {'a': 2})

    def test_lossless_multi_image_export_and_rerun(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        blobs = []
        for color in ('red', 'blue'):
            f = io.BytesIO()
            Image.new('RGB', (12, 16), color).save(f, format='PNG')
            blobs.append(f.getvalue())
        rows = [dict(id='same-original-id', question_type='layout', answer=['A', 'a'],
                     messages=json.dumps([dict(role='user', content='Question?')]),
                     media=[dict(bytes=b, path=f'{i}.png') for i, b in enumerate(blobs)]),
                dict(id='same-original-id', question_type='layout', answer=['B'],
                     messages=json.dumps([dict(role='user', content='Question?')]),
                     media=[dict(bytes=blobs[0], path='another-name.png')])]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / 'validation-00000-of-00001.parquet'
            pq.write_table(pa.Table.from_pylist(rows), parquet)
            guard = p.Guard(root, root, 0)
            for _ in range(2):
                result = p.export_dataset('InfoVQA-val', 'validation', 2, [parquet], root, guard,
                                          dict(dataset='InfoVQA-val', split='validation', expected_qa=2,
                                               local_path=str(root), source_files=[dict(
                                                   local_path=str(parquet), bytes=parquet.stat().st_size,
                                                   sha256=p.sha(parquet), filename='data/validation-00000-of-00001.parquet')]))
                self.assertEqual(result['actual_qa'], 2)
                self.assertEqual(result['unique_images'], 2)
                self.assertEqual(result['image_references'], 3)
                from finalize import verify
                self.assertTrue(verify(result)['original_parquet_field_and_media_equivalence'])
            for info in json.loads((root / 'image_manifest.json').read_text()):
                self.assertIn(Path(info['image_path']).read_bytes(), blobs)
            inputs = [json.loads(line) for line in (root / 'inputs.jsonl').read_text().splitlines()]
            self.assertNotEqual(inputs[0]['question_id'], inputs[1]['question_id'])
            self.assertTrue(all('answers' not in r for r in inputs))
            self.assertEqual(len(inputs[0]['images']), 2)

    def test_disk_floor_blocks_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = p.Guard(Path(tmp), Path(tmp), 2**63)
            with self.assertRaises(p.SpaceError):
                guard.check('must-stop', 1)


if __name__ == '__main__':
    unittest.main()

"""Review guards for dataset namespaces and independent multi-token scoring."""
import unittest
import numpy as np

from benchmarks.publication import image_overlap, logp
from benchmarks.fetch_resources import RangeReader


class Response:
    def __init__(self, status, headers, data=b''):
        import io
        self.status_code, self.headers, self.raw = status, headers, io.BytesIO(data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ValueError('HTTP error')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Session:
    def __init__(self, ignore_range=False):
        self.ignore_range = ignore_range

    def head(self, *args, **kwargs):
        return Response(200, {'Content-Length': '5', 'ETag': 'pinned'})

    def get(self, *args, **kwargs):
        start, end = map(int, kwargs['headers']['Range'][6:].split('-'))
        return Response(200 if self.ignore_range else 206,
                        {'ETag': 'pinned', 'Content-Range': f'bytes {start}-{end}/5'}, b'abcde'[start:end+1])


class PublicationTests(unittest.TestCase):
    def test_numeric_id_collision_is_not_same_image(self):
        a = [{'dataset': 'gqa_chooseattr', 'image_id': '2993', 'image_sha256': 'first'}]
        b = [{'dataset': 'scienceqa_img', 'image_id': '2993', 'image_sha256': 'second'}]
        result = image_overlap(a, b)
        self.assertEqual(result['same_namespace_image_ids'], [])
        self.assertEqual(result['verified_image_sha256_overlap'], [])
        self.assertEqual(result['raw_id_string_collisions_not_proof_of_same_image'], ['2993'])

    def test_gqa_tasks_share_image_namespace(self):
        a = [{'dataset': 'gqa_chooseattr', 'image_id': '001', 'image_sha256': 'same'}]
        b = [{'dataset': 'gqa_logicalobj', 'image_id': '001', 'image_sha256': 'same'}]
        result = image_overlap(a, b)
        self.assertEqual(result['same_namespace_image_ids'], [['GQA', '001']])
        self.assertEqual(result['verified_image_sha256_overlap'], ['same'])

    def test_independent_multitoken_full_vocabulary(self):
        self.assertAlmostEqual(logp(np.zeros((2, 5)), [1, 3]), -2*np.log(5))
        with self.assertRaises(ValueError):
            logp(np.zeros((2, 5)), [1])

    def test_zip_range_seek_and_pin(self):
        reader = RangeReader(Session(), 'https://official.example/archive.zip')
        reader.seek(-3, 2)
        self.assertEqual(reader.read(3), b'cde')
        self.assertEqual(reader.transferred_bytes, 3)

    def test_ignored_range_refuses_full_archive(self):
        reader = RangeReader(Session(ignore_range=True), 'https://official.example/archive.zip')
        with self.assertRaises(ValueError):
            reader.read(1)


if __name__ == '__main__':
    unittest.main()
